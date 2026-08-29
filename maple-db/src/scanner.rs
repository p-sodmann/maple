//! Background library scanner.
//!
//! Runs in a dedicated OS thread and wakes up every [`SCAN_INTERVAL`] seconds
//! to reconcile the library directory with the database:
//!
//! * Files in the DB that no longer exist on disk → marked `missing`.
//! * Files marked `missing` that have reappeared → marked `present`.
//! * Image groups found on disk that have no DB record → hashed and inserted.
//!
//! The scan uses [`scan_grouped_excluding`] so that application-internal
//! subdirectories (`aligned_faces`, `.thumbcache`, any hidden dir) are never
//! accidentally ingested as user photos.
//!
//! # Telling somebody
//!
//! A scan that changes the library and says nothing leaves whatever is on
//! screen wrong until the app is restarted — drop photos into `library_dir`
//! with the window open and the grid keeps showing what it read at startup.
//! [`LibraryScanner::on_change`] is the hook that closes that; `maple-ui`
//! points it at the grid. It fires **only when a scan actually changed
//! something**, because it runs every 60 seconds forever and a UI that
//! re-read the library that often for no reason would be worse than a stale
//! one.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_import::{content_hash, scan_grouped_excluding};

use maple_import::ImageGroup;

use crate::{Database, ImageStatus, ThumbnailCache};

/// Called after a scan that changed the library.
///
/// Runs on the scanner thread, so a UI caller must marshal
/// (`slint::Weak::upgrade_in_event_loop`).
pub type LibraryChanged = Arc<dyn Fn() + Send + Sync>;

const SCAN_INTERVAL: Duration = Duration::from_secs(60);

/// Subdirectory names inside `library_dir` that the scanner never ingests.
const EXCLUDED_DIRS: &[&str] = &["aligned_faces", ".thumbcache"];

/// When set, [`LibraryScanner::run_scan`] becomes a no-op for the duration.
///
/// Used to keep the periodic reconciliation from racing a library
/// restructure (`maple_ui::path_template_window`), which moves files and
/// updates their DB row one at a time — a scan firing mid-operation could
/// see a file "missing" at its old path and "new" at its new path before
/// the corresponding DB update lands, inserting a duplicate row.
static PAUSED: AtomicBool = AtomicBool::new(false);

/// Pause or resume the periodic scanner. There is only ever one
/// [`LibraryScanner`] per process, so this is process-global rather than
/// tied to a particular instance. Callers must not overlap pause/resume
/// spans (not reference-counted).
pub fn set_scanner_paused(paused: bool) {
    PAUSED.store(paused, Ordering::SeqCst);
}

pub struct LibraryScanner {
    db: Arc<Mutex<Database>>,
    library_dir: PathBuf,
    cache: Option<Arc<ThumbnailCache>>,
    on_change: Option<LibraryChanged>,
}

impl LibraryScanner {
    pub fn new(
        db: Arc<Mutex<Database>>,
        library_dir: PathBuf,
        cache: Option<Arc<ThumbnailCache>>,
    ) -> Self {
        Self { db, library_dir, cache, on_change: None }
    }

    /// Run `hook` after any scan that inserted a row or changed one's status.
    ///
    /// Also handed to the metadata filler this scanner spawns: the filler is
    /// what fills in `taken_at`, and a photo that appears undated and then
    /// silently re-sorts an hour later is not better than one that appears
    /// late.
    pub fn on_change(mut self, hook: LibraryChanged) -> Self {
        self.on_change = Some(hook);
        self
    }

    pub fn spawn(self) {
        std::thread::Builder::new()
            .name("maple-library-scanner".into())
            .spawn(move || {
                tracing::info!(
                    "Library scanner started, monitoring {}",
                    self.library_dir.display()
                );
                loop {
                    std::thread::sleep(SCAN_INTERVAL);
                    self.run_scan();
                }
            })
            .expect("Failed to spawn library scanner thread");
    }

    /// Reconcile the library directory against the database, once.
    ///
    /// Public so a test outside this crate can assert what a scan does *not*
    /// do — the failure mode worth guarding is a file the scanner adopts as a
    /// photograph nobody claimed, and that only shows up by running one.
    pub fn run_scan(&self) {
        if PAUSED.load(Ordering::SeqCst) {
            tracing::debug!("Library scan skipped: paused for a restructure");
            return;
        }

        let dir = &self.library_dir;
        if !dir.is_dir() {
            tracing::debug!(
                "Library dir {} does not exist yet, skipping scan",
                dir.display()
            );
            return;
        }

        tracing::info!("Library scan: reconciling {}", dir.display());

        // ── 1. Load all DB records ───────────────────────────────
        let db_records: Vec<crate::ImagePathStatusRow> =
            crate::lock_db(&self.db).all_paths().unwrap_or_default();
        // Both path columns, deliberately. A row's *status* is decided by its
        // display file alone (step 3 below), but a companion this library
        // already claims must never look like a file nobody claims — step 4
        // inserts one of those as a photograph of its own, and since a raw
        // groups as its own `ImageGroup` the duplicate then stamps and
        // replicates. See `Database::all_paths`.
        let db_path_set: HashSet<&PathBuf> = db_records
            .iter()
            .flat_map(|(path, _, _, raw)| std::iter::once(path).chain(raw.as_ref()))
            .collect();

        // ── 2. Scan library directory, skipping internal subdirs ──
        let groups = match scan_grouped_excluding(dir, EXCLUDED_DIRS) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("Library scan error scanning {}: {e}", dir.display());
                return;
            }
        };
        let found_map: HashMap<PathBuf, (u64, Option<PathBuf>)> =
            groups.into_iter().map(group_map_entry).collect();

        // ── 3. Reconcile DB records against disk ─────────────────
        let mut restatused = 0usize;
        for (path, status, hash, _) in &db_records {
            let on_disk = found_map.contains_key(path);
            match (on_disk, status) {
                (false, ImageStatus::Present) => {
                    tracing::info!("Library scan: marking missing {}", path.display());
                    if let (Some(cache), Some(hash)) = (&self.cache, hash) {
                        if let Err(e) = cache.remove(hash) {
                            tracing::warn!(
                                "Library scan: failed to evict thumbnail for {}: {e}",
                                path.display()
                            );
                        }
                    }
                    match crate::lock_db(&self.db).mark_missing(path) {
                        Ok(()) => restatused += 1,
                        Err(e) => tracing::warn!(
                            "Library scan: failed to mark missing {}: {e}",
                            path.display()
                        ),
                    }
                }
                (true, ImageStatus::Missing) => {
                    tracing::info!("Library scan: marking present {}", path.display());
                    match crate::lock_db(&self.db).mark_present(path) {
                        Ok(()) => restatused += 1,
                        Err(e) => tracing::warn!(
                            "Library scan: failed to mark present {}: {e}",
                            path.display()
                        ),
                    }
                }
                _ => {}
            }
        }

        // ── 4. Insert newly discovered groups ─────────────────────
        let mut inserted = 0usize;
        for (display_path, (size, raw_path)) in &found_map {
            if db_path_set.contains(display_path) {
                continue;
            }
            match content_hash(display_path) {
                Ok(hash) => {
                    let result = crate::lock_db(&self.db).insert_image_with_raw(
                        display_path,
                        &hash,
                        *size,
                        raw_path.as_deref(),
                    );
                    if let Err(e) = result {
                        tracing::warn!(
                            "Library scan: failed to insert {}: {e}",
                            display_path.display()
                        );
                    } else {
                        inserted += 1;
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Library scan: failed to hash {}: {e}",
                        display_path.display()
                    );
                }
            }
        }

        tracing::info!(
            "Library scan complete: {} DB records, {} groups on disk, {} newly inserted",
            db_records.len(),
            found_map.len(),
            inserted,
        );

        if inserted > 0 {
            crate::metadata::spawn_metadata_filler(self.db.clone(), self.on_change.clone());
        }

        // Last, and only if the library really moved: this runs every minute
        // for the life of the process, and the whole point of counting rather
        // than notifying unconditionally is that an idle library costs the UI
        // nothing at all.
        if inserted > 0 || restatused > 0 {
            if let Some(hook) = &self.on_change {
                hook();
            }
        }
    }
}

/// Convert an [`ImageGroup`] into the `(display_path, (size, raw_path))` entry
/// used in the scanner's `found_map`.
///
/// Only the first companion is stored in `raw_path` — the DB schema currently
/// has a single `raw_path TEXT` column.  When more than one companion is present
/// a warning is emitted so the data loss is never silent.  A schema migration
/// (separate `companions` table) will be needed before multi-companion support
/// (XMP sidecars, DNG+JPG) can land.
fn group_map_entry(g: ImageGroup) -> (PathBuf, (u64, Option<PathBuf>)) {
    if g.companions.len() > 1 {
        tracing::warn!(
            "{} has {} companions; only '{}' will be stored as raw_path — \
             remaining companions are dropped (schema supports one companion per image)",
            g.display.path.display(),
            g.companions.len(),
            g.companions[0].path.display(),
        );
    }
    let raw = g.companions.into_iter().next().map(|c| c.path);
    (g.display.path, (g.display.size, raw))
}

#[cfg(test)]
mod tests {
    use super::*;
    use maple_import::ImageFile;

    /// A library dir with one real photo in it, plus a database holding both
    /// that photo and one relayed from another device.
    fn library_with_a_remote_row() -> (tempfile::TempDir, Arc<Mutex<Database>>, [u8; 32]) {
        let dir = tempfile::tempdir().expect("tempdir");
        let lib = dir.path().join("library");
        std::fs::create_dir_all(&lib).expect("library dir");
        let local = lib.join("mine.jpg");
        std::fs::write(&local, b"\xff\xd8\xffnot-really-a-jpeg").expect("write");

        let db = Database::open(&dir.path().join("library.db")).expect("open");
        let local_hash = content_hash(&local).expect("hash");
        db.insert_image(&local, &local_hash, 4).expect("insert local");

        // The remote row: `path` is the *master's*, which is exactly why the
        // scanner must not look for it on this disk.
        let remote_hash = [0x5Au8; 32];
        db.conn
            .execute(
                "INSERT INTO images(path, hash, file_size, added_at, status, filename,
                                    guid, rev, rev_dev, locality, origin_device)
                 VALUES ('/workstation/photos/theirs.jpg', ?1, 9, 100, 'present',
                         'theirs.jpg', 'guid-remote', 1, 'dev-master', 'remote', 'dev-master')",
                rusqlite::params![remote_hash.as_slice()],
            )
            .expect("insert remote");

        (dir, Arc::new(Mutex::new(db)), remote_hash)
    }

    /// The whole point of the `all_paths` filter: a relayed photo has no file
    /// here, and a scanner that reconciled it would blank the grid one minute
    /// after it filled — and evict the thumbnail that made it usable.
    #[test]
    fn a_remote_row_survives_repeated_scans() {
        let (dir, db, remote_hash) = library_with_a_remote_row();
        let cache = Arc::new(
            ThumbnailCache::open(&dir.path().join("thumbs")).expect("thumbnail cache"),
        );
        cache.insert(&remote_hash, b"webp-bytes").expect("seed cache");

        let scanner = LibraryScanner::new(
            db.clone(),
            dir.path().join("library"),
            Some(cache.clone()),
        );
        scanner.run_scan();
        scanner.run_scan();

        let (status, locality): (String, String) = db
            .lock()
            .unwrap()
            .conn
            .query_row(
                "SELECT status, locality FROM images WHERE guid = 'guid-remote'",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("the remote row is still there");
        assert_eq!(status, "present", "a relayed photo is not a missing photo");
        assert_eq!(locality, "remote");
        assert!(
            cache.get(&remote_hash).is_some(),
            "its thumbnail must not have been evicted"
        );
    }

    /// ...while a local row that really did vanish is still reconciled.
    #[test]
    fn a_local_row_whose_file_vanished_is_still_marked_missing() {
        let (dir, db, _) = library_with_a_remote_row();
        std::fs::remove_file(dir.path().join("library/mine.jpg")).expect("delete");

        LibraryScanner::new(db.clone(), dir.path().join("library"), None).run_scan();

        let status: String = db
            .lock()
            .unwrap()
            .conn
            .query_row(
                "SELECT status FROM images WHERE filename = 'mine.jpg'",
                [],
                |r| r.get(0),
            )
            .expect("row");
        assert_eq!(status, "missing");
    }

    /// A companion this library already claims is never adopted as a photo of
    /// its own — even when it is nowhere near its display file.
    ///
    /// `place_pair` makes that separation impossible for a *newly* transferred
    /// photo, but a library that already diverged (or a user who moved a file
    /// by hand) still has orphans on disk, and the scanner regroups from disk
    /// by directory and stem: a lone RAF is its own `ImageGroup`. Building the
    /// known set from `path` alone left the second row nothing to collide
    /// with, so the scanner minted it, `insert_image_with_raw` stamped it, and
    /// sync replicated the ghost to every peer.
    #[test]
    fn an_orphaned_companion_is_not_adopted_as_a_second_photo() {
        let dir = tempfile::tempdir().expect("tempdir");
        let library = dir.path().join("library");
        std::fs::create_dir_all(library.join("2019/07")).expect("month");
        std::fs::create_dir_all(library.join("2026/08")).expect("other month");

        // The shape a diverged transfer leaves behind: one row, two files,
        // two directories.
        let display = library.join("2019/07/DSCF0001.JPG");
        let raw = library.join("2026/08/DSCF0001.RAF");
        std::fs::write(&display, b"\xff\xd8\xffthe display file").expect("write");
        std::fs::write(&raw, b"the negative").expect("write");

        let db = Database::open(&dir.path().join("library.db")).expect("open");
        let hash = content_hash(&display).expect("hash");
        db.insert_image_with_raw(&display, &hash, 4, Some(&raw))
            .expect("insert");
        let db = Arc::new(Mutex::new(db));

        LibraryScanner::new(db.clone(), library.clone(), None).run_scan();
        LibraryScanner::new(db.clone(), library, None).run_scan();

        assert_eq!(
            db.lock().unwrap().count().unwrap(),
            1,
            "the companion is already spoken for; it is not a new photograph"
        );
    }

    /// A scanner whose change hook counts its calls.
    fn counting(
        db: Arc<Mutex<Database>>,
        library: PathBuf,
    ) -> (LibraryScanner, Arc<std::sync::atomic::AtomicUsize>) {
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let hook = {
            let calls = calls.clone();
            Arc::new(move || {
                calls.fetch_add(1, Ordering::SeqCst);
            })
        };
        (LibraryScanner::new(db, library, None).on_change(hook), calls)
    }

    #[test]
    fn a_scan_that_found_nothing_new_says_nothing() {
        // This runs every 60 seconds for the life of the process. A hook that
        // fired unconditionally would have the grid re-read the library once
        // a minute forever, for a library that had not moved.
        let (dir, db, _) = library_with_a_remote_row();
        let (scanner, calls) = counting(db, dir.path().join("library"));

        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 0, "nothing changed");
    }

    #[test]
    fn a_scan_that_inserted_a_photo_says_so() {
        // The gap this exists for: drop photos into `library_dir` with the
        // window open, and until now the grid kept showing what it read at
        // startup.
        let (dir, db, _) = library_with_a_remote_row();
        let library = dir.path().join("library");
        let (scanner, calls) = counting(db, library.clone());

        std::fs::write(library.join("new.jpg"), b"\xff\xd8\xffnew").expect("write");
        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 1);

        // ...and once it has settled, it goes quiet again.
        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_photo_that_vanished_or_came_back_says_so() {
        let (dir, db, _) = library_with_a_remote_row();
        let library = dir.path().join("library");
        let (scanner, calls) = counting(db, library.clone());

        let photo = library.join("mine.jpg");
        let bytes = std::fs::read(&photo).expect("read");
        std::fs::remove_file(&photo).expect("delete");
        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 1, "marked missing");

        std::fs::write(&photo, &bytes).expect("restore");
        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "marked present again");

        scanner.run_scan();
        assert_eq!(calls.load(Ordering::SeqCst), 2, "and then quiet");
    }

    fn img(path: &str) -> ImageFile {
        ImageFile { path: PathBuf::from(path), size: 42 }
    }

    #[test]
    fn map_entry_no_companions() {
        let g = ImageGroup {
            display: img("/lib/DSCF0001.JPG"),
            companions: vec![],
        };
        let (path, (size, raw)) = group_map_entry(g);
        assert_eq!(path, PathBuf::from("/lib/DSCF0001.JPG"));
        assert_eq!(size, 42);
        assert!(raw.is_none());
    }

    #[test]
    fn map_entry_one_companion_stored() {
        let g = ImageGroup {
            display: img("/lib/DSCF0001.JPG"),
            companions: vec![img("/lib/DSCF0001.RAF")],
        };
        let (_, (_, raw)) = group_map_entry(g);
        assert_eq!(raw, Some(PathBuf::from("/lib/DSCF0001.RAF")));
    }

    #[test]
    fn map_entry_multiple_companions_keeps_first() {
        // Current schema supports only one companion per image.  The first
        // companion is kept; extras are dropped with a warning.  This test
        // pins the current behavior so any future multi-companion schema
        // change is deliberate.
        let g = ImageGroup {
            display: img("/lib/DSCF0001.JPG"),
            companions: vec![img("/lib/DSCF0001.RAF"), img("/lib/DSCF0001.XMP")],
        };
        let (_, (_, raw)) = group_map_entry(g);
        assert_eq!(raw, Some(PathBuf::from("/lib/DSCF0001.RAF")));
    }
}
