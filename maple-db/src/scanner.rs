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

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_import::{content_hash, scan_grouped_excluding};

use maple_import::ImageGroup;

use crate::{Database, ImageStatus, ThumbnailCache};

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
}

impl LibraryScanner {
    pub fn new(
        db: Arc<Mutex<Database>>,
        library_dir: PathBuf,
        cache: Option<Arc<ThumbnailCache>>,
    ) -> Self {
        Self { db, library_dir, cache }
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

    fn run_scan(&self) {
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
        let db_records: Vec<(PathBuf, ImageStatus, Option<[u8; 32]>)> =
            crate::lock_db(&self.db).all_paths().unwrap_or_default();
        let db_path_set: HashSet<&PathBuf> = db_records.iter().map(|(p, _, _)| p).collect();

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
        for (path, status, hash) in &db_records {
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
                    if let Err(e) = crate::lock_db(&self.db).mark_missing(path) {
                        tracing::warn!(
                            "Library scan: failed to mark missing {}: {e}",
                            path.display()
                        );
                    }
                }
                (true, ImageStatus::Missing) => {
                    tracing::info!("Library scan: marking present {}", path.display());
                    if let Err(e) = crate::lock_db(&self.db).mark_present(path) {
                        tracing::warn!(
                            "Library scan: failed to mark present {}: {e}",
                            path.display()
                        );
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
            crate::metadata::spawn_metadata_filler(self.db.clone());
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
