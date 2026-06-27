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
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_import::{content_hash, scan_grouped_excluding};

use maple_import::ImageGroup;

use crate::{Database, ImageStatus, ThumbnailCache};

const SCAN_INTERVAL: Duration = Duration::from_secs(60);

/// Subdirectory names inside `library_dir` that the scanner never ingests.
const EXCLUDED_DIRS: &[&str] = &["aligned_faces", ".thumbcache"];

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
