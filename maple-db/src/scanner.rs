//! Background library scanner.
//!
//! Runs in a dedicated OS thread and wakes up every `SCAN_INTERVAL` seconds
//! to reconcile the on-disk state of the library directory with the database:
//!
//! * Files in the DB that no longer exist on disk → marked `missing`.
//! * Files that are `missing` in the DB but have reappeared → marked `present`.
//! * Image files found on disk that have no DB record → hashed and inserted.

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_import::{content_hash, scan_images};

use crate::{Database, ImageStatus};

/// How long the scanner sleeps between reconciliation passes.
const SCAN_INTERVAL: Duration = Duration::from_secs(60);

pub struct LibraryScanner {
    db: Arc<Mutex<Database>>,
    library_dir: PathBuf,
}

impl LibraryScanner {
    pub fn new(db: Arc<Mutex<Database>>, library_dir: PathBuf) -> Self {
        Self { db, library_dir }
    }

    /// Spawn the scanner as a background thread.
    ///
    /// The thread sleeps for `SCAN_INTERVAL` between passes and is
    /// automatically killed when the process exits.
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
        let db_records: Vec<(PathBuf, ImageStatus)> = match self.db.lock() {
            Ok(db) => db.all_paths().unwrap_or_default(),
            Err(e) => {
                tracing::warn!("Library scan: DB lock poisoned: {e}");
                return;
            }
        };
        // Build a set of all paths known to the DB.
        let db_path_set: HashSet<&PathBuf> = db_records.iter().map(|(p, _)| p).collect();

        // ── 2. Scan the library directory on disk ────────────────
        let found = match scan_images(dir) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("Library scan error scanning {}: {e}", dir.display());
                return;
            }
        };
        // Map path → size for O(1) lookup when inserting new files.
        let found_map: HashMap<PathBuf, u64> =
            found.into_iter().map(|f| (f.path, f.size)).collect();

        // ── 3. Reconcile DB records against disk ─────────────────
        for (path, status) in &db_records {
            let on_disk = found_map.contains_key(path);
            match (on_disk, status) {
                (false, ImageStatus::Present) => {
                    tracing::info!("Library scan: marking missing {}", path.display());
                    if let Ok(db) = self.db.lock() {
                        let _ = db.mark_missing(path);
                    }
                }
                (true, ImageStatus::Missing) => {
                    tracing::info!("Library scan: marking present {}", path.display());
                    if let Ok(db) = self.db.lock() {
                        let _ = db.mark_present(path);
                    }
                }
                _ => {} // already consistent
            }
        }

        // ── 4. Insert newly discovered files ─────────────────────
        let mut inserted = 0usize;
        for (path, size) in &found_map {
            if db_path_set.contains(path) {
                continue;
            }
            match content_hash(path) {
                Ok(hash) => {
                    if let Ok(db) = self.db.lock() {
                        if let Err(e) = db.insert_image(path, &hash, *size) {
                            tracing::warn!(
                                "Library scan: failed to insert {}: {e}",
                                path.display()
                            );
                        } else {
                            inserted += 1;
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Library scan: failed to hash {}: {e}",
                        path.display()
                    );
                }
            }
        }

        tracing::info!(
            "Library scan complete: {} DB records, {} on disk, {} newly inserted",
            db_records.len(),
            found_map.len(),
            inserted,
        );
    }
}
