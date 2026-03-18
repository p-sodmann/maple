//! Background library scanner.
//!
//! Runs in a dedicated OS thread and wakes up every `SCAN_INTERVAL` seconds
//! to reconcile the on-disk state of the library directory with the database:
//!
//! * Files in the DB that no longer exist on disk → marked `missing`.
//! * Files that are `missing` in the DB but have reappeared → marked `present`.
//! * Image groups found on disk that have no DB record → hashed and inserted
//!   (one row per group, with `raw_path` set for any companion raw file).

use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_import::{content_hash, scan_grouped};

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
        // Build a set of all display paths known to the DB.
        let db_path_set: HashSet<&PathBuf> = db_records.iter().map(|(p, _)| p).collect();

        // ── 2. Scan the library directory on disk (grouped) ───────
        let groups = match scan_grouped(dir) {
            Ok(g) => g,
            Err(e) => {
                tracing::warn!("Library scan error scanning {}: {e}", dir.display());
                return;
            }
        };
        // Map display_path → (size, optional raw_path) for quick lookup.
        let found_map: HashMap<PathBuf, (u64, Option<PathBuf>)> = groups
            .into_iter()
            .map(|g| {
                let raw = g.companions.first().map(|c| c.path.clone());
                (g.display.path, (g.display.size, raw))
            })
            .collect();

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

        // ── 4. Insert newly discovered groups ────────────────────
        let mut inserted = 0usize;
        for (display_path, (size, raw_path)) in &found_map {
            if db_path_set.contains(display_path) {
                continue;
            }
            match content_hash(display_path) {
                Ok(hash) => {
                    if let Ok(db) = self.db.lock() {
                        let result = db.insert_image_with_raw(
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
