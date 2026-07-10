//! Library restructure — plan and execute moving already-imported files to
//! match a (possibly changed) destination-path template. Window controllers
//! call into this instead of locking the DB inline; see `services/mod.rs`.

use std::path::Path;
use std::sync::{Arc, Mutex};

use maple_import::{MoveResult, PlannedMove, RestructureSummary};

/// Compute the restructure plan for every present library image against
/// `folder_template`/`filename_template`. An empty result means every file
/// already sits where the template would put it.
pub fn plan(
    db: &Arc<Mutex<maple_db::Database>>,
    library_dir: &Path,
    folder_template: &str,
    filename_template: &str,
) -> Vec<PlannedMove> {
    let Ok(guard) = db.lock() else { return Vec::new() };
    let Ok(candidates) = guard.restructure_candidates() else { return Vec::new() };
    drop(guard);
    maple_import::plan_moves(&candidates, library_dir, folder_template, filename_template)
}

/// Pauses the periodic library scanner (`maple_db::set_scanner_paused`) for
/// its lifetime, resuming on drop (including on panic-unwind) so a
/// restructure can never leave it stuck paused.
struct ScannerPauseGuard;

impl ScannerPauseGuard {
    fn new() -> Self {
        maple_db::set_scanner_paused(true);
        Self
    }
}

impl Drop for ScannerPauseGuard {
    fn drop(&mut self) {
        maple_db::set_scanner_paused(false);
    }
}

/// Execute a previously computed plan and update each moved file's DB row.
///
/// Pauses the periodic library scanner for the duration — it reconciles DB
/// paths against disk, and a scan firing mid-restructure could see a file
/// "missing" at its old path before the corresponding DB update lands,
/// inserting a duplicate row. Calls `on_progress(done, total)` after each
/// planned move.
pub fn execute(
    db: &Arc<Mutex<maple_db::Database>>,
    planned: &[PlannedMove],
    on_progress: impl FnMut(usize, usize),
) -> RestructureSummary {
    let _guard = ScannerPauseGuard::new();
    let summary = maple_import::execute_moves(planned, on_progress);

    if let Ok(guard) = db.lock() {
        for result in &summary.results {
            if let MoveResult::Moved { id, new_path, new_raw_path, new_filename } = result {
                if let Err(e) =
                    guard.update_image_location(*id, new_path, new_raw_path.as_deref(), new_filename)
                {
                    tracing::warn!("restructure: failed to update DB row {id}: {e}");
                }
            }
        }
    }

    summary
}
