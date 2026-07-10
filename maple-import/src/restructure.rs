//! Move already-imported library files to match a (possibly new) path
//! template.
//!
//! Two-phase, mirroring [`crate::copy`]'s style: [`plan_moves`] is a pure
//! planning pass over metadata already known to the caller (no filesystem
//! writes — collisions are resolved in-memory), and [`execute_moves`]
//! performs the actual renames. Splitting the two lets a caller show a
//! "N files would move" confirmation before touching anything.

use std::collections::HashSet;
use std::path::{Path, PathBuf};

use crate::path_template::{self, TemplateContext};
use crate::ExifDateTime;

/// One image already in the library, as known from the database — no
/// filesystem/EXIF re-read is needed to plan its target path.
pub struct RestructureCandidate {
    pub id: i64,
    pub current_path: PathBuf,
    /// Companion RAW file, if any (see `raw_path` in the `images` table).
    pub current_raw_path: Option<PathBuf>,
    pub datetime: Option<ExifDateTime>,
    pub camera: Option<String>,
}

/// A single planned move: `id`'s current location and its resolved new one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedMove {
    pub id: i64,
    pub current_path: PathBuf,
    pub current_raw_path: Option<PathBuf>,
    pub new_path: PathBuf,
    pub new_raw_path: Option<PathBuf>,
    /// New file name (with extension) for the `images.filename` column.
    pub new_filename: String,
}

/// Compute which candidates need to move to match `folder_template` /
/// `filename_template`, and where.
///
/// Candidates that already sit at their template-derived location are
/// omitted from the result — so an empty return means "nothing to do",
/// which callers use to skip the confirmation prompt entirely.
///
/// Collisions (two candidates rendering to the same target, or a target
/// colliding with a library file that isn't moving) are resolved with the
/// same numeric-suffix scheme `copy_images` uses, but against an in-memory
/// set of paths rather than the filesystem — nothing here reads or writes
/// disk state.
pub fn plan_moves(
    candidates: &[RestructureCandidate],
    library_dir: &Path,
    folder_template: &str,
    filename_template: &str,
) -> Vec<PlannedMove> {
    let mut claimed: HashSet<PathBuf> = candidates
        .iter()
        .flat_map(|c| {
            std::iter::once(c.current_path.clone()).chain(c.current_raw_path.clone())
        })
        .collect();

    let mut planned = Vec::new();
    for (i, c) in candidates.iter().enumerate() {
        let original_stem = c
            .current_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("file");
        let extension = c
            .current_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("");

        let datetime = c.datetime.or_else(|| crate::copy::mtime_fallback(&c.current_path));
        let ctx = TemplateContext {
            datetime,
            original_stem,
            counter: i + 1,
            camera: c.camera.as_deref(),
        };

        let target_dir = if folder_template.is_empty() {
            library_dir.to_path_buf()
        } else {
            library_dir.join(path_template::render_folder(folder_template, &ctx))
        };

        let stem = path_template::render_filename_stem(filename_template, &ctx);
        let stem = if stem.is_empty() { original_stem.to_owned() } else { stem };

        let new_path = unique_path(&stem, extension, &target_dir, &claimed, &c.current_path);

        if new_path == c.current_path {
            continue;
        }

        claimed.remove(&c.current_path);
        claimed.insert(new_path.clone());

        let new_raw_path = c.current_raw_path.as_ref().map(|raw| {
            let raw_ext = raw.extension().and_then(|e| e.to_str()).unwrap_or("");
            target_dir.join(file_name(&stem, raw_ext))
        });
        if let Some(raw) = &new_raw_path {
            claimed.insert(raw.clone());
        }

        let new_filename = new_path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or(&stem)
            .to_owned();

        planned.push(PlannedMove {
            id: c.id,
            current_path: c.current_path.clone(),
            current_raw_path: c.current_raw_path.clone(),
            new_path,
            new_raw_path,
            new_filename,
        });
    }

    planned
}

/// Resolve `stem.extension` inside `dir` to a path not present in `claimed`,
/// suffixing with `_1`, `_2`, … on collision. `own_current_path` is always
/// accepted even if `claimed` contains it (it's this candidate's own,
/// about-to-be-vacated, current location).
fn unique_path(
    stem: &str,
    extension: &str,
    dir: &Path,
    claimed: &HashSet<PathBuf>,
    own_current_path: &Path,
) -> PathBuf {
    let candidate = dir.join(file_name(stem, extension));
    if candidate == own_current_path || !claimed.contains(&candidate) {
        return candidate;
    }

    for n in 1..u32::MAX {
        let candidate = dir.join(file_name(&format!("{stem}_{n}"), extension));
        if candidate == own_current_path || !claimed.contains(&candidate) {
            return candidate;
        }
    }

    // Extremely unlikely fallback.
    dir.join(file_name(stem, extension))
}

fn file_name(stem: &str, extension: &str) -> String {
    if extension.is_empty() {
        stem.to_owned()
    } else {
        format!("{stem}.{extension}")
    }
}

/// Outcome of a single planned move.
#[derive(Debug, Clone)]
pub enum MoveResult {
    Moved {
        id: i64,
        new_path: PathBuf,
        new_raw_path: Option<PathBuf>,
        new_filename: String,
    },
    Failed {
        id: i64,
        current_path: PathBuf,
        error: String,
    },
}

/// Summary of a batch restructure operation.
#[derive(Debug, Clone, Default)]
pub struct RestructureSummary {
    pub moved: usize,
    pub failed: usize,
    pub results: Vec<MoveResult>,
}

/// Execute a previously computed move plan.
///
/// Each move creates its target directory, renames the display file (and its
/// RAW companion, if any) into place, and calls `on_progress(done, total)`.
/// A failed move leaves that file untouched — the caller should skip the DB
/// update for it and keep tracking it at `current_path`.
pub fn execute_moves<F>(planned: &[PlannedMove], mut on_progress: F) -> RestructureSummary
where
    F: FnMut(usize, usize),
{
    let total = planned.len();
    let mut moved = 0usize;
    let mut failed = 0usize;
    let mut results = Vec::with_capacity(total);

    for (i, mv) in planned.iter().enumerate() {
        let outcome = move_one(mv);
        match outcome {
            Ok(()) => {
                tracing::info!(
                    "Restructured {} → {}",
                    mv.current_path.display(),
                    mv.new_path.display()
                );
                results.push(MoveResult::Moved {
                    id: mv.id,
                    new_path: mv.new_path.clone(),
                    new_raw_path: mv.new_raw_path.clone(),
                    new_filename: mv.new_filename.clone(),
                });
                moved += 1;
            }
            Err(error) => {
                tracing::warn!(
                    "Restructure failed {} → {}: {error}",
                    mv.current_path.display(),
                    mv.new_path.display()
                );
                results.push(MoveResult::Failed {
                    id: mv.id,
                    current_path: mv.current_path.clone(),
                    error,
                });
                failed += 1;
            }
        }
        on_progress(i + 1, total);
    }

    RestructureSummary { moved, failed, results }
}

fn move_one(mv: &PlannedMove) -> Result<(), String> {
    if let Some(parent) = mv.new_path.parent() {
        std::fs::create_dir_all(parent)
            .map_err(|e| format!("failed to create {}: {e}", parent.display()))?;
    }
    move_file(&mv.current_path, &mv.new_path)?;
    if let (Some(old_raw), Some(new_raw)) = (&mv.current_raw_path, &mv.new_raw_path) {
        move_file(old_raw, new_raw)?;
    }
    Ok(())
}

/// Rename `from` to `to`, falling back to copy+remove on cross-device
/// errors (or any other rename failure).
fn move_file(from: &Path, to: &Path) -> Result<(), String> {
    if std::fs::rename(from, to).is_ok() {
        return Ok(());
    }
    std::fs::copy(from, to).map_err(|e| format!("{e}"))?;
    std::fs::remove_file(from).map_err(|e| format!("{e}"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn candidate(id: i64, path: &Path) -> RestructureCandidate {
        RestructureCandidate {
            id,
            current_path: path.to_path_buf(),
            current_raw_path: None,
            datetime: ExifDateTime::parse("2024:03:15 14:30:45"),
            camera: None,
        }
    }

    #[test]
    fn plan_skips_candidates_already_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("2024").join("03").join("a.jpg");
        let candidates = vec![candidate(1, &path)];

        let planned = plan_moves(&candidates, dir.path(), "{YYYY}/{MM}", "{original}");
        assert!(planned.is_empty());
    }

    #[test]
    fn plan_moves_file_to_new_folder() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jpg");
        let candidates = vec![candidate(1, &path)];

        let planned = plan_moves(&candidates, dir.path(), "{YYYY}/{MM}", "{original}");
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0].new_path, dir.path().join("2024").join("03").join("a.jpg"));
        assert_eq!(planned[0].new_filename, "a.jpg");
    }

    #[test]
    fn plan_suffixes_colliding_targets() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.jpg");
        let b = dir.path().join("b.jpg");
        let candidates = vec![candidate(1, &a), candidate(2, &b)];

        // Both render to the same stem under a sequential template.
        let planned = plan_moves(&candidates, dir.path(), "", "photo");
        assert_eq!(planned.len(), 2);
        assert_eq!(planned[0].new_path, dir.path().join("photo.jpg"));
        assert_eq!(planned[1].new_path, dir.path().join("photo_1.jpg"));
    }

    #[test]
    fn plan_moves_raw_companion_alongside_display() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jpg");
        let mut c = candidate(1, &path);
        c.current_raw_path = Some(dir.path().join("a.RAF"));
        let candidates = vec![c];

        let planned = plan_moves(&candidates, dir.path(), "{YYYY}", "{original}");
        assert_eq!(planned.len(), 1);
        assert_eq!(
            planned[0].new_raw_path,
            Some(dir.path().join("2024").join("a.RAF"))
        );
    }

    #[test]
    fn execute_moves_renames_file_and_updates_progress() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.jpg"), b"image-a").unwrap();

        let planned = vec![PlannedMove {
            id: 1,
            current_path: dir.path().join("a.jpg"),
            current_raw_path: None,
            new_path: dir.path().join("2024").join("a.jpg"),
            new_raw_path: None,
            new_filename: "a.jpg".into(),
        }];

        let mut progress_calls = Vec::new();
        let summary = execute_moves(&planned, |done, total| progress_calls.push((done, total)));

        assert_eq!(summary.moved, 1);
        assert_eq!(summary.failed, 0);
        assert!(!dir.path().join("a.jpg").exists());
        assert!(dir.path().join("2024").join("a.jpg").exists());
        assert_eq!(progress_calls, vec![(1, 1)]);
    }

    #[test]
    fn execute_moves_raw_companion_too() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.jpg"), b"image-a").unwrap();
        fs::write(dir.path().join("a.RAF"), b"raw-a").unwrap();

        let planned = vec![PlannedMove {
            id: 1,
            current_path: dir.path().join("a.jpg"),
            current_raw_path: Some(dir.path().join("a.RAF")),
            new_path: dir.path().join("2024").join("a.jpg"),
            new_raw_path: Some(dir.path().join("2024").join("a.RAF")),
            new_filename: "a.jpg".into(),
        }];

        let summary = execute_moves(&planned, |_, _| {});
        assert_eq!(summary.moved, 1);
        assert!(dir.path().join("2024").join("a.jpg").exists());
        assert!(dir.path().join("2024").join("a.RAF").exists());
    }

    #[test]
    fn execute_moves_reports_failure_for_missing_source() {
        let dir = tempfile::tempdir().unwrap();
        let planned = vec![PlannedMove {
            id: 1,
            current_path: dir.path().join("missing.jpg"),
            current_raw_path: None,
            new_path: dir.path().join("2024").join("missing.jpg"),
            new_raw_path: None,
            new_filename: "missing.jpg".into(),
        }];

        let summary = execute_moves(&planned, |_, _| {});
        assert_eq!(summary.moved, 0);
        assert_eq!(summary.failed, 1);
        assert!(matches!(&summary.results[0], MoveResult::Failed { .. }));
    }
}
