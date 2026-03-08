//! Copy selected images to the destination directory.

use std::path::{Path, PathBuf};

/// Result of a single file copy operation.
#[derive(Debug, Clone)]
pub enum CopyResult {
    /// File was copied successfully. Contains the destination path.
    Ok(PathBuf),
    /// File copy failed. Contains the source path and error message.
    Failed { source: PathBuf, error: String },
}

/// Summary of a batch copy operation.
#[derive(Debug, Clone)]
pub struct CopySummary {
    pub copied: usize,
    pub failed: usize,
    pub results: Vec<CopyResult>,
}

/// Copy the given source files into `destination`.
///
/// Files are placed directly in the destination directory (flat copy).
/// If a file with the same name already exists, a numeric suffix is
/// appended (e.g. `photo_1.jpg`, `photo_2.jpg`).
///
/// Calls `on_progress(copied_so_far, total)` after each file.
pub fn copy_images<F>(
    sources: &[PathBuf],
    destination: &Path,
    mut on_progress: F,
) -> anyhow::Result<CopySummary>
where
    F: FnMut(usize, usize),
{
    anyhow::ensure!(
        destination.is_dir(),
        "{} is not a directory",
        destination.display()
    );

    let total = sources.len();
    let mut results = Vec::with_capacity(total);
    let mut copied = 0usize;
    let mut failed = 0usize;

    for (i, src) in sources.iter().enumerate() {
        let dest_path = unique_dest_path(src, destination);
        match std::fs::copy(src, &dest_path) {
            Ok(_) => {
                tracing::info!("Copied {} → {}", src.display(), dest_path.display());
                results.push(CopyResult::Ok(dest_path));
                copied += 1;
            }
            Err(e) => {
                let msg = format!("{e}");
                tracing::warn!("Copy failed {} → {}: {e}", src.display(), dest_path.display());
                results.push(CopyResult::Failed {
                    source: src.clone(),
                    error: msg,
                });
                failed += 1;
            }
        }
        on_progress(i + 1, total);
    }

    Ok(CopySummary {
        copied,
        failed,
        results,
    })
}

/// Determine a destination path that does not collide with existing files.
///
/// Given `source = /photos/IMG_001.jpg` and `destination = /output/`,
/// returns `/output/IMG_001.jpg` if it doesn't exist, otherwise
/// `/output/IMG_001_1.jpg`, `/output/IMG_001_2.jpg`, etc.
fn unique_dest_path(source: &Path, destination: &Path) -> PathBuf {
    let file_name = source
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("file");
    let extension = source
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    let candidate = destination.join(source.file_name().unwrap_or_default());
    if !candidate.exists() {
        return candidate;
    }

    // Append _1, _2, … until we find a free name.
    for n in 1..u32::MAX {
        let name = if extension.is_empty() {
            format!("{file_name}_{n}")
        } else {
            format!("{file_name}_{n}.{extension}")
        };
        let candidate = destination.join(&name);
        if !candidate.exists() {
            return candidate;
        }
    }

    // Extremely unlikely fallback
    destination.join(source.file_name().unwrap_or_default())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn copy_images_to_destination() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        fs::write(src_dir.path().join("a.jpg"), b"image-a").unwrap();
        fs::write(src_dir.path().join("b.png"), b"image-b").unwrap();

        let sources = vec![
            src_dir.path().join("a.jpg"),
            src_dir.path().join("b.png"),
        ];

        let mut progress_calls = Vec::new();
        let summary = copy_images(&sources, dst_dir.path(), |done, total| {
            progress_calls.push((done, total));
        })
        .unwrap();

        assert_eq!(summary.copied, 2);
        assert_eq!(summary.failed, 0);
        assert!(dst_dir.path().join("a.jpg").exists());
        assert!(dst_dir.path().join("b.png").exists());
        assert_eq!(progress_calls, vec![(1, 2), (2, 2)]);
    }

    #[test]
    fn copy_avoids_name_collision() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        fs::write(src_dir.path().join("photo.jpg"), b"original").unwrap();
        // Pre-create a file in the destination with the same name.
        fs::write(dst_dir.path().join("photo.jpg"), b"existing").unwrap();

        let sources = vec![src_dir.path().join("photo.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), |_, _| {}).unwrap();

        assert_eq!(summary.copied, 1);
        // Original file in destination should be untouched.
        assert_eq!(fs::read(dst_dir.path().join("photo.jpg")).unwrap(), b"existing");
        // New copy should have a suffixed name.
        assert!(dst_dir.path().join("photo_1.jpg").exists());
        assert_eq!(
            fs::read(dst_dir.path().join("photo_1.jpg")).unwrap(),
            b"original"
        );
    }

    #[test]
    fn copy_invalid_destination_returns_error() {
        let sources = vec![PathBuf::from("/nonexistent/photo.jpg")];
        let result = copy_images(&sources, Path::new("/nonexistent/dir"), |_, _| {});
        assert!(result.is_err());
    }

    #[test]
    fn copy_missing_source_records_failure() {
        let dst_dir = tempfile::tempdir().unwrap();
        let sources = vec![PathBuf::from("/nonexistent/photo.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), |_, _| {}).unwrap();

        assert_eq!(summary.copied, 0);
        assert_eq!(summary.failed, 1);
        matches!(&summary.results[0], CopyResult::Failed { .. });
    }
}
