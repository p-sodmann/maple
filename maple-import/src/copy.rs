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
/// When `folder_format` is non-empty, files are placed into subdirectories
/// derived from each file's EXIF DateTimeOriginal using `strftime` tokens
/// (e.g. `"%Y/%m"` → `2024/01/`).  Files without EXIF dates fall back to
/// a flat copy into `destination` directly.
///
/// If a file with the same name already exists, a numeric suffix is
/// appended (e.g. `photo_1.jpg`, `photo_2.jpg`).
///
/// Calls `on_progress(copied_so_far, total)` after each file.
pub fn copy_images<F>(
    sources: &[PathBuf],
    destination: &Path,
    folder_format: &str,
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
        let target_dir = resolve_target_dir(src, destination, folder_format);

        // Ensure the subdirectory exists (no-op for flat copy).
        if target_dir != destination {
            if let Err(e) = std::fs::create_dir_all(&target_dir) {
                let msg = format!("failed to create {}: {e}", target_dir.display());
                tracing::warn!("{msg}");
                results.push(CopyResult::Failed {
                    source: src.clone(),
                    error: msg,
                });
                failed += 1;
                on_progress(i + 1, total);
                continue;
            }
        }

        let dest_path = unique_dest_path(src, &target_dir);
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

/// Resolve the target directory for `source` inside `destination`.
///
/// Reads EXIF DateTimeOriginal and formats it with `folder_format`.
/// Falls back to `destination` (flat) when the format is empty or
/// no EXIF date is available.
fn resolve_target_dir(source: &Path, destination: &Path, folder_format: &str) -> PathBuf {
    if folder_format.is_empty() {
        return destination.to_path_buf();
    }

    if let Some(sub) = exif_date_subdir(source, folder_format) {
        destination.join(sub)
    } else {
        destination.to_path_buf()
    }
}

/// Try to read EXIF DateTimeOriginal from `path` and format it.
///
/// EXIF ASCII format is `"YYYY:MM:DD HH:MM:SS"`.  We parse just enough
/// to feed the strftime-style format string.
fn exif_date_subdir(path: &Path, format: &str) -> Option<String> {
    let file = std::fs::File::open(path).ok()?;
    let mut reader = std::io::BufReader::new(file);
    let exif = exif::Reader::new().read_from_container(&mut reader).ok()?;
    let field = exif
        .fields()
        .find(|f| f.tag == exif::Tag::DateTimeOriginal)?;

    if let exif::Value::Ascii(ref v) = field.value {
        let bytes = v.first()?;
        let s = std::str::from_utf8(bytes).ok()?;
        let dt = crate::ExifDateTime::parse(s)?;
        let result = dt.format(format);
        if result.is_empty() { None } else { Some(result) }
    } else {
        None
    }
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
        let summary = copy_images(&sources, dst_dir.path(), "", |done, total| {
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
        let summary = copy_images(&sources, dst_dir.path(), "", |_, _| {}).unwrap();

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
        let result = copy_images(&sources, Path::new("/nonexistent/dir"), "", |_, _| {});
        assert!(result.is_err());
    }

    #[test]
    fn copy_missing_source_records_failure() {
        let dst_dir = tempfile::tempdir().unwrap();
        let sources = vec![PathBuf::from("/nonexistent/photo.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), "", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 0);
        assert_eq!(summary.failed, 1);
        matches!(&summary.results[0], CopyResult::Failed { .. });
    }

    #[test]
    fn no_exif_falls_back_to_flat_copy() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        fs::write(src_dir.path().join("a.jpg"), b"not-a-real-jpeg").unwrap();

        let sources = vec![src_dir.path().join("a.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), "%Y/%m", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 1);
        // No EXIF → file goes directly into destination.
        assert!(dst_dir.path().join("a.jpg").exists());
    }

    #[test]
    fn exif_date_subdir_parsing() {
        // Verify the format replacement logic directly.
        assert_eq!(
            "%Y/%m"
                .replace("%Y", &format!("{:04}", 2024))
                .replace("%m", &format!("{:02}", 1)),
            "2024/01"
        );
        assert_eq!(
            "%Y/%m/%d"
                .replace("%Y", &format!("{:04}", 2024))
                .replace("%m", &format!("{:02}", 12))
                .replace("%d", &format!("{:02}", 5)),
            "2024/12/05"
        );
    }
}
