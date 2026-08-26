//! Copy selected images to the destination directory.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::path_template::{self, TemplateContext};
use crate::ExifDateTime;

/// Result of a single file copy operation.
#[derive(Debug, Clone)]
pub enum CopyResult {
    /// File was copied successfully.
    ///
    /// Carries `source` as well as `dest` so callers can correlate the two
    /// without relying on positional alignment with the input slice — the
    /// library DB records the *destination*, but everything the import UI
    /// knows about a photo (its content hash, its raw companion) is keyed by
    /// the source path it was scanned under.
    Ok { source: PathBuf, dest: PathBuf },
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

impl CopySummary {
    /// Map every successfully copied source path to where it landed.
    ///
    /// Failed copies are absent, so a lookup miss means "this file never
    /// made it to the destination" — which is exactly when the caller should
    /// skip inserting a library row for it.
    pub fn destination_map(&self) -> HashMap<PathBuf, PathBuf> {
        self.results
            .iter()
            .filter_map(|r| match r {
                CopyResult::Ok { source, dest } => Some((source.clone(), dest.clone())),
                CopyResult::Failed { .. } => None,
            })
            .collect()
    }
}

/// Copy the given source files into `destination`.
///
/// `folder_template` and `filename_template` use `{TOKEN}` placeholders
/// (see [`crate::path_template`]) resolved from each file's EXIF
/// `DateTimeOriginal` (falling back to its filesystem mtime when absent).
/// An empty `folder_template` copies flat into `destination`. The source
/// file's extension is always preserved regardless of `filename_template`.
///
/// If the resulting filename already exists at the target, a numeric
/// suffix is appended (e.g. `photo_1.jpg`, `photo_2.jpg`).
///
/// Calls `on_progress(copied_so_far, total)` after each file.
pub fn copy_images<F>(
    sources: &[PathBuf],
    destination: &Path,
    folder_template: &str,
    filename_template: &str,
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
        let original_stem = src
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("file");
        let extension = src.extension().and_then(|e| e.to_str()).unwrap_or("");

        let ctx = crate::exif_read::read(src);
        let (datetime, camera) = (ctx.datetime, ctx.camera);
        let datetime = datetime.or_else(|| mtime_fallback(src));
        let ctx = TemplateContext {
            datetime,
            original_stem,
            counter: i + 1,
            camera: camera.as_deref(),
        };

        let target_dir = if folder_template.is_empty() {
            destination.to_path_buf()
        } else {
            destination.join(path_template::render_folder(folder_template, &ctx))
        };

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

        let stem = path_template::render_filename_stem(filename_template, &ctx);
        let stem = if stem.is_empty() { original_stem.to_owned() } else { stem };
        let dest_path = unique_dest_path(&stem, extension, &target_dir);

        match std::fs::copy(src, &dest_path) {
            Ok(_) => {
                tracing::info!("Copied {} → {}", src.display(), dest_path.display());
                results.push(CopyResult::Ok {
                    source: src.clone(),
                    dest: dest_path,
                });
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

/// Move one already-materialised file into a library, organised by the same
/// templates a card import uses.
///
/// This is the seam sync needs and [`copy_images`] cannot provide. A
/// downloaded original arrives as bytes with a *sender's* filename attached
/// and no path of its own; the caller stages it somewhere (verifying its
/// hash while it does), and this puts it where the library's own rules say it
/// belongs. Doing it any other way — a flat drop, or a path echoing the
/// sender's folders — would leave two machines in "full" mode organised
/// differently, which is the one thing §3.8 asks for.
///
/// `original_name` is the *sender's* file name, extension included: it
/// supplies the `{original}` token and the extension the copy keeps. The
/// date and camera tokens are read from `staged` itself, so a photo files
/// under the day it was taken on both machines rather than the day it
/// happened to arrive.
///
/// `staged` is **moved**, not copied — a rename within the library directory
/// is atomic and does not double the disk cost of a large raw file. It falls
/// back to copy-and-remove when the two are on different filesystems, which
/// is what a caller staging in a system temp dir will hit.
///
/// The `{counter}` token renders as 1: there is no batch here, only ever one
/// file. A template that leans entirely on it collides, and
/// [`unique_dest_path`] resolves that the same way it does for an import.
pub fn place_file(
    staged: &Path,
    destination: &Path,
    folder_template: &str,
    filename_template: &str,
    original_name: &str,
) -> anyhow::Result<PathBuf> {
    let original = Path::new(original_name);
    let original_stem = original
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("file");
    let extension = original.extension().and_then(|e| e.to_str()).unwrap_or("");

    let ctx = crate::exif_read::read(staged);
    let (datetime, camera) = (ctx.datetime, ctx.camera);
    let datetime = datetime.or_else(|| mtime_fallback(staged));
    let ctx = TemplateContext {
        datetime,
        original_stem,
        counter: 1,
        camera: camera.as_deref(),
    };

    let target_dir = if folder_template.is_empty() {
        destination.to_path_buf()
    } else {
        destination.join(path_template::render_folder(folder_template, &ctx))
    };
    std::fs::create_dir_all(&target_dir)
        .map_err(|e| anyhow::anyhow!("failed to create {}: {e}", target_dir.display()))?;

    let stem = path_template::render_filename_stem(filename_template, &ctx);
    let stem = if stem.is_empty() { original_stem.to_owned() } else { stem };
    let dest = unique_dest_path(&stem, extension, &target_dir);

    if std::fs::rename(staged, &dest).is_err() {
        // Different filesystems, most likely. Copy first and only then drop
        // the staged file, so a failure here leaves the caller's bytes intact
        // rather than losing a download that has already been verified.
        std::fs::copy(staged, &dest).map_err(|e| {
            anyhow::anyhow!("failed to place {} at {}: {e}", staged.display(), dest.display())
        })?;
        if let Err(e) = std::fs::remove_file(staged) {
            tracing::warn!("could not remove staged file {}: {e}", staged.display());
        }
    }
    tracing::info!("Placed {} → {}", original_name, dest.display());
    Ok(dest)
}

/// Fall back to the file's mtime when no EXIF date is available.
///
/// Also reused by [`crate::restructure`] so a library restructure resolves
/// missing dates identically to a fresh import.
pub(crate) fn mtime_fallback(path: &Path) -> Option<ExifDateTime> {
    let metadata = std::fs::metadata(path).ok()?;
    let modified = metadata.modified().ok()?;
    let secs = modified
        .duration_since(std::time::UNIX_EPOCH)
        .ok()?
        .as_secs() as i64;
    Some(ExifDateTime::from_unix_timestamp(secs))
}

/// Determine a destination path that does not collide with existing files.
///
/// Given `stem = "photo"`, `extension = "jpg"`, `destination = /output/`,
/// returns `/output/photo.jpg` if it doesn't exist, otherwise
/// `/output/photo_1.jpg`, `/output/photo_2.jpg`, etc.
fn unique_dest_path(stem: &str, extension: &str, destination: &Path) -> PathBuf {
    let file_name = |name: &str| -> String {
        if extension.is_empty() {
            name.to_owned()
        } else {
            format!("{name}.{extension}")
        }
    };

    let candidate = destination.join(file_name(stem));
    if !candidate.exists() {
        return candidate;
    }

    // Append _1, _2, … until we find a free name.
    for n in 1..u32::MAX {
        let candidate = destination.join(file_name(&format!("{stem}_{n}")));
        if !candidate.exists() {
            return candidate;
        }
    }

    // Extremely unlikely fallback
    destination.join(file_name(stem))
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
        let summary = copy_images(&sources, dst_dir.path(), "", "{original}", |done, total| {
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
    fn place_file_moves_the_staged_file_under_the_template() {
        let staged_dir = tempfile::tempdir().unwrap();
        let lib = tempfile::tempdir().unwrap();
        let staged = staged_dir.path().join("blob.part");
        fs::write(&staged, b"downloaded bytes").unwrap();
        // No EXIF in those bytes, so the mtime fallback supplies the date —
        // which is why the assertion below is on the name, not the folder.
        let dest = place_file(&staged, lib.path(), "", "{original}", "DSCF0001.JPG").unwrap();

        assert_eq!(dest, lib.path().join("DSCF0001.JPG"));
        assert_eq!(fs::read(&dest).unwrap(), b"downloaded bytes");
        assert!(!staged.exists(), "the staged file is moved, not copied");
    }

    #[test]
    fn place_file_keeps_the_senders_extension_and_avoids_collisions() {
        let staged_dir = tempfile::tempdir().unwrap();
        let lib = tempfile::tempdir().unwrap();
        fs::write(lib.path().join("DSCF0001.RAF"), b"already here").unwrap();

        let staged = staged_dir.path().join("blob.part");
        fs::write(&staged, b"the raw file").unwrap();
        let dest = place_file(&staged, lib.path(), "", "{original}", "DSCF0001.RAF").unwrap();

        assert_eq!(dest, lib.path().join("DSCF0001_1.RAF"));
        assert_eq!(fs::read(lib.path().join("DSCF0001.RAF")).unwrap(), b"already here");
    }

    #[test]
    fn place_file_creates_the_folders_the_template_asks_for() {
        let staged_dir = tempfile::tempdir().unwrap();
        let lib = tempfile::tempdir().unwrap();
        let staged = staged_dir.path().join("blob.part");
        fs::write(&staged, b"bytes").unwrap();

        // The mtime of a file written a moment ago is "now", so the year
        // folder is whatever year it is — assert on the shape, not a literal.
        let dest = place_file(&staged, lib.path(), "{YYYY}/{MM}", "{original}", "a.jpg").unwrap();
        let relative = dest.strip_prefix(lib.path()).unwrap();
        let parts: Vec<_> = relative.components().collect();
        assert_eq!(parts.len(), 3, "{}", relative.display());
        assert!(dest.exists());
    }

    #[test]
    fn copy_avoids_name_collision() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        fs::write(src_dir.path().join("photo.jpg"), b"original").unwrap();
        // Pre-create a file in the destination with the same name.
        fs::write(dst_dir.path().join("photo.jpg"), b"existing").unwrap();

        let sources = vec![src_dir.path().join("photo.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), "", "{original}", |_, _| {}).unwrap();

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
    fn destination_map_pairs_each_source_with_where_it_landed() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        // A raw + display pair, plus a name that collides in the destination
        // so the mapping cannot be inferred from the filename alone.
        fs::write(src_dir.path().join("photo.jpg"), b"display").unwrap();
        fs::write(src_dir.path().join("photo.raf"), b"raw").unwrap();
        fs::write(dst_dir.path().join("photo.jpg"), b"existing").unwrap();

        let jpg = src_dir.path().join("photo.jpg");
        let raf = src_dir.path().join("photo.raf");
        let missing = PathBuf::from("/nonexistent/photo.jpg");

        let sources = vec![jpg.clone(), raf.clone(), missing.clone()];
        let summary =
            copy_images(&sources, dst_dir.path(), "", "{original}", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 2);
        assert_eq!(summary.failed, 1);

        let map = summary.destination_map();
        // The display file was renamed around the collision — the map has to
        // report the suffixed name, not the source's basename.
        assert_eq!(map.get(&jpg).unwrap(), &dst_dir.path().join("photo_1.jpg"));
        assert_eq!(map.get(&raf).unwrap(), &dst_dir.path().join("photo.raf"));
        // Failed copies are absent, so callers skip them rather than
        // recording a library row for a file that isn't there.
        assert!(!map.contains_key(&missing));
        assert_eq!(map.len(), 2);
    }

    #[test]
    fn copy_invalid_destination_returns_error() {
        let sources = vec![PathBuf::from("/nonexistent/photo.jpg")];
        let result = copy_images(&sources, Path::new("/nonexistent/dir"), "", "{original}", |_, _| {});
        assert!(result.is_err());
    }

    #[test]
    fn copy_missing_source_records_failure() {
        let dst_dir = tempfile::tempdir().unwrap();
        let sources = vec![PathBuf::from("/nonexistent/photo.jpg")];
        let summary = copy_images(&sources, dst_dir.path(), "", "{original}", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 0);
        assert_eq!(summary.failed, 1);
        matches!(&summary.results[0], CopyResult::Failed { .. });
    }

    #[test]
    fn no_exif_folder_template_falls_back_to_mtime() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        let src_path = src_dir.path().join("a.jpg");
        fs::write(&src_path, b"not-a-real-jpeg").unwrap();
        let mtime = fs::metadata(&src_path).unwrap().modified().unwrap();
        let secs = mtime
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs() as i64;
        let expected_year = format!("{:04}", ExifDateTime::from_unix_timestamp(secs).year);

        let sources = vec![src_path];
        let summary = copy_images(&sources, dst_dir.path(), "{YYYY}", "{original}", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 1);
        // No EXIF date, but mtime fallback still produces a real year subdir.
        assert!(dst_dir.path().join(&expected_year).join("a.jpg").exists());
    }

    #[test]
    fn custom_filename_template_renames_files() {
        let src_dir = tempfile::tempdir().unwrap();
        let dst_dir = tempfile::tempdir().unwrap();

        fs::write(src_dir.path().join("a.jpg"), b"image-a").unwrap();
        fs::write(src_dir.path().join("b.jpg"), b"image-b").unwrap();

        let sources = vec![
            src_dir.path().join("a.jpg"),
            src_dir.path().join("b.jpg"),
        ];
        let summary = copy_images(&sources, dst_dir.path(), "", "photo_{counter}", |_, _| {}).unwrap();

        assert_eq!(summary.copied, 2);
        assert!(dst_dir.path().join("photo_0001.jpg").exists());
        assert!(dst_dir.path().join("photo_0002.jpg").exists());
    }
}
