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
///
/// A photo **with a companion raw must not use this twice**; see
/// [`place_pair`].
pub fn place_file(
    staged: &Path,
    destination: &Path,
    folder_template: &str,
    filename_template: &str,
    original_name: &str,
) -> anyhow::Result<PathBuf> {
    let (dest, _) = place_pair(
        staged,
        destination,
        folder_template,
        filename_template,
        original_name,
        None,
    )?;
    Ok(dest)
}

/// Place a display file and, in the same call, its companion raw **beside
/// it**: same directory, same stem, only the extension differing.
///
/// This is [`place_file`] plus the one invariant a companion has to satisfy,
/// and it exists because deriving the companion's own destination cannot
/// produce that invariant — only approximate it. The library scanner
/// regroups from *disk*, by directory and stem, so a RAF that does not sit
/// beside its JPEG under a matching stem is not a companion at all: it is a
/// photograph the scanner has never seen, and it inserts a second `images`
/// row for it, which then replicates to every peer.
///
/// Two files placed independently diverge for at least three reasons, and
/// each of them was reachable:
///
/// - **The date.** A staged blob has a synthetic name, so the template
///   context was read from a raw container by an EXIF path that recognises
///   raws *by extension* — it read nothing, fell back to the file's mtime,
///   and filed the companion under the month it arrived while the display
///   file went under the month it was taken.
/// - **The camera.** `{camera}` comes from `Make`/`Model`, and a raw's
///   embedded preview need not carry the same strings its JPEG sibling does.
/// - **The collision suffix.** [`unique_dest_path`] appends `_1` to whichever
///   of the two happens to collide, independently of the other.
///
/// So the pair is placed together: one template context, read from the
/// display file, one target directory, and one stem chosen free for *both*
/// extensions at once. `companion` is `(staged path, sender's file name)`;
/// only the name's extension is used, since the stem is the display file's by
/// construction.
///
/// Both files are **moved**, display file first. A companion that cannot be
/// moved is an error, and the caller's contract (`transfer::commit`,
/// `server::blob_upload`) is to clear staging and try the whole photo again
/// on the next pass rather than adopt half of it.
pub fn place_pair(
    staged: &Path,
    destination: &Path,
    folder_template: &str,
    filename_template: &str,
    original_name: &str,
    companion: Option<(&Path, &str)>,
) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
    let original = Path::new(original_name);
    let original_stem = original
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("file");
    let extension = original.extension().and_then(|e| e.to_str()).unwrap_or("");
    let companion_ext = companion
        .map(|(_, name)| Path::new(name))
        .and_then(|name| name.extension())
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_owned();

    // Read against the *sender's* name, not the staged one: a blob staged as
    // `<hash>.orig` tells `is_raw_format` nothing, and a raw handed to the
    // JPEG parser yields an empty context and the arrival-time fallback.
    let ctx = crate::exif_read::read_named(staged, original);
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
    let wanted_companion = companion.map(|_| companion_ext.as_str());
    let (dest, companion_dest) =
        unique_pair_path(&stem, extension, wanted_companion, &target_dir);

    move_into_place(staged, &dest)?;
    tracing::info!("Placed {} → {}", original_name, dest.display());

    let companion_dest = match (companion, companion_dest) {
        (Some((staged_raw, raw_name)), Some(raw_dest)) => {
            move_into_place(staged_raw, &raw_dest)?;
            tracing::info!("Placed companion {} → {}", raw_name, raw_dest.display());
            Some(raw_dest)
        }
        _ => None,
    };
    Ok((dest, companion_dest))
}

/// Move `staged` to `dest`, falling back to copy-and-remove across
/// filesystems.
fn move_into_place(staged: &Path, dest: &Path) -> anyhow::Result<()> {
    if std::fs::rename(staged, dest).is_ok() {
        return Ok(());
    }
    // Different filesystems, most likely. Copy first and only then drop the
    // staged file, so a failure here leaves the caller's bytes intact rather
    // than losing a download that has already been verified.
    std::fs::copy(staged, dest).map_err(|e| {
        anyhow::anyhow!("failed to place {} at {}: {e}", staged.display(), dest.display())
    })?;
    if let Err(e) = std::fs::remove_file(staged) {
        tracing::warn!("could not remove staged file {}: {e}", staged.display());
    }
    Ok(())
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

/// Pick one stem that is free for a display file **and** its companion.
///
/// The pair version of [`unique_dest_path`], and the reason that one cannot
/// simply be called twice: resolving each collision separately can hand the
/// two files different suffixes, which is exactly the divergence
/// [`place_pair`] exists to make impossible. A stem is a candidate only when
/// *neither* name is taken, so a stray `DSCF0001.RAF` sitting alone still
/// pushes an arriving pair to `DSCF0001_1.JPG` + `DSCF0001_1.RAF` and keeps
/// them together.
fn unique_pair_path(
    stem: &str,
    extension: &str,
    companion_ext: Option<&str>,
    destination: &Path,
) -> (PathBuf, Option<PathBuf>) {
    let Some(companion_ext) = companion_ext else {
        return (unique_dest_path(stem, extension, destination), None);
    };

    let name = |stem: &str, ext: &str| -> PathBuf {
        destination.join(if ext.is_empty() {
            stem.to_owned()
        } else {
            format!("{stem}.{ext}")
        })
    };
    let free = |candidate: &str| -> Option<(PathBuf, PathBuf)> {
        let display = name(candidate, extension);
        let companion = name(candidate, companion_ext);
        // `display != companion` guards the degenerate case of a sender whose
        // two files differ only in case, or not at all: placing both would
        // otherwise mean the second overwriting the first.
        (display != companion && !display.exists() && !companion.exists())
            .then_some((display, companion))
    };

    if let Some(pair) = free(stem) {
        return (pair.0, Some(pair.1));
    }
    for n in 1..u32::MAX {
        if let Some(pair) = free(&format!("{stem}_{n}")) {
            return (pair.0, Some(pair.1));
        }
    }
    // Extremely unlikely fallback.
    (name(stem, extension), Some(name(stem, companion_ext)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    /// The smallest JPEG that carries a `DateTimeOriginal`.
    ///
    /// Built rather than checked in as a fixture because the *only* thing
    /// under test is the date: a real photo would make these tests depend on
    /// a binary nobody can read in a diff. Little-endian TIFF, IFD0 holding
    /// one pointer to an Exif IFD, which holds one ASCII tag.
    fn jpeg_taken_on(stamp: &str) -> Vec<u8> {
        assert_eq!(stamp.len(), 19, "EXIF stamps are `YYYY:MM:DD HH:MM:SS`");
        let mut date = stamp.as_bytes().to_vec();
        date.push(0);

        let entry = |tag: u16, kind: u16, count: u32, value: u32| {
            let mut e = Vec::new();
            e.extend_from_slice(&tag.to_le_bytes());
            e.extend_from_slice(&kind.to_le_bytes());
            e.extend_from_slice(&count.to_le_bytes());
            e.extend_from_slice(&value.to_le_bytes());
            e
        };
        let ifd = |e: Vec<u8>| {
            let mut ifd = 1u16.to_le_bytes().to_vec();
            ifd.extend_from_slice(&e);
            ifd.extend_from_slice(&0u32.to_le_bytes()); // no next IFD
            ifd
        };

        let mut tiff = b"II".to_vec();
        tiff.extend_from_slice(&42u16.to_le_bytes());
        tiff.extend_from_slice(&8u32.to_le_bytes()); // IFD0 starts here
        // IFD0 is 18 bytes at offset 8, so the Exif IFD begins at 26 and its
        // one out-of-line value at 44.
        tiff.extend_from_slice(&ifd(entry(0x8769, 4, 1, 26))); // ExifIFDPointer
        tiff.extend_from_slice(&ifd(entry(0x9003, 2, date.len() as u32, 44)));
        tiff.extend_from_slice(&date);

        let mut app1 = b"Exif\0\0".to_vec();
        app1.extend_from_slice(&tiff);
        let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
        jpeg.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
        jpeg.extend_from_slice(&app1);
        jpeg.extend_from_slice(&[0xFF, 0xD9]);
        jpeg
    }

    #[test]
    fn the_test_jpeg_really_carries_its_date() {
        // If this ever stops holding, every assertion below about *where* a
        // photo files silently starts testing the mtime fallback instead.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.jpg");
        fs::write(&path, jpeg_taken_on("2019:07:04 10:11:12")).unwrap();
        let ctx = crate::exif_read::read(&path);
        assert_eq!(ctx.datetime.map(|d| (d.year, d.month)), Some((2019, 7)));
    }

    /// A companion follows its display file into whatever folder the template
    /// sent that one to — which is the whole invariant, and the one that was
    /// broken: staged as `<hash>.raw`, a RAF was handed to the JPEG parser,
    /// read as having no date at all, and filed under the month it arrived.
    #[test]
    fn place_pair_files_a_companion_beside_its_display_file() {
        let staged_dir = tempfile::tempdir().unwrap();
        let lib = tempfile::tempdir().unwrap();

        // Staged the way sync stages: synthetic names, no extension anyone
        // could learn a format from.
        let display = staged_dir.path().join("abcdef.orig");
        let raw = staged_dir.path().join("abcdef.raw");
        fs::write(&display, jpeg_taken_on("2019:07:04 10:11:12")).unwrap();
        fs::write(&raw, b"not a parseable raw container").unwrap();

        let (placed, placed_raw) = place_pair(
            &display,
            lib.path(),
            "{YYYY}/{MM}",
            "{original}",
            "DSCF0001.JPG",
            Some((raw.as_path(), "DSCF0001.RAF")),
        )
        .unwrap();
        let placed_raw = placed_raw.expect("the companion was placed");

        assert_eq!(placed, lib.path().join("2019/07/DSCF0001.JPG"));
        assert_eq!(placed_raw, lib.path().join("2019/07/DSCF0001.RAF"));
        assert_eq!(placed.parent(), placed_raw.parent());
        assert_eq!(placed.file_stem(), placed_raw.file_stem());
        assert!(!display.exists() && !raw.exists(), "both are moved");
    }

    /// A collision moves *both* names, together. Resolving each file's
    /// collision on its own would give the pair different suffixes, which is
    /// the same divergence by another route.
    #[test]
    fn place_pair_keeps_one_stem_across_a_collision() {
        let staged_dir = tempfile::tempdir().unwrap();
        let lib = tempfile::tempdir().unwrap();
        // Only the JPEG name is taken; the RAF name is free.
        fs::write(lib.path().join("DSCF0001.JPG"), b"already here").unwrap();

        let display = staged_dir.path().join("abcdef.orig");
        let raw = staged_dir.path().join("abcdef.raw");
        fs::write(&display, b"the display file").unwrap();
        fs::write(&raw, b"the negative").unwrap();

        let (placed, placed_raw) = place_pair(
            &display,
            lib.path(),
            "",
            "{original}",
            "DSCF0001.JPG",
            Some((raw.as_path(), "DSCF0001.RAF")),
        )
        .unwrap();

        assert_eq!(placed, lib.path().join("DSCF0001_1.JPG"));
        assert_eq!(
            placed_raw.unwrap(),
            lib.path().join("DSCF0001_1.RAF"),
            "the companion follows the display file's suffix, free name or not"
        );
        assert_eq!(fs::read(lib.path().join("DSCF0001.JPG")).unwrap(), b"already here");
    }

    /// A raw with no JPEG beside it is its own display file, and sync stages
    /// it as `<hash>.orig` — so the format has to come from the sender's name
    /// or the capture date is lost and the photo files under the day it
    /// arrived.
    #[test]
    fn a_staged_blob_is_read_as_the_format_its_sender_named() {
        let staged_dir = tempfile::tempdir().unwrap();
        let staged = staged_dir.path().join("abcdef.orig");
        fs::write(&staged, jpeg_taken_on("2019:07:04 10:11:12")).unwrap();

        let ctx = crate::exif_read::read_named(&staged, Path::new("DSCF0001.JPG"));
        assert_eq!(ctx.datetime.map(|d| (d.year, d.month)), Some((2019, 7)));
        // And the staged name on its own tells nobody anything, which is
        // precisely why the sender's has to be carried.
        let blind = crate::exif_read::read(&staged);
        assert_eq!(blind, ctx, "a .orig JPEG still parses as a JPEG");
    }

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
