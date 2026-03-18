//! Recursive image file scanner.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use crate::image_source::is_raw_format;

/// Supported image file extensions.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png", "raf"];

/// Metadata about a discovered image file.
#[derive(Debug, Clone)]
pub struct ImageFile {
    /// Absolute path to the image.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
}

/// A group of image files that share the same stem (filename without
/// extension) in the same directory — e.g. `DSCF3883.JPG` + `DSCF3883.RAF`.
///
/// The `display` file is the one shown in the browser (prefers standard
/// formats over raw).  `companions` holds the other files in the group.
#[derive(Debug, Clone)]
pub struct ImageGroup {
    /// The file to display (preview/thumbnail).  Prefers JPG/PNG over RAW.
    pub display: ImageFile,
    /// Other files in the group (e.g. the RAF when display is JPG).
    pub companions: Vec<ImageFile>,
}

/// What to include when copying a selected image group.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CopyMode {
    /// Copy all files in the group (display + companions).  Default.
    All,
    /// Copy only the raw file(s).  Falls back to display if no raw companion.
    RawOnly,
    /// Copy only the display file (JPG/PNG).
    DisplayOnly,
}

impl Default for CopyMode {
    fn default() -> Self {
        Self::All
    }
}

impl ImageGroup {
    /// Return the list of paths to copy based on the given [`CopyMode`].
    pub fn paths_for_copy(&self, mode: CopyMode) -> Vec<PathBuf> {
        match mode {
            CopyMode::All => {
                let mut paths = vec![self.display.path.clone()];
                paths.extend(self.companions.iter().map(|c| c.path.clone()));
                paths
            }
            CopyMode::RawOnly => {
                let raws: Vec<PathBuf> = self
                    .companions
                    .iter()
                    .filter(|c| is_raw_format(&c.path))
                    .map(|c| c.path.clone())
                    .collect();
                if raws.is_empty() {
                    // No raw companion — check if the display itself is raw.
                    if is_raw_format(&self.display.path) {
                        vec![self.display.path.clone()]
                    } else {
                        vec![self.display.path.clone()]
                    }
                } else {
                    raws
                }
            }
            CopyMode::DisplayOnly => vec![self.display.path.clone()],
        }
    }
}

/// Recursively scan `root` for image files.
///
/// Returns a sorted list of discovered images.
pub fn scan_images(root: &Path) -> anyhow::Result<Vec<ImageFile>> {
    anyhow::ensure!(root.is_dir(), "{} is not a directory", root.display());
    let mut results = Vec::new();
    scan_dir(root, &mut results)?;
    results.sort_by(|a, b| a.path.cmp(&b.path));
    tracing::info!("Scanned {} images in {}", results.len(), root.display());
    Ok(results)
}

/// Scan `root` and group files that share the same stem in the same directory.
///
/// For each group the preferred display file is a standard format (JPG/PNG);
/// raw files become companions.  Groups are sorted by display path.
pub fn scan_grouped(root: &Path) -> anyhow::Result<Vec<ImageGroup>> {
    let files = scan_images(root)?;

    // Key = (parent_dir, stem_lowercase).  BTreeMap keeps insertion order stable.
    let mut groups: BTreeMap<(PathBuf, String), Vec<ImageFile>> = BTreeMap::new();

    for f in files {
        let parent = f.path.parent().unwrap_or(Path::new("")).to_path_buf();
        let stem = f
            .path
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        groups.entry((parent, stem)).or_default().push(f);
    }

    let mut result: Vec<ImageGroup> = groups
        .into_values()
        .map(|mut members| {
            // Prefer a standard (non-raw) file as the display image.
            let display_idx = members
                .iter()
                .position(|f| !is_raw_format(&f.path))
                .unwrap_or(0);
            let display = members.remove(display_idx);
            ImageGroup {
                display,
                companions: members,
            }
        })
        .collect();

    result.sort_by(|a, b| a.display.path.cmp(&b.display.path));
    tracing::info!(
        "Grouped into {} entries ({} with companions)",
        result.len(),
        result.iter().filter(|g| !g.companions.is_empty()).count()
    );
    Ok(result)
}

fn scan_dir(dir: &Path, out: &mut Vec<ImageFile>) -> anyhow::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::warn!("Cannot read {}: {}", dir.display(), e);
            return Ok(());
        }
    };

    for entry in entries {
        let entry = entry?;
        let ft = entry.file_type()?;
        let path = entry.path();

        if ft.is_dir() {
            scan_dir(&path, out)?;
        } else if ft.is_file() && is_image(&path) {
            let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
            out.push(ImageFile { path, size });
        }
    }

    Ok(())
}

fn is_image(path: &Path) -> bool {
    // Skip macOS resource forks (._*) and other dotfiles.
    let dominated_by_dot = path
        .file_name()
        .and_then(|n| n.to_str())
        .is_some_and(|n| n.starts_with('.'));
    if dominated_by_dot {
        return false;
    }

    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| IMAGE_EXTENSIONS.contains(&e.to_ascii_lowercase().as_str()))
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_is_image() {
        assert!(is_image(Path::new("photo.jpg")));
        assert!(is_image(Path::new("photo.JPG")));
        assert!(is_image(Path::new("photo.jpeg")));
        assert!(is_image(Path::new("photo.png")));
        assert!(is_image(Path::new("photo.raf")));
        assert!(is_image(Path::new("photo.RAF")));
        assert!(!is_image(Path::new("photo.gif")));
        assert!(!is_image(Path::new("readme.txt")));
        // macOS resource forks
        assert!(!is_image(Path::new("._DSCF3883.JPG")));
        assert!(!is_image(Path::new(".hidden.jpg")));
    }

    #[test]
    fn scan_finds_images_recursively() {
        let dir = tempfile::tempdir().unwrap();
        // root level
        fs::write(dir.path().join("a.jpg"), b"fake").unwrap();
        fs::write(dir.path().join("b.png"), b"fake").unwrap();
        fs::write(dir.path().join("notes.txt"), b"not an image").unwrap();
        // nested
        let sub = dir.path().join("sub");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("c.jpeg"), b"fake").unwrap();

        let images = scan_images(dir.path()).unwrap();
        let names: Vec<&str> = images
            .iter()
            .map(|i| i.path.file_name().unwrap().to_str().unwrap())
            .collect();
        assert_eq!(names, &["a.jpg", "b.png", "c.jpeg"]);
    }

    #[test]
    fn scan_empty_dir_returns_empty_vec() {
        let dir = tempfile::tempdir().unwrap();
        let images = scan_images(dir.path()).unwrap();
        assert!(images.is_empty());
    }

    #[test]
    fn scan_non_dir_returns_error() {
        let file = tempfile::NamedTempFile::new().unwrap();
        let result = scan_images(file.path());
        assert!(result.is_err());
    }

    #[test]
    fn scan_records_file_size() {
        let dir = tempfile::tempdir().unwrap();
        let data = b"hello world jpeg";
        fs::write(dir.path().join("photo.jpg"), data).unwrap();

        let images = scan_images(dir.path()).unwrap();
        assert_eq!(images.len(), 1);
        assert_eq!(images[0].size, data.len() as u64);
    }

    #[test]
    fn scan_grouped_pairs_jpg_and_raf() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("DSCF0001.JPG"), b"jpg").unwrap();
        fs::write(dir.path().join("DSCF0001.RAF"), b"raf").unwrap();
        fs::write(dir.path().join("DSCF0002.JPG"), b"solo").unwrap();

        let groups = scan_grouped(dir.path()).unwrap();
        assert_eq!(groups.len(), 2);

        // First group: DSCF0001 — display is JPG, companion is RAF.
        let g = &groups[0];
        assert!(g.display.path.to_str().unwrap().contains("DSCF0001.JPG"));
        assert_eq!(g.companions.len(), 1);
        assert!(g.companions[0].path.to_str().unwrap().contains("DSCF0001.RAF"));

        // Second group: DSCF0002 — solo JPG, no companions.
        assert!(groups[1].companions.is_empty());
    }

    #[test]
    fn scan_grouped_raw_only_becomes_display() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("DSCF0003.RAF"), b"raw only").unwrap();

        let groups = scan_grouped(dir.path()).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].display.path.to_str().unwrap().contains("DSCF0003.RAF"));
        assert!(groups[0].companions.is_empty());
    }

    #[test]
    fn paths_for_copy_modes() {
        let group = ImageGroup {
            display: ImageFile {
                path: PathBuf::from("/photos/DSCF0001.JPG"),
                size: 100,
            },
            companions: vec![ImageFile {
                path: PathBuf::from("/photos/DSCF0001.RAF"),
                size: 200,
            }],
        };

        let all = group.paths_for_copy(CopyMode::All);
        assert_eq!(all.len(), 2);

        let raw = group.paths_for_copy(CopyMode::RawOnly);
        assert_eq!(raw, vec![PathBuf::from("/photos/DSCF0001.RAF")]);

        let display = group.paths_for_copy(CopyMode::DisplayOnly);
        assert_eq!(display, vec![PathBuf::from("/photos/DSCF0001.JPG")]);
    }
}
