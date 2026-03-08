//! Recursive image file scanner.

use std::path::{Path, PathBuf};

/// Supported image file extensions.
const IMAGE_EXTENSIONS: &[&str] = &["jpg", "jpeg", "png"];

/// Metadata about a discovered image file.
#[derive(Debug, Clone)]
pub struct ImageFile {
    /// Absolute path to the image.
    pub path: PathBuf,
    /// File size in bytes.
    pub size: u64,
}

/// Recursively scan `root` for image files (jpg, jpeg, png).
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
        assert!(!is_image(Path::new("photo.gif")));
        assert!(!is_image(Path::new("photo.raw")));
        assert!(!is_image(Path::new("readme.txt")));
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
}
