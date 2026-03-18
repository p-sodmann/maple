//! Image source abstraction — factory for loadable image bytes.
//!
//! Different file formats require different handling to produce decodable
//! image bytes.  Standard formats (JPEG, PNG) are read as-is.  Raw formats
//! (Fujifilm RAF) have an embedded JPEG preview that must be extracted first.
//!
//! All consumers that need to *decode* an image should call
//! [`loadable_image_bytes`] instead of `std::fs::read` directly.

use std::path::Path;

use crate::raw;

/// Returns `true` if the file extension indicates a camera raw format whose
/// embedded preview should be extracted rather than decoding the raw sensor
/// data.
pub fn is_raw_format(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| matches!(e.to_ascii_lowercase().as_str(), "raf"))
        .unwrap_or(false)
}

/// Read decodable image bytes for the file at `path`.
///
/// For standard image formats this is equivalent to [`std::fs::read`].
/// For raw formats the embedded JPEG preview is extracted instead.
///
/// The returned bytes are always a standard image format (JPEG/PNG) that
/// gdk-pixbuf and the `image` crate can decode directly.
pub fn loadable_image_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    if is_raw_format(path) {
        raw::extract_raf_preview(path)
    } else {
        std::fs::read(path)
            .map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_raw_detects_raf() {
        assert!(is_raw_format(Path::new("photo.raf")));
        assert!(is_raw_format(Path::new("photo.RAF")));
        assert!(is_raw_format(Path::new("dir/photo.Raf")));
    }

    #[test]
    fn is_raw_rejects_standard() {
        assert!(!is_raw_format(Path::new("photo.jpg")));
        assert!(!is_raw_format(Path::new("photo.png")));
    }

    #[test]
    fn loadable_bytes_reads_standard_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jpg");
        std::fs::write(&path, b"fake jpeg data").unwrap();
        let bytes = loadable_image_bytes(&path).unwrap();
        assert_eq!(bytes, b"fake jpeg data");
    }
}
