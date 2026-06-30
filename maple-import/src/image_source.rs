//! Image source abstraction — factory for loadable image bytes.
//!
//! Different file formats require different handling to produce decodable
//! image bytes.  Standard formats (JPEG, PNG) are read as-is.  Raw formats
//! whose preview extraction is implemented (currently only Fujifilm RAF) have
//! their embedded JPEG extracted.  Recognised-but-unimplemented raw formats
//! (CR2, CR3) return an error from [`loadable_image_bytes`]; callers can check
//! [`raw_preview_supported`] to distinguish "broken file" from "not yet
//! implemented".
//!
//! # Adding a new raw format
//!
//! 1. Implement [`RawHandler`] for a new zero-sized struct.
//! 2. Add it to [`HANDLERS`].
//! 3. Add the extension string(s) to `IMAGE_EXTENSIONS` in `scan.rs`.

use std::path::Path;

use crate::raw;

// ── Handler trait ────────────────────────────────────────────────

trait RawHandler: Send + Sync {
    /// Return `true` if `ext` (lower-case, no leading dot) belongs to this format.
    fn matches(&self, ext: &str) -> bool;
    /// Return `true` if this handler can actually extract a preview.
    /// `false` means the format is recognised but extraction is not yet implemented.
    fn preview_supported(&self) -> bool;
    /// Extract decodable image bytes (typically an embedded JPEG preview).
    fn extract_preview(&self, path: &Path) -> anyhow::Result<Vec<u8>>;
}

// ── Concrete handlers ────────────────────────────────────────────

struct RafHandler;
impl RawHandler for RafHandler {
    fn matches(&self, ext: &str) -> bool { ext == "raf" }
    fn preview_supported(&self) -> bool { true }
    fn extract_preview(&self, path: &Path) -> anyhow::Result<Vec<u8>> {
        raw::extract_raf_preview(path)
    }
}

struct Cr2Handler;
impl RawHandler for Cr2Handler {
    fn matches(&self, ext: &str) -> bool { ext == "cr2" }
    fn preview_supported(&self) -> bool { false }
    fn extract_preview(&self, _path: &Path) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("CR2 preview extraction is not yet implemented")
    }
}

struct Cr3Handler;
impl RawHandler for Cr3Handler {
    fn matches(&self, ext: &str) -> bool { ext == "cr3" }
    fn preview_supported(&self) -> bool { false }
    fn extract_preview(&self, _path: &Path) -> anyhow::Result<Vec<u8>> {
        anyhow::bail!("CR3 preview extraction is not yet implemented")
    }
}

static HANDLERS: &[&dyn RawHandler] = &[&RafHandler, &Cr2Handler, &Cr3Handler];

// ── Public API ───────────────────────────────────────────────────

/// Returns `true` if the file extension indicates a camera raw format whose
/// embedded preview should be extracted rather than decoding the raw sensor data.
pub fn is_raw_format(path: &Path) -> bool {
    with_ext(path, |e| HANDLERS.iter().any(|h| h.matches(e)))
}

/// Returns `true` if the raw format at `path` has a working preview extractor.
///
/// A return value of `false` means the extension is a recognised raw format
/// but extraction is not yet implemented — use this to show a "not yet
/// supported" placeholder rather than a generic error.
///
/// Returns `true` for non-raw paths (they are always loadable directly).
pub fn raw_preview_supported(path: &Path) -> bool {
    with_ext(path, |e| {
        HANDLERS
            .iter()
            .find(|h| h.matches(e))
            .is_none_or(|h| h.preview_supported())
    })
}

/// Read decodable image bytes for the file at `path`.
///
/// For standard image formats this is equivalent to [`std::fs::read`].
/// For supported raw formats the embedded JPEG preview is extracted instead.
/// For recognised-but-unsupported raw formats an error is returned; check
/// [`raw_preview_supported`] to distinguish this case from a read error.
///
/// The returned bytes are always a standard image format (JPEG/PNG) that
/// gdk-pixbuf and the `image` crate can decode directly.
pub fn loadable_image_bytes(path: &Path) -> anyhow::Result<Vec<u8>> {
    let ext = ext_str(path);
    if let Some(e) = ext.as_deref() {
        if let Some(handler) = HANDLERS.iter().find(|h| h.matches(e)) {
            return handler.extract_preview(path);
        }
    }
    std::fs::read(path).map_err(|e| anyhow::anyhow!("reading {}: {e}", path.display()))
}

// ── Helpers ──────────────────────────────────────────────────────

fn ext_str(path: &Path) -> Option<String> {
    path.extension().and_then(|e| e.to_str()).map(|e| e.to_ascii_lowercase())
}

fn with_ext(path: &Path, f: impl FnOnce(&str) -> bool) -> bool {
    ext_str(path).as_deref().is_some_and(f)
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
    fn is_raw_detects_cr2_cr3() {
        assert!(is_raw_format(Path::new("photo.cr2")));
        assert!(is_raw_format(Path::new("photo.CR2")));
        assert!(is_raw_format(Path::new("photo.cr3")));
        assert!(is_raw_format(Path::new("photo.CR3")));
    }

    #[test]
    fn is_raw_rejects_standard() {
        assert!(!is_raw_format(Path::new("photo.jpg")));
        assert!(!is_raw_format(Path::new("photo.png")));
    }

    #[test]
    fn raf_preview_is_supported() {
        assert!(raw_preview_supported(Path::new("photo.raf")));
    }

    #[test]
    fn cr2_cr3_preview_not_supported() {
        assert!(!raw_preview_supported(Path::new("photo.cr2")));
        assert!(!raw_preview_supported(Path::new("photo.CR2")));
        assert!(!raw_preview_supported(Path::new("photo.cr3")));
        assert!(!raw_preview_supported(Path::new("photo.CR3")));
    }

    #[test]
    fn non_raw_preview_considered_supported() {
        assert!(raw_preview_supported(Path::new("photo.jpg")));
        assert!(raw_preview_supported(Path::new("photo.png")));
    }

    #[test]
    fn loadable_bytes_reads_standard_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jpg");
        std::fs::write(&path, b"fake jpeg data").unwrap();
        let bytes = loadable_image_bytes(&path).unwrap();
        assert_eq!(bytes, b"fake jpeg data");
    }

    #[test]
    fn loadable_bytes_cr2_errors() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("IMG_0001.CR2");
        std::fs::write(&path, b"not a real cr2").unwrap();
        let result = loadable_image_bytes(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not yet implemented"));
    }
}
