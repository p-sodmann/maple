//! Raw file preview extraction.
//!
//! Fujifilm RAF files embed a full-resolution JPEG preview.  The RAF header
//! stores the JPEG offset and length at fixed positions, so we can extract
//! the preview without decoding the actual sensor data.

use std::path::Path;

use anyhow::{bail, Context};

/// RAF header magic — first 16 bytes of every `.raf` file.
const RAF_MAGIC: &[u8; 16] = b"FUJIFILMCCD-RAW ";

/// Byte offset within the RAF header where the embedded JPEG offset is stored
/// (big-endian `u32`).
const JPEG_OFFSET_POS: usize = 84;

/// Byte offset within the RAF header where the embedded JPEG length is stored
/// (big-endian `u32`).
const JPEG_LENGTH_POS: usize = 88;

/// Minimum header size we need to read the JPEG location fields.
const MIN_HEADER: usize = JPEG_LENGTH_POS + 4;

/// Extract the embedded JPEG preview from a Fujifilm RAF file.
///
/// Returns the raw JPEG bytes (starting with `0xFF 0xD8`).
pub fn extract_raf_preview(path: &Path) -> anyhow::Result<Vec<u8>> {
    let data = std::fs::read(path)
        .with_context(|| format!("reading RAF file: {}", path.display()))?;

    if data.len() < MIN_HEADER || &data[..16] != RAF_MAGIC {
        bail!("not a valid RAF file: {}", path.display());
    }

    let jpeg_offset =
        u32::from_be_bytes(data[JPEG_OFFSET_POS..JPEG_OFFSET_POS + 4].try_into().unwrap())
            as usize;
    let jpeg_length =
        u32::from_be_bytes(data[JPEG_LENGTH_POS..JPEG_LENGTH_POS + 4].try_into().unwrap())
            as usize;

    if jpeg_offset == 0 || jpeg_length == 0 {
        bail!("RAF file has no embedded JPEG preview: {}", path.display());
    }

    let jpeg_end = jpeg_offset
        .checked_add(jpeg_length)
        .context("JPEG offset+length overflow")?;

    if jpeg_end > data.len() {
        bail!(
            "RAF JPEG preview extends past EOF (offset={jpeg_offset}, len={jpeg_length}, file={})",
            data.len()
        );
    }

    let jpeg = &data[jpeg_offset..jpeg_end];

    // Sanity-check: JPEG files start with 0xFF 0xD8.
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        bail!("extracted RAF preview does not start with JPEG SOI marker");
    }

    Ok(jpeg.to_vec())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal fake RAF file with an embedded JPEG stub.
    fn make_fake_raf(jpeg: &[u8]) -> Vec<u8> {
        // Header: 108 bytes minimum, then JPEG data follows.
        let jpeg_offset: u32 = 108;
        let jpeg_length: u32 = jpeg.len() as u32;

        let mut buf = vec![0u8; jpeg_offset as usize + jpeg.len()];
        buf[..16].copy_from_slice(RAF_MAGIC);
        buf[JPEG_OFFSET_POS..JPEG_OFFSET_POS + 4]
            .copy_from_slice(&jpeg_offset.to_be_bytes());
        buf[JPEG_LENGTH_POS..JPEG_LENGTH_POS + 4]
            .copy_from_slice(&jpeg_length.to_be_bytes());
        buf[jpeg_offset as usize..].copy_from_slice(jpeg);
        buf
    }

    #[test]
    fn extracts_embedded_jpeg() {
        let fake_jpeg = [0xFF, 0xD8, 0xFF, 0xE0, 1, 2, 3, 4];
        let raf = make_fake_raf(&fake_jpeg);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.raf");
        std::fs::write(&path, &raf).unwrap();

        let result = extract_raf_preview(&path).unwrap();
        assert_eq!(result, fake_jpeg);
    }

    #[test]
    fn rejects_non_raf() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.raf");
        std::fs::write(&path, b"not a raf file at all, nope").unwrap();
        assert!(extract_raf_preview(&path).is_err());
    }

    #[test]
    fn rejects_truncated_header() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.raf");
        std::fs::write(&path, &b"FUJIFILMCCD-RAW "[..]).unwrap();
        assert!(extract_raf_preview(&path).is_err());
    }
}
