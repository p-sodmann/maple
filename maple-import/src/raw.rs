//! Raw file preview extraction.
//!
//! Fujifilm RAF files embed a full-resolution JPEG preview.  The RAF header
//! stores the JPEG offset and length at fixed positions, so we can extract
//! the preview without decoding the actual sensor data.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
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
///
/// Only the header and the JPEG block itself are read — RAF files are
/// 20-50 MB (sensor data dwarfs the preview), so seeking straight to the
/// JPEG offset instead of reading the whole file keeps this to a few
/// hundred KB of I/O and avoids buffering the sensor data at all.
pub fn extract_raf_preview(path: &Path) -> anyhow::Result<Vec<u8>> {
    let mut file = File::open(path)
        .with_context(|| format!("opening RAF file: {}", path.display()))?;

    let mut header = [0u8; MIN_HEADER];
    if file.read_exact(&mut header).is_err() || &header[..16] != RAF_MAGIC {
        bail!("not a valid RAF file: {}", path.display());
    }

    let jpeg_offset =
        u32::from_be_bytes(header[JPEG_OFFSET_POS..JPEG_OFFSET_POS + 4].try_into().unwrap())
            as usize;
    let jpeg_length =
        u32::from_be_bytes(header[JPEG_LENGTH_POS..JPEG_LENGTH_POS + 4].try_into().unwrap())
            as usize;

    if jpeg_offset == 0 || jpeg_length == 0 {
        bail!("RAF file has no embedded JPEG preview: {}", path.display());
    }

    // Guards against offset+length overflowing usize on 32-bit targets.
    jpeg_offset
        .checked_add(jpeg_length)
        .context("JPEG offset+length overflow")?;

    file.seek(SeekFrom::Start(jpeg_offset as u64))
        .with_context(|| format!("seeking to JPEG offset in RAF file: {}", path.display()))?;

    let mut jpeg = vec![0u8; jpeg_length];
    file.read_exact(&mut jpeg).map_err(|_| {
        anyhow::anyhow!(
            "RAF JPEG preview extends past EOF (offset={jpeg_offset}, len={jpeg_length})"
        )
    })?;

    // Sanity-check: JPEG files start with 0xFF 0xD8.
    if jpeg.len() < 2 || jpeg[0] != 0xFF || jpeg[1] != 0xD8 {
        bail!("extracted RAF preview does not start with JPEG SOI marker");
    }

    Ok(jpeg)
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

    /// Mimics a real RAF: a multi-megabyte sensor-data block sits between the
    /// header and the embedded JPEG. Extraction must land on the JPEG bytes
    /// via the offset field, not by reading sequentially from the start.
    #[test]
    fn extracts_jpeg_far_from_start_of_large_file() {
        let fake_jpeg = [0xFF, 0xD8, 0xFF, 0xE1, 9, 9, 9];
        let jpeg_offset: u32 = 2_000_000;
        let jpeg_length: u32 = fake_jpeg.len() as u32;

        let mut buf = vec![0u8; jpeg_offset as usize + fake_jpeg.len()];
        buf[..16].copy_from_slice(RAF_MAGIC);
        buf[JPEG_OFFSET_POS..JPEG_OFFSET_POS + 4].copy_from_slice(&jpeg_offset.to_be_bytes());
        buf[JPEG_LENGTH_POS..JPEG_LENGTH_POS + 4].copy_from_slice(&jpeg_length.to_be_bytes());
        buf[jpeg_offset as usize..].copy_from_slice(&fake_jpeg);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("big.raf");
        std::fs::write(&path, &buf).unwrap();

        let result = extract_raf_preview(&path).unwrap();
        assert_eq!(result, fake_jpeg);
    }
}
