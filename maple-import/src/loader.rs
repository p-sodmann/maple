//! Canonical image loader — one decode path for the entire codebase.
//!
//! [`decode_image`] is the single entry point for loading any image Maple
//! understands into a `DynamicImage` with EXIF orientation already applied:
//!
//! * Raw files (RAF etc.) have their embedded JPEG preview extracted first via
//!   [`crate::loadable_image_bytes`].
//! * JPEG bytes (identified by SOI magic) are decoded with `zune-jpeg` directly,
//!   skipping the `image` crate's dispatch overhead.
//! * All other formats fall back to the `image` crate which handles orientation
//!   through its own `ImageDecoder::orientation()` API.

use std::io::Cursor;
use std::path::Path;

use anyhow::{Context, Result};

// ── Public API ────────────────────────────────────────────────────────────────

/// Decode `path` into a full-resolution `DynamicImage` with EXIF orientation
/// applied.
///
/// Raw files have their embedded JPEG preview extracted by
/// [`crate::loadable_image_bytes`] before decoding, so callers never need to
/// handle raw formats separately.
pub fn decode_image(path: &Path) -> Result<image::DynamicImage> {
    let bytes = crate::loadable_image_bytes(path)
        .with_context(|| format!("reading {}", path.display()))?;
    decode_image_bytes(&bytes)
        .with_context(|| format!("decoding {}", path.display()))
}

/// Decode already-fetched image bytes with EXIF orientation applied.
///
/// JPEG is identified by SOI magic bytes (`FF D8 FF`) and decoded via
/// `zune-jpeg`; every other format goes through the `image` crate.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<image::DynamicImage> {
    if is_jpeg(bytes) {
        decode_jpeg(bytes)
    } else {
        decode_other(bytes)
    }
}

/// Apply an EXIF orientation value (1–8) to a `DynamicImage`.
///
/// Orientation 1 is the identity; all other values map to the standard EXIF
/// rotation/flip operations.  Out-of-range values are treated as 1.
pub fn apply_orientation(img: image::DynamicImage, orientation: u32) -> image::DynamicImage {
    match orientation {
        1 => img,
        2 => img.fliph(),
        3 => img.rotate180(),
        4 => img.flipv(),
        5 => img.fliph().rotate270(),
        6 => img.rotate90(),
        7 => img.fliph().rotate90(),
        8 => img.rotate270(),
        _ => img,
    }
}

// ── JPEG via zune-jpeg ────────────────────────────────────────────────────────

fn decode_jpeg(bytes: &[u8]) -> Result<image::DynamicImage> {
    // zune-jpeg 0.5 requires a seekable reader.
    let mut dec = zune_jpeg::JpegDecoder::new(Cursor::new(bytes));
    let pixels = dec.decode().map_err(|e| anyhow::anyhow!("zune-jpeg: {e:?}"))?;
    let (w, h) = dec
        .dimensions()
        .ok_or_else(|| anyhow::anyhow!("zune-jpeg: no dimensions after decode"))?;
    let img = image::RgbImage::from_raw(w as u32, h as u32, pixels)
        .ok_or_else(|| anyhow::anyhow!("zune-jpeg: pixel buffer length mismatch"))?;
    let dyn_img = image::DynamicImage::ImageRgb8(img);
    let orientation = read_jpeg_orientation(bytes);
    Ok(apply_orientation(dyn_img, orientation))
}

// ── Other formats via the image crate ────────────────────────────────────────

fn decode_other(bytes: &[u8]) -> Result<image::DynamicImage> {
    use image::metadata::Orientation;
    use image::ImageDecoder as _;
    let reader = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()
        .context("detecting image format")?;
    let mut decoder = reader.into_decoder().context("creating decoder")?;
    let orientation = decoder.orientation().unwrap_or(Orientation::NoTransforms);
    let mut img = image::DynamicImage::from_decoder(decoder).context("decoding pixels")?;
    img.apply_orientation(orientation);
    Ok(img)
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// JPEG SOI magic: `FF D8 FF`.
fn is_jpeg(bytes: &[u8]) -> bool {
    bytes.starts_with(&[0xFF, 0xD8, 0xFF])
}

/// Read the EXIF Orientation tag (1–8) from JPEG bytes via `kamadak-exif`.
/// Returns 1 (normal) on any failure.
fn read_jpeg_orientation(bytes: &[u8]) -> u32 {
    match exif::Reader::new().read_from_container(&mut Cursor::new(bytes)) {
        Ok(e) => e
            .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .unwrap_or(1),
        Err(_) => 1,
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;

    fn make_png(w: u32, h: u32) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let mut buf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        buf.into_inner()
    }

    #[test]
    fn decode_image_bytes_png() {
        let bytes = make_png(64, 48);
        let img = decode_image_bytes(&bytes).unwrap();
        assert_eq!(img.dimensions(), (64, 48));
    }

    #[test]
    fn apply_orientation_identity() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img, 1).dimensions(), (4, 2));
    }

    #[test]
    fn apply_orientation_rotate90_swaps_dims() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img, 6).dimensions(), (2, 4));
    }

    #[test]
    fn apply_orientation_flip_preserves_dims() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img, 2).dimensions(), (4, 2));
    }

    #[test]
    fn is_jpeg_detection() {
        assert!(is_jpeg(&[0xFF, 0xD8, 0xFF, 0xE0]));
        assert!(!is_jpeg(&[0x89, 0x50, 0x4E, 0x47])); // PNG
        assert!(!is_jpeg(&[]));
    }
}
