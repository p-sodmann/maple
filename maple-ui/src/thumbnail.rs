//! Thumbnail generation and codec helpers.
//!
//! # Pipeline (cache miss)
//!
//! 1. Decode via the pure-Rust `image` crate (no system GTK / gdk-pixbuf).
//! 2. Apply EXIF orientation (`apply_orientation`, driven by the file's — or, for
//!    raw files, the embedded preview's — Orientation tag).
//! 3. Convert to a tight RGB buffer.
//! 4. `fast_image_resize` Lanczos3 → target dimensions.
//! 5. Encode as lossy WebP at the configured quality.
//!
//! # Cache round-trip
//!
//! On a cache hit, `decode_webp_rgb` turns stored WebP bytes back into an RGB
//! pixel buffer that the UI uploads to Slint as an `Image`.
//!
//! # Note on peak memory
//!
//! The former gdk-pixbuf path decoded JPEGs at 2× the target size using
//! libjpeg's DCT downscaling, keeping peak memory low for very large files. The
//! `image` crate does not expose scale-on-decode, so this path decodes at full
//! resolution before the Lanczos3 downsample. Correct, but heavier for huge
//! sources — a future optimisation could drop to `zune-jpeg`'s DCT scaling.

use std::fs::File;
use std::io::{BufRead, BufReader, Cursor, Seek};
use std::path::Path;

use fast_image_resize as fir;
use maple_import::{is_raw_format, loadable_image_bytes};

// ── Public entry points ───────────────────────────────────────────

/// Decode and resize `path` to at most `max_size`px on the longest edge.
///
/// Returns `(rgb_pixels, width, height)` — 24-bit tightly-packed RGB.
/// EXIF orientation is applied before resizing.
/// Lanczos3 filter is used for the final downsample step.
pub fn render_to_rgb(path: &Path, max_size: u32) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let rgb_img = decode_oriented(path)?.into_rgb8();
    let (src_w, src_h) = rgb_img.dimensions();
    let (dst_w, dst_h) = fit_dims(src_w, src_h, max_size);

    let src_img = fir::images::ImageRef::new(src_w, src_h, rgb_img.as_raw(), fir::PixelType::U8x3)
        .map_err(|e| anyhow::anyhow!("fir src: {e}"))?;
    let mut dst_img = fir::images::Image::new(dst_w, dst_h, fir::PixelType::U8x3);
    fir::Resizer::new()
        .resize(
            &src_img,
            &mut dst_img,
            &fir::ResizeOptions::new()
                .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Lanczos3)),
        )
        .map_err(|e| anyhow::anyhow!("fir resize: {e}"))?;

    Ok((dst_img.into_vec(), dst_w, dst_h))
}

/// Decode `path` (or its raw preview) into a full-resolution image with EXIF
/// orientation already applied.
///
/// Shared by the thumbnail pipeline and the full-resolution image loader.
pub fn decode_oriented(path: &Path) -> anyhow::Result<image::DynamicImage> {
    if is_raw_format(path) {
        // `loadable_image_bytes` returns the embedded JPEG preview for raw files.
        let bytes = loadable_image_bytes(path)
            .map_err(|e| anyhow::anyhow!("read raw preview {}: {e}", path.display()))?;
        let img = image::load_from_memory(&bytes)
            .map_err(|e| anyhow::anyhow!("decode raw preview {}: {e}", path.display()))?;
        // The preview JPEG carries its own EXIF orientation.
        let orientation = read_orientation_from(&mut Cursor::new(&bytes));
        Ok(apply_orientation(img, orientation))
    } else {
        let img = image::ImageReader::open(path)
            .map_err(|e| anyhow::anyhow!("open {}: {e}", path.display()))?
            .with_guessed_format()
            .map_err(|e| anyhow::anyhow!("probe {}: {e}", path.display()))?
            .decode()
            .map_err(|e| anyhow::anyhow!("decode {}: {e}", path.display()))?;
        let orientation = read_exif_orientation(path);
        Ok(apply_orientation(img, orientation))
    }
}

/// Encode `rgb` pixels as lossy WebP at the given quality (0–100).
pub fn encode_webp_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    webp::Encoder::from_rgb(rgb, width, height)
        .encode(quality as f32)
        .to_vec()
}

/// Decode cached WebP bytes back to tight RGB pixels.
///
/// Returns `(rgb_pixels, width, height)`.
pub fn decode_webp_rgb(webp: &[u8]) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    let img = image::load_from_memory(webp)?;
    let rgb = img.into_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

/// Convenience: render + encode WebP in one call.
///
/// Returns WebP bytes ready to store in the cache.
pub fn generate_thumbnail(path: &Path, max_size: u32, quality: u8) -> anyhow::Result<Vec<u8>> {
    let (rgb, w, h) = render_to_rgb(path, max_size)?;
    Ok(encode_webp_rgb(&rgb, w, h, quality))
}

// ── EXIF orientation ──────────────────────────────────────────────

/// Read the EXIF orientation tag (1–8) from `path`. Returns 1 (normal) on any
/// failure (missing tag, unreadable file, no EXIF).
pub fn read_exif_orientation(path: &Path) -> u32 {
    let Ok(file) = File::open(path) else {
        return 1;
    };
    read_orientation_from(&mut BufReader::new(file))
}

/// Read the EXIF orientation tag from any seekable reader (file or in-memory
/// preview bytes). Returns 1 on any failure.
fn read_orientation_from<R: BufRead + Seek>(reader: &mut R) -> u32 {
    match exif::Reader::new().read_from_container(reader) {
        Ok(e) => e
            .get_field(exif::Tag::Orientation, exif::In::PRIMARY)
            .and_then(|f| f.value.get_uint(0))
            .unwrap_or(1),
        Err(_) => 1,
    }
}

/// Apply an EXIF orientation transform (1–8) to a `DynamicImage`.
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

// ── Internals ─────────────────────────────────────────────────────

/// Compute output dimensions preserving aspect ratio.
fn fit_dims(w: u32, h: u32, max: u32) -> (u32, u32) {
    if w == 0 || h == 0 {
        return (max, max);
    }
    if w <= max && h <= max {
        return (w, h);
    }
    if w >= h {
        (max, ((h as u64 * max as u64) / w as u64).max(1) as u32)
    } else {
        (((w as u64 * max as u64) / h as u64).max(1) as u32, max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;
    use std::io::Cursor;

    fn create_test_png(w: u32, h: u32) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Cursor::new(Vec::new());
        dyn_img.write_to(&mut buf, image::ImageFormat::Png).unwrap();
        std::fs::write(&path, buf.get_ref()).unwrap();
        dir
    }

    fn test_png_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("test.png")
    }

    #[test]
    fn render_to_rgb_packs_tight_rgb() {
        let dir = create_test_png(640, 480);
        let (rgb, w, h) = render_to_rgb(&test_png_path(&dir), 128).unwrap();
        assert_eq!(w, 128);
        assert_eq!(h, 96);
        assert_eq!(rgb.len(), (w * h * 3) as usize);
    }

    #[test]
    fn generate_thumbnail_produces_webp() {
        let dir = create_test_png(640, 480);
        let bytes = generate_thumbnail(&test_png_path(&dir), 128, 80).unwrap();
        // WebP container: "RIFF" .... "WEBP".
        assert_eq!(&bytes[0..4], b"RIFF");
        assert_eq!(&bytes[8..12], b"WEBP");
    }

    #[test]
    fn render_to_rgb_respects_max_size() {
        let dir = create_test_png(800, 400);
        let (_rgb, w, h) = render_to_rgb(&test_png_path(&dir), 100).unwrap();
        assert_eq!(w, 100);
        assert_eq!(h, 50);
    }

    #[test]
    fn render_to_rgb_bad_path_errors() {
        let result = render_to_rgb(Path::new("/nonexistent/photo.jpg"), 128);
        assert!(result.is_err());
    }

    #[test]
    fn webp_round_trips_to_rgb() {
        let dir = create_test_png(64, 48);
        let webp = generate_thumbnail(&test_png_path(&dir), 64, 90).unwrap();
        let (rgb, w, h) = decode_webp_rgb(&webp).unwrap();
        assert_eq!((w, h), (64, 48));
        assert_eq!(rgb.len(), (w * h * 3) as usize);
    }

    #[test]
    fn apply_orientation_rotate90_swaps_dims() {
        // Orientation 6 = rotate 90° CW: a 4×2 image becomes 2×4.
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        let rotated = apply_orientation(img, 6);
        assert_eq!((rotated.width(), rotated.height()), (2, 4));
    }

    #[test]
    fn apply_orientation_identity_and_flip_preserve_dims() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img.clone(), 1).dimensions(), (4, 2));
        assert_eq!(apply_orientation(img, 2).dimensions(), (4, 2)); // horizontal flip
    }

    #[test]
    fn fit_dims_landscape() {
        assert_eq!(fit_dims(800, 400, 200), (200, 100));
    }

    #[test]
    fn fit_dims_portrait() {
        assert_eq!(fit_dims(400, 800, 200), (100, 200));
    }

    #[test]
    fn fit_dims_small_image_unchanged() {
        assert_eq!(fit_dims(100, 80, 200), (100, 80));
    }
}
