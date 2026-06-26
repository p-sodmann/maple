//! Thumbnail generation and codec helpers.
//!
//! # Pipeline (cache miss)
//!
//! 1. Decode via gdk-pixbuf at 2× target size — JPEGs use DCT downscaling
//!    inside libjpeg-turbo, keeping peak memory low for large files.
//! 2. Apply EXIF orientation (`apply_embedded_orientation`).
//! 3. Strip alpha and row-padding → tight RGB byte buffer.
//! 4. `fast_image_resize` Lanczos3 → target dimensions.
//! 5. Encode as lossy WebP at the configured quality.
//!
//! # Cache round-trip
//!
//! On a cache hit, `decode_webp_rgb` turns stored WebP bytes back into an RGB
//! pixel buffer that the grid sends to GTK without going through gdk-pixbuf
//! format loaders (no system WebP loader required).

use std::fs::File;
use std::io::{BufReader, Cursor};
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
    // Decode to 2× target so Lanczos3 has enough source detail.
    let hint = (max_size * 2) as i32;

    let pixbuf = if is_raw_format(path) {
        let bytes = loadable_image_bytes(path)?;
        let stream =
            gtk4::gio::MemoryInputStream::from_bytes(&gtk4::glib::Bytes::from(&bytes));
        gtk4::gdk_pixbuf::Pixbuf::from_stream_at_scale(
            &stream,
            hint,
            hint,
            true,
            gtk4::gio::Cancellable::NONE,
        )
    } else {
        gtk4::gdk_pixbuf::Pixbuf::from_file_at_scale(path, hint, hint, true)
    }
    .map_err(|e| anyhow::anyhow!("decode {}: {e}", path.display()))?;

    let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);

    let src_w = pixbuf.width() as u32;
    let src_h = pixbuf.height() as u32;
    let rowstride = pixbuf.rowstride() as usize;
    let has_alpha = pixbuf.has_alpha();
    let src_ch: usize = if has_alpha { 4 } else { 3 };

    let Some(raw) = pixbuf.pixel_bytes() else {
        anyhow::bail!("no pixel data for {}", path.display());
    };

    // Unpack rows (strip row padding) and convert to tight RGB.
    let mut rgb = Vec::with_capacity((src_w * src_h * 3) as usize);
    for y in 0..src_h as usize {
        let row = &raw[y * rowstride..y * rowstride + src_w as usize * src_ch];
        if has_alpha {
            for px in row.chunks_exact(4) {
                rgb.extend_from_slice(&px[..3]);
            }
        } else {
            rgb.extend_from_slice(row);
        }
    }
    drop(raw);
    drop(pixbuf);

    let (dst_w, dst_h) = fit_dims(src_w, src_h, max_size);

    let src_img =
        fir::images::ImageRef::new(src_w, src_h, &rgb, fir::PixelType::U8x3)
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

// ── Legacy fallback (pure-Rust, used in tests) ────────────────────

/// Generate a thumbnail using the `image` crate (pure Rust, no GTK).
///
/// Slower than the gdk-pixbuf path but usable in test contexts without a
/// GTK main loop.  Returns PNG-encoded bytes.
#[allow(dead_code)]
pub fn generate_thumbnail_image_crate(path: &Path, max_size: u32) -> anyhow::Result<Vec<u8>> {
    let img = image::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to decode {}: {}", path.display(), e))?;

    let orientation = read_exif_orientation(path);
    let img = apply_orientation(img, orientation);

    let thumb = img.thumbnail(max_size, max_size);

    let mut cursor = Cursor::new(Vec::new());
    thumb.write_to(&mut cursor, image::ImageFormat::Png)?;

    Ok(cursor.into_inner())
}

/// Read the EXIF orientation tag (1–8). Returns 1 (normal) on any failure.
pub fn read_exif_orientation(path: &Path) -> u32 {
    let file = match File::open(path) {
        Ok(f) => f,
        Err(_) => return 1,
    };
    let mut reader = BufReader::new(file);
    let exif = match exif::Reader::new().read_from_container(&mut reader) {
        Ok(e) => e,
        Err(_) => return 1,
    };
    exif.get_field(exif::Tag::Orientation, exif::In::PRIMARY)
        .and_then(|f| f.value.get_uint(0))
        .unwrap_or(1)
}

/// Apply EXIF orientation transform to a `DynamicImage`.
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn create_test_png(w: u32, h: u32) -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.png");
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            image::Rgb([(x % 256) as u8, (y % 256) as u8, 128])
        });
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Cursor::new(Vec::new());
        dyn_img
            .write_to(&mut buf, image::ImageFormat::Png)
            .unwrap();
        std::fs::write(&path, buf.get_ref()).unwrap();
        dir
    }

    fn test_png_path(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("test.png")
    }

    #[test]
    fn thumbnail_produces_valid_png() {
        let dir = create_test_png(640, 480);
        let bytes = generate_thumbnail_image_crate(&test_png_path(&dir), 128).unwrap();
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn thumbnail_respects_max_size() {
        let dir = create_test_png(800, 400);
        let bytes = generate_thumbnail_image_crate(&test_png_path(&dir), 100).unwrap();

        let img = image::load_from_memory(&bytes).unwrap();
        assert!(img.width() <= 100);
        assert!(img.height() <= 100);
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn thumbnail_bad_path_errors() {
        let result = generate_thumbnail_image_crate(Path::new("/nonexistent/photo.jpg"), 128);
        assert!(result.is_err());
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
