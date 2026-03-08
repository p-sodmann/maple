//! Off-thread thumbnail generation.
//!
//! Uses gdk-pixbuf (libjpeg-turbo) for fast scaled decoding — for JPEGs
//! this leverages DCT downsampling so a 4000×3000 image can be thumbnailed
//! without ever fully decoding all 12 megapixels.
//!
//! EXIF orientation is applied via `apply_embedded_orientation()`.

use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::Path;

/// Generate a PNG thumbnail for the given image file.
///
/// The thumbnail preserves aspect ratio with the longest edge ≤ `max_size` pixels.
/// EXIF orientation is applied before resizing.
/// Returns PNG-encoded bytes.
///
/// Uses gdk-pixbuf under the hood (`Pixbuf::from_file_at_scale`) which
/// is backed by libjpeg-turbo and can downscale JPEGs during DCT decode.
pub fn generate_thumbnail(path: &Path, max_size: u32) -> anyhow::Result<Vec<u8>> {
    let size = max_size as i32;

    // Load at reduced resolution — gdk-pixbuf tells libjpeg to use
    // IDCT scaling (1/2, 1/4, 1/8) when possible, vastly reducing work.
    let pixbuf = gtk4::gdk_pixbuf::Pixbuf::from_file_at_scale(
        path,
        size,
        size,
        true, // preserve aspect ratio
    )
    .map_err(|e| anyhow::anyhow!("Failed to decode {}: {}", path.display(), e))?;

    // Apply EXIF orientation (rotation/flip).
    let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);

    // Encode to PNG bytes.
    let buf = pixbuf
        .save_to_bufferv("png", &[])
        .map_err(|e| anyhow::anyhow!("PNG encode failed: {e}"))?;

    Ok(buf)
}

/// Fallback: generate a thumbnail using the `image` crate (pure Rust).
///
/// Kept for any format gdk-pixbuf can't handle, and for tests that
/// don't have a GTK main loop.
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

/// Apply EXIF orientation transform.
///
/// See <https://www.exif.org/Exif2-2.PDF> (page 18) for the 8 values:
///   1 = normal
///   2 = flipped horizontally
///   3 = rotated 180°
///   4 = flipped vertically
///   5 = transposed (flip H + rotate 270° CW)
///   6 = rotated 90° CW
///   7 = transverse (flip H + rotate 90° CW)
///   8 = rotated 270° CW
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
        _ => img, // unknown value → leave as-is
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Create a minimal valid PNG in a temp file with a `.png` extension.
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
        // Should start with PNG magic bytes
        assert!(bytes.starts_with(&[0x89, b'P', b'N', b'G']));
    }

    #[test]
    fn thumbnail_respects_max_size() {
        let dir = create_test_png(800, 400);
        let bytes = generate_thumbnail_image_crate(&test_png_path(&dir), 100).unwrap();

        let img = image::load_from_memory(&bytes).unwrap();
        assert!(img.width() <= 100);
        assert!(img.height() <= 100);
        // Aspect ratio: 800×400 → 100×50
        assert_eq!(img.width(), 100);
        assert_eq!(img.height(), 50);
    }

    #[test]
    fn thumbnail_bad_path_errors() {
        let result = generate_thumbnail_image_crate(Path::new("/nonexistent/photo.jpg"), 128);
        assert!(result.is_err());
    }
}
