//! Thumbnail generation for the library, and the codec helpers it shares
//! with the rest of the app.
//!
//! The render and WebP primitives themselves live in
//! [`maple_import::preview`] — the import scan, the session-detection lab
//! and these thumbnails must all resample the same way, and one of them
//! having its own copy is how the lab and the scan drifted apart once
//! already. This module keeps only what is specific to the library's own
//! images: the cover crop, and the render-then-encode convenience.
//!
//! # Pipeline (cache miss)
//!
//! 1. Decode via [`maple_import::decode_image`] (zune-jpeg for JPEG, `image`
//!    crate for everything else; EXIF orientation applied).
//! 2. Convert to a tight RGB buffer.
//! 3. `fast_image_resize` Lanczos3 → target dimensions.
//! 4. Encode as lossy WebP at the configured quality.
//!
//! # Cache round-trip
//!
//! On a cache hit, `decode_webp_rgb` turns stored WebP bytes back into an RGB
//! pixel buffer that the UI uploads to Slint as an `Image`.

use std::path::Path;

pub use maple_import::preview::{
    decode_webp_rgb, encode_webp_rgb, render_bytes_to_rgb, render_to_rgb,
};

/// Convenience: render + encode WebP in one call.
///
/// Returns WebP bytes ready to store in the cache.
pub fn generate_thumbnail(path: &Path, max_size: u32, quality: u8) -> anyhow::Result<Vec<u8>> {
    let (rgb, w, h) = render_to_rgb(path, max_size)?;
    Ok(encode_webp_rgb(&rgb, w, h, quality))
}

/// Decode `path`, crop to the largest centred square, and resize to
/// `out_px × out_px` RGB pixels.
///
/// Used for representative "cover" crops (e.g. a Collection's central image)
/// where — unlike [`render_to_rgb`] — the output must be square to match the
/// round/rounded-square tile it's rendered into, and there's no face bbox to
/// crop around (see [`crate::face_crop::extract_crop`] for that case).
pub fn center_square_crop(path: &Path, out_px: u32) -> anyhow::Result<Vec<u8>> {
    let rgb = maple_import::decode_image(path)?.into_rgb8();
    let (w, h) = rgb.dimensions();
    let side = w.min(h).max(1);
    let x = (w - side) / 2;
    let y = (h - side) / 2;
    let cropped = image::imageops::crop_imm(&rgb, x, y, side, side).to_image();
    let resized = image::imageops::resize(&cropped, out_px, out_px, image::imageops::FilterType::Lanczos3);
    Ok(resized.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::GenericImageView;
    use maple_import::apply_orientation;
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
    fn center_square_crop_produces_requested_square() {
        let dir = create_test_png(800, 400);
        let pixels = center_square_crop(&test_png_path(&dir), 64).unwrap();
        assert_eq!(pixels.len(), (64 * 64 * 3) as usize);
    }

    #[test]
    fn center_square_crop_bad_path_errors() {
        let result = center_square_crop(Path::new("/nonexistent/photo.jpg"), 64);
        assert!(result.is_err());
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
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img, 6).dimensions(), (2, 4));
    }

    #[test]
    fn apply_orientation_identity_and_flip_preserve_dims() {
        let img = image::DynamicImage::ImageRgb8(image::RgbImage::new(4, 2));
        assert_eq!(apply_orientation(img.clone(), 1).dimensions(), (4, 2));
        assert_eq!(apply_orientation(img, 2).dimensions(), (4, 2));
    }
}
