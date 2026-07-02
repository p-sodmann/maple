//! Face-crop extraction — decode an image and cut a square, padded crop
//! centred on a detection bounding box. No DB or Slint dependency.

use std::path::Path;

/// Decode `path`, extract a square face crop padded by one bbox-diameter, and
/// resize to `out_px × out_px` RGB pixels.
///
/// The crop is centred on the face centre.  Padding outside the image is filled
/// with mid-grey so the face is always at a visually consistent size.
pub fn extract_crop(path: &Path, bbox: [f32; 4], out_px: u32) -> anyhow::Result<Vec<u8>> {
    let rgb = maple_import::decode_image(path)?.into_rgb8();
    let (iw, ih) = rgb.dimensions();
    let [x1, y1, x2, y2] = bbox;

    // Face centre and size in source pixels.
    let fx1 = x1 * iw as f32;
    let fy1 = y1 * ih as f32;
    let fx2 = x2 * iw as f32;
    let fy2 = y2 * ih as f32;
    let fw = (fx2 - fx1).max(1.0);
    let fh = (fy2 - fy1).max(1.0);
    let cx = (fx1 + fx2) * 0.5;
    let cy = (fy1 + fy2) * 0.5;

    // Pad = one bbox-diameter (largest axis) on every side.
    let d = fw.max(fh);
    let half = d * 1.5; // d/2 face half + d padding

    let side = ((half * 2.0).round() as u32).max(4);

    // Desired crop window top-left in image-space (may be negative near edge).
    let desired_x0 = (cx - half).round() as i32;
    let desired_y0 = (cy - half).round() as i32;

    // Clamp to image bounds.
    let actual_x0 = desired_x0.clamp(0, iw as i32 - 1) as u32;
    let actual_y0 = desired_y0.clamp(0, ih as i32 - 1) as u32;
    let actual_x1 = (desired_x0 + side as i32).clamp(0, iw as i32) as u32;
    let actual_y1 = (desired_y0 + side as i32).clamp(0, ih as i32) as u32;
    let crop_w = actual_x1.saturating_sub(actual_x0).max(1);
    let crop_h = actual_y1.saturating_sub(actual_y0).max(1);

    // Where inside the output square the cropped pixels begin.
    let paste_x = (actual_x0 as i32 - desired_x0) as u32;
    let paste_y = (actual_y0 as i32 - desired_y0) as u32;

    let cropped =
        image::imageops::crop_imm(&rgb, actual_x0, actual_y0, crop_w, crop_h).to_image();

    // Build the square output, grey-filling any border that fell outside the image.
    let square: image::RgbImage =
        if paste_x == 0 && paste_y == 0 && crop_w >= side && crop_h >= side {
            cropped
        } else {
            let mut bg =
                image::RgbImage::from_pixel(side, side, image::Rgb([64u8, 64, 64]));
            image::imageops::overlay(&mut bg, &cropped, paste_x as i64, paste_y as i64);
            bg
        };

    let final_img = image::imageops::resize(
        &square,
        out_px,
        out_px,
        image::imageops::FilterType::Lanczos3,
    );
    Ok(final_img.into_raw())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_png(w: u32, h: u32) -> (tempfile::TempDir, std::path::PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("face.png");
        let img = image::RgbImage::from_pixel(w, h, image::Rgb([200, 100, 50]));
        image::DynamicImage::ImageRgb8(img).save(&path).unwrap();
        (dir, path)
    }

    #[test]
    fn extract_crop_returns_requested_output_size() {
        let (_dir, path) = create_test_png(200, 200);
        let pixels = extract_crop(&path, [0.4, 0.4, 0.6, 0.6], 64).unwrap();
        assert_eq!(pixels.len(), 64 * 64 * 3);
    }

    #[test]
    fn extract_crop_pads_bbox_near_image_edge_with_grey() {
        // Face bbox touching the top-left corner — padding must fall outside
        // the source image and be filled with mid-grey rather than panicking.
        let (_dir, path) = create_test_png(100, 100);
        let pixels = extract_crop(&path, [0.0, 0.0, 0.1, 0.1], 32).unwrap();
        assert_eq!(pixels.len(), 32 * 32 * 3);
        // Top-left-most pixel is outside the source image bounds → grey fill.
        assert_eq!(&pixels[0..3], &[64, 64, 64]);
    }

    #[test]
    fn extract_crop_missing_file_errors() {
        let result = extract_crop(std::path::Path::new("/nonexistent/face.jpg"), [0.0, 0.0, 1.0, 1.0], 32);
        assert!(result.is_err());
    }
}
