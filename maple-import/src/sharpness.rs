//! Blur/sharpness scoring via variance-of-Laplacian.
//!
//! Used to pick the "best" (sharpest) photo within a detected burst group.
//! Operates directly on an already-decoded RGB8 buffer (e.g. the thumbnail
//! produced during an import scan) — no image decode of its own.

/// Sharpness score for a tightly-packed RGB8 buffer. Higher = sharper.
///
/// Converts to grayscale, applies a 3x3 Laplacian kernel, and returns the
/// variance of the response — the classic "variance of Laplacian" blur
/// metric. A flat/blurry image has little high-frequency response and low
/// variance; a sharp image has strong edges and high variance.
///
/// Returns `0.0` for images too small to convolve (< 3px in either
/// dimension) or a buffer shorter than `width * height * 3` bytes.
pub fn laplacian_variance(rgb: &[u8], width: u32, height: u32) -> f32 {
    let (w, h) = (width as usize, height as usize);
    if w < 3 || h < 3 || rgb.len() < w * h * 3 {
        return 0.0;
    }

    let gray: Vec<f32> = (0..w * h)
        .map(|i| {
            let r = rgb[i * 3] as f32;
            let g = rgb[i * 3 + 1] as f32;
            let b = rgb[i * 3 + 2] as f32;
            0.299 * r + 0.587 * g + 0.114 * b
        })
        .collect();

    let mut responses = Vec::with_capacity((w - 2) * (h - 2));
    for y in 1..h - 1 {
        for x in 1..w - 1 {
            let center = gray[y * w + x];
            let up = gray[(y - 1) * w + x];
            let down = gray[(y + 1) * w + x];
            let left = gray[y * w + x - 1];
            let right = gray[y * w + x + 1];
            responses.push(up + down + left + right - 4.0 * center);
        }
    }

    variance(&responses)
}

fn variance(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let mean = values.iter().sum::<f32>() / values.len() as f32;
    values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / values.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(size: usize, color: [u8; 3]) -> Vec<u8> {
        (0..size * size).flat_map(|_| color).collect()
    }

    #[test]
    fn flat_image_has_zero_variance() {
        let rgb = solid(8, [128, 128, 128]);
        assert_eq!(laplacian_variance(&rgb, 8, 8), 0.0);
    }

    #[test]
    fn checkerboard_has_higher_variance_than_flat() {
        let size = 8usize;
        let mut rgb = Vec::with_capacity(size * size * 3);
        for y in 0..size {
            for x in 0..size {
                let v = if (x + y) % 2 == 0 { 255 } else { 0 };
                rgb.extend_from_slice(&[v, v, v]);
            }
        }
        let sharp = laplacian_variance(&rgb, size as u32, size as u32);
        let flat = laplacian_variance(&solid(size, [128, 128, 128]), size as u32, size as u32);
        assert!(sharp > flat, "checkerboard ({sharp}) should be sharper than flat ({flat})");
    }

    #[test]
    fn tiny_image_returns_zero() {
        let rgb = solid(2, [10, 10, 10]);
        assert_eq!(laplacian_variance(&rgb, 2, 2), 0.0);
    }

    #[test]
    fn short_buffer_returns_zero() {
        let rgb = vec![0u8; 4]; // shorter than 8*8*3
        assert_eq!(laplacian_variance(&rgb, 8, 8), 0.0);
    }
}
