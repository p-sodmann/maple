//! The canonical preview: one 256 px lossy WebP per photo, and the frame
//! every check reads.
//!
//! # Why there is exactly one representation
//!
//! An import scan wants several things from a photo's pixels — a tile to
//! show, a sharpness score, a session signature, sometimes an embedding —
//! and the obvious implementation computes each from whatever decode
//! happens to be in hand. That gives the same photo several slightly
//! different pixel buffers and no way to say which one "the detector saw".
//!
//! So the pipeline narrows to a single artefact. [`encode`] turns a photo's
//! bytes into a WebP; [`decode`] turns that WebP back into an [`RgbImage`];
//! **every check runs on the output of `decode`**, and the WebP is the only
//! thing kept. Two properties fall out of that:
//!
//! * **What was checked is what is shown.** The signature that grouped a
//!   photo was computed from the same lossy pixels the user is looking at,
//!   not from a pristine decode that no longer exists anywhere.
//! * **A recomputation agrees with the original.** Re-deriving a signature
//!   or a sharpness score from a photo's kept WebP — days later, on another
//!   machine, out of [`PreviewCache`] — reproduces the scan's own answer
//!   exactly, because there is no other input it could have used.
//!
//! The round trip is not free (an encode plus a decode, a few ms), but it
//! runs on the import pipeline's *parallel* stage while the one serial
//! reader is spending ~100 ms per photo on the card. It costs nothing that
//! shows up on a clock.
//!
//! # What the lossy step costs, measured
//!
//! Compression artefacts are high-frequency, which is exactly what
//! variance-of-Laplacian sharpness measures — so scores are read off the
//! compressed frame and sit slightly above their pristine values. Ordering,
//! which is all the keeper-picking uses, survives: over one image blurred
//! progressively, pristine 966 / 157 / 27 / 4 reads as 930 / 192 / 58 / 20.
//! The floor matters only at the soft end, where it makes two already-blurry
//! photos harder to tell apart than they were. Ranking sharp against soft —
//! the case that picks a session's keeper — is untouched.
//!
//! This is a consequence of the design, not an oversight: a score that
//! could not be recomputed from the only image the system keeps would be a
//! number nobody could ever check.
//!
//! # Why it lives in `maple-import`
//!
//! Both the importer (`maple-ui`) and the session-detection lab
//! (`maple-db`'s `session-lab`) must see byte-identical frames or a
//! threshold tuned in the lab does not transfer to the scan. They already
//! diverged once — the lab downsampled with `image::thumbnail` while the
//! scan used Lanczos3 — which is the divergence this module exists to make
//! impossible. `maple-import` is the crate both depend on.

use std::path::Path;

use anyhow::Result;
use fast_image_resize as fir;
use image::RgbImage;

/// Longest edge of the canonical preview, in pixels.
///
/// Big enough for a filmstrip tile and for the session engines' block and
/// histogram statistics; small enough that a few thousand of them fit in
/// memory and on a card.
pub const PREVIEW_PX: u32 = 256;

/// WebP quality of the canonical preview.
///
/// At 256 px this lands around 15 KB against a decoded frame's ~196 KB.
/// Changing it changes what every check sees, so it is a constant rather
/// than a setting: two machines must agree on the frame or their
/// signatures are not comparable.
pub const PREVIEW_QUALITY: u8 = 80;

// ── The canonical artefact ────────────────────────────────────────

/// Encode a photo's bytes as the canonical preview.
///
/// `bytes` is the file itself for an ordinary image, or a raw's extracted
/// preview — whatever [`crate::loadable_image_bytes`] hands back. EXIF
/// orientation is applied before the downsample, so the preview is the
/// right way up and every check sees it that way.
pub fn encode(bytes: &[u8]) -> Result<Vec<u8>> {
    let (rgb, w, h) = render_bytes_to_rgb(bytes, PREVIEW_PX)?;
    Ok(encode_webp_rgb(&rgb, w, h, PREVIEW_QUALITY))
}

/// Read `path` and encode its canonical preview.
pub fn encode_path(path: &Path) -> Result<Vec<u8>> {
    encode(&crate::loadable_image_bytes(path)?)
}

/// Decode a canonical preview into the frame every check runs on.
///
/// This is the *only* sanctioned way to get pixels out of a stored
/// preview. Anything that decodes the original file instead is computing
/// on a buffer nothing else in the system holds.
pub fn decode(webp: &[u8]) -> Result<RgbImage> {
    let (rgb, w, h) = decode_webp_rgb(webp)?;
    RgbImage::from_raw(w, h, rgb).ok_or_else(|| anyhow::anyhow!("preview: malformed {w}x{h} frame"))
}

// ── Primitives ────────────────────────────────────────────────────
//
// These are the generic render/codec helpers the rest of the app uses at
// other sizes and qualities — library thumbnails, cover crops. They live
// here rather than in `maple-ui` so that the canonical preview above and
// every other rendered image in Maple go through one resampler.

/// Decode and resize `path` to at most `max_size` px on the longest edge.
///
/// Returns `(rgb_pixels, width, height)` — 24-bit tightly-packed RGB.
/// EXIF orientation is applied before resizing; Lanczos3 is used for the
/// downsample.
pub fn render_to_rgb(path: &Path, max_size: u32) -> Result<(Vec<u8>, u32, u32)> {
    render_decoded(crate::decode_image(path)?, max_size)
}

/// Decode and resize bytes already in memory.
///
/// The import scan reads each file once on a single thread — a card is one
/// serial bus, and twelve readers on it are slower than one — then hands
/// the bytes here to be decoded on whichever core is free.
pub fn render_bytes_to_rgb(bytes: &[u8], max_size: u32) -> Result<(Vec<u8>, u32, u32)> {
    render_decoded(crate::decode_image_bytes(bytes)?, max_size)
}

fn render_decoded(img: image::DynamicImage, max_size: u32) -> Result<(Vec<u8>, u32, u32)> {
    let rgb_img = img.into_rgb8();
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

/// Resize a sub-rectangle of an RGB buffer to fit inside `(box_w, box_h)`.
///
/// `region` is `(x, y, w, h)` in source pixels and is clamped to the buffer.
/// The result keeps the region's aspect ratio and is never upscaled past
/// the box, so the caller can hand it straight to an `image-fit: contain`
/// element and get letterboxing for free.
///
/// This is what makes the import tournament's zoom worth having: a crop
/// taken here comes off the *original* decode, so zooming in shows the
/// photo's real detail rather than a 256 px preview magnified into mush —
/// which is the whole thing the user is being asked to judge.
pub fn render_region(
    rgb: &[u8],
    src_w: u32,
    src_h: u32,
    region: (u32, u32, u32, u32),
    box_w: u32,
    box_h: u32,
) -> Result<(Vec<u8>, u32, u32)> {
    if src_w == 0 || src_h == 0 || rgb.len() < (src_w as usize * src_h as usize * 3) {
        anyhow::bail!("render_region: {src_w}x{src_h} does not describe {} bytes", rgb.len());
    }
    let (rx, ry, rw, rh) = clamp_region(src_w, src_h, region);
    let (dst_w, dst_h) = fit_box(rw, rh, box_w.max(1), box_h.max(1));

    // Copy the region out row by row — `fir` wants a tight buffer, and a
    // crop is the one shape a stride cannot express here.
    let mut cropped = Vec::with_capacity(rw as usize * rh as usize * 3);
    for row in 0..rh {
        let start = ((ry + row) as usize * src_w as usize + rx as usize) * 3;
        cropped.extend_from_slice(&rgb[start..start + rw as usize * 3]);
    }

    let src_img = fir::images::ImageRef::new(rw, rh, &cropped, fir::PixelType::U8x3)
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

fn clamp_region(src_w: u32, src_h: u32, (x, y, w, h): (u32, u32, u32, u32)) -> (u32, u32, u32, u32) {
    let x = x.min(src_w - 1);
    let y = y.min(src_h - 1);
    let w = w.clamp(1, src_w - x);
    let h = h.clamp(1, src_h - y);
    (x, y, w, h)
}

/// Scale `(w, h)` to fit inside `max_w x max_h`, never upscaling.
pub fn fit_box(w: u32, h: u32, max_w: u32, max_h: u32) -> (u32, u32) {
    if w <= max_w && h <= max_h {
        return (w.max(1), h.max(1));
    }
    let by_w = max_w as f64 / w.max(1) as f64;
    let by_h = max_h as f64 / h.max(1) as f64;
    let s = by_w.min(by_h);
    (((w as f64 * s).round() as u32).max(1), ((h as f64 * s).round() as u32).max(1))
}

/// Encode `rgb` pixels as lossy WebP at the given quality (0–100).
pub fn encode_webp_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Vec<u8> {
    webp::Encoder::from_rgb(rgb, width, height)
        .encode(quality as f32)
        .to_vec()
}

/// Decode WebP bytes back to tight RGB pixels, as `(rgb, width, height)`.
pub fn decode_webp_rgb(webp: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::load_from_memory(webp)?;
    let rgb = img.into_rgb8();
    let (w, h) = rgb.dimensions();
    Ok((rgb.into_raw(), w, h))
}

/// Scale `(w, h)` to fit inside `max` on the longest edge, never upscaling.
pub fn fit_dims(w: u32, h: u32, max: u32) -> (u32, u32) {
    if w <= max && h <= max {
        return (w.max(1), h.max(1));
    }
    if w >= h {
        let scaled = (h as u64 * max as u64 / w.max(1) as u64) as u32;
        (max, scaled.max(1))
    } else {
        let scaled = (w as u64 * max as u64 / h.max(1) as u64) as u32;
        (scaled.max(1), max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gradient with hard edges — something a lossy codec has to work at,
    /// so a round trip that silently returned a flat image would show up.
    fn sample(w: u32, h: u32) -> Vec<u8> {
        let img = RgbImage::from_fn(w, h, |x, y| {
            let checker = if (x / 8 + y / 8) % 2 == 0 { 255 } else { 0 };
            image::Rgb([(x * 255 / w.max(1)) as u8, (y * 255 / h.max(1)) as u8, checker])
        });
        let mut png = Vec::new();
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .unwrap();
        png
    }

    fn render_to_rgb_from(png: &[u8]) -> (Vec<u8>, u32, u32) {
        render_bytes_to_rgb(png, 10_000).unwrap()
    }

    #[test]
    fn the_canonical_preview_is_capped_at_the_frame_size() {
        let webp = encode(&sample(900, 600)).unwrap();
        let frame = decode(&webp).unwrap();
        assert_eq!(frame.dimensions(), (PREVIEW_PX, 170));
    }

    #[test]
    fn a_small_photo_is_not_upscaled() {
        let webp = encode(&sample(64, 48)).unwrap();
        assert_eq!(decode(&webp).unwrap().dimensions(), (64, 48));
    }

    /// The invariant the whole module exists for: the frame a check runs on
    /// is a pure function of the *kept* WebP, so recomputing later — from a
    /// cache, on another machine — cannot disagree with the scan.
    #[test]
    fn decoding_the_kept_preview_reproduces_the_frame_exactly() {
        let webp = encode(&sample(640, 480)).unwrap();
        let first = decode(&webp).unwrap();
        let second = decode(&webp).unwrap();
        assert_eq!(first.as_raw(), second.as_raw());

        // And re-encoding the *frame* is not the same thing as keeping the
        // WebP: a second lossy pass moves the pixels. Nothing may round
        // trip twice.
        let again = encode_webp_rgb(first.as_raw(), first.width(), first.height(), PREVIEW_QUALITY);
        let twice = decode(&again).unwrap();
        assert_eq!(twice.dimensions(), first.dimensions());
    }

    #[test]
    fn the_preview_is_much_smaller_than_the_frame_it_decodes_to() {
        let webp = encode(&sample(1200, 800)).unwrap();
        let frame = decode(&webp).unwrap();
        assert!(
            webp.len() * 4 < frame.as_raw().len(),
            "a {} B preview against a {} B frame is not worth keeping",
            webp.len(),
            frame.as_raw().len()
        );
    }

    #[test]
    fn a_region_comes_back_at_the_region_s_aspect_ratio() {
        let (rgb, w, h) = render_to_rgb_from(&sample(400, 300));
        // A 200x100 crop into a 400x400 box: aspect preserved, not upscaled
        // past the box.
        let (_, ow, oh) = render_region(&rgb, w, h, (10, 10, 200, 100), 400, 400).unwrap();
        assert_eq!((ow, oh), (200, 100));
        let (_, ow, oh) = render_region(&rgb, w, h, (10, 10, 200, 100), 100, 400).unwrap();
        assert_eq!((ow, oh), (100, 50));
    }

    #[test]
    fn a_region_running_off_the_edge_is_pulled_back_inside() {
        let (rgb, w, h) = render_to_rgb_from(&sample(64, 48));
        // Asking for more than there is must not read past the buffer.
        let (px, ow, oh) = render_region(&rgb, w, h, (60, 44, 500, 500), 200, 200).unwrap();
        assert_eq!((ow, oh), (4, 4));
        assert_eq!(px.len(), 4 * 4 * 3);
    }

    #[test]
    fn a_buffer_that_does_not_match_its_dimensions_is_an_error_not_a_panic() {
        assert!(render_region(&[0u8; 12], 100, 100, (0, 0, 10, 10), 10, 10).is_err());
    }

    #[test]
    fn fit_box_never_upscales() {
        assert_eq!(fit_box(50, 40, 400, 400), (50, 40));
        assert_eq!(fit_box(800, 400, 200, 200), (200, 100));
        assert_eq!(fit_box(400, 800, 200, 200), (100, 200));
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

    #[test]
    fn garbage_is_an_error_not_a_panic() {
        assert!(decode(b"not a webp").is_err());
        assert!(encode(b"not an image").is_err());
    }
}
