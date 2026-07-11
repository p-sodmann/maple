//! Shared "representative crop" loading — the People page's face-avatar
//! crops and the Collections gallery's cover-image crops both boil down to
//! the same operation: decode-or-cache a square thumbnail from a source
//! image, optionally cut to a bounding box.
//!
//! Two layers of caching, shared by both callers:
//! - **redb** (`maple_db::ThumbnailCache`'s `face_crops`/`covers` tables) —
//!   persists across app restarts, keyed by entity id (face id / collection
//!   id). Callers pass the pre-fetched bytes in and a `store` closure out,
//!   since which table to hit differs per caller.
//! - **In-memory `CropCache`** — avoids even the WebP decode cost on repeat
//!   tab visits within one session; invalidated by comparing the *source* id
//!   (representative face id / representative image id) the crop was last
//!   rendered from, so a changed representative naturally busts the cache.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};

use slint::{Image, Rgb8Pixel, SharedPixelBuffer};

use crate::{face_crop, thumbnail};

/// Per-session in-memory memoization: entity id → (source id the crop was
/// rendered from, tightly-packed `out_px × out_px` RGB pixels).
pub type CropCache = Arc<Mutex<HashMap<i64, (i64, Arc<Vec<u8>>)>>>;

/// Decode-or-cache a square RGB crop.
///
/// `bbox` selects the operation: `Some` crops to a face bounding box via
/// [`face_crop::extract_crop`]; `None` takes the largest centred square of
/// the whole image via [`thumbnail::center_square_crop`].
///
/// `cached` is the redb-fetched WebP bytes (or `None` on a miss); on a miss
/// this renders fresh pixels from `path`, encodes them as WebP, and passes
/// the bytes to `store` so the caller can write them into whichever redb
/// table applies (`face_crops` vs `covers`).
pub fn extract_and_cache(
    path: &Path,
    bbox: Option<[f32; 4]>,
    out_px: u32,
    quality: u8,
    cached: Option<Vec<u8>>,
    store: impl FnOnce(&[u8]),
) -> anyhow::Result<Vec<u8>> {
    if let Some(webp) = cached {
        return Ok(thumbnail::decode_webp_rgb(&webp)?.0);
    }
    let rgb = match bbox {
        Some(b) => face_crop::extract_crop(path, b, out_px)?,
        None => thumbnail::center_square_crop(path, out_px)?,
    };
    let webp = thumbnail::encode_webp_rgb(&rgb, out_px, out_px, quality);
    store(&webp);
    Ok(rgb)
}

/// Build a Slint `Image` from a tightly-packed `px × px` RGB buffer.
pub fn image_from_rgb(pixels: &[u8], px: u32) -> Image {
    let mut pb = SharedPixelBuffer::<Rgb8Pixel>::new(px, px);
    pb.make_mut_bytes().copy_from_slice(pixels);
    Image::from_rgb8(pb)
}
