//! Perceptual hashing for stack / near-duplicate detection.
//!
//! Wraps `image_hasher` to compute a DCT-based pHash for any image that
//! Maple can load (including raw files via `loadable_image_bytes`).
//!
//! The returned [`ImageHash`] is opaque; use [`phash_similarity`] to convert
//! a pairwise Hamming distance into a normalised [0, 1] similarity score.

use std::path::Path;

use anyhow::{Context, Result};
pub use image_hasher::ImageHash;
use image_hasher::{HashAlg, HasherConfig};

/// Compute a pHash for the image at `path`.
///
/// Raw files (RAF, etc.) are handled transparently via
/// [`loadable_image_bytes`](crate::loadable_image_bytes).
///
/// `hash_size` controls the hash grid (default 8 → 8×8 = 64-bit hash).
/// Larger values produce finer hashes at a small speed cost.
pub fn compute_phash(path: &Path, hash_size: u32) -> Result<ImageHash> {
    let bytes = crate::loadable_image_bytes(path)
        .with_context(|| format!("reading image for pHash: {}", path.display()))?;

    let img = image::load_from_memory(&bytes)
        .with_context(|| format!("decoding image for pHash: {}", path.display()))?;

    // pHash: Median + DCT preprocessing (classic perceptual hash).
    let hasher = HasherConfig::new()
        .hash_alg(HashAlg::Median)
        .preproc_dct()
        .hash_size(hash_size, hash_size)
        .to_hasher();

    Ok(hasher.hash_image(&img))
}

/// Normalised similarity between two pHashes in [0, 1].
///
/// Returns `1.0` for identical hashes and approaches `0.0` as hashes diverge.
/// The denominator is `hash_size²` (total number of bits).
pub fn phash_similarity(a: &ImageHash, b: &ImageHash, hash_size: u32) -> f32 {
    let total_bits = (hash_size * hash_size) as f32;
    let dist = a.dist(b) as f32;
    (total_bits - dist) / total_bits
}
