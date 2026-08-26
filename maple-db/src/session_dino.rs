//! Engine 4 — DINOv2, the baseline the cheap engines are measured against.
//!
//! It lives here rather than beside the other three in `maple-import`
//! because it needs the ONNX session, and `maple-db` is the crate that owns
//! inference. `maple-import` must stay free of `ort` — it is on the import
//! path of every build, including ones with no model at all.
//!
//! This is the *reference*, not the recommendation. A DINOv2 pass is tens
//! of milliseconds per photo on CPU against microseconds for a colour
//! histogram, and it runs serially behind a single `&mut` session, which on
//! the import path throttles the entire scan to its own rate. If a cheap
//! engine segments a real card the same way this one does, the cheap engine
//! wins; where they differ is where the argument for paying for inference
//! actually has to be made.

use anyhow::Result;
use image::DynamicImage;
use maple_import::session::{Frame, Signature, SessionEngine};
use maple_state::StackSettings;

const NAME: &str = "dinov2";

pub struct DinoEngine {
    embedder: crate::models::OnnxImageEmbedder,
    model: String,
}

impl DinoEngine {
    /// Load the embedder named by `[stacks]` in settings.toml, fetching it
    /// from the Hub on first use.
    pub fn load(settings: &StackSettings) -> Result<Self> {
        Ok(Self {
            embedder: crate::load_onnx_embedder(settings)?,
            model: settings.model_repo.clone(),
        })
    }
}

impl SessionEngine for DinoEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        format!("{} ({} dims), cosine distance", self.model, self.embedder.embedding_dim())
    }

    fn default_cut(&self) -> f32 {
        // `1 - cosine`, so this is the 0.90 similarity that
        // `StackSettings::default_threshold` has always used — the same
        // number the existing stack detector is tuned around, kept so a
        // comparison starts from what ships today.
        0.10
    }

    fn signature(&mut self, frame: &Frame<'_>) -> Result<Signature> {
        // The embedder does its own resize and centre crop from whatever
        // it is handed, so the ~256 px frame the harness gives every engine
        // is already what it wants.
        let raw = self.embedder.embed(&DynamicImage::ImageRgb8(frame.rgb.clone()))?;
        Ok(Signature::new(NAME, pool(raw, self.embedder.embedding_dim())))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() || a.values().len() != b.values().len() {
            return 1.0;
        }
        // Cosine runs to -1 for opposed vectors, which would put the
        // distance at 2. Clamping keeps every engine on one 0..=1 scale so
        // the harness can print them in the same column.
        (1.0 - crate::models::image_cosine_similarity(a.values(), b.values())).clamp(0.0, 1.0)
    }
}

/// Reduce a raw DINOv2 output to one global descriptor.
///
/// `OnnxImageEmbedder::embed` returns output 0 verbatim, and for
/// `onnx-community/dinov2-small` that is `last_hidden_state`:
/// `[1, 257, 384]` — a CLS token followed by one token per 14×14 patch,
/// L2-normalised as a single 98 688-float vector. Two things follow, and
/// both are wrong for this job:
///
/// - **385 KB per photo.** A 3000-photo card would hold 1.1 GB of
///   signatures. The cheap engines are in the 80 B – 1.5 KB range.
/// - **Cosine over it compares patches positionally.** Token 96 of one
///   frame is only ever compared against token 96 of the other, so a
///   subject that moved across the frame reads as a scene change even
///   though every engine here is meant to be invariant to exactly that.
///
/// So take the CLS token, the descriptor DINOv2 is actually trained to
/// summarise an image with, and renormalise. Anything whose length is not
/// a whole number of tokens is passed through untouched — a model that
/// already pools is already giving us what we want.
///
/// This is deliberately local to the session engine. `stacker.rs` stores
/// the unpooled vector in `image_hashes` and compares it the same
/// positional way; that is a real bug in the shipped stack detector, but
/// changing it re-keys every stored embedding and is its own change.
fn pool(raw: Vec<f32>, dim: usize) -> Vec<f32> {
    if dim == 0 || raw.len() <= dim || !raw.len().is_multiple_of(dim) {
        return raw;
    }
    let mut cls: Vec<f32> = raw[..dim].to_vec();
    let norm = cls.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > f32::EPSILON {
        for v in cls.iter_mut() {
            *v /= norm;
        }
    }
    cls
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pooling_keeps_the_cls_token_and_renormalises() {
        // Two tokens of 3 dims; the first is the CLS token.
        let raw = vec![3.0, 4.0, 0.0, 9.0, 9.0, 9.0];
        let pooled = pool(raw, 3);
        assert_eq!(pooled.len(), 3);
        let norm = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "got {norm}");
        assert!((pooled[0] - 0.6).abs() < 1e-6 && (pooled[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn an_already_pooled_vector_is_left_alone() {
        let raw = vec![1.0, 0.0, 0.0];
        assert_eq!(pool(raw.clone(), 3), raw);
        // Not a whole number of tokens: not ours to interpret.
        let odd = vec![1.0; 7];
        assert_eq!(pool(odd.clone(), 3), odd);
    }

    #[test]
    fn a_zero_vector_survives_renormalisation() {
        assert_eq!(pool(vec![0.0, 0.0, 5.0, 5.0], 2), vec![0.0, 0.0]);
    }
}
