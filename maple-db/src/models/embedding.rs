//! Embedding model traits and implementations.
//!
//! # Traits
//!
//! - [`EmbeddingModel`] — takes a pre-cropped face image and returns a
//!   512-dim L2-normalised identity embedding (e.g. ArcFace).
//! - [`TextEmbeddingModel`] — takes a text string and returns a dense
//!   embedding vector.  Reserved for future sentence-transformer support to
//!   improve semantic search.
//!
//! # Concrete implementation
//!
//! [`OnnxFaceEmbedder`] wraps an ArcFace ONNX model and implements
//! [`EmbeddingModel`].
//!
//! Expected model I/O (InsightFace ArcFace):
//! - Input  (first) : `[1, 3, 112, 112]` float32 — BGR, normalised to `[-1, 1]`
//! - Output (first) : `[1, 512]`         float32 — raw embedding (L2-normalised here)
//!
//! The preprocessing pipeline applied inside [`OnnxFaceEmbedder::embed_face_crop`]:
//! 1. `LinearScale { scale: 1/127.5, offset: -1.0 }` — remap `[0, 255]` → `[-1, 1]`
//! 2. `HwcToChw`    — `[112, 112, 3]` → `[3, 112, 112]`
//! 3. `AddBatchDim` — `[3, 112, 112]` → `[1, 3, 112, 112]`

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::ArrayView3;

use super::{
    device::ModelDevice,
    preprocessor::{Preprocessor, PreprocessStep},
    session::OnnxSession,
};

// ── Traits ─────────────────────────────────────────────────────────────────────

/// Produces a dense identity embedding from a face-crop image.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// background threads or wrapped in `Arc`.
pub trait EmbeddingModel: Send + Sync {
    /// Compute an L2-normalised embedding from a `[H, W, C]` BGR float32 crop.
    ///
    /// For ArcFace the expected crop size is 112 × 112.
    fn embed_face_crop(&mut self, crop: ArrayView3<f32>) -> Result<Vec<f32>>;

    /// Dimensionality of the returned embedding vector (e.g. 512 for ArcFace).
    fn embedding_dim(&self) -> usize;
}

/// Produces a dense embedding from a text string.
///
/// Intended for sentence-transformer models that will improve semantic search
/// by indexing AI-generated captions and EXIF metadata as dense vectors.
pub trait TextEmbeddingModel: Send + Sync {
    fn embed_text(&self, text: &str) -> Result<Vec<f32>>;

    fn embedding_dim(&self) -> usize;
}

// ── OnnxFaceEmbedder ───────────────────────────────────────────────────────────

/// ArcFace-compatible ONNX face embedder.
///
/// Loads any InsightFace-compatible ArcFace ONNX model and implements
/// [`EmbeddingModel`].  The preprocessing pipeline (scale → CHW → batch) is
/// applied internally — callers pass raw BGR `[0, 255]` crops.
pub struct OnnxFaceEmbedder {
    session: OnnxSession,
    /// Preprocessing: scale to [-1,1], HWC→CHW, add batch dim.
    preprocessor: Preprocessor,
    embedding_dim: usize,
}

impl OnnxFaceEmbedder {
    /// Load an ArcFace ONNX model.
    pub fn load(path: &Path, device: &ModelDevice) -> Result<Self> {
        let session = OnnxSession::load(path, device)
            .with_context(|| format!("loading face embedder: {}", path.display()))?;

        // Infer embedding dimension from the model's first output shape.
        // ArcFace outputs [1, 512]; we default to 512 if inspection fails.
        let embedding_dim = session
            .session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(512);

        // ArcFace preprocessing: BGR→RGB, HWC→CHW, add batch.
        // Input from atksh detector is BGR [0,255]; model expects RGB CHW [0,255].
        let preprocessor = Preprocessor::new()
            .add(PreprocessStep::SwapChannels)
            .add(PreprocessStep::HwcToChw)
            .add(PreprocessStep::AddBatchDim);

        Ok(Self { session, preprocessor, embedding_dim })
    }
}

impl EmbeddingModel for OnnxFaceEmbedder {
    fn embed_face_crop(&mut self, crop: ArrayView3<f32>) -> Result<Vec<f32>> {
        // Apply preprocessing pipeline → [1, 3, 112, 112].
        let tensor_dyn = self.preprocessor.run(crop.to_owned())?;

        // Convert from dynamic rank to concrete Ix4 so ort::inputs! accepts it.
        let tensor = tensor_dyn
            .into_dimensionality::<ndarray::Ix4>()
            .context("embedder preprocessing must produce a 4-D tensor [1,C,H,W]")?;

        // Run the ONNX session.
        let input_name = &self.session.input_names[0];
        let tensor_ref =
            ort::value::TensorRef::from_array_view(tensor.view()).context("creating embedder input tensor")?;
        let outputs = self
            .session
            .session
            .run(ort::inputs![input_name.as_str() => tensor_ref])
            .context("running face embedder")?;

        // Extract the first output as a flat Vec<f32>.
        let output_name = &self.session.output_names[0];
        let (_, raw_data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extracting embedding tensor")?;
        let raw: Vec<f32> = raw_data.iter().copied().collect();

        Ok(l2_normalize(raw))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

// ── Helpers ────────────────────────────────────────────────────────────────────

/// L2-normalise a vector in-place.  Returns the input unchanged if the norm
/// is near zero (prevents NaN embeddings for blank / degenerate crops).
pub(crate) fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in &mut v {
            *x /= norm;
        }
    }
    v
}
