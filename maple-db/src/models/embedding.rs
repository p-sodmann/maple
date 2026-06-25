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
//! - Input  (first) : `[1, 3, 112, 112]` float32 — RGB `[0, 255]`
//! - Output (first) : `[1, 512]`         float32 — raw embedding (L2-normalised here)
//!
//! The preprocessing pipeline applied inside [`OnnxFaceEmbedder::embed_face_crop`]:
//! 1. `SwapChannels` — BGR → RGB
//! 2. `HwcToChw`     — `[112, 112, 3]` → `[3, 112, 112]`
//! 3. `AddBatchDim`  — `[3, 112, 112]` → `[1, 3, 112, 112]`

use std::path::Path;

use anyhow::{anyhow, Context, Result};
use ndarray::{Array2, ArrayView3};
use tokenizers::Tokenizer;

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
/// Used by semantic search to index each sentence of an AI description (and to
/// embed the search query) as a dense vector.  Takes `&mut self` to match
/// `ort::Session::run`, which mutates internal IO-binding state.
pub trait TextEmbeddingModel: Send + Sync {
    /// Compute an L2-normalised sentence embedding for `text`.
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>>;

    /// Dimensionality of the returned embedding (e.g. 384 for all-MiniLM-L6-v2).
    fn embedding_dim(&self) -> usize;
}

// ── OnnxFaceEmbedder ───────────────────────────────────────────────────────────

/// ArcFace-compatible ONNX face embedder.
///
/// Loads any InsightFace-compatible ArcFace ONNX model and implements
/// [`EmbeddingModel`].  The preprocessing pipeline (BGR→RGB → CHW → batch) is
/// applied internally — callers pass raw BGR `[0, 255]` crops.
pub struct OnnxFaceEmbedder {
    session: OnnxSession,
    /// Preprocessing: BGR→RGB, HWC→CHW, add batch dim.
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
        // Preprocess [H,W,C] BGR → [1,C,H,W] RGB.
        let tensor = self
            .preprocessor
            .run(crop.to_owned())?
            .into_dimensionality::<ndarray::Ix4>()
            .context("embedder preprocessing must produce a 4-D tensor [1,C,H,W]")?;

        // Run inference.
        let input_name = &self.session.input_names[0];
        let tensor_ref =
            ort::value::TensorRef::from_array_view(tensor.view()).context("creating embedder input tensor")?;
        let outputs = self
            .session
            .session
            .run(ort::inputs![input_name.as_str() => tensor_ref])
            .context("running face embedder")?;

        // Extract embedding and L2-normalise.
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

// ── OnnxTextEmbedder ─────────────────────────────────────────────────────────

/// Maximum number of tokens fed to the model.  Sentences (and short search
/// queries) are well under this; longer inputs are truncated.
const MAX_TOKENS: usize = 256;

/// Sentence-transformer ONNX embedder (e.g. all-MiniLM-L6-v2, all-mpnet-base-v2).
///
/// Tokenizes input with a HuggingFace `tokenizer.json`, runs the transformer,
/// applies attention-masked mean pooling over the token embeddings, and
/// L2-normalises the result — matching the sentence-transformers default.
pub struct OnnxTextEmbedder {
    session: OnnxSession,
    tokenizer: Tokenizer,
    /// Resolved model input names.
    input_ids_name: String,
    attention_mask_name: String,
    /// Present only for models that take token type ids (BERT-family; MPNet omits it).
    token_type_ids_name: Option<String>,
    embedding_dim: usize,
}

impl OnnxTextEmbedder {
    /// Load a sentence-transformer ONNX model + its tokenizer.
    pub fn load(onnx_path: &Path, tokenizer_path: &Path, device: &ModelDevice) -> Result<Self> {
        let session = OnnxSession::load(onnx_path, device)
            .with_context(|| format!("loading text embedder: {}", onnx_path.display()))?;

        let tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow!("loading tokenizer {}: {e}", tokenizer_path.display()))?;

        // Resolve input names (fall back to positional order if unconventional).
        let find = |needle: &str| session.input_names.iter().find(|n| n.contains(needle)).cloned();
        let input_ids_name = find("input_ids")
            .or_else(|| session.input_names.first().cloned())
            .ok_or_else(|| anyhow!("text model has no inputs"))?;
        let attention_mask_name = find("attention_mask")
            .or_else(|| session.input_names.get(1).cloned())
            .ok_or_else(|| anyhow!("text model has no attention_mask input"))?;
        let token_type_ids_name = find("token_type_ids");

        // Last hidden dimension is static even though sequence length is dynamic.
        let embedding_dim = session
            .session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(384);

        Ok(Self {
            session,
            tokenizer,
            input_ids_name,
            attention_mask_name,
            token_type_ids_name,
            embedding_dim,
        })
    }
}

impl TextEmbeddingModel for OnnxTextEmbedder {
    fn embed_text(&mut self, text: &str) -> Result<Vec<f32>> {
        use ort::value::TensorRef;

        // ── Tokenize ──────────────────────────────────────────────
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow!("tokenizing text: {e}"))?;

        let mut ids: Vec<i64> = encoding.get_ids().iter().map(|&x| x as i64).collect();
        let mut mask: Vec<i64> = encoding.get_attention_mask().iter().map(|&x| x as i64).collect();
        let mut types: Vec<i64> = encoding.get_type_ids().iter().map(|&x| x as i64).collect();

        if ids.is_empty() {
            return Ok(vec![0.0; self.embedding_dim]);
        }
        if ids.len() > MAX_TOKENS {
            ids.truncate(MAX_TOKENS);
            mask.truncate(MAX_TOKENS);
            types.truncate(MAX_TOKENS);
        }

        let seq = ids.len();
        let ids_arr = Array2::from_shape_vec((1, seq), ids)?;
        let mask_arr = Array2::from_shape_vec((1, seq), mask.clone())?;

        // ── Inference ─────────────────────────────────────────────
        let ids_ref = TensorRef::from_array_view(ids_arr.view())
            .context("creating input_ids tensor")?;
        let mask_ref = TensorRef::from_array_view(mask_arr.view())
            .context("creating attention_mask tensor")?;

        let outputs = match &self.token_type_ids_name {
            Some(types_name) => {
                let types_arr = Array2::from_shape_vec((1, seq), types)?;
                let types_ref = TensorRef::from_array_view(types_arr.view())
                    .context("creating token_type_ids tensor")?;
                self.session.session.run(ort::inputs![
                    self.input_ids_name.as_str()      => ids_ref,
                    self.attention_mask_name.as_str() => mask_ref,
                    types_name.as_str()               => types_ref,
                ])
            }
            None => self.session.session.run(ort::inputs![
                self.input_ids_name.as_str()      => ids_ref,
                self.attention_mask_name.as_str() => mask_ref,
            ]),
        }
        .context("running text embedder")?;

        // ── Pool ──────────────────────────────────────────────────
        let output_name = &self.session.output_names[0];
        let (shape, data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extracting text embedding tensor")?;

        // 3-D [1, seq, hidden] → mean-pool over tokens; 2-D [1, hidden] →
        // already a sentence vector, use as-is.
        let pooled = if shape.len() >= 3 {
            mean_pool(data, seq, &mask)
        } else {
            data.to_vec()
        };

        Ok(l2_normalize(pooled))
    }

    fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }
}

/// Attention-masked mean pooling of token embeddings `[seq, hidden]`
/// (flattened row-major), weighted by the 0/1 attention `mask`.
fn mean_pool(data: &[f32], seq: usize, mask: &[i64]) -> Vec<f32> {
    let hidden = data.len() / seq.max(1);
    let mut pooled = vec![0.0f32; hidden];
    let mut denom = 0.0f32;
    for t in 0..seq {
        let m = *mask.get(t).unwrap_or(&1) as f32;
        denom += m;
        let base = t * hidden;
        for h in 0..hidden {
            pooled[h] += data[base + h] * m;
        }
    }
    if denom > 0.0 {
        for v in &mut pooled {
            *v /= denom;
        }
    }
    pooled
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
