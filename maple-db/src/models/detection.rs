//! Detection model trait and atksh-based ONNX face detector.
//!
//! # Model I/O (atksh joined model)
//!
//! - Input  `input`           : `[H, W, 3]` float32 — BGR pixel values `[0, 255]`
//! - Output 0 `scores`        : `[N]`               — confidence per face
//! - Output 1 `bboxes`        : `[N, 4]`            — `[x1, y1, x2, y2]` in pixels
//! - Output 2 `keypoints`     : `[N, 5, 2]`         — 5 facial landmarks (pixels)
//! - Output 3 `aligned_imgs`  : `[N, 112, 112, 3]`  — pre-aligned BGR `[0, 255]` face crops
//! - Output 4 `landmarks`     : `[N, *, 2]`
//! - Output 5 `affine_matrices`: `[N, 2, 3]`
//!
//! `aligned_imgs[i]` is fed directly to the optional ArcFace embedder —
//! no manual cropping or resizing is required.
//!
//! Download: <https://github.com/atksh/onnx-facial-lmk-detector/releases>

use std::path::Path;

use anyhow::{Context, Result};
use ndarray::{s, Array3, ArrayView2, ArrayView4, Ix3};
use tracing::debug;

use super::{
    device::ModelDevice,
    embedding::EmbeddingModel,
    session::OnnxSession,
};
use crate::face_detector::DetectedFace;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// Detects faces in an image and returns one [`DetectedFace`] per person.
///
/// Implementations must be `Send + Sync` so they can be moved into background
/// threads or shared behind `Arc`.
pub trait DetectionModel: Send + Sync {
    /// Run detection (and optional embedding) on the image at `path`.
    fn detect(&mut self, path: &Path) -> Result<Vec<DetectedFace>>;
}

// ── OnnxFaceDetector ──────────────────────────────────────────────────────────

/// ONNX-backed face detector using the atksh joined model.
///
/// Optionally holds an [`EmbeddingModel`] (typically [`OnnxFaceEmbedder`]).
/// When an embedder is present, `aligned_imgs` from the detector are fed
/// through it to produce 512-dim ArcFace identity embeddings.
pub struct OnnxFaceDetector {
    /// atksh joined detector: SCRFD detection + landmark alignment.
    pub(super) detector: OnnxSession,
    /// Optional ArcFace embedder for per-face identity vectors.
    pub(super) embedder: Option<Box<dyn EmbeddingModel>>,
}

impl OnnxFaceDetector {
    /// Create from a pre-loaded `OnnxSession` and optional embedder.
    pub fn new(detector: OnnxSession, embedder: Option<Box<dyn EmbeddingModel>>) -> Self {
        debug!(
            inputs  = ?detector.input_names,
            outputs = ?detector.output_names,
            "atksh detector loaded"
        );
        Self { detector, embedder }
    }

    /// Load the atksh detector directly from a path.
    pub fn load(
        path: &Path,
        device: &ModelDevice,
        embedder: Option<Box<dyn EmbeddingModel>>,
    ) -> Result<Self> {
        let session = OnnxSession::load(path, device)
            .with_context(|| format!("loading face detector: {}", path.display()))?;
        Ok(Self::new(session, embedder))
    }
}

impl DetectionModel for OnnxFaceDetector {
    fn detect(&mut self, path: &Path) -> Result<Vec<DetectedFace>> {
        // ── Load image and build BGR float tensor ──────────────────────────
        let img = image::open(path)
            .with_context(|| format!("opening image: {}", path.display()))?
            .to_rgb8();
        let (img_w, img_h) = (img.width() as usize, img.height() as usize);

        // Build [H, W, 3] BGR float32 array directly.
        // atksh expects BGR [0, 255]; image crate gives us RGB, so swap channels.
        let mut img_arr = Array3::<f32>::zeros((img_h, img_w, 3));
        for (x, y, p) in img.enumerate_pixels() {
            img_arr[[y as usize, x as usize, 0]] = p[2] as f32; // B
            img_arr[[y as usize, x as usize, 1]] = p[1] as f32; // G
            img_arr[[y as usize, x as usize, 2]] = p[0] as f32; // R
        }

        // ── Run atksh detector ─────────────────────────────────────────────
        let input_name = &self.detector.input_names[0];
        let tensor =
            ort::value::TensorRef::from_array_view(img_arr.view()).context("creating detector input tensor")?;
        let outputs = self
            .detector
            .session
            .run(ort::inputs![input_name.as_str() => tensor])
            .context("running face detector")?;

        // Use cached output names (positional order: scores, bboxes, _, aligned_imgs, …).
        let scores_name = &self.detector.output_names[0];
        let bboxes_name = &self.detector.output_names[1];
        let aligned_name = &self.detector.output_names[3];

        let (scores_shape, scores_data) = outputs[scores_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("scores")?;
        let (_, bboxes_data) = outputs[bboxes_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("bboxes")?;
        let (_, aligned_data) = outputs[aligned_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("aligned_imgs")?;

        let n = scores_shape[0] as usize;
        let bboxes = ArrayView2::from_shape((n, 4), bboxes_data)
            .context("reshaping bboxes to [N, 4]")?;
        // aligned_imgs shape: [N, 112, 112, 3]
        let aligned4 = ArrayView4::from_shape((n, 112, 112, 3), aligned_data)
            .context("reshaping aligned_imgs to [N, 112, 112, 3]")?;
        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let conf = scores_data[i];

            // Normalise pixel bbox → [0, 1].
            let x1 = (bboxes[[i, 0]] / img_w as f32).clamp(0.0, 1.0);
            let y1 = (bboxes[[i, 1]] / img_h as f32).clamp(0.0, 1.0);
            let x2 = (bboxes[[i, 2]] / img_w as f32).clamp(0.0, 1.0);
            let y2 = (bboxes[[i, 3]] / img_h as f32).clamp(0.0, 1.0);

            // ── Optional ArcFace embedding pass ───────────────────────────
            // aligned4[i] is [112, 112, 3] BGR [0, 255].
            let embedding = if let Some(ref mut embedder) = self.embedder {
                let crop = aligned4
                    .slice(s![i, .., .., ..])
                    .into_dimensionality::<Ix3>()
                    .context("extracting aligned crop")?;
                embedder.embed_face_crop(crop.view())?
            } else {
                vec![]
            };

            result.push(DetectedFace { bbox: [x1, y1, x2, y2], embedding, confidence: conf });
        }

        Ok(result)
    }
}
