//! [`ModelFactory`] — constructs ONNX models from paths and a device spec.
//!
//! The factory is the primary entry point for creating inference models.
//! Preprocessing pipelines are wired automatically based on model type; callers
//! do not need to configure them directly.
//!
//! # Example
//!
//! ```ignore
//! use maple_db::models::{ModelDevice, ModelFactory};
//!
//! let factory = ModelFactory::new()
//!     .with_device("cuda:0".parse().unwrap());
//!
//! let detector = factory.build_face_detector(
//!     Path::new("/models/atksh.onnx"),
//!     Some(Path::new("/models/arcface.onnx")),
//! )?;
//!
//! let faces = detector.detect(Path::new("/photos/portrait.jpg"))?;
//! ```

use std::path::Path;

use anyhow::Result;

use std::path::PathBuf;

use maple_state::DetectorKind;

use super::{
    detection::{DetectionModel, OnnxFaceDetector},
    device::ModelDevice,
    embedding::{EmbeddingModel, OnnxFaceEmbedder},
    scrfd::ScrfdDetector,
    session::OnnxSession,
};

// ── ModelFactory ──────────────────────────────────────────────────────────────

/// Builder that constructs inference models targeted at a specific device.
///
/// Use [`with_device`](ModelFactory::with_device) to select GPU execution;
/// defaults to CPU when not specified.
///
/// Use [`with_debug_dir`](ModelFactory::with_debug_dir) to enable saving
/// aligned face crops to disk for debugging.
#[derive(Debug, Default)]
pub struct ModelFactory {
    device: ModelDevice,
    /// When `Some`, detectors write 112×112 aligned face crops here.
    debug_dir: Option<PathBuf>,
}

impl ModelFactory {
    /// Create a factory targeting the CPU (default).
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the execution device for all models built by this factory.
    ///
    /// ```ignore
    /// let factory = ModelFactory::new().with_device("cuda:0".parse()?);
    /// ```
    pub fn with_device(mut self, device: ModelDevice) -> Self {
        self.device = device;
        self
    }

    /// Enable debug crop saving.  Detectors will write one 112×112 PNG per
    /// detected face into `dir` after each image is processed.
    pub fn with_debug_dir(mut self, dir: PathBuf) -> Self {
        self.debug_dir = Some(dir);
        self
    }

    // ── Face detection ────────────────────────────────────────────────────

    /// Build a face detector of the given `kind`, optionally combined with an
    /// ArcFace embedder for identity embeddings.
    ///
    /// - `detector_path`: path to the detector ONNX model.
    /// - `embedder_path`: optional path to an InsightFace ArcFace ONNX model.
    ///   Pass `None` to run detection without embedding (person grouping
    ///   will be unavailable but bounding boxes are still stored).
    /// - `kind`: which detector backend to instantiate.
    pub fn build_face_detector(
        &self,
        detector_path: &Path,
        embedder_path: Option<&Path>,
        kind: DetectorKind,
    ) -> Result<Box<dyn DetectionModel>> {
        let embedder: Option<Box<dyn EmbeddingModel>> = embedder_path
            .map(|p| -> Result<Box<dyn EmbeddingModel>> {
                Ok(Box::new(OnnxFaceEmbedder::load(p, &self.device)?))
            })
            .transpose()?;

        match kind {
            DetectorKind::Atksh => {
                let detector = OnnxFaceDetector::load(
                    detector_path,
                    &self.device,
                    embedder,
                    self.debug_dir.clone(),
                )?;
                Ok(Box::new(detector))
            }
            DetectorKind::Scrfd => {
                let detector = ScrfdDetector::load(
                    detector_path,
                    &self.device,
                    embedder,
                    self.debug_dir.clone(),
                )?;
                Ok(Box::new(detector))
            }
        }
    }

    // ── Face embedding (standalone) ───────────────────────────────────────

    /// Build a standalone face embedder.
    ///
    /// Useful for re-embedding face crops that were detected by a separate
    /// pipeline (e.g. a custom detector).
    pub fn build_face_embedder(&self, model_path: &Path) -> Result<Box<dyn EmbeddingModel>> {
        Ok(Box::new(OnnxFaceEmbedder::load(model_path, &self.device)?))
    }

    // ── Raw ONNX session ──────────────────────────────────────────────────

    /// Load a raw ONNX session for custom inference pipelines.
    ///
    /// Use this when the higher-level builders don't cover a specific model
    /// architecture (e.g. a custom detection head or a depth estimator).
    pub fn load_session(&self, model_path: &Path) -> Result<OnnxSession> {
        OnnxSession::load(model_path, &self.device)
    }
}
