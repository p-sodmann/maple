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

use super::{
    detection::{DetectionModel, OnnxFaceDetector},
    device::ModelDevice,
    embedding::{EmbeddingModel, OnnxFaceEmbedder},
    session::OnnxSession,
};

// ── ModelFactory ──────────────────────────────────────────────────────────────

/// Builder that constructs inference models targeted at a specific device.
///
/// Use [`with_device`](ModelFactory::with_device) to select GPU execution;
/// defaults to CPU when not specified.
#[derive(Debug, Default)]
pub struct ModelFactory {
    device: ModelDevice,
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

    // ── Face detection ────────────────────────────────────────────────────

    /// Build an atksh face detector, optionally combined with an ArcFace
    /// embedder for identity embeddings.
    ///
    /// - `detector_path`: path to the atksh joined ONNX model.
    /// - `embedder_path`: optional path to an InsightFace ArcFace ONNX model.
    ///   Pass `None` to run detection without embedding (person grouping
    ///   will be unavailable but bounding boxes are still stored).
    pub fn build_face_detector(
        &self,
        detector_path: &Path,
        embedder_path: Option<&Path>,
    ) -> Result<Box<dyn DetectionModel>> {
        let embedder: Option<Box<dyn EmbeddingModel>> = embedder_path
            .map(|p| -> Result<Box<dyn EmbeddingModel>> {
                Ok(Box::new(OnnxFaceEmbedder::load(p, &self.device)?))
            })
            .transpose()?;

        let detector = OnnxFaceDetector::load(detector_path, &self.device, embedder)?;
        Ok(Box::new(detector))
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
