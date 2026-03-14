//! Device-aware `ort::Session` wrapper.
//!
//! [`OnnxSession`] is a thin newtype around `ort::Session` that:
//! 1. Applies the requested [`ModelDevice`]'s execution providers.
//! 2. Caches the model's input and output names so callers don't have to
//!    query `session.inputs[i].name` in hot paths.
//!
//! # Graph optimization
//!
//! ORT's default `All` (Level3+) optimization is extremely slow for large
//! dynamic-shape models like the atksh joined detector.  We use `Level1`
//! (basic intra-node fusions only) which loads in a fraction of the time with
//! negligible inference overhead on CPU.
//!
//! Memory-pattern optimization is also disabled because the atksh model has
//! dynamic H×W input; ORT requires stable shapes to use memory patterns.

use std::path::Path;

use anyhow::{Context, Result};
use ort::session::builder::GraphOptimizationLevel;
use tracing::info;

use super::device::ModelDevice;

/// A loaded ONNX model ready for inference.
pub struct OnnxSession {
    pub session: ort::session::Session,
    /// Ordered input names (converted from `Arc<str>` at construction time).
    pub input_names: Vec<String>,
    /// Ordered output names.
    pub output_names: Vec<String>,
}

impl OnnxSession {
    /// Load a model from `path`, targeting `device`.
    ///
    /// Execution providers are registered in priority order; CPU is always the
    /// final fallback so this never fails due to missing GPU support.
    pub fn load(path: &Path, device: &ModelDevice) -> Result<Self> {
        info!(path = %path.display(), "loading ONNX model…");

        let builder = ort::session::builder::SessionBuilder::new()
            .context("creating ort session builder")?;

        info!("step2");
        // Level1 = basic intra-node fusions only; much faster to load than
        // the default All/Level3 without measurable inference slowdown on CPU.
        let builder = builder
            .with_optimization_level(GraphOptimizationLevel::Level1)
            .map_err(|e| anyhow::anyhow!("setting graph optimization level: {e}"))?;

        // Disable memory-pattern optimization: required for dynamic-shape
        // inputs (atksh model accepts arbitrary H×W).
        let builder = builder
            .with_memory_pattern(false)
            .map_err(|e| anyhow::anyhow!("disabling memory pattern: {e}"))?;

        let mut builder = device
            .apply_to_builder(builder)
            .with_context(|| format!("configuring execution providers for {device}"))?;

        info!(path = %path.display(), "building ONNX session (graph compile)…");
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("loading ONNX model: {}", path.display()))?;

        info!(path = %path.display(), "ONNX session ready");

        let input_names: Vec<String> = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names: Vec<String> = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        Ok(Self { session, input_names, output_names })
    }
}
