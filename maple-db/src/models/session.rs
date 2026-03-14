//! Device-aware `ort::Session` wrapper.
//!
//! [`OnnxSession`] is a thin newtype around `ort::Session` that:
//! 1. Applies the requested [`ModelDevice`]'s execution providers.
//! 2. Caches the model's input and output names so callers don't have to
//!    query `session.inputs[i].name` in hot paths.

use std::path::Path;

use anyhow::{Context, Result};

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
        let builder = ort::session::builder::SessionBuilder::new()
            .context("creating ort session builder")?;
        let mut builder = device
            .apply_to_builder(builder)
            .with_context(|| format!("configuring execution providers for {device}"))?;
        let session = builder
            .commit_from_file(path)
            .with_context(|| format!("loading ONNX model: {}", path.display()))?;

        let input_names = session
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect();
        let output_names = session
            .outputs()
            .iter()
            .map(|o| o.name().to_string())
            .collect();

        Ok(Self { session, input_names, output_names })
    }
}
