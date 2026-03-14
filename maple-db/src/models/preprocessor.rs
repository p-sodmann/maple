//! Config-driven image preprocessing pipeline.
//!
//! A [`Preprocessor`] is a sequence of [`PreprocessStep`]s that transforms a
//! raw `[H, W, C]` float32 image array into the tensor shape and value range
//! expected by a particular ONNX model.
//!
//! # Typical pipelines
//!
//! **atksh face detector** (BGR [0, 255], dynamic HWC):
//! ```ignore
//! Preprocessor::new()
//!     .add(PreprocessStep::SwapChannels)   // RGB → BGR
//! ```
//!
//! **ArcFace embedder** (BGR [-1, 1], fixed NCHW 1×3×112×112):
//! ```ignore
//! Preprocessor::new()
//!     .add(PreprocessStep::LinearScale { scale: 1.0 / 127.5, offset: -1.0 })
//!     .add(PreprocessStep::HwcToChw)
//!     .add(PreprocessStep::AddBatchDim)
//! ```
//!
//! **ImageNet models** (RGB [0, 1] normalised, NCHW):
//! ```ignore
//! Preprocessor::new()
//!     .add(PreprocessStep::LinearScale { scale: 1.0 / 255.0, offset: 0.0 })
//!     .add(PreprocessStep::Normalize {
//!         mean: [0.485, 0.456, 0.406],
//!         std:  [0.229, 0.224, 0.225],
//!     })
//!     .add(PreprocessStep::HwcToChw)
//!     .add(PreprocessStep::AddBatchDim)
//! ```

use anyhow::{bail, Result};
use ndarray::{s, Array3, ArrayD, Axis};
use serde::{Deserialize, Serialize};

// ── Steps ─────────────────────────────────────────────────────────────────────

/// A single image transformation applied in sequence.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum PreprocessStep {
    /// Swap channel 0 and channel 2 (RGB ↔ BGR).
    ///
    /// Requires a 3-D `[H, W, C]` array.
    SwapChannels,

    /// Element-wise affine scale: `output = input * scale + offset`.
    ///
    /// Examples:
    /// - `[0, 255]` → `[-1, 1]`: `scale = 1/127.5`, `offset = -1.0`
    /// - `[0, 255]` → `[0, 1]`:  `scale = 1/255.0`, `offset = 0.0`
    LinearScale { scale: f32, offset: f32 },

    /// Per-channel normalise: `output[c] = (input[c] - mean[c]) / std[c]`.
    ///
    /// Requires a 3-D `[H, W, C]` array (apply before [`HwcToChw`]).
    Normalize { mean: [f32; 3], std: [f32; 3] },

    /// Transpose `[H, W, C]` → `[C, H, W]` (HWC to CHW).
    ///
    /// Requires a 3-D array.
    HwcToChw,

    /// Insert a leading batch dimension: `[…]` → `[1, …]`.
    AddBatchDim,
}

// ── Preprocessor ──────────────────────────────────────────────────────────────

/// An ordered preprocessing pipeline.
///
/// Build it with [`new`](Preprocessor::new) + [`add`](Preprocessor::add), then
/// call [`run`](Preprocessor::run) to transform an image array.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Preprocessor {
    pub steps: Vec<PreprocessStep>,
}

impl Preprocessor {
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a step and return `self` for chaining.
    pub fn add(mut self, step: PreprocessStep) -> Self {
        self.steps.push(step);
        self
    }

    /// Apply the pipeline to a `[H, W, C]` float32 image array.
    ///
    /// Returns the transformed array as a dynamically-ranked `ArrayD<f32>`.
    /// The shape depends on which steps are included — for example, adding
    /// [`HwcToChw`] + [`AddBatchDim`] yields `[1, C, H, W]`.
    pub fn run(&self, img: Array3<f32>) -> Result<ArrayD<f32>> {
        let mut arr: ArrayD<f32> = img.into_dyn();
        for step in &self.steps {
            arr = apply_step(arr, step)?;
        }
        Ok(arr)
    }
}

// ── Step implementations ───────────────────────────────────────────────────────

fn apply_step(arr: ArrayD<f32>, step: &PreprocessStep) -> Result<ArrayD<f32>> {
    match step {
        PreprocessStep::SwapChannels => {
            let mut arr3: Array3<f32> = arr
                .into_dimensionality()
                .map_err(|_| anyhow::anyhow!("SwapChannels requires a 3-D [H,W,C] array"))?;
            // Swap channels 0 and 2 in-place via slicing.
            let ch0 = arr3.slice(s![.., .., 0]).to_owned();
            let ch2 = arr3.slice(s![.., .., 2]).to_owned();
            arr3.slice_mut(s![.., .., 0]).assign(&ch2);
            arr3.slice_mut(s![.., .., 2]).assign(&ch0);
            Ok(arr3.into_dyn())
        }

        PreprocessStep::LinearScale { scale, offset } => {
            Ok(arr.mapv(|v| v * scale + offset))
        }

        PreprocessStep::Normalize { mean, std } => {
            let mut arr3: Array3<f32> = arr
                .into_dimensionality()
                .map_err(|_| anyhow::anyhow!("Normalize requires a 3-D [H,W,C] array"))?;
            for c in 0..3usize {
                let m = mean[c];
                let s = std[c];
                arr3.slice_mut(s![.., .., c]).mapv_inplace(|v| (v - m) / s);
            }
            Ok(arr3.into_dyn())
        }

        PreprocessStep::HwcToChw => {
            if arr.ndim() != 3 {
                bail!("HwcToChw requires a 3-D [H,W,C] array, got {}D", arr.ndim());
            }
            // permute_axes [H, W, C] → [C, H, W]
            let arr3 = arr
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|_| anyhow::anyhow!("HwcToChw: failed to reshape to Ix3"))?;
            Ok(arr3.permuted_axes([2usize, 0, 1]).into_dyn())
        }

        PreprocessStep::AddBatchDim => {
            // Insert axis 0: [d0, d1, …] → [1, d0, d1, …]
            Ok(arr.insert_axis(Axis(0)).to_owned())
        }
    }
}
