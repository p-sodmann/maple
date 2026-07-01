//! Execution device selection for ONNX inference.
//!
//! Configured via the `device` key in `settings.toml`:
//!
//! ```toml
//! [face]
//! device = "cpu"         # default — always available
//! device = "cuda:0"      # first NVIDIA GPU (requires CUDA-enabled ORT)
//! device = "tensorrt:0"  # TensorRT (fastest, falls back to CUDA→CPU)
//! ```
//!
//! GPU execution only takes effect when this crate is built with the `gpu`
//! Cargo feature (`cargo build --features gpu` — see `maple-db/Cargo.toml`),
//! which switches `ort` to dynamic loading. Point `ORT_DYLIB_PATH` at an
//! ONNX Runtime build with the relevant execution providers compiled in
//! before launching. On the default `cpu` feature build, CUDA/TensorRT
//! devices always fall back to CPU.

use std::fmt;
use std::str::FromStr;

use anyhow::Result;
use serde::{Deserialize, Serialize};

/// Where to run ONNX inference.
///
/// If the requested device is unavailable at runtime, ONNX Runtime
/// automatically falls back toward CPU.
#[derive(Debug, Clone, Default, PartialEq)]
pub enum ModelDevice {
    /// CPU inference — always available.
    #[default]
    Cpu,
    /// NVIDIA CUDA GPU.  The `u32` is the GPU device index (0 = first GPU).
    Cuda(u32),
    /// NVIDIA TensorRT — highest throughput for fixed-shape models.
    /// Falls back to CUDA, then CPU, when TRT is unavailable.
    TensorRt(u32),
}

impl fmt::Display for ModelDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cpu => f.write_str("cpu"),
            Self::Cuda(id) => write!(f, "cuda:{id}"),
            Self::TensorRt(id) => write!(f, "tensorrt:{id}"),
        }
    }
}

impl FromStr for ModelDevice {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let s = s.trim().to_ascii_lowercase();
        if s == "cpu" {
            return Ok(Self::Cpu);
        }
        if let Some(device) = parse_indexed_device(&s, "cuda:", "CUDA", Self::Cuda)? {
            return Ok(device);
        }
        if let Some(device) = parse_indexed_device(&s, "tensorrt:", "TensorRT", Self::TensorRt)? {
            return Ok(device);
        }
        anyhow::bail!("unknown device {s:?}; valid: 'cpu', 'cuda:N', 'tensorrt:N'")
    }
}

fn parse_indexed_device(
    raw: &str,
    prefix: &str,
    label: &str,
    constructor: fn(u32) -> ModelDevice,
) -> Result<Option<ModelDevice>> {
    let Some(id_str) = raw.strip_prefix(prefix) else {
        return Ok(None);
    };

    let id: u32 = id_str
        .parse()
        .map_err(|_| anyhow::anyhow!("invalid {label} device index: {id_str:?}"))?;
    Ok(Some(constructor(id)))
}

// String-based serde so settings.toml can use `device = "cuda:0"`.

impl Serialize for ModelDevice {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl<'de> Deserialize<'de> for ModelDevice {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

impl ModelDevice {
    /// Register execution providers on an `ort::SessionBuilder`.
    ///
    /// CPU is always appended as the final fallback.
    pub(super) fn apply_to_builder(
        &self,
        builder: ort::session::builder::SessionBuilder,
    ) -> Result<ort::session::builder::SessionBuilder> {
        use ort::execution_providers::{
            CPUExecutionProvider, CUDAExecutionProvider, TensorRTExecutionProvider,
        };

        let cpu = CPUExecutionProvider::default().build();

        let builder = match self {
            ModelDevice::Cpu => builder
                .with_execution_providers([cpu])
                .map_err(|e| anyhow::anyhow!("{e}"))?,

            // For CUDA/TensorRT, use default device (index 0).
            // Multi-device selection can be added when ort stabilises the EP API.
            ModelDevice::Cuda(_) => builder
                .with_execution_providers([CUDAExecutionProvider::default().build(), cpu])
                .map_err(|e| anyhow::anyhow!("{e}"))?,

            ModelDevice::TensorRt(_) => builder
                .with_execution_providers([
                    TensorRTExecutionProvider::default().build(),
                    CUDAExecutionProvider::default().build(),
                    cpu,
                ])
                .map_err(|e| anyhow::anyhow!("{e}"))?,
        };

        Ok(builder)
    }
}
