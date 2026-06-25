//! HuggingFace Hub model download (blocking, `ureq`-based).
//!
//! Sentence-transformer models are referenced by repo id in settings; this
//! module fetches the ONNX model and tokenizer files and caches them under
//! `~/.config/maple/models/` so subsequent runs are offline.
//!
//! Uses `hf_hub`'s synchronous API (no tokio) to match the std::thread
//! architecture used throughout maple-db's background workers.

use std::path::PathBuf;

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use tracing::info;

/// Download (or reuse the cached copy of) the ONNX model and tokenizer for a
/// sentence-transformer repo.
///
/// - `repo` — HuggingFace repo id, e.g. `"sentence-transformers/all-MiniLM-L6-v2"`.
/// - `onnx_file` — model path within the repo, e.g. `"onnx/model.onnx"`.
/// - `tokenizer_file` — tokenizer path within the repo, e.g. `"tokenizer.json"`.
///
/// Returns `(onnx_path, tokenizer_path)` as local filesystem paths.  This is a
/// blocking network call; run it on a background thread.
pub fn fetch_sentence_model(
    repo: &str,
    onnx_file: &str,
    tokenizer_file: &str,
) -> Result<(PathBuf, PathBuf)> {
    let cache_dir = maple_state::config_dir().join("models");
    std::fs::create_dir_all(&cache_dir).ok();

    info!(repo, "fetching sentence-transformer model from HuggingFace Hub…");

    let api = ApiBuilder::new()
        .with_cache_dir(cache_dir)
        .build()
        .context("building HuggingFace Hub API client")?;

    let model = api.model(repo.to_string());

    let onnx_path = model
        .get(onnx_file)
        .with_context(|| format!("downloading {repo}/{onnx_file}"))?;
    let tokenizer_path = model
        .get(tokenizer_file)
        .with_context(|| format!("downloading {repo}/{tokenizer_file}"))?;

    info!(
        onnx = %onnx_path.display(),
        tokenizer = %tokenizer_path.display(),
        "sentence-transformer model ready"
    );
    Ok((onnx_path, tokenizer_path))
}
