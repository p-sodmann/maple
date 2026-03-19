//! AI image description — trait abstraction + LM Studio (OpenAI-compatible) impl.
//!
//! # Architecture
//!
//! [`AiDescriber`] is a Send trait so that new AI backends can be added without
//! touching the runner.  [`LmStudioDescriber`] is the first implementation; it
//! targets LM Studio's OpenAI-compatible `/v1/chat/completions` endpoint using
//! a vision-capable model.
//!
//! [`spawn_ai_tagger`] starts a background thread that continuously queries the
//! database for images without a description from the given model, calls the
//! describer, and writes results back.  The returned [`AiTagger`] handle lets
//! callers stop the thread gracefully between images.

use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use base64::{engine::general_purpose::STANDARD, Engine as _};
use maple_import::loadable_image_bytes;
use tracing::{info, warn};

use crate::Database;

// ── Trait ─────────────────────────────────────────────────────────

/// Implemented by any AI backend that can describe an image given its path.
///
/// Implementations must be `Send` so they can be moved into a background thread.
pub trait AiDescriber: Send {
    /// A stable identifier for this model/provider combination.
    ///
    /// Used as the `model_id` column in `ai_descriptions`.  Two different
    /// configurations of the same model should use distinct IDs.
    fn model_id(&self) -> &str;

    /// Produce a textual description for the image at `path`.
    ///
    /// Returns `Err` on any network or parsing failure; the tagger will log
    /// the error and retry the image on the next pass.
    fn describe_image(&self, path: &Path) -> anyhow::Result<String>;
}

// ── LM Studio (OpenAI-compatible) ─────────────────────────────────

/// Describes images via an OpenAI-compatible vision endpoint (e.g. LM Studio).
///
/// Reads the `server_url` and `model` from [`maple_state::AiSettings`] and
/// sends each image as a base64-encoded data-URL inside a chat completion
/// request.
pub struct LmStudioDescriber {
    /// Full endpoint URL, e.g. `http://localhost:1234/v1/chat/completions`.
    endpoint: String,
    model: String,
    prompt: String,
    agent: ureq::Agent,
}

impl LmStudioDescriber {
    /// Create a new describer.
    ///
    /// `server_url` should be the base address of the LM Studio server,
    /// e.g. `http://localhost:1234`.  The `/v1/chat/completions` path is
    /// appended automatically.
    pub fn new(server_url: &str, model: &str, prompt: &str) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(120))
            .build();
        Self {
            endpoint: format!("{}/v1/chat/completions", server_url.trim_end_matches('/')),
            model: model.to_owned(),
            prompt: prompt.to_owned(),
            agent,
        }
    }
}

impl AiDescriber for LmStudioDescriber {
    fn model_id(&self) -> &str {
        &self.model
    }

    fn describe_image(&self, path: &Path) -> anyhow::Result<String> {
        let bytes = loadable_image_bytes(path)?;
        let mime = mime_type_for_path(path);
        let data_url = format!("data:{mime};base64,{}", STANDARD.encode(&bytes));

        let body = serde_json::json!({
            "model": self.model,
            "messages": [{
                "role": "user",
                "content": [
                    { "type": "text",      "text": self.prompt },
                    { "type": "image_url", "image_url": { "url": data_url } }
                ]
            }],
            "max_tokens": 512
        });

        let response: serde_json::Value = self
            .agent
            .post(&self.endpoint)
            .set("Content-Type", "application/json")
            .send_json(body)?
            .into_json()?;

        let description = response["choices"][0]["message"]["content"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("unexpected response shape: {response}"))?
            .trim()
            .to_owned();

        Ok(description)
    }
}

/// Map a file extension to a MIME type for the data-URL.
fn mime_type_for_path(path: &Path) -> &'static str {
    match path
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_ascii_lowercase())
        .as_deref()
    {
        Some("jpg") | Some("jpeg") => "image/jpeg",
        Some("png") => "image/png",
        Some("gif") => "image/gif",
        Some("webp") => "image/webp",
        Some("tiff") | Some("tif") => "image/tiff",
        Some("raf") => "image/jpeg", // embedded JPEG preview
        _ => "image/jpeg",
    }
}

// ── Runner ────────────────────────────────────────────────────────

/// Handle to a running AI tagger thread.  Call [`stop`](AiTagger::stop) to
/// request a graceful shutdown (the thread finishes the current image first).
pub struct AiTagger {
    handle: crate::worker::WorkerHandle,
}

impl AiTagger {
    /// Signal the tagger thread to stop after the current image.
    pub fn stop(&self) {
        self.handle.stop();
    }
}

/// Spawn a background thread that fills `ai_descriptions` for every present
/// image that has not yet been described by `describer`.
///
/// The thread runs indefinitely, sleeping 60 s between passes once all
/// existing images are described.  It checks for a stop signal between each
/// image so shutdown is prompt.
///
/// Returns an [`AiTagger`] whose [`stop`](AiTagger::stop) method signals the
/// thread to exit.
pub fn spawn_ai_tagger(
    db: Arc<Mutex<Database>>,
    describer: impl AiDescriber + 'static,
) -> AiTagger {
    let model_id = describer.model_id().to_owned();

    let handle = crate::worker::spawn_db_worker(
        &format!("ai-tagger[{model_id}]"),
        db,
        describer,
        Duration::from_secs(60),
        // fetch
        move |db_guard| {
            db_guard
                .images_needing_ai_description(&model_id)
                .unwrap_or_else(|e| {
                    warn!("ai_tagger: DB query failed: {e}");
                    vec![]
                })
        },
        // process
        |describer, db, (id, path)| match describer.describe_image(&path) {
            Ok(desc) => {
                let result =
                    crate::lock_db(db).insert_ai_description(id, describer.model_id(), &desc);
                if let Err(e) = result {
                    warn!(path = %path.display(), "AI tagger: failed to store description: {e}");
                } else {
                    info!(path = %path.display(), "AI tagger: described image");
                }
            }
            Err(e) => {
                warn!(path = %path.display(), "AI tagger: description failed: {e}");
            }
        },
    );

    AiTagger { handle }
}
