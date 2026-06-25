//! Sentence embedding — query encoder + background worker for semantic search.
//!
//! # Pipeline
//!
//! A background thread queries the database for AI descriptions that have not
//! been embedded by the active encoder, splits each into sentences, embeds
//! every sentence, and stores the vectors (see [`crate::semantic_db`]).  The
//! same encoder is shared (behind a mutex) with the UI so search queries can
//! be embedded on demand.
//!
//! Follows the same spawn→fetch→process→sleep pattern as the AI and face
//! taggers via [`crate::worker::spawn_db_worker`].

use std::sync::{Arc, Mutex};
use std::time::Duration;

use tracing::{info, warn};

use crate::models::TextEmbeddingModel;
use crate::Database;

// ── Sentence splitting ──────────────────────────────────────────────

/// Split a description into sentences for individual embedding.
///
/// Rule-based: a boundary occurs after a terminator (`.`, `!`, `?`, newline)
/// that is followed by whitespace or end-of-text.  Fragments are trimmed and
/// those shorter than 3 bytes are dropped.  Text with no boundary returns as a
/// single sentence.
pub fn split_sentences(text: &str) -> Vec<String> {
    let mut sentences = Vec::new();
    let mut current = String::new();
    let chars: Vec<char> = text.chars().collect();

    for (i, &c) in chars.iter().enumerate() {
        current.push(c);
        // A newline always ends a line; `.`/`!`/`?` end a sentence only when
        // followed by whitespace or end-of-text (avoids splitting "3.5mm").
        let punct_boundary = matches!(c, '.' | '!' | '?')
            && chars.get(i + 1).map(|n| n.is_whitespace()).unwrap_or(true);
        if c == '\n' || punct_boundary {
            let s = current.trim();
            if s.len() >= 3 {
                sentences.push(s.to_string());
            }
            current.clear();
        }
    }

    let tail = current.trim();
    if tail.len() >= 3 {
        sentences.push(tail.to_string());
    }

    sentences
}

// ── Shared encoder ──────────────────────────────────────────────────

/// A loaded sentence-transformer shared between the embedding worker and the
/// UI query path.  Cheap to clone (the model lives behind an `Arc<Mutex>`).
#[derive(Clone)]
pub struct SemanticEncoder {
    inner: Arc<Mutex<Box<dyn TextEmbeddingModel>>>,
    model_id: String,
    dim: usize,
}

impl SemanticEncoder {
    /// Wrap a loaded text-embedding model, tagging it with its `model_id`
    /// (the HuggingFace repo id) for tracking which vectors belong to it.
    pub fn new(model: Box<dyn TextEmbeddingModel>, model_id: impl Into<String>) -> Self {
        let dim = model.embedding_dim();
        Self {
            inner: Arc::new(Mutex::new(model)),
            model_id: model_id.into(),
            dim,
        }
    }

    /// HuggingFace repo id of the active encoder.
    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Embed a search query.  Returns `None` on failure so callers can fall
    /// back to keyword-only search.
    pub fn embed_query(&self, text: &str) -> Option<Vec<f32>> {
        let mut guard = lock(&self.inner);
        match guard.embed_text(text) {
            Ok(v) => Some(v),
            Err(e) => {
                warn!("semantic: query embedding failed: {e}");
                None
            }
        }
    }
}

/// Lock the encoder mutex, recovering from poison (a panic mid-inference must
/// not wedge the whole encoder).
fn lock(m: &Mutex<Box<dyn TextEmbeddingModel>>) -> std::sync::MutexGuard<'_, Box<dyn TextEmbeddingModel>> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

// ── Background worker ───────────────────────────────────────────────

/// Handle to a running sentence-embedder thread.  Call [`stop`](SentenceEmbedder::stop)
/// to request a graceful shutdown (the thread finishes the current description).
pub struct SentenceEmbedder {
    handle: crate::worker::WorkerHandle,
}

impl SentenceEmbedder {
    /// Signal the embedder thread to stop after the current description.
    pub fn stop(&self) {
        self.handle.stop();
    }
}

/// Spawn the background thread that embeds AI-description sentences.
///
/// The caller is responsible for calling [`Database::ensure_vec_table`] with
/// `encoder.model_id()` / `encoder.dim()` *before* spawning, so the vector
/// table matches the active model.
pub fn spawn_sentence_embedder(
    db: Arc<Mutex<Database>>,
    encoder: SemanticEncoder,
) -> SentenceEmbedder {
    let model_id = encoder.model_id().to_owned();

    let handle = crate::worker::spawn_db_worker(
        &format!("sentence-embedder[{model_id}]"),
        db,
        encoder,
        Duration::from_secs(60),
        // fetch — descriptions not yet embedded by this encoder.
        {
            let model_id = model_id.clone();
            move |db_guard| {
                db_guard
                    .descriptions_needing_embedding(&model_id)
                    .unwrap_or_else(|e| {
                        warn!("sentence-embedder: DB query failed: {e}");
                        vec![]
                    })
            }
        },
        // process — split into sentences, embed each, store.
        move |encoder, db, (desc_id, image_id, description)| {
            let sentences = split_sentences(&description);

            let mut embedded: Vec<(String, Vec<f32>)> = Vec::with_capacity(sentences.len());
            {
                let mut model = lock(&encoder.inner);
                for s in sentences {
                    match model.embed_text(&s) {
                        Ok(v) => embedded.push((s, v)),
                        Err(e) => warn!("sentence-embedder: embedding failed: {e}"),
                    }
                }
            }

            let result = crate::lock_db(db).insert_sentence_embeddings(
                image_id,
                desc_id,
                encoder.model_id(),
                &embedded,
            );
            match result {
                Ok(_) => info!(
                    image_id,
                    "sentence-embedder: stored {} sentence vector(s)",
                    embedded.len()
                ),
                Err(e) => warn!(image_id, "sentence-embedder: store failed: {e}"),
            }
        },
    );

    SentenceEmbedder { handle }
}

// ── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_on_terminators() {
        let s = split_sentences("A dog runs. The cat sleeps! Is it raining?");
        assert_eq!(
            s,
            vec!["A dog runs.", "The cat sleeps!", "Is it raining?"]
        );
    }

    #[test]
    fn keeps_unterminated_tail() {
        let s = split_sentences("First sentence. Trailing fragment without period");
        assert_eq!(
            s,
            vec!["First sentence.", "Trailing fragment without period"]
        );
    }

    #[test]
    fn single_sentence_no_punctuation() {
        assert_eq!(split_sentences("a quiet meadow"), vec!["a quiet meadow"]);
    }

    #[test]
    fn drops_tiny_fragments_and_blank() {
        assert!(split_sentences("   ").is_empty());
        // "Hi." is 3 bytes → kept; a lone "x" tail is dropped.
        assert_eq!(split_sentences("Hi. x"), vec!["Hi."]);
    }

    #[test]
    fn splits_on_newlines() {
        let s = split_sentences("line one\nline two\n");
        assert_eq!(s, vec!["line one", "line two"]);
    }
}
