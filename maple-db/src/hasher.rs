//! Background perceptual hasher.
//!
//! [`spawn_hasher`] starts a thread that continuously polls the database for
//! images that do not yet have a hash for the active algorithm, computes the
//! hash (pHash or ONNX embedding), stores it, and triggers the stacker to
//! update group assignments.
//!
//! The thread runs for the lifetime of the process and can pick up images that
//! arrive during an active import — newly inserted rows appear in the poll
//! immediately.
//!
//! # Algorithm key
//!
//! The algorithm is identified by [`StackSettings::algorithm_key`] (e.g.
//! `"phash:8"` or `"onnx:facebook/dinov2-with-registers-base"`).  If the user
//! changes the algorithm in settings, the key changes and the background hasher
//! starts working on the new key; old rows under the previous key are ignored.

use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use maple_import::{compute_phash, loadable_image_bytes};
use maple_state::{StackMode, StackSettings};
use tracing::{info, warn};

use crate::worker::WorkerHandle;
use crate::{lock_db, Database};

/// How many images to hash per poll cycle before re-running the stacker.
const BATCH_SIZE: usize = 20;

/// How long to sleep when there is no pending work.
const IDLE_SLEEP: Duration = Duration::from_secs(10);

/// How long to sleep between batches when work is available.
const WORK_SLEEP: Duration = Duration::from_millis(100);

/// Spawn the background hasher thread.
///
/// Returns a [`WorkerHandle`] whose [`stop`](WorkerHandle::stop) method
/// requests a graceful shutdown.
pub fn spawn_hasher(db: Arc<Mutex<Database>>, settings: StackSettings) -> WorkerHandle {
    let (stop_tx, stop_rx) = mpsc::sync_channel::<()>(1);

    std::thread::Builder::new()
        .name("maple-hasher".into())
        .spawn(move || {
            info!("Background hasher started (algorithm: {})", settings.algorithm_key());

            // For ONNX mode, load the embedder once for the lifetime of the thread.
            let mut onnx_embedder: Option<crate::models::OnnxImageEmbedder> =
                if settings.mode == StackMode::Onnx {
                    match load_onnx_embedder(&settings) {
                        Ok(e) => Some(e),
                        Err(err) => {
                            warn!("Hasher: failed to load ONNX embedder, falling back to pHash: {err}");
                            None
                        }
                    }
                } else {
                    None
                };

            // Effective mode — falls back to pHash if ONNX embedder failed to load.
            let effective_mode = if settings.mode == StackMode::Onnx && onnx_embedder.is_none() {
                StackMode::PHash
            } else {
                settings.mode
            };

            let algorithm = match effective_mode {
                StackMode::PHash => format!("phash:{}", settings.hash_size),
                StackMode::Onnx => format!("onnx:{}", settings.model_repo),
            };

            loop {
                if stop_rx.try_recv().is_ok() {
                    info!("Background hasher stopped");
                    break;
                }

                let pending = {
                    let guard = lock_db(&db);
                    guard.images_without_hash(&algorithm, BATCH_SIZE).unwrap_or_default()
                };

                if pending.is_empty() {
                    std::thread::sleep(IDLE_SLEEP);
                    continue;
                }

                info!("Hasher: processing {} image(s)", pending.len());

                let mut newly_hashed: Vec<i64> = Vec::new();

                for (image_id, path) in &pending {
                    if stop_rx.try_recv().is_ok() {
                        break;
                    }

                    let result = match effective_mode {
                        StackMode::PHash => hash_phash(*image_id, path, &settings, &db, &algorithm),
                        StackMode::Onnx => {
                            if let Some(ref mut embedder) = onnx_embedder {
                                hash_onnx(*image_id, path, embedder, &db, &algorithm)
                            } else {
                                hash_phash(*image_id, path, &settings, &db, &algorithm)
                            }
                        }
                    };

                    match result {
                        Ok(()) => newly_hashed.push(*image_id),
                        Err(e) => warn!("Hasher: failed to hash {}: {e}", path.display()),
                    }
                }

                // Re-run the stacker over all currently hashed images so that
                // newly hashed images are merged into existing stacks.
                if !newly_hashed.is_empty() {
                    if let Err(e) = crate::stacker::update_stacks(&db, &algorithm, &settings) {
                        warn!("Hasher: stacker update failed: {e}");
                    }
                }

                std::thread::sleep(WORK_SLEEP);
            }
        })
        .expect("failed to spawn maple-hasher thread");

    WorkerHandle::from_sync_sender(stop_tx)
}

// ── Per-algorithm hashing ─────────────────────────────────────────────────────

fn hash_phash(
    image_id: i64,
    path: &std::path::Path,
    settings: &StackSettings,
    db: &Arc<Mutex<Database>>,
    algorithm: &str,
) -> anyhow::Result<()> {
    let hash = compute_phash(path, settings.hash_size)?;
    let blob = hash.as_bytes().to_vec();
    lock_db(db).insert_image_hash(image_id, algorithm, &blob)?;
    Ok(())
}

fn hash_onnx(
    image_id: i64,
    path: &std::path::Path,
    embedder: &mut crate::models::OnnxImageEmbedder,
    db: &Arc<Mutex<Database>>,
    algorithm: &str,
) -> anyhow::Result<()> {
    let bytes = loadable_image_bytes(path)?;
    let img = image::load_from_memory(&bytes)?;
    let embedding = embedder.embed(&img)?;
    // Serialise Vec<f32> as little-endian bytes (same layout as face embeddings).
    let blob: Vec<u8> = embedding
        .iter()
        .flat_map(|f| f.to_le_bytes())
        .collect();
    lock_db(db).insert_image_hash(image_id, algorithm, &blob)?;
    Ok(())
}

fn load_onnx_embedder(settings: &StackSettings) -> anyhow::Result<crate::models::OnnxImageEmbedder> {
    let onnx_path = {
        let p = std::path::Path::new(&settings.onnx_file);
        if p.is_absolute() && p.exists() {
            p.to_path_buf()
        } else {
            crate::models::fetch_image_model(&settings.model_repo, &settings.onnx_file)?
        }
    };
    let device: crate::models::ModelDevice = settings.device.parse().unwrap_or_default();
    crate::models::OnnxImageEmbedder::load(
        &onnx_path,
        &device,
        settings.shortest_edge,
        settings.image_size,
        settings.image_mean,
        settings.image_std,
    )
}
