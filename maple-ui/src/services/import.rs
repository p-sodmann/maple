//! Post-copy DB insertion for the import flow.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Plain (path, hash) pair for one copied file, ready to insert into the
/// library DB. Deliberately excludes UI-only fields (decoded thumbnail,
/// selection state) carried on the import window's own per-entry struct.
pub struct ImportEntry {
    pub path: PathBuf,
    pub content_hash: [u8; 32],
    /// DINOv2 embedding computed during the SD-card scan, if stack detection
    /// was enabled and inference succeeded. Transferring it here means the
    /// background hasher never needs to recompute it for this image.
    pub embedding: Option<Vec<f32>>,
}

/// Insert freshly-copied files into the library DB. Best-effort per entry —
/// a metadata-read or insert failure for one file doesn't stop the rest.
///
/// `algorithm_key` identifies the embedding model (`StackSettings::algorithm_key`,
/// e.g. `"onnx:onnx-community/dinov2-small"`) under which any `embedding` on
/// an entry was computed.
pub fn insert_imported_images(
    db: &Arc<Mutex<maple_db::Database>>,
    entries: &[ImportEntry],
    algorithm_key: &str,
) {
    let Ok(guard) = db.lock() else { return };
    for e in entries {
        if let Ok(meta) = e.path.metadata() {
            if let Err(err) = guard.insert_image_with_raw(&e.path, &e.content_hash, meta.len(), None) {
                tracing::warn!("insert_imported_images {}: {err}", e.path.display());
                continue;
            }
            if let Some(embedding) = &e.embedding {
                match guard.image_id_for_path(&e.path) {
                    Ok(Some(image_id)) => {
                        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
                        if let Err(err) = guard.insert_image_hash(image_id, algorithm_key, &blob) {
                            tracing::warn!(
                                "insert_imported_images: failed to store embedding for {}: {err}",
                                e.path.display()
                            );
                        }
                    }
                    Ok(None) => tracing::warn!(
                        "insert_imported_images: no row found for just-inserted {}",
                        e.path.display()
                    ),
                    Err(err) => tracing::warn!(
                        "insert_imported_images: image_id_for_path failed for {}: {err}",
                        e.path.display()
                    ),
                }
            }
        }
    }
}
