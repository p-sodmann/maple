//! Post-copy DB insertion for the import flow.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

/// Plain (path, hash) pair for one copied file, ready to insert into the
/// library DB. Deliberately excludes UI-only fields (decoded thumbnail,
/// selection state) carried on the import window's own per-entry struct.
pub struct ImportEntry {
    pub path: PathBuf,
    pub content_hash: [u8; 32],
}

/// Insert freshly-copied files into the library DB. Best-effort per entry —
/// a metadata-read or insert failure for one file doesn't stop the rest.
pub fn insert_imported_images(db: &Arc<Mutex<maple_db::Database>>, entries: &[ImportEntry]) {
    let Ok(guard) = db.lock() else { return };
    for e in entries {
        if let Ok(meta) = e.path.metadata() {
            if let Err(err) = guard.insert_image_with_raw(&e.path, &e.content_hash, meta.len(), None) {
                tracing::warn!("insert_imported_images {}: {err}", e.path.display());
            }
        }
    }
}
