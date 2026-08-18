//! Face assignment/removal DB operations, shared by the detail-window face
//! overlay ([`crate::face_overlay`]) and the dedicated face-tagging review
//! window ([`crate::face_tag`]).
//!
//! Both callers used to hit the DB inline and had drifted: the overlay's
//! assign functions didn't refresh the person's representative-face crop
//! after a reassignment, while the review window did. Routing both through
//! these functions keeps that behavior in one place.

use std::sync::{Arc, Mutex};

use maple_db::lock_db;

use crate::transforms::EmbeddingMatrix;

/// Load all assigned face embeddings and the full person list into an
/// in-memory matrix for cosine-similarity suggestion matching (built once
/// per image load; see [`EmbeddingMatrix`]).
pub fn load_embedding_matrix(db: &Arc<Mutex<maple_db::Database>>) -> EmbeddingMatrix {
    let guard = lock_db(db);
    let known = guard.all_assigned_face_embeddings().unwrap_or_default();
    let persons: Vec<(i64, String)> = guard
        .search_persons("")
        .unwrap_or_default()
        .into_iter()
        .map(|p| (p.id, p.name))
        .collect();
    drop(guard);
    EmbeddingMatrix::from_rows(known, persons)
}

/// Assign `face_id` to an existing person and refresh their representative
/// face crop. Returns `false` on DB error (best-effort).
pub fn assign_face_to_person(db: &Arc<Mutex<maple_db::Database>>, face_id: i64, person_id: i64) -> bool {
    let guard = lock_db(db);
    if let Err(e) = guard.assign_face_to_person(face_id, Some(person_id)) {
        tracing::warn!("assign_face_to_person {face_id} -> {person_id}: {e}");
        return false;
    }
    if let Err(e) = guard.update_person_representative(person_id) {
        tracing::warn!("update_person_representative {person_id}: {e}");
    }
    true
}

/// Upsert a person by `name`, assign `face_id` to them, and refresh their
/// representative face crop. Returns the person id on success.
pub fn assign_face_to_name(db: &Arc<Mutex<maple_db::Database>>, face_id: i64, name: &str) -> Option<i64> {
    let guard = lock_db(db);
    let person_id = guard
        .upsert_person(name)
        .map_err(|e| tracing::warn!("upsert_person '{name}': {e}"))
        .ok()?;
    guard
        .assign_face_to_person(face_id, Some(person_id))
        .map_err(|e| tracing::warn!("assign_face_to_name {face_id} -> {person_id}: {e}"))
        .ok()?;
    if let Err(e) = guard.update_person_representative(person_id) {
        tracing::warn!("update_person_representative {person_id}: {e}");
    }
    Some(person_id)
}

/// Delete a face detection. Best-effort (logs on DB error).
pub fn delete_face(db: &Arc<Mutex<maple_db::Database>>, face_id: i64) {
    let guard = lock_db(db);
    if let Err(e) = guard.delete_face_detection(face_id) {
        tracing::warn!("delete_face {face_id}: {e}");
    }
}

/// Mark a face as skipped, excluding it from the untagged-face queue without
/// deleting the detection. Best-effort (logs on DB error).
pub fn skip_face(db: &Arc<Mutex<maple_db::Database>>, face_id: i64) {
    let guard = lock_db(db);
    if let Err(e) = guard.mark_face_skipped(face_id, true) {
        tracing::warn!("skip_face {face_id}: {e}");
    }
}

/// Insert a manually-drawn face box. Returns the new face id on success.
pub fn insert_face(db: &Arc<Mutex<maple_db::Database>>, image_id: i64, bbox: [f32; 4]) -> Option<i64> {
    lock_db(db)
        .insert_face_detection(image_id, bbox, &[], 1.0)
        .map_err(|e| tracing::warn!("insert_face image={image_id}: {e}"))
        .ok()
}

/// Rename a person. Returns `false` on DB error (best-effort).
pub fn rename_person(db: &Arc<Mutex<maple_db::Database>>, id: i64, name: &str) -> bool {
    lock_db(db).rename_person(id, name).is_ok()
}

/// Delete a person and best-effort evict their cached representative face
/// crop (looked up before the row disappears).
pub fn delete_person(db: &Arc<Mutex<maple_db::Database>>, cache: &maple_db::ThumbnailCache, id: i64) -> bool {
    let guard = lock_db(db);
    let face_id = guard
        .all_persons_with_representatives()
        .unwrap_or_default()
        .into_iter()
        .find(|p| p.id == id)
        .and_then(|p| p.face_id);
    let ok = guard.delete_person(id).is_ok();
    drop(guard);
    if ok {
        if let Some(fid) = face_id {
            if let Err(e) = cache.remove_face_crop(fid) {
                tracing::warn!("remove_face_crop {fid}: {e}");
            }
        }
    }
    ok
}
