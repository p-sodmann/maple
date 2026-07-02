//! Face overlay helpers — GTK-free port of `views/library/face_shared.rs`.
//!
//! Thin Slint-facing layer: builds [`ModelRc`]s from the pure transforms in
//! [`crate::transforms`] ([`EmbeddingMatrix`], `faces_to_boxes`,
//! `faces_to_suggestions`) that drive the overlay in `ui/detail.slint`, and
//! wraps face-assignment actions to keep in-memory face lists in sync with
//! the DB writes in `services::faces`.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ModelRc, VecModel};

use maple_db::FaceDetection;

use crate::transforms::{self, EmbeddingMatrix};
use crate::{FaceBox, FacePersonSuggestion};

/// Similarity threshold below which a match is shown as "?" rather than a
/// confident suggestion.  Mirrors the GTK code's `similarity_threshold`.
fn suggestion_threshold() -> f32 {
    maple_state::Settings::load().face.similarity_threshold
}

/// Convert all loaded faces into [`FaceBox`] structs for the Slint model.
pub fn build_face_boxes(faces: &[FaceDetection], known: &EmbeddingMatrix) -> ModelRc<FaceBox> {
    let boxes = transforms::faces_to_boxes(faces, known, suggestion_threshold());
    ModelRc::from(Rc::new(VecModel::from(boxes)))
}

/// Build person suggestions for the assignment panel (ranked by similarity).
pub fn build_suggestions(embedding: &[f32], known: &EmbeddingMatrix) -> ModelRc<FacePersonSuggestion> {
    let sugs = transforms::faces_to_suggestions(embedding, known);
    ModelRc::from(Rc::new(VecModel::from(sugs)))
}

/// Assign `face_id` to an existing person and return the updated face list.
pub fn assign_to_person(
    face_id: i64,
    person_id: i64,
    faces: &mut [FaceDetection],
    db: &Arc<Mutex<maple_db::Database>>,
) -> bool {
    if !crate::services::faces::assign_face_to_person(db, face_id, person_id) {
        return false;
    }
    if let Some(f) = faces.iter_mut().find(|f| f.id == face_id) {
        f.person_id = Some(person_id);
    }
    true
}

/// Upsert person by `name` and assign `face_id` to them.
/// Returns the person_id on success.
pub fn assign_to_name(
    face_id: i64,
    name: &str,
    faces: &mut [FaceDetection],
    known: &mut EmbeddingMatrix,
    db: &Arc<Mutex<maple_db::Database>>,
) -> Option<i64> {
    let person_id = crate::services::faces::assign_face_to_name(db, face_id, name)?;
    if let Some(f) = faces.iter_mut().find(|f| f.id == face_id) {
        f.person_id = Some(person_id);
        known.add(person_id, name.to_owned(), &f.embedding.clone());
    }
    Some(person_id)
}

/// Delete `face_id` from DB and remove from the in-memory list.
pub fn delete_face(
    face_id: i64,
    faces: &mut Vec<FaceDetection>,
    db: &Arc<Mutex<maple_db::Database>>,
) {
    crate::services::faces::delete_face(db, face_id);
    faces.retain(|f| f.id != face_id);
}

/// Insert a manually-drawn face box and return its id.
pub fn insert_new_face(
    image_id: i64,
    bbox: [f32; 4],
    faces: &mut Vec<FaceDetection>,
    db: &Arc<Mutex<maple_db::Database>>,
) -> Option<i64> {
    let id = crate::services::faces::insert_face(db, image_id, bbox)?;
    faces.push(FaceDetection {
        id,
        image_id,
        bbox,
        embedding: vec![],
        person_id: None,
        confidence: 1.0,
        skipped: false,
    });
    Some(id)
}
