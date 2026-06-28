//! Face overlay helpers — GTK-free port of `views/library/face_shared.rs`.
//!
//! Provides [`EmbeddingMatrix`] for cosine-similarity ranking of person
//! suggestions, and helpers that translate [`maple_db::FaceDetection`] rows
//! into the Slint [`FaceBox`] / [`FacePersonSuggestion`] structs that drive
//! the overlay in `ui/detail.slint`.

use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ModelRc, VecModel};

use maple_db::{cosine_similarity, FaceDetection};

use crate::{FaceBox, FacePersonSuggestion};

// ── EmbeddingMatrix ────────────────────────────────────────────────

/// In-memory embedding matrix for fast cosine-similarity search.
///
/// Built once per image load from all currently assigned face embeddings,
/// so DB queries are not repeated while the user cycles through suggestions.
/// Persons with no embedding (tagged before the embedder was configured) are
/// included as fallback rows with `sim = f32::NEG_INFINITY`.
pub struct EmbeddingMatrix {
    data: Vec<f32>,
    dim: usize,
    rows: Vec<(i64, String)>,
    persons: Vec<(i64, String)>,
}

impl EmbeddingMatrix {
    pub fn empty() -> Self {
        Self { data: vec![], dim: 512, rows: vec![], persons: vec![] }
    }

    /// Build by loading all persons and assigned embeddings from the DB.
    pub fn build(db: &Arc<Mutex<maple_db::Database>>) -> Self {
        let Ok(guard) = db.lock() else { return Self::empty() };
        let known = guard.all_assigned_face_embeddings().unwrap_or_default();
        let persons: Vec<(i64, String)> = guard
            .search_persons("")
            .unwrap_or_default()
            .into_iter()
            .map(|p| (p.id, p.name))
            .collect();
        drop(guard);

        let dim = known
            .iter()
            .find_map(|(_, _, e)| if !e.is_empty() { Some(e.len()) } else { None })
            .unwrap_or(512);

        let mut mat = Self {
            data: Vec::with_capacity(known.len() * dim),
            dim,
            rows: Vec::with_capacity(known.len()),
            persons,
        };
        for (pid, name, emb) in &known {
            mat.add(*pid, name.clone(), emb);
        }
        mat
    }

    /// Register a person and optionally append their embedding row.
    pub fn add(&mut self, person_id: i64, name: String, embedding: &[f32]) {
        if !self.persons.iter().any(|(pid, _)| *pid == person_id) {
            self.persons.push((person_id, name.clone()));
        }
        if embedding.is_empty() {
            return;
        }
        if self.dim == 0 {
            self.dim = embedding.len();
        }
        if embedding.len() != self.dim {
            return;
        }
        self.data.extend_from_slice(embedding);
        self.rows.push((person_id, name));
    }

    /// Top-k persons by cosine similarity.
    ///
    /// When no ArcFace data is available, all known persons are returned with
    /// `sim = f32::NEG_INFINITY` so the UI can still show name buttons.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(i64, String, f32)> {
        if k == 0 {
            return vec![];
        }

        if query.is_empty() || self.dim == 0 || query.len() != self.dim {
            return self
                .persons
                .iter()
                .take(k)
                .map(|(pid, name)| (*pid, name.clone(), f32::NEG_INFINITY))
                .collect();
        }

        let mut best: HashMap<i64, (String, f32)> = HashMap::new();
        for (i, (pid, name)) in self.rows.iter().enumerate() {
            let row = &self.data[i * self.dim..(i + 1) * self.dim];
            let sim = cosine_similarity(query, row);
            let entry = best
                .entry(*pid)
                .or_insert_with(|| (name.clone(), f32::NEG_INFINITY));
            if sim > entry.1 {
                entry.0 = name.clone();
                entry.1 = sim;
            }
        }
        for (pid, name) in &self.persons {
            best.entry(*pid).or_insert_with(|| (name.clone(), f32::NEG_INFINITY));
        }

        let mut results: Vec<(i64, String, f32)> =
            best.into_iter().map(|(pid, (name, sim))| (pid, name, sim)).collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

// ── Helpers ────────────────────────────────────────────────────────

/// `true` when this row is a real detection (not a zero-confidence sentinel).
pub fn is_real_detection(face: &FaceDetection) -> bool {
    face.confidence >= 0.0 && face.bbox != [0.0, 0.0, 0.0, 0.0]
}

/// How many top-k suggestions to fetch (enough for a compact panel).
const SUGGESTION_LIMIT: usize = 12;

/// Similarity threshold below which a match is shown as "?" rather than a
/// confident suggestion.  Mirrors the GTK code's `similarity_threshold`.
fn suggestion_threshold() -> f32 {
    maple_state::Settings::load().face.similarity_threshold
}

/// Build the face label for a detection: person name, a suggestion, or "?".
fn face_label(
    face: &FaceDetection,
    db: &Arc<Mutex<maple_db::Database>>,
    known: &EmbeddingMatrix,
) -> (String, bool, bool) {
    if let Some(pid) = face.person_id {
        let name = db
            .lock()
            .ok()
            .and_then(|g| g.person_name(pid).ok().flatten())
            .unwrap_or_else(|| "?".into());
        return (name, true, false);
    }
    if face.embedding.is_empty() {
        return ("?".into(), false, false);
    }
    let threshold = suggestion_threshold();
    let matches = known.top_k(&face.embedding, 1);
    if let Some((_pid, name, sim)) = matches.first() {
        if sim.is_finite() && *sim >= threshold {
            return (name.clone(), false, true);
        }
    }
    ("?".into(), false, false)
}

/// Convert all loaded faces into [`FaceBox`] structs for the Slint model.
pub fn build_face_boxes(
    faces: &[FaceDetection],
    db: &Arc<Mutex<maple_db::Database>>,
    known: &EmbeddingMatrix,
) -> ModelRc<FaceBox> {
    let boxes: Vec<FaceBox> = faces
        .iter()
        .filter(|f| is_real_detection(f))
        .map(|f| {
            let [x1, y1, x2, y2] = f.bbox;
            let (label, is_assigned, is_suggestion) = face_label(f, db, known);
            FaceBox {
                face_id: f.id as i32,
                x1,
                y1,
                x2,
                y2,
                label: label.into(),
                is_assigned,
                is_suggestion,
            }
        })
        .collect();
    ModelRc::from(Rc::new(VecModel::from(boxes)))
}

/// Build person suggestions for the assignment panel (ranked by similarity).
pub fn build_suggestions(
    embedding: &[f32],
    known: &EmbeddingMatrix,
) -> ModelRc<FacePersonSuggestion> {
    let raw = known.top_k(embedding, SUGGESTION_LIMIT);
    let sugs: Vec<FacePersonSuggestion> = raw
        .into_iter()
        .map(|(pid, name, sim)| {
            let sim_text: slint::SharedString = if sim.is_finite() && sim >= 0.0 {
                format!("{:.0}%", sim * 100.0).into()
            } else {
                "".into()
            };
            FacePersonSuggestion {
                person_id: pid as i32,
                name: name.into(),
                sim_text,
            }
        })
        .collect();
    ModelRc::from(Rc::new(VecModel::from(sugs)))
}

/// Assign `face_id` to an existing person and return the updated face list.
pub fn assign_to_person(
    face_id: i64,
    person_id: i64,
    faces: &mut Vec<FaceDetection>,
    db: &Arc<Mutex<maple_db::Database>>,
) -> bool {
    let Ok(guard) = db.lock() else { return false };
    if let Err(e) = guard.assign_face_to_person(face_id, Some(person_id)) {
        tracing::warn!("assign_to_person {face_id} → {person_id}: {e}");
        return false;
    }
    drop(guard);
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
    faces: &mut Vec<FaceDetection>,
    known: &mut EmbeddingMatrix,
    db: &Arc<Mutex<maple_db::Database>>,
) -> Option<i64> {
    let Ok(guard) = db.lock() else { return None };
    let person_id = match guard.upsert_person(name) {
        Ok(id) => id,
        Err(e) => {
            tracing::warn!("upsert_person '{name}': {e}");
            return None;
        }
    };
    if let Err(e) = guard.assign_face_to_person(face_id, Some(person_id)) {
        tracing::warn!("assign_to_name {face_id} → {person_id}: {e}");
        return None;
    }
    drop(guard);
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
    let Ok(guard) = db.lock() else { return };
    let _ = guard.delete_face_detection(face_id);
    drop(guard);
    faces.retain(|f| f.id != face_id);
}

/// Insert a manually-drawn face box and return its id.
pub fn insert_new_face(
    image_id: i64,
    bbox: [f32; 4],
    faces: &mut Vec<FaceDetection>,
    db: &Arc<Mutex<maple_db::Database>>,
) -> Option<i64> {
    let Ok(guard) = db.lock() else { return None };
    let id = guard.insert_face_detection(image_id, bbox, &[], 1.0).ok()?;
    drop(guard);
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
