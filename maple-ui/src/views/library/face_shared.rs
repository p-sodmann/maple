//! Shared helpers for face overlay rendering and person tagging flows.
//!
//! Used by both the detail-window face overlay and the standalone face-tagging view.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use maple_db::{cosine_similarity, FaceDetection};
use tracing::debug;

/// Return `true` when this row represents a real detection (not a sentinel).
pub fn is_real_detection(face: &FaceDetection) -> bool {
    face.confidence >= 0.0 && face.bbox != [0.0, 0.0, 0.0, 0.0]
}

// ── Embedding matrix ──────────────────────────────────────────────

/// In-memory embedding matrix for fast cosine-similarity search.
///
/// Built once per tagging session from all currently assigned face embeddings,
/// then updated incrementally as new faces are tagged.  Avoids repeated DB
/// queries while iterating through many faces.
///
/// All known persons are tracked separately so that suggestion buttons are
/// shown even when no ArcFace embedder is configured (empty embeddings).
/// In that case `top_k` returns all known persons with `sim = -1.0`; the UI
/// hides the percentage label for those entries.
pub struct EmbeddingMatrix {
    /// Row-major flat storage: row `i` → `data[i*dim .. (i+1)*dim]`.
    data: Vec<f32>,
    /// Embedding dimensionality (512 for ArcFace).
    dim: usize,
    /// Per-row (person_id, name) — only rows that have a real embedding.
    rows: Vec<(i64, String)>,
    /// All known persons (superset of `rows`); used as fallback when embeddings
    /// are unavailable.
    persons: Vec<(i64, String)>,
}

impl EmbeddingMatrix {
    /// Empty matrix (no known persons yet).
    pub fn empty() -> Self {
        Self { data: vec![], dim: 512, rows: vec![], persons: vec![] }
    }

    /// Build by loading all persons and their face embeddings from the database.
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

        // Use the first *non-empty* embedding to determine dimensionality.
        // Faces tagged before the embedder was configured have 0-byte blobs;
        // using those would set dim=0 and suppress all similarity scores.
        let dim = known
            .iter()
            .find_map(|(_, _, e)| if !e.is_empty() { Some(e.len()) } else { None })
            .unwrap_or(512);
        debug!(
            "EmbeddingMatrix::build: {} assigned face embeddings, {} persons, dim={}",
            known.len(),
            persons.len(),
            dim,
        );
        let mut mat = Self {
            data: Vec::with_capacity(known.len() * dim),
            dim,
            rows: Vec::with_capacity(known.len()),
            persons,
        };
        for (pid, name, emb) in &known {
            debug!(
                "  embedding row: person_id={}, name={:?}, embedding_len={}",
                pid, name, emb.len(),
            );
            mat.add(*pid, name.clone(), emb);
        }
        debug!(
            "EmbeddingMatrix::build: final matrix has {} rows, data len={}",
            mat.rows.len(),
            mat.data.len(),
        );
        mat
    }

    /// Register a person and optionally append their embedding row.
    ///
    /// The person is always added to the `persons` fallback list (deduped by
    /// id).  The embedding row is only added when non-empty and the right
    /// dimensionality.
    pub fn add(&mut self, person_id: i64, name: String, embedding: &[f32]) {
        // Keep persons list up to date (for fallback display).
        if !self.persons.iter().any(|(pid, _)| *pid == person_id) {
            self.persons.push((person_id, name.clone()));
        }
        // Add embedding row when available.
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

    /// Top-k persons by cosine similarity against `query`.
    ///
    /// When multiple embeddings belong to the same person the highest similarity
    /// per person is used.  Results are always sorted by similarity (highest
    /// first).  Persons that have no embedding yet are included at the bottom
    /// with `sim = -1.0` — the UI omits the percentage label for those.
    ///
    /// When no embedding data is available at all (no embedder configured),
    /// returns up to `k` known persons each with `sim = -1.0`.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(i64, String, f32)> {
        if k == 0 {
            return vec![];
        }

        // No embedding data — list all known persons without scores.
        if query.is_empty() || self.dim == 0 || query.len() != self.dim {
            debug!(
                "top_k: falling back to persons list (query_len={}, dim={}, rows={})",
                query.len(),
                self.dim,
                self.rows.len(),
            );
            return self.persons
                .iter()
                .take(k)
                .map(|(pid, name)| (*pid, name.clone(), f32::NEG_INFINITY))
                .collect();
        }

        // Best similarity per person across all their stored embeddings.
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

        // Include persons with no embeddings yet at the bottom.
        for (pid, name) in &self.persons {
            best.entry(*pid).or_insert_with(|| (name.clone(), f32::NEG_INFINITY));
        }

        let mut results: Vec<(i64, String, f32)> = best
            .into_iter()
            .map(|(pid, (name, sim))| (pid, name, sim))
            .collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

/// Map a normalised face bbox to screen coordinates inside the overlay widget.
///
/// Returns `None` when image dimensions are unknown.
///
/// At zoom <= 1.0 the image is in ContentFit::Contain (letterboxed).
/// At zoom > 1.0 the image is ContentFit::Fill with a size_request and
/// the scrolled window's adjustments carry the scroll offset.
pub fn face_screen_rect(
    [x1, y1, x2, y2]: [f32; 4],
    img_w: i32,
    img_h: i32,
    vw: f64,
    vh: f64,
    zoom: f64,
    scroll_x: f64,
    scroll_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    if img_w == 0 || img_h == 0 || vw == 0.0 || vh == 0.0 {
        return None;
    }
    let fit = f64::min(vw / img_w as f64, vh / img_h as f64);

    let (sx, sy, sw, sh) = if zoom <= 1.0 {
        let draw_w = img_w as f64 * fit;
        let draw_h = img_h as f64 * fit;
        let off_x = (vw - draw_w) / 2.0;
        let off_y = (vh - draw_h) / 2.0;
        (
            off_x + x1 as f64 * draw_w,
            off_y + y1 as f64 * draw_h,
            (x2 - x1) as f64 * draw_w,
            (y2 - y1) as f64 * draw_h,
        )
    } else {
        let ppx = fit * zoom;
        (
            x1 as f64 * img_w as f64 * ppx - scroll_x,
            y1 as f64 * img_h as f64 * ppx - scroll_y,
            (x2 - x1) as f64 * img_w as f64 * ppx,
            (y2 - y1) as f64 * img_h as f64 * ppx,
        )
    };

    Some((sx, sy, sw, sh))
}

/// Find the next unassigned real detection index, starting from `start_idx`.
pub fn next_untagged_index(faces: &[FaceDetection], start_idx: usize) -> Option<usize> {
    faces
        .iter()
        .enumerate()
        .skip(start_idx)
        .find_map(|(idx, face)| {
            if is_real_detection(face) && face.person_id.is_none() {
                Some(idx)
            } else {
                None
            }
        })
}

/// Resolve person name for an optional person id.
pub fn person_name(
    db: &Arc<Mutex<maple_db::Database>>,
    person_id: Option<i64>,
) -> Option<String> {
    person_id.and_then(|pid| db.lock().ok()?.person_name(pid).ok().flatten())
}

/// Assign a face to an existing person and update the in-memory faces list.
pub fn assign_face_to_person(
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    drawing_area: &gtk4::DrawingArea,
    face_idx: usize,
    face_id: i64,
    person_id: i64,
) {
    let Ok(guard) = db.lock() else { return };
    if let Err(e) = guard.assign_face_to_person(face_id, Some(person_id)) {
        tracing::warn!("failed to assign face {} to person {}: {}", face_id, person_id, e);
        return;
    }
    drop(guard);

    if let Some(face) = faces.borrow_mut().get_mut(face_idx) {
        face.person_id = Some(person_id);
    }
    drawing_area.queue_draw();
}

/// Upsert person by name and assign to the selected face.
///
/// Returns the `person_id` that was used, or `None` on failure.
pub fn assign_face_to_name(
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    drawing_area: &gtk4::DrawingArea,
    face_idx: usize,
    face_id: i64,
    name: &str,
) -> Option<i64> {
    let Ok(guard) = db.lock() else { return None };
    match guard.upsert_person(name) {
        Ok(person_id) => {
            if let Err(e) = guard.assign_face_to_person(face_id, Some(person_id)) {
                tracing::warn!("failed to assign face {} to person {}: {}", face_id, person_id, e);
                return None;
            }
            drop(guard);
            if let Some(face) = faces.borrow_mut().get_mut(face_idx) {
                face.person_id = Some(person_id);
            }
            drawing_area.queue_draw();
            Some(person_id)
        }
        Err(e) => {
            tracing::warn!("failed to upsert person '{}': {}", name, e);
            None
        }
    }
}

/// Clear a face's person assignment in both DB and the in-memory list.
pub fn unassign_face(
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    face_id: i64,
    face_idx: usize,
) {
    let Ok(guard) = db.lock() else { return };
    if let Err(e) = guard.assign_face_to_person(face_id, None) {
        tracing::warn!("failed to clear assignment for face {}: {}", face_id, e);
    }
    drop(guard);
    if let Some(face) = faces.borrow_mut().get_mut(face_idx) {
        face.person_id = None;
    }
}
