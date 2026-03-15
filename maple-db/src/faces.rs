//! Face detection DB operations — persons, face_detections, cosine similarity.
//!
//! # Tables
//!
//! `persons(id, name, created_at)` — named identities.
//!
//! `face_detections(id, image_id, bbox_x1/y1/x2/y2, embedding BLOB, person_id,
//!  confidence)` — one row per detected face.  `embedding` is 512 × f32
//!  little-endian (2048 bytes).  `bbox_*` are normalised [0, 1] coordinates.
//!
//! # Cosine similarity
//!
//! [`cosine_similarity`] computes the dot product of two L2-normalised
//! 512-dim vectors.  The face detector stores L2-normalised embeddings, so
//! similarity == 1.0 is an identical face, ≥ ~0.4 is the same person for
//! ArcFace-R100.

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};
use std::collections::HashMap;

use rusqlite::params;

use crate::Database;

// ── Public types ─────────────────────────────────────────────────

/// A detected face record read from the database.
#[derive(Debug, Clone)]
pub struct FaceDetection {
    pub id: i64,
    pub image_id: i64,
    /// Normalised bounding box: [x1, y1, x2, y2] each in [0, 1].
    pub bbox: [f32; 4],
    /// L2-normalised 512-dim ArcFace embedding.
    pub embedding: Vec<f32>,
    pub person_id: Option<i64>,
    pub confidence: f32,
}

/// A named person identity.
#[derive(Debug, Clone)]
pub struct Person {
    pub id: i64,
    pub name: String,
}

// ── Cosine similarity ─────────────────────────────────────────────

/// Dot product of two L2-normalised vectors — equals cosine similarity when
/// both are already normalised (as stored in the DB).
///
/// Returns 0.0 if either slice is empty or lengths differ.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Find the best-matching person given a query embedding and a list of
/// `(person_id, person_name, embedding)` known faces.
///
/// Returns `Some((person_id, person_name, similarity))` when the best match
/// exceeds `threshold`, otherwise `None`.
pub fn best_person_match(
    query: &[f32],
    known: &[(i64, String, Vec<f32>)],
    threshold: f32,
) -> Option<(i64, String, f32)> {
    best_person_matches(query, known, threshold, 1).into_iter().next()
}

/// Find the top `k` matching persons for a query embedding.
///
/// Multiple known embeddings for the same person are merged by taking the
/// highest similarity per person.
pub fn best_person_matches(
    query: &[f32],
    known: &[(i64, String, Vec<f32>)],
    threshold: f32,
    k: usize,
) -> Vec<(i64, String, f32)> {
    if k == 0 {
        return vec![];
    }

    let mut best_per_person: HashMap<i64, (String, f32)> = HashMap::new();
    for (pid, name, emb) in known {
        let sim = cosine_similarity(query, emb);
        if sim < threshold {
            continue;
        }
        match best_per_person.get_mut(pid) {
            Some((saved_name, saved_sim)) => {
                if sim > *saved_sim {
                    *saved_name = name.clone();
                    *saved_sim = sim;
                }
            }
            None => {
                best_per_person.insert(*pid, (name.clone(), sim));
            }
        }
    }

    let mut matches: Vec<(i64, String, f32)> = best_per_person
        .into_iter()
        .map(|(pid, (name, sim))| (pid, name, sim))
        .collect();

    matches.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    matches.truncate(k);
    matches
}

// ── Database impl ─────────────────────────────────────────────────

impl Database {
    // ── Write operations ──────────────────────────────────────────

    /// Insert a detected face.  Returns the new row's `id`.
    ///
    /// `bbox` is `[x1, y1, x2, y2]` normalised to [0, 1].
    /// `embedding` must be a 512-dim L2-normalised f32 vector.
    pub fn insert_face_detection(
        &self,
        image_id: i64,
        bbox: [f32; 4],
        embedding: &[f32],
        confidence: f32,
    ) -> anyhow::Result<i64> {
        let blob = embedding_to_blob(embedding);
        self.conn.execute(
            "INSERT INTO face_detections
                 (image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, confidence)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                image_id,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                blob,
                confidence,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Assign `face_id` to `person_id` (or unassign if `None`).
    pub fn assign_face_to_person(
        &self,
        face_id: i64,
        person_id: Option<i64>,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE face_detections SET person_id = ?1 WHERE id = ?2",
            params![person_id, face_id],
        )?;
        Ok(())
    }

    /// Insert or retrieve a person by name.  Returns the person's `id`.
    pub fn upsert_person(&self, name: &str) -> anyhow::Result<i64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.conn.execute(
            "INSERT INTO persons(name, created_at) VALUES (?1, ?2)
             ON CONFLICT(name) DO NOTHING",
            params![name, now],
        )?;
        let id: i64 = self
            .conn
            .query_row("SELECT id FROM persons WHERE name = ?1", params![name], |r| {
                r.get(0)
            })?;
        Ok(id)
    }

    // ── Read operations ───────────────────────────────────────────

    /// Return all face detections for `image_id`, including person info.
    pub fn faces_for_image(&self, image_id: i64) -> anyhow::Result<Vec<FaceDetection>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2,
                    embedding, person_id, confidence
             FROM face_detections
             WHERE image_id = ?1",
        )?;
        let rows = stmt
            .query_map(params![image_id], |row| {
                let blob: Vec<u8> = row.get(6)?;
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, f32>(2)?,
                    row.get::<_, f32>(3)?,
                    row.get::<_, f32>(4)?,
                    row.get::<_, f32>(5)?,
                    blob,
                    row.get::<_, Option<i64>>(7)?,
                    row.get::<_, f32>(8)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, image_id, x1, y1, x2, y2, blob, person_id, confidence)| {
                FaceDetection {
                    id,
                    image_id,
                    bbox: [x1, y1, x2, y2],
                    embedding: blob_to_embedding(&blob),
                    person_id,
                    confidence,
                }
            })
            .collect();
        Ok(rows)
    }

    /// Return `(id, path)` for images that have no face_detections row yet.
    ///
    /// Used by the background face tagger to determine what to process.
    pub fn images_needing_face_detection(&self) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.id, i.path
             FROM images i
             WHERE i.status = 'present'
               AND NOT EXISTS (
                   SELECT 1 FROM face_detections fd WHERE fd.image_id = i.id
               )",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    PathBuf::from(row.get::<_, String>(1)?),
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return all `(person_id, person_name, embedding)` tuples for faces that
    /// have been assigned to a person.  Used for cosine-similarity grouping.
    pub fn all_assigned_face_embeddings(&self) -> anyhow::Result<Vec<(i64, String, Vec<f32>)>> {
        let mut stmt = self.conn.prepare(
            "SELECT fd.person_id, p.name, fd.embedding
             FROM face_detections fd
             JOIN persons p ON p.id = fd.person_id",
        )?;
        let rows = stmt
            .query_map([], |row| {
                let blob: Vec<u8> = row.get(2)?;
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    blob,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(pid, name, blob)| (pid, name, blob_to_embedding(&blob)))
            .collect();
        Ok(rows)
    }

    /// Search persons by name substring (case-insensitive).
    pub fn search_persons(&self, query: &str) -> anyhow::Result<Vec<Person>> {
        let pattern = format!("%{}%", query.to_lowercase());
        let mut stmt = self.conn.prepare(
            "SELECT id, name FROM persons
             WHERE LOWER(name) LIKE ?1
             ORDER BY name",
        )?;
        let rows = stmt
            .query_map(params![pattern], |row| {
                Ok(Person {
                    id: row.get(0)?,
                    name: row.get(1)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return IDs of present images that have at least one real untagged face.
    pub fn images_with_untagged_faces(&self) -> anyhow::Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT fd.image_id
             FROM face_detections fd
             JOIN images i ON i.id = fd.image_id
             WHERE fd.person_id IS NULL
               AND fd.confidence >= 0.0
               AND NOT (fd.bbox_x1 = 0.0 AND fd.bbox_y1 = 0.0
                        AND fd.bbox_x2 = 0.0 AND fd.bbox_y2 = 0.0)
               AND i.status = 'present'
             ORDER BY fd.image_id",
        )?;
        let rows = stmt
            .query_map([], |row| row.get::<_, i64>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return image ids that contain a face assigned to any of `person_ids`.
    pub fn image_ids_for_persons(&self, person_ids: &[i64]) -> anyhow::Result<Vec<i64>> {
        if person_ids.is_empty() {
            return Ok(vec![]);
        }
        // Build `IN (?, ?, ...)` clause dynamically.
        let placeholders: String = person_ids
            .iter()
            .enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect::<Vec<_>>()
            .join(", ");
        let sql = format!(
            "SELECT DISTINCT image_id FROM face_detections WHERE person_id IN ({placeholders})"
        );
        use rusqlite::types::Value;
        let params: Vec<Value> = person_ids.iter().map(|id| Value::Integer(*id)).collect();
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params), |row| {
                row.get::<_, i64>(0)
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return the person name for `person_id`, if any.
    pub fn person_name(&self, person_id: i64) -> anyhow::Result<Option<String>> {
        let mut stmt = self
            .conn
            .prepare("SELECT name FROM persons WHERE id = ?1")?;
        let mut rows = stmt.query(params![person_id])?;
        Ok(rows.next()?.map(|r| r.get::<_, String>(0)).transpose()?)
    }
}

// ── Blob encoding helpers ─────────────────────────────────────────

/// Encode a `[f32]` slice as a little-endian byte vector.
pub(crate) fn embedding_to_blob(embedding: &[f32]) -> Vec<u8> {
    let mut buf = Vec::with_capacity(embedding.len() * 4);
    for &v in embedding {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    buf
}

/// Decode a little-endian byte vector back to `Vec<f32>`.
pub(crate) fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect()
}
