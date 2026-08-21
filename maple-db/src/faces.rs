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
    /// True when the user explicitly skipped this face during tagging.
    /// Skipped faces are hidden from the default tagging queue and only shown
    /// in the "Review Skipped Faces" mode.  Cleared automatically when a
    /// person is assigned.
    pub skipped: bool,
}

/// A named person identity.
#[derive(Debug, Clone)]
pub struct Person {
    pub id: i64,
    pub name: String,
}

/// A person together with the data needed to render their representative face.
#[derive(Debug, Clone)]
pub struct PersonWithRep {
    pub id: i64,
    pub name: String,
    /// Path to the image that contains the representative face.
    pub image_path: Option<PathBuf>,
    /// Bounding box `[x1, y1, x2, y2]` in [0,1] of the representative face.
    pub bbox: Option<[f32; 4]>,
    pub face_id: Option<i64>,
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
        let (rev, rev_dev) = self.stamp()?;
        let guid = self.new_guid()?;
        self.conn.execute(
            "INSERT INTO face_detections
                 (image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding, confidence,
                  guid, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            params![
                image_id,
                bbox[0],
                bbox[1],
                bbox[2],
                bbox[3],
                blob,
                confidence,
                guid,
                rev,
                rev_dev,
            ],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Assign `face_id` to `person_id` (or unassign if `None`).
    ///
    /// Also clears the `skipped` flag — assigning a name to a face implicitly
    /// un-skips it.
    pub fn assign_face_to_person(
        &self,
        face_id: i64,
        person_id: Option<i64>,
    ) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE face_detections
             SET person_id = ?1, skipped = 0, rev = ?2, rev_dev = ?3
             WHERE id = ?4",
            params![person_id, rev, rev_dev, face_id],
        )?;
        Ok(())
    }

    /// Mark a face as skipped (or un-skip it).
    ///
    /// Skipped faces are excluded from the default tagging queue and only
    /// shown when the user explicitly opens "Review Skipped Faces".
    pub fn mark_face_skipped(&self, face_id: i64, skipped: bool) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE face_detections SET skipped = ?1, rev = ?2, rev_dev = ?3 WHERE id = ?4",
            params![skipped as i32, rev, rev_dev, face_id],
        )?;
        Ok(())
    }

    /// Update the bounding box of an existing face detection.
    pub fn update_face_bbox(&self, face_id: i64, bbox: [f32; 4]) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE face_detections
             SET bbox_x1 = ?1, bbox_y1 = ?2, bbox_x2 = ?3, bbox_y2 = ?4,
                 rev = ?5, rev_dev = ?6
             WHERE id = ?7",
            params![bbox[0], bbox[1], bbox[2], bbox[3], rev, rev_dev, face_id],
        )?;
        Ok(())
    }

    /// Delete a face detection and its person assignment.
    pub fn delete_face_detection(&self, face_id: i64) -> anyhow::Result<()> {
        self.tombstone("face_detections", &[face_id])?;
        self.conn
            .execute("DELETE FROM face_detections WHERE id = ?1", params![face_id])?;
        Ok(())
    }

    /// Delete **all** face data: every detection and every named person.
    ///
    /// Resets face recognition to a clean slate.  The background face tagger
    /// re-detects every image on its next pass, since
    /// [`images_needing_face_detection`](Self::images_needing_face_detection)
    /// keys off the absence of any `face_detections` row.  Used by the Settings
    /// "Delete All Face Data" debug action (e.g. to recover from duplicate
    /// detections left over from earlier detection runs).
    ///
    /// Returns `(faces_deleted, persons_deleted)`.
    pub fn clear_all_face_data(&self) -> anyhow::Result<(usize, usize)> {
        // Tombstone everything first: this is a deliberate user action, so it
        // must propagate rather than being silently undone by the next sync
        // refilling the tables from a peer.
        for table in ["face_detections", "persons"] {
            let ids: Vec<i64> = self
                .conn
                .prepare(&format!("SELECT id FROM {table}"))?
                .query_map([], |r| r.get(0))?
                .filter_map(|r| r.ok())
                .collect();
            self.tombstone(table, &ids)?;
        }
        let faces = self.conn.execute("DELETE FROM face_detections", [])?;
        let persons = self.conn.execute("DELETE FROM persons", [])?;
        Ok((faces, persons))
    }

    /// Rename an existing person.
    pub fn rename_person(&self, id: i64, name: &str) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE persons SET name = ?1, rev = ?2, rev_dev = ?3 WHERE id = ?4",
            params![name, rev, rev_dev, id],
        )?;
        Ok(())
    }

    /// Delete a person. Their face detections are un-assigned (`person_id`
    /// set to `NULL`) rather than deleted, so the photos and detected
    /// bounding boxes remain — only the name association is removed.
    pub fn delete_person(&self, id: i64) -> anyhow::Result<()> {
        // The un-assignment is a real edit to each face row, not a cascade,
        // so those rows need their own stamps — otherwise a peer would keep
        // showing the deleted name against them.
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE face_detections
             SET person_id = NULL, rev = ?1, rev_dev = ?2
             WHERE person_id = ?3",
            params![rev, rev_dev, id],
        )?;
        self.tombstone("persons", &[id])?;
        self.conn.execute("DELETE FROM persons WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Insert or retrieve a person by name.  Returns the person's `id`.
    pub fn upsert_person(&self, name: &str) -> anyhow::Result<i64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        let (rev, rev_dev) = self.stamp()?;
        let guid = self.new_guid()?;
        self.conn.execute(
            "INSERT INTO persons(name, created_at, guid, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4, ?5)
             ON CONFLICT(name) DO NOTHING",
            params![name, now, guid, rev, rev_dev],
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
                    embedding, person_id, confidence, skipped
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
                    row.get::<_, bool>(9)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, image_id, x1, y1, x2, y2, blob, person_id, confidence, skipped)| {
                FaceDetection {
                    id,
                    image_id,
                    bbox: [x1, y1, x2, y2],
                    embedding: blob_to_embedding(&blob),
                    person_id,
                    confidence,
                    skipped,
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
                    crate::path_from_db(row.get::<_, String>(1)?),
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

    /// Return IDs of present images that have at least one real untagged, non-skipped face.
    pub fn images_with_untagged_faces(&self) -> anyhow::Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT fd.image_id
             FROM face_detections fd
             JOIN images i ON i.id = fd.image_id
             WHERE fd.person_id IS NULL
               AND fd.skipped = 0
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

    /// Return IDs of present images that have at least one real skipped face.
    ///
    /// Used by the "Review Skipped Faces" tagging mode.
    pub fn images_with_skipped_faces(&self) -> anyhow::Result<Vec<i64>> {
        let mut stmt = self.conn.prepare(
            "SELECT DISTINCT fd.image_id
             FROM face_detections fd
             JOIN images i ON i.id = fd.image_id
             WHERE fd.person_id IS NULL
               AND fd.skipped = 1
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

    /// Count real skipped faces across all present images.
    pub fn count_skipped_faces(&self) -> anyhow::Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*)
             FROM face_detections fd
             JOIN images i ON i.id = fd.image_id
             WHERE fd.person_id IS NULL
               AND fd.skipped = 1
               AND fd.confidence >= 0.0
               AND NOT (fd.bbox_x1 = 0.0 AND fd.bbox_y1 = 0.0
                        AND fd.bbox_x2 = 0.0 AND fd.bbox_y2 = 0.0)
               AND i.status = 'present'",
            [],
            |r| r.get(0),
        )?;
        Ok(count as usize)
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

    /// Recompute the centroid embedding and representative face for `person_id`.
    ///
    /// Fetches all assigned embeddings, averages them (then L2-normalises),
    /// picks the face whose embedding is closest to the centroid, and writes
    /// both back to the `persons` row.  No-op (clears both fields) when the
    /// person has no assigned faces.
    pub fn update_person_representative(&self, person_id: i64) -> anyhow::Result<()> {
        // Collect all (face_id, embedding) pairs for this person.
        let mut stmt = self.conn.prepare(
            "SELECT id, embedding FROM face_detections WHERE person_id = ?1",
        )?;
        let faces: Vec<(i64, Vec<f32>)> = stmt
            .query_map(params![person_id], |row| {
                let blob: Vec<u8> = row.get(1)?;
                Ok((row.get::<_, i64>(0)?, blob))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, blob)| (id, blob_to_embedding(&blob)))
            .collect();

        let (centroid, best_id) = crate::embedding::centroid_and_nearest(&faces);
        let centroid_blob = centroid.as_deref().map(embedding_to_blob);
        // Not stamped: both columns are derived from this device's face rows,
        // and `representative_face_id` is a *local* rowid that would be
        // meaningless on a peer. Each side recomputes them after applying a
        // sync batch.
        self.conn.execute(
            "UPDATE persons SET centroid_embedding = ?1, representative_face_id = ?2 WHERE id = ?3",
            params![centroid_blob, best_id, person_id],
        )?;
        Ok(())
    }

    /// Return every person together with image path and bbox of their
    /// representative face (the detection closest to the person centroid).
    ///
    /// Persons with no assigned faces still appear in the list but with
    /// `image_path = None` and `bbox = None`.
    pub fn all_persons_with_representatives(&self) -> anyhow::Result<Vec<PersonWithRep>> {
        let mut stmt = self.conn.prepare(
            "SELECT p.id, p.name,
                    i.path,
                    fd.bbox_x1, fd.bbox_y1, fd.bbox_x2, fd.bbox_y2,
                    fd.id
             FROM persons p
             LEFT JOIN face_detections fd ON fd.id = p.representative_face_id
             LEFT JOIN images i           ON i.id  = fd.image_id
             ORDER BY p.name",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<String>>(2)?,
                    row.get::<_, Option<f32>>(3)?,
                    row.get::<_, Option<f32>>(4)?,
                    row.get::<_, Option<f32>>(5)?,
                    row.get::<_, Option<f32>>(6)?,
                    row.get::<_, Option<i64>>(7)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, name, path, x1, y1, x2, y2, face_id)| PersonWithRep {
                id,
                name,
                image_path: path.map(crate::path_from_db),
                bbox: match (x1, y1, x2, y2) {
                    (Some(a), Some(b), Some(c), Some(d)) => Some([a, b, c, d]),
                    _ => None,
                },
                face_id,
            })
            .collect();
        Ok(rows)
    }

    /// Count real untagged non-skipped faces across all present images.
    pub fn untagged_face_count(&self) -> anyhow::Result<usize> {
        let count: i64 = self.conn.query_row(
            "SELECT COUNT(*)
             FROM face_detections fd
             JOIN images i ON i.id = fd.image_id
             WHERE fd.person_id IS NULL
               AND fd.skipped = 0
               AND fd.confidence >= 0.0
               AND NOT (fd.bbox_x1 = 0.0 AND fd.bbox_y1 = 0.0
                        AND fd.bbox_x2 = 0.0 AND fd.bbox_y2 = 0.0)
               AND i.status = 'present'",
            [],
            |r| r.get(0),
        )?;
        Ok(count as usize)
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

// ── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn tmp_db() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        (dir, db)
    }

    fn insert_image(db: &Database, name: &str) -> i64 {
        let path = PathBuf::from(format!("/photos/{name}"));
        db.insert_image(&path, &[0u8; 32], 1024).unwrap();
        db.image_id_for_path(&path).unwrap().unwrap()
    }

    fn norm_vec(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn rename_person_updates_name() {
        let (_dir, db) = tmp_db();
        let pid = db.upsert_person("Alice").unwrap();
        db.rename_person(pid, "Alicia").unwrap();
        assert_eq!(db.person_name(pid).unwrap(), Some("Alicia".to_string()));
    }

    #[test]
    fn delete_person_unassigns_faces_but_keeps_detections() {
        let (_dir, db) = tmp_db();
        let pid = db.upsert_person("Bob").unwrap();
        let img = insert_image(&db, "a.jpg");
        let face_id = db
            .insert_face_detection(img, [0.0, 0.0, 1.0, 1.0], &[1.0, 0.0], 0.9)
            .unwrap();
        db.assign_face_to_person(face_id, Some(pid)).unwrap();

        db.delete_person(pid).unwrap();

        assert_eq!(db.person_name(pid).unwrap(), None);
        let faces = db.faces_for_image(img).unwrap();
        assert_eq!(faces.len(), 1);
        assert_eq!(faces[0].person_id, None);
    }

    #[test]
    fn update_person_representative_picks_nearest_to_centroid() {
        let (_dir, db) = tmp_db();
        let pid = db.upsert_person("Carol").unwrap();
        let img = insert_image(&db, "b.jpg");

        let f1 = db
            .insert_face_detection(img, [0.0, 0.0, 0.1, 0.1], &norm_vec(vec![1.0, 0.0]), 0.9)
            .unwrap();
        let f2 = db
            .insert_face_detection(img, [0.1, 0.1, 0.2, 0.2], &norm_vec(vec![0.0, 1.0]), 0.9)
            .unwrap();
        let f3 = db
            .insert_face_detection(
                img,
                [0.2, 0.2, 0.3, 0.3],
                &norm_vec(vec![0.9, 0.436]),
                0.9,
            )
            .unwrap();
        for f in [f1, f2, f3] {
            db.assign_face_to_person(f, Some(pid)).unwrap();
        }

        db.update_person_representative(pid).unwrap();

        let reps = db.all_persons_with_representatives().unwrap();
        let rep = reps.iter().find(|p| p.id == pid).unwrap();
        assert_eq!(rep.face_id, Some(f3));
    }

    #[test]
    fn update_person_representative_clears_when_no_faces() {
        let (_dir, db) = tmp_db();
        let pid = db.upsert_person("Dave").unwrap();
        db.update_person_representative(pid).unwrap();
        let reps = db.all_persons_with_representatives().unwrap();
        let rep = reps.iter().find(|p| p.id == pid).unwrap();
        assert_eq!(rep.face_id, None);
        assert_eq!(rep.image_path, None);
    }
}
