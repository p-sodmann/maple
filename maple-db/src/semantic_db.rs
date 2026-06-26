//! Database operations for semantic search (schema V7).
//!
//! # Tables
//!
//! `sentence_embeddings(id, image_id, description_id, encoder_model,
//!  sentence_index, text, created_at)` — one row per embedded sentence.  A
//!  row with `sentence_index = -1` is a **sentinel** marking a description that
//!  was processed but yielded no embeddable sentences (prevents reprocessing).
//!
//! `vec_sentences` — sqlite-vec `vec0` table holding the float32 vectors; its
//!  `rowid` equals the `sentence_embeddings.id` of the same sentence.
//!
//! `semantic_meta(id=1, encoder_model, dim)` — the active encoder + vector
//!  dimension, used to rebuild the index when the model changes.

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::{params, types::Value};

use crate::faces::embedding_to_blob;
use crate::{Database, LibraryImage};

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

impl Database {
    /// Return `(description_id, image_id, description)` for present-image
    /// descriptions that have not yet been embedded by `encoder_model`.
    ///
    /// Mirrors `images_needing_ai_description`: a description is "needing
    /// embedding" while no `sentence_embeddings` row exists for it under the
    /// active encoder (real rows *or* the sentinel both count as processed).
    pub fn descriptions_needing_embedding(
        &self,
        encoder_model: &str,
    ) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT ad.id, ad.image_id, ad.description
             FROM ai_descriptions ad
             JOIN images i ON i.id = ad.image_id AND i.status = 'present'
             WHERE NOT EXISTS (
                 SELECT 1 FROM sentence_embeddings se
                 WHERE se.description_id = ad.id AND se.encoder_model = ?1
             )",
        )?;
        let rows = stmt
            .query_map(params![encoder_model], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, i64>(1)?,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Store the per-sentence embeddings for one description.
    ///
    /// `sentences` is `(text, embedding)` pairs in order.  Each is written to
    /// `sentence_embeddings` and its vector to `vec_sentences` (shared rowid).
    /// If `sentences` is empty, a sentinel row is written so the description is
    /// not reprocessed on the next pass.
    pub fn insert_sentence_embeddings(
        &self,
        image_id: i64,
        description_id: i64,
        encoder_model: &str,
        sentences: &[(String, Vec<f32>)],
    ) -> anyhow::Result<()> {
        let now = now_secs();
        let tx = self.conn.unchecked_transaction()?;

        if sentences.is_empty() {
            tx.execute(
                "INSERT INTO sentence_embeddings
                     (image_id, description_id, encoder_model, sentence_index, text, created_at)
                 VALUES (?1, ?2, ?3, -1, '', ?4)",
                params![image_id, description_id, encoder_model, now],
            )?;
        } else {
            for (idx, (text, embedding)) in sentences.iter().enumerate() {
                tx.execute(
                    "INSERT INTO sentence_embeddings
                         (image_id, description_id, encoder_model, sentence_index, text, created_at)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![image_id, description_id, encoder_model, idx as i64, text, now],
                )?;
                let row_id = tx.last_insert_rowid();
                let blob = embedding_to_blob(embedding);
                tx.execute(
                    "INSERT INTO vec_sentences(rowid, embedding) VALUES (?1, ?2)",
                    params![row_id, blob],
                )?;
            }
        }

        tx.commit()?;
        Ok(())
    }

    /// Ensure the `vec_sentences` table matches the active encoder + dimension.
    ///
    /// If the stored `(encoder_model, dim)` differs, the index is rebuilt from
    /// scratch: all sentence rows are cleared (the delete trigger drops their
    /// vectors) and the vec0 table is recreated at the new dimension.  This is
    /// the model-change invalidation path; call it once when an encoder loads.
    pub fn ensure_vec_table(&self, encoder_model: &str, dim: usize) -> anyhow::Result<()> {
        let current: Option<(String, i64)> = self
            .conn
            .query_row(
                "SELECT encoder_model, dim FROM semantic_meta WHERE id = 1",
                [],
                |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)?)),
            )
            .ok();

        if current.as_ref() == Some(&(encoder_model.to_owned(), dim as i64)) {
            return Ok(());
        }

        tracing::info!(
            model = encoder_model,
            dim,
            "(re)building semantic vector index"
        );

        let tx = self.conn.unchecked_transaction()?;
        // Delete sentence rows first so the AFTER DELETE trigger can clean the
        // (still-existing) vec table; then drop and recreate it at the new dim.
        tx.execute_batch("DELETE FROM sentence_embeddings;")?;
        tx.execute_batch("DROP TABLE IF EXISTS vec_sentences;")?;
        tx.execute_batch(&crate::schema::vec_table_ddl(dim as i64))?;
        tx.execute(
            "INSERT INTO semantic_meta(id, encoder_model, dim) VALUES (1, ?1, ?2)
             ON CONFLICT(id) DO UPDATE SET
                 encoder_model = excluded.encoder_model,
                 dim           = excluded.dim",
            params![encoder_model, dim as i64],
        )?;
        tx.commit()?;
        Ok(())
    }

    /// K-nearest-neighbour search over sentence vectors.
    ///
    /// Returns `(image_id, distance, sentence)` for present images, one row per
    /// image (best/lowest distance across its sentences), ordered nearest-first.
    /// `sentence` is the embedded text from the winning row.  SQLite's min/max
    /// bare-column extension guarantees that the bare columns (`se.text`) come
    /// from the same row as `MIN(knn.distance)`.
    /// `query_embedding` must have the active encoder's dimension.
    pub fn semantic_search(
        &self,
        query_embedding: &[f32],
        k: usize,
    ) -> anyhow::Result<Vec<(i64, f32, String)>> {
        let blob = embedding_to_blob(query_embedding);
        // `k` is a trusted internal value (no injection risk) and sqlite-vec
        // wants it as a literal constraint.
        let sql = format!(
            "WITH knn AS (
                 SELECT rowid, distance
                 FROM vec_sentences
                 WHERE embedding MATCH ?1 AND k = {k}
             )
             SELECT se.image_id, MIN(knn.distance) AS dist, se.text
             FROM knn
             JOIN sentence_embeddings se ON se.id = knn.rowid
             JOIN images i ON i.id = se.image_id AND i.status = 'present'
             GROUP BY se.image_id
             ORDER BY dist ASC"
        );
        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(params![blob], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, f64>(1)? as f32,
                    row.get::<_, String>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Fetch present images for `ids`, returned in the same order as `ids`
    /// (any missing/non-present ids are skipped).  Optionally restricted to a
    /// collection.  Used to materialise a fused (ranked) result list.
    pub fn images_by_ids_ordered(
        &self,
        ids: &[i64],
        collection_id: Option<i64>,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        if ids.is_empty() {
            return Ok(vec![]);
        }

        let placeholders: String = ids.iter().map(|_| "?").collect::<Vec<_>>().join(", ");
        let coll_clause = if collection_id.is_some() {
            " AND id IN (SELECT image_id FROM collection_images WHERE collection_id = ?)"
        } else {
            ""
        };
        let sql = format!(
            "SELECT id, path, added_at, status,
                    filename, taken_at, make, model, lens,
                    focal_length, aperture, iso,
                    width, height, orientation, raw_path, hash
             FROM images
             WHERE id IN ({placeholders}) AND status = 'present'{coll_clause}"
        );

        let mut sql_params: Vec<Value> = ids.iter().map(|id| Value::Integer(*id)).collect();
        if let Some(cid) = collection_id {
            sql_params.push(Value::Integer(cid));
        }

        let mut stmt = self.conn.prepare(&sql)?;
        let fetched: Vec<LibraryImage> = stmt
            .query_map(rusqlite::params_from_iter(sql_params), crate::row_to_library_image)?
            .filter_map(|r| r.ok())
            .collect();

        // Reorder to match `ids`.
        let mut by_id: std::collections::HashMap<i64, LibraryImage> =
            fetched.into_iter().map(|img| (img.id, img)).collect();
        Ok(ids.iter().filter_map(|id| by_id.remove(id)).collect())
    }

    /// Hybrid search: direct keyword hits first, then semantic-only hits.
    ///
    /// Images matched by keywords are ranked ahead of images found only via
    /// vector similarity.  Within each group the keyword group keeps its
    /// recency order; the semantic group keeps its similarity order.
    /// Images that appear in both groups are treated as direct hits and are
    /// not repeated in the semantic group.
    ///
    /// Keyword images are reused directly from `search_images_text` so their
    /// pre-computed `search_hit` (including description snippet) is preserved.
    pub(crate) fn search_images_hybrid(
        &self,
        text: &str,
        query_embedding: &[f32],
        k: usize,
        limit: Option<usize>,
        offset: Option<usize>,
        collection_id: Option<i64>,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        let limit = limit.unwrap_or(500);
        let offset = offset.unwrap_or(0);
        let pool = (limit + offset).max(200);

        // Keyword hits already carry SearchHit::Direct (with snippet) from
        // search_images_text.
        let keyword = self.search_images_text(text, Some(pool), Some(0), collection_id)?;
        // Semantic hits: (image_id, cosine_distance, best_sentence).
        let semantic = self.semantic_search(query_embedding, k)?;

        // Build ordered ID list: keyword first, then semantic-only.
        let mut ordered: Vec<i64> = Vec::with_capacity(keyword.len() + semantic.len());
        let mut seen: std::collections::HashSet<i64> = std::collections::HashSet::new();
        for img in &keyword {
            if seen.insert(img.id) {
                ordered.push(img.id);
            }
        }
        for (id, _dist, _sent) in &semantic {
            if seen.insert(*id) {
                ordered.push(*id);
            }
        }

        let page: Vec<i64> = ordered.into_iter().skip(offset).take(limit).collect();

        // Keep keyword images as-is (their search_hit is already set).
        let keyword_map: std::collections::HashMap<i64, LibraryImage> =
            keyword.into_iter().map(|img| (img.id, img)).collect();

        // For semantic-only images on the page, fetch and tag them separately.
        let semantic_only_page: Vec<i64> = page
            .iter()
            .filter(|id| !keyword_map.contains_key(*id))
            .copied()
            .collect();

        let semantic_info: std::collections::HashMap<i64, (f32, String)> = semantic
            .into_iter()
            .map(|(id, dist, sent)| (id, (1.0 - dist, sent)))
            .collect();

        let mut semantic_only_images =
            self.images_by_ids_ordered(&semantic_only_page, collection_id)?;
        for img in &mut semantic_only_images {
            if let Some((sim, sent)) = semantic_info.get(&img.id) {
                img.search_hit = Some(crate::SearchHit::Semantic {
                    similarity: *sim,
                    sentence: sent.clone(),
                });
            }
        }
        let semantic_only_map: std::collections::HashMap<i64, LibraryImage> =
            semantic_only_images.into_iter().map(|img| (img.id, img)).collect();

        // Assemble the final page in order.
        let result = page
            .iter()
            .filter_map(|id| {
                keyword_map
                    .get(id)
                    .or_else(|| semantic_only_map.get(id))
                    .cloned()
            })
            .collect();
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── sqlite-vec integration ───────────────────────────────────

    use crate::SearchQuery;
    use std::path::Path;

    /// Open a fresh DB, register two images each with an AI description, and
    /// return `(db, dir, [(image_id, description_id)])` for the given model.
    fn db_with_descriptions(model: &str) -> (Database, tempfile::TempDir, Vec<(i64, i64)>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        db.insert_image(Path::new("/p/a.jpg"), &[1u8; 32], 10).unwrap();
        db.insert_image(Path::new("/p/b.jpg"), &[2u8; 32], 10).unwrap();

        // image ids, keyed by filename
        let imgs = db.search_images(&SearchQuery::default()).unwrap();
        for img in &imgs {
            db.insert_ai_description(img.id, model, "an apple on a table.")
                .unwrap();
        }

        // (desc_id, image_id) pairs for the model
        let pairs: Vec<(i64, i64)> = db
            .descriptions_needing_embedding(model)
            .unwrap()
            .into_iter()
            .map(|(desc_id, image_id, _)| (image_id, desc_id))
            .collect();
        (db, dir, pairs)
    }

    #[test]
    fn knn_returns_nearest_image() {
        let model = "test-model";
        let (db, _dir, pairs) = db_with_descriptions(model);
        db.ensure_vec_table(model, 4).unwrap();

        // Plant one orthogonal vector per image.
        let (img_a, desc_a) = pairs[0];
        let (img_b, desc_b) = pairs[1];
        db.insert_sentence_embeddings(img_a, desc_a, model, &[("s".into(), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        db.insert_sentence_embeddings(img_b, desc_b, model, &[("s".into(), vec![0.0, 1.0, 0.0, 0.0])])
            .unwrap();

        // A query aligned with image A's vector ranks A first.
        let hits = db.semantic_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap();
        assert_eq!(hits.first().map(|(id, _, _)| *id), Some(img_a));

        // Both descriptions are now processed → none pending.
        assert!(db.descriptions_needing_embedding(model).unwrap().is_empty());
    }

    #[test]
    fn changing_description_invalidates_embeddings() {
        let model = "test-model";
        let (db, _dir, pairs) = db_with_descriptions(model);
        db.ensure_vec_table(model, 4).unwrap();
        let (img_a, desc_a) = pairs[0];
        db.insert_sentence_embeddings(img_a, desc_a, model, &[("s".into(), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        assert_eq!(db.semantic_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap().len() >= 1, true);

        // Editing the description text must drop its sentence + vector rows
        // (via the AFTER UPDATE trigger) so it is re-embedded next pass.
        db.insert_ai_description(img_a, model, "a completely different caption.")
            .unwrap();

        let pending = db.descriptions_needing_embedding(model).unwrap();
        assert!(pending.iter().any(|(_, image_id, _)| *image_id == img_a));
        assert!(db.semantic_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap().is_empty());
    }

    #[test]
    fn changing_model_rebuilds_index() {
        let model = "test-model";
        let (db, _dir, pairs) = db_with_descriptions(model);
        db.ensure_vec_table(model, 4).unwrap();
        let (img_a, desc_a) = pairs[0];
        db.insert_sentence_embeddings(img_a, desc_a, model, &[("s".into(), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        assert!(!db.semantic_search(&[1.0, 0.0, 0.0, 0.0], 10).unwrap().is_empty());

        // Switching to a model with a different dimension clears all vectors.
        db.ensure_vec_table("other-model", 8).unwrap();
        assert!(db.semantic_search(&[0.0; 8], 10).unwrap().is_empty());
        // All descriptions need re-embedding under the new model.
        assert_eq!(db.descriptions_needing_embedding("other-model").unwrap().len(), 2);
    }

    #[test]
    fn empty_description_gets_sentinel() {
        // A description that yields no sentences is marked processed via a
        // sentinel and not returned as pending again.
        let model = "test-model";
        let (db, _dir, pairs) = db_with_descriptions(model);
        db.ensure_vec_table(model, 4).unwrap();
        let (img_a, desc_a) = pairs[0];
        db.insert_sentence_embeddings(img_a, desc_a, model, &[]).unwrap();
        assert!(!db
            .descriptions_needing_embedding(model)
            .unwrap()
            .iter()
            .any(|(_, image_id, _)| *image_id == img_a));
    }

    #[test]
    fn hybrid_direct_hits_before_semantic() {
        let model = "test-model";
        let (db, _dir, pairs) = db_with_descriptions(model);
        db.ensure_vec_table(model, 4).unwrap();
        let (img_a, desc_a) = pairs[0];
        let (img_b, desc_b) = pairs[1];
        db.insert_sentence_embeddings(img_a, desc_a, model, &[("s".into(), vec![1.0, 0.0, 0.0, 0.0])])
            .unwrap();
        db.insert_sentence_embeddings(img_b, desc_b, model, &[("s".into(), vec![0.0, 1.0, 0.0, 0.0])])
            .unwrap();

        // Both descriptions contain "apple" so both are keyword (direct) hits.
        // Direct hits appear before semantic-only hits; both images surface here.
        let results = db
            .search_images_hybrid("apple", &[0.9, 0.1, 0.0, 0.0], 10, None, None, None)
            .unwrap();
        let mut ids: Vec<i64> = results.iter().map(|i| i.id).collect();
        ids.sort();
        assert_eq!(ids, vec![img_a, img_b]);

        // Verify all results are tagged as Direct hits (not Semantic).
        for r in &results {
            assert!(matches!(r.search_hit, Some(crate::SearchHit::Direct { .. })));
        }
    }
}
