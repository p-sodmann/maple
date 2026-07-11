//! Collections — named, colour-coded groups of images.
//!
//! Each collection has a unique name and a hex colour string.  Images are
//! linked via the `collection_images` join table (many-to-many).

use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

use crate::faces::{blob_to_embedding, embedding_to_blob};
use crate::Database;

// ── Types ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct Collection {
    pub id: i64,
    pub name: String,
    /// Hex colour, e.g. `"#3584e4"`.
    pub color: String,
    pub created_at: i64,
    /// Number of images in this collection (populated by list queries).
    pub image_count: u64,
    /// Optional parent collection id (`None` = top-level).
    pub parent_id: Option<i64>,
}

/// A collection together with the data needed to render its representative
/// (cover) image — the [`PersonWithRep`](crate::PersonWithRep) equivalent for
/// collections.
#[derive(Debug, Clone)]
pub struct CollectionWithRep {
    pub id: i64,
    pub name: String,
    pub color: String,
    pub image_count: u64,
    pub parent_id: Option<i64>,
    /// Id of the representative member image, if one has been computed.
    /// Used as a cheap cache-invalidation key (mirrors
    /// [`PersonWithRep::face_id`](crate::PersonWithRep)) — the path alone
    /// changes identity too, but comparing an `i64` is cheaper than a `PathBuf`.
    pub image_id: Option<i64>,
    /// Path to the representative member image, if one has been computed.
    pub image_path: Option<PathBuf>,
}

// ── Database methods ────────────────────────────────────────────

impl Database {
    // ── Write ────────────────────────────────────────────────────

    /// Create a new collection.  Returns the new row id.
    pub fn create_collection(&self, name: &str, color: &str, parent_id: Option<i64>) -> anyhow::Result<i64> {
        let now = now_secs();
        self.conn.execute(
            "INSERT INTO collections (name, color, created_at, parent_id) VALUES (?1, ?2, ?3, ?4)",
            params![name, color, now, parent_id],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Rename an existing collection.
    pub fn rename_collection(&self, id: i64, name: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE collections SET name = ?1 WHERE id = ?2",
            params![name, id],
        )?;
        Ok(())
    }

    /// Change a collection's colour.
    pub fn set_collection_color(&self, id: i64, color: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE collections SET color = ?1 WHERE id = ?2",
            params![color, id],
        )?;
        Ok(())
    }

    /// Delete a collection.  Memberships are removed via `ON DELETE CASCADE`.
    pub fn delete_collection(&self, id: i64) -> anyhow::Result<()> {
        self.conn
            .execute("DELETE FROM collections WHERE id = ?1", params![id])?;
        Ok(())
    }

    /// Add an image to a collection.  No-op if already a member.
    pub fn add_image_to_collection(
        &self,
        collection_id: i64,
        image_id: i64,
    ) -> anyhow::Result<()> {
        let now = now_secs();
        self.conn.execute(
            "INSERT OR IGNORE INTO collection_images (collection_id, image_id, added_at)
             VALUES (?1, ?2, ?3)",
            params![collection_id, image_id, now],
        )?;
        Ok(())
    }

    /// Remove an image from a collection.
    pub fn remove_image_from_collection(
        &self,
        collection_id: i64,
        image_id: i64,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "DELETE FROM collection_images
             WHERE collection_id = ?1 AND image_id = ?2",
            params![collection_id, image_id],
        )?;
        Ok(())
    }

    // ── Read ─────────────────────────────────────────────────────

    /// Return all collections ordered by name, with image counts.
    pub fn all_collections(&self) -> anyhow::Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.color, c.created_at,
                    (SELECT COUNT(*) FROM collection_images ci
                     WHERE ci.collection_id = c.id) AS cnt,
                    c.parent_id
             FROM collections c
             ORDER BY c.name",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok(Collection {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    color: row.get(2)?,
                    created_at: row.get(3)?,
                    image_count: row.get::<_, i64>(4)? as u64,
                    parent_id: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Fetch a single collection by id.
    pub fn collection_by_id(&self, id: i64) -> anyhow::Result<Option<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.color, c.created_at,
                    (SELECT COUNT(*) FROM collection_images ci
                     WHERE ci.collection_id = c.id) AS cnt,
                    c.parent_id
             FROM collections c
             WHERE c.id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], |row| {
            Ok(Collection {
                id: row.get(0)?,
                name: row.get(1)?,
                color: row.get(2)?,
                created_at: row.get(3)?,
                image_count: row.get::<_, i64>(4)? as u64,
                parent_id: row.get(5)?,
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    /// Return all collections that `image_id` belongs to.
    pub fn collections_for_image(&self, image_id: i64) -> anyhow::Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.color, c.created_at,
                    (SELECT COUNT(*) FROM collection_images ci2
                     WHERE ci2.collection_id = c.id) AS cnt,
                    c.parent_id
             FROM collections c
             INNER JOIN collection_images ci ON ci.collection_id = c.id
             WHERE ci.image_id = ?1
             ORDER BY c.name",
        )?;
        let rows = stmt
            .query_map(params![image_id], |row| {
                Ok(Collection {
                    id: row.get(0)?,
                    name: row.get(1)?,
                    color: row.get(2)?,
                    created_at: row.get(3)?,
                    image_count: row.get::<_, i64>(4)? as u64,
                    parent_id: row.get(5)?,
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Recompute the centroid embedding and representative (cover) image for
    /// `collection_id`, using whole-image DINOv2 embeddings from
    /// `image_hashes` (see [`crate::hasher`]).
    ///
    /// A library can carry `image_hashes` rows under more than one
    /// `algorithm` key (e.g. after `model_repo` was changed in settings), so
    /// member images are grouped by algorithm and the group covering the
    /// most members is used — this matches how the stacker treats
    /// `algorithm` as the active-model selector.
    ///
    /// When none of the collection's members have any stored embedding yet
    /// (background hasher still running, or a brand-new collection), falls
    /// back to the most-recently-added member as the cover, leaving
    /// `centroid_embedding` `NULL`. An empty collection clears both columns.
    pub fn update_collection_representative(&self, collection_id: i64) -> anyhow::Result<()> {
        let mut stmt = self.conn.prepare(
            "SELECT ih.algorithm, ih.image_id, ih.hash_blob
             FROM image_hashes ih
             JOIN collection_images ci ON ci.image_id = ih.image_id
             WHERE ci.collection_id = ?1",
        )?;
        let mut by_algorithm: HashMap<String, Vec<(i64, Vec<f32>)>> = HashMap::new();
        let rows = stmt.query_map(params![collection_id], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, Vec<u8>>(2)?,
            ))
        })?;
        for row in rows.filter_map(|r| r.ok()) {
            let (algorithm, image_id, blob) = row;
            by_algorithm
                .entry(algorithm)
                .or_default()
                .push((image_id, blob_to_embedding(&blob)));
        }

        let best_group = by_algorithm
            .into_values()
            .max_by_key(|items| items.len());

        let (centroid, representative_id) = match best_group {
            Some(items) if !items.is_empty() => crate::embedding::centroid_and_nearest(&items),
            _ => {
                // `added_at` has one-second resolution, so ties (e.g. a batch
                // add) fall back to `id DESC`, which follows insertion order.
                let fallback: Option<i64> = self.conn.query_row(
                    "SELECT image_id FROM collection_images
                     WHERE collection_id = ?1
                     ORDER BY added_at DESC, id DESC LIMIT 1",
                    params![collection_id],
                    |r| r.get(0),
                ).ok();
                (None, fallback)
            }
        };

        let centroid_blob = centroid.as_deref().map(embedding_to_blob);
        self.conn.execute(
            "UPDATE collections SET centroid_embedding = ?1, representative_image_id = ?2 WHERE id = ?3",
            params![centroid_blob, representative_id, collection_id],
        )?;
        Ok(())
    }

    /// Return every collection together with the path of its representative
    /// (cover) image, if one has been computed.
    pub fn all_collections_with_representatives(&self) -> anyhow::Result<Vec<CollectionWithRep>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.color, c.parent_id,
                    (SELECT COUNT(*) FROM collection_images ci
                     WHERE ci.collection_id = c.id) AS cnt,
                    c.representative_image_id, i.path
             FROM collections c
             LEFT JOIN images i ON i.id = c.representative_image_id
             ORDER BY c.name",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, String>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                    row.get::<_, i64>(4)?,
                    row.get::<_, Option<i64>>(5)?,
                    row.get::<_, Option<String>>(6)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, name, color, parent_id, cnt, image_id, path)| CollectionWithRep {
                id,
                name,
                color,
                image_count: cnt as u64,
                parent_id,
                image_id,
                image_path: path.map(crate::path_from_db),
            })
            .collect();
        Ok(rows)
    }
}

// ── Helpers ─────────────────────────────────────────────────────

fn now_secs() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
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
        db.search_images(&crate::SearchQuery::default())
            .unwrap()
            .iter()
            .find(|img| img.path == path)
            .unwrap()
            .id
    }

    #[test]
    fn create_and_list() {
        let (_dir, db) = tmp_db();
        let id = db.create_collection("Favourites", "#e01b24", None).unwrap();
        let all = db.all_collections().unwrap();
        assert_eq!(all.len(), 1);
        assert_eq!(all[0].id, id);
        assert_eq!(all[0].name, "Favourites");
        assert_eq!(all[0].color, "#e01b24");
        assert_eq!(all[0].image_count, 0);
    }

    #[test]
    fn rename_and_recolor() {
        let (_dir, db) = tmp_db();
        let id = db.create_collection("Old", "#000000", None).unwrap();
        db.rename_collection(id, "New").unwrap();
        db.set_collection_color(id, "#ffffff").unwrap();
        let c = db.collection_by_id(id).unwrap().unwrap();
        assert_eq!(c.name, "New");
        assert_eq!(c.color, "#ffffff");
    }

    #[test]
    fn add_remove_image() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Test", "#3584e4", None).unwrap();
        let img = insert_image(&db, "a.jpg");

        db.add_image_to_collection(cid, img).unwrap();
        assert_eq!(db.collections_for_image(img).unwrap().len(), 1);

        // Duplicate add is a no-op.
        db.add_image_to_collection(cid, img).unwrap();
        assert_eq!(db.collections_for_image(img).unwrap().len(), 1);

        db.remove_image_from_collection(cid, img).unwrap();
        assert!(db.collections_for_image(img).unwrap().is_empty());
    }

    #[test]
    fn image_count_updates() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Count", "#3584e4", None).unwrap();
        let img1 = insert_image(&db, "x.jpg");
        let img2 = insert_image(&db, "y.jpg");

        db.add_image_to_collection(cid, img1).unwrap();
        db.add_image_to_collection(cid, img2).unwrap();
        let c = db.collection_by_id(cid).unwrap().unwrap();
        assert_eq!(c.image_count, 2);
    }

    #[test]
    fn delete_collection_cascades() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Gone", "#3584e4", None).unwrap();
        let img = insert_image(&db, "z.jpg");
        db.add_image_to_collection(cid, img).unwrap();
        db.delete_collection(cid).unwrap();
        assert!(db.collections_for_image(img).unwrap().is_empty());
    }

    fn norm_vec(v: Vec<f32>) -> Vec<f32> {
        let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn representative_picks_image_nearest_centroid() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Trip", "#3584e4", None).unwrap();
        let a = insert_image(&db, "a.jpg");
        let b = insert_image(&db, "b.jpg");
        let c = insert_image(&db, "c.jpg");
        for img in [a, b, c] {
            db.add_image_to_collection(cid, img).unwrap();
        }
        db.insert_image_hash(a, "onnx:test", &embedding_to_blob(&norm_vec(vec![1.0, 0.0]))).unwrap();
        db.insert_image_hash(b, "onnx:test", &embedding_to_blob(&norm_vec(vec![0.0, 1.0]))).unwrap();
        db.insert_image_hash(c, "onnx:test", &embedding_to_blob(&norm_vec(vec![0.9, 0.436]))).unwrap();

        db.update_collection_representative(cid).unwrap();

        let reps = db.all_collections_with_representatives().unwrap();
        let rep = reps.iter().find(|r| r.id == cid).unwrap();
        assert_eq!(rep.image_path, Some(PathBuf::from("/photos/c.jpg")));
    }

    #[test]
    fn representative_falls_back_to_most_recent_without_embeddings() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("NoEmbeds", "#3584e4", None).unwrap();
        let a = insert_image(&db, "a.jpg");
        let b = insert_image(&db, "b.jpg");
        db.add_image_to_collection(cid, a).unwrap();
        db.add_image_to_collection(cid, b).unwrap();

        db.update_collection_representative(cid).unwrap();

        let reps = db.all_collections_with_representatives().unwrap();
        let rep = reps.iter().find(|r| r.id == cid).unwrap();
        assert_eq!(rep.image_path, Some(PathBuf::from("/photos/b.jpg")));
    }

    #[test]
    fn representative_clears_for_empty_collection() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Empty", "#3584e4", None).unwrap();
        db.update_collection_representative(cid).unwrap();
        let reps = db.all_collections_with_representatives().unwrap();
        let rep = reps.iter().find(|r| r.id == cid).unwrap();
        assert_eq!(rep.image_path, None);
    }

    #[test]
    fn representative_uses_algorithm_with_most_coverage() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Mixed", "#3584e4", None).unwrap();
        let a = insert_image(&db, "a.jpg");
        let b = insert_image(&db, "b.jpg");
        db.add_image_to_collection(cid, a).unwrap();
        db.add_image_to_collection(cid, b).unwrap();
        // Stale single-image algorithm from an old model_repo.
        db.insert_image_hash(a, "onnx:old-model", &embedding_to_blob(&norm_vec(vec![1.0, 0.0]))).unwrap();
        // Current algorithm covers both members — should win.
        db.insert_image_hash(a, "onnx:new-model", &embedding_to_blob(&norm_vec(vec![1.0, 0.0]))).unwrap();
        db.insert_image_hash(b, "onnx:new-model", &embedding_to_blob(&norm_vec(vec![0.0, 1.0]))).unwrap();

        db.update_collection_representative(cid).unwrap();

        let reps = db.all_collections_with_representatives().unwrap();
        let rep = reps.iter().find(|r| r.id == cid).unwrap();
        // With only two points, the centroid is equidistant — either member is a
        // valid nearest pick, but it must have come from the 2-image group.
        assert!(rep.image_path == Some(PathBuf::from("/photos/a.jpg"))
            || rep.image_path == Some(PathBuf::from("/photos/b.jpg")));
    }
}
