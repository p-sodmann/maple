//! Collections — named, colour-coded groups of images.
//!
//! Each collection has a unique name and a hex colour string.  Images are
//! linked via the `collection_images` join table (many-to-many).

use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::params;

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
}

// ── Database methods ────────────────────────────────────────────

impl Database {
    // ── Write ────────────────────────────────────────────────────

    /// Create a new collection.  Returns the new row id.
    pub fn create_collection(&self, name: &str, color: &str) -> anyhow::Result<i64> {
        let now = now_secs();
        self.conn.execute(
            "INSERT INTO collections (name, color, created_at) VALUES (?1, ?2, ?3)",
            params![name, color, now],
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
                     WHERE ci.collection_id = c.id) AS cnt
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
                     WHERE ci.collection_id = c.id) AS cnt
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
            })
        })?;
        Ok(rows.next().transpose()?)
    }

    /// Return all collections that `image_id` belongs to.
    pub fn collections_for_image(&self, image_id: i64) -> anyhow::Result<Vec<Collection>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.id, c.name, c.color, c.created_at,
                    (SELECT COUNT(*) FROM collection_images ci2
                     WHERE ci2.collection_id = c.id) AS cnt
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
                })
            })?
            .filter_map(|r| r.ok())
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
        let id = db.create_collection("Favourites", "#e01b24").unwrap();
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
        let id = db.create_collection("Old", "#000000").unwrap();
        db.rename_collection(id, "New").unwrap();
        db.set_collection_color(id, "#ffffff").unwrap();
        let c = db.collection_by_id(id).unwrap().unwrap();
        assert_eq!(c.name, "New");
        assert_eq!(c.color, "#ffffff");
    }

    #[test]
    fn add_remove_image() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Test", "#3584e4").unwrap();
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
        let cid = db.create_collection("Count", "#3584e4").unwrap();
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
        let cid = db.create_collection("Gone", "#3584e4").unwrap();
        let img = insert_image(&db, "z.jpg");
        db.add_image_to_collection(cid, img).unwrap();
        db.delete_collection(cid).unwrap();
        assert!(db.collections_for_image(img).unwrap().is_empty());
    }
}
