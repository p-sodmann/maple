//! maple-db — SQLite-backed image library database.
//!
//! Stores every image that has been imported into the library, keyed by its
//! file-system path and BLAKE3 content hash.  A background scanner
//! (`LibraryScanner`) periodically reconciles the on-disk state with the
//! database, marking records as `missing` when the file is deleted and
//! re-marking them `present` when it reappears.

mod scanner;
mod schema;

pub mod metadata;
pub mod query;

pub use metadata::{extract_metadata, spawn_metadata_filler, ImageMetadata};
pub use query::SearchQuery;
pub use scanner::LibraryScanner;

use rusqlite::{params, Connection};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

// ── Status ───────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum ImageStatus {
    Present,
    Missing,
}

impl ImageStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            ImageStatus::Present => "present",
            ImageStatus::Missing => "missing",
        }
    }

    fn from_str(s: &str) -> Self {
        match s {
            "missing" => ImageStatus::Missing,
            _ => ImageStatus::Present,
        }
    }
}

// ── Records ──────────────────────────────────────────────────────

/// Minimal record used by the scanner and import code.
#[derive(Debug, Clone)]
pub struct ImageRecord {
    pub id: i64,
    pub path: PathBuf,
    /// Full 32-byte BLAKE3 content hash.
    pub hash: [u8; 32],
    pub file_size: u64,
    /// Unix timestamp (seconds) when the record was first inserted.
    pub added_at: i64,
    pub status: ImageStatus,
}

/// Full record including EXIF metadata — returned by `Database::search_images`.
#[derive(Debug, Clone)]
pub struct LibraryImage {
    pub id: i64,
    pub path: PathBuf,
    pub added_at: i64,
    pub status: ImageStatus,
    pub meta: ImageMetadata,
}

// ── Database ─────────────────────────────────────────────────────

pub struct Database {
    conn: Connection,
}

impl Database {
    /// Open (or create) the database at `path`.
    ///
    /// Creates parent directories, enables WAL mode, and applies all pending
    /// schema migrations automatically.
    pub fn open(path: &Path) -> anyhow::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        schema::ensure_schema(&conn)?;
        Ok(Self { conn })
    }

    // ── Write operations ─────────────────────────────────────────

    /// Insert an image record.  No-op if `path` already exists in the DB.
    ///
    /// The file basename is stored immediately as `filename` so that FTS-based
    /// filename search works before full EXIF extraction runs.
    pub fn insert_image(
        &self,
        path: &Path,
        hash: &[u8; 32],
        file_size: u64,
    ) -> anyhow::Result<()> {
        let added_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let filename = path
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("")
            .to_owned();

        self.conn.execute(
            "INSERT OR IGNORE INTO images
                 (path, hash, file_size, added_at, status, filename)
             VALUES (?1, ?2, ?3, ?4, 'present', ?5)",
            params![
                path.to_string_lossy().as_ref(),
                hash.as_slice(),
                file_size as i64,
                added_at,
                filename,
            ],
        )?;
        Ok(())
    }

    /// Populate / overwrite EXIF metadata for the record with `id`.
    pub fn update_metadata(&self, id: i64, meta: &ImageMetadata) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET
                 filename     = ?1,
                 taken_at     = ?2,
                 make         = ?3,
                 model        = ?4,
                 lens         = ?5,
                 focal_length = ?6,
                 aperture     = ?7,
                 iso          = ?8,
                 width        = ?9,
                 height       = ?10,
                 orientation  = ?11
             WHERE id = ?12",
            params![
                meta.filename,
                meta.taken_at,
                meta.make,
                meta.model,
                meta.lens,
                meta.focal_length,
                meta.aperture,
                meta.iso,
                meta.width,
                meta.height,
                meta.orientation,
                id,
            ],
        )?;
        Ok(())
    }

    /// Mark a record as missing (file deleted from disk).
    pub fn mark_missing(&self, path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET status = 'missing' WHERE path = ?1",
            params![path.to_string_lossy().as_ref()],
        )?;
        Ok(())
    }

    /// Mark a record as present (file has reappeared on disk).
    pub fn mark_present(&self, path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET status = 'present' WHERE path = ?1",
            params![path.to_string_lossy().as_ref()],
        )?;
        Ok(())
    }

    // ── Read operations ──────────────────────────────────────────

    /// Search the library and return matching images with their metadata.
    ///
    /// With no text filter: returns all present images, newest-first.
    /// With a text filter: uses FTS5 prefix matching across filename, make,
    ///   model, and lens, ordered by recency.
    pub fn search_images(&self, query: &SearchQuery) -> anyhow::Result<Vec<LibraryImage>> {
        let limit = query.limit.unwrap_or(500) as i64;
        let offset = query.offset.unwrap_or(0) as i64;

        let rows: Vec<LibraryImage> = if let Some(ref text) = query.text {
            // Build one LIKE pattern per space-separated token.
            // Each token must appear as a substring in at least one text field;
            // all tokens must match (AND).  Backslash is the LIKE escape char.
            let like_patterns: Vec<String> = text
                .split_whitespace()
                .map(|t| {
                    let escaped = t
                        .to_lowercase()
                        .replace('\\', "\\\\")
                        .replace('%', "\\%")
                        .replace('_', "\\_");
                    format!("%{escaped}%")
                })
                .collect();

            if like_patterns.is_empty() {
                return Ok(vec![]);
            }

            // For each token produce one AND-clause that checks all text fields.
            let field_expr =
                "LOWER(COALESCE(i.filename,'') || ' ' || \
                       COALESCE(i.make,'')     || ' ' || \
                       COALESCE(i.model,'')    || ' ' || \
                       COALESCE(i.lens,''))";

            let token_conditions: String = like_patterns
                .iter()
                .map(|_| format!("{field_expr} LIKE ? ESCAPE '\\'"))
                .collect::<Vec<_>>()
                .join(" AND ");

            let sql = format!(
                "SELECT i.id, i.path, i.added_at, i.status,
                        i.filename, i.taken_at, i.make, i.model, i.lens,
                        i.focal_length, i.aperture, i.iso,
                        i.width, i.height, i.orientation
                 FROM images i
                 WHERE i.status = 'present'
                   AND {token_conditions}
                 ORDER BY i.added_at DESC
                 LIMIT ? OFFSET ?"
            );

            use rusqlite::types::Value;
            let params: Vec<Value> = like_patterns
                .into_iter()
                .map(Value::Text)
                .chain([Value::Integer(limit), Value::Integer(offset)])
                .collect();

            let mut stmt = self.conn.prepare(&sql)?;
            let rows: Vec<LibraryImage> = stmt
                .query_map(rusqlite::params_from_iter(params), row_to_library_image)?
                .filter_map(|r| r.ok())
                .collect();
            rows
        } else {
            let mut stmt = self.conn.prepare(
                "SELECT id, path, added_at, status,
                        filename, taken_at, make, model, lens,
                        focal_length, aperture, iso,
                        width, height, orientation
                 FROM images
                 WHERE status = 'present'
                 ORDER BY added_at DESC
                 LIMIT ?1 OFFSET ?2",
            )?;
            let rows: Vec<LibraryImage> = stmt
                .query_map(params![limit, offset], row_to_library_image)?
                .filter_map(|r| r.ok())
                .collect();
            rows
        };
        Ok(rows)
    }

    /// Return `(id, path)` for all records where EXIF has not been extracted
    /// yet (`filename IS NULL`).  Used by `spawn_metadata_filler`.
    pub fn records_needing_metadata(&self) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path FROM images WHERE filename IS NULL")?;
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

    /// Return all records (for scanner reconciliation).
    pub fn all_images(&self) -> anyhow::Result<Vec<ImageRecord>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, hash, file_size, added_at, status
             FROM images ORDER BY added_at DESC",
        )?;
        let records = stmt
            .query_map([], |row| {
                let id: i64 = row.get(0)?;
                let path: String = row.get(1)?;
                let hash_bytes: Vec<u8> = row.get(2)?;
                let file_size: i64 = row.get(3)?;
                let added_at: i64 = row.get(4)?;
                let status_str: String = row.get(5)?;
                Ok((id, path, hash_bytes, file_size, added_at, status_str))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(id, path, hash_bytes, file_size, added_at, status_str)| {
                let hash: [u8; 32] = hash_bytes.try_into().ok()?;
                Some(ImageRecord {
                    id,
                    path: PathBuf::from(path),
                    hash,
                    file_size: file_size as u64,
                    added_at,
                    status: ImageStatus::from_str(&status_str),
                })
            })
            .collect();
        Ok(records)
    }

    /// Return all `(path, status)` pairs — used by the scanner for
    /// reconciliation.
    pub fn all_paths(&self) -> anyhow::Result<Vec<(PathBuf, ImageStatus)>> {
        let mut stmt = self.conn.prepare("SELECT path, status FROM images")?;
        let rows = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let status: String = row.get(1)?;
                Ok((PathBuf::from(path), ImageStatus::from_str(&status)))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Total number of records in the library.
    pub fn count(&self) -> anyhow::Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
        Ok(n as u64)
    }
}

// ── Row-mapping helper ───────────────────────────────────────────

fn row_to_library_image(row: &rusqlite::Row<'_>) -> rusqlite::Result<LibraryImage> {
    let status_str: String = row.get(3)?;
    let meta = ImageMetadata {
        filename: row.get(4)?,
        taken_at: row.get(5)?,
        make: row.get(6)?,
        model: row.get(7)?,
        lens: row.get(8)?,
        focal_length: row.get(9)?,
        aperture: row.get(10)?,
        iso: row.get(11)?,
        width: row.get(12)?,
        height: row.get(13)?,
        orientation: row.get(14)?,
    };
    Ok(LibraryImage {
        id: row.get(0)?,
        path: PathBuf::from(row.get::<_, String>(1)?),
        added_at: row.get(2)?,
        status: ImageStatus::from_str(&status_str),
        meta,
    })
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_db() -> (tempfile::TempDir, Database) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        (dir, db)
    }

    fn fake_hash(seed: u8) -> [u8; 32] {
        [seed; 32]
    }

    #[test]
    fn insert_and_count() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024).unwrap();
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn insert_or_ignore_duplicate_path() {
        let (_dir, db) = tmp_db();
        let path = PathBuf::from("/photos/a.jpg");
        db.insert_image(&path, &fake_hash(1), 1024).unwrap();
        db.insert_image(&path, &fake_hash(2), 2048).unwrap();
        assert_eq!(db.count().unwrap(), 1);
    }

    #[test]
    fn mark_missing_and_present() {
        let (_dir, db) = tmp_db();
        let path = PathBuf::from("/photos/b.jpg");
        db.insert_image(&path, &fake_hash(3), 512).unwrap();
        db.mark_missing(&path).unwrap();
        assert_eq!(db.all_paths().unwrap()[0].1, ImageStatus::Missing);
        db.mark_present(&path).unwrap();
        assert_eq!(db.all_paths().unwrap()[0].1, ImageStatus::Present);
    }

    #[test]
    fn filename_stored_on_insert() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/c.jpg"), &fake_hash(4), 1024).unwrap();
        let results = db.search_images(&SearchQuery::default()).unwrap();
        assert_eq!(results[0].meta.filename.as_deref(), Some("c.jpg"));
    }

    #[test]
    fn search_by_filename_fts() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/sunset.jpg"), &fake_hash(5), 1024).unwrap();
        db.insert_image(&PathBuf::from("/photos/portrait.jpg"), &fake_hash(6), 1024).unwrap();

        let q = SearchQuery::default().with_text("sunset");
        let results = db.search_images(&q).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].meta.filename.as_deref(), Some("sunset.jpg"));
    }

    #[test]
    fn update_metadata_round_trip() {
        let (_dir, db) = tmp_db();
        let path = PathBuf::from("/photos/d.jpg");
        db.insert_image(&path, &fake_hash(7), 2048).unwrap();

        let id = db.search_images(&SearchQuery::default()).unwrap()[0].id;
        let meta = ImageMetadata {
            filename: Some("d.jpg".into()),
            make: Some("Canon".into()),
            model: Some("EOS R5".into()),
            iso: Some(800),
            ..Default::default()
        };
        db.update_metadata(id, &meta).unwrap();

        let q = SearchQuery::default().with_text("Canon");
        let results = db.search_images(&q).unwrap();
        assert_eq!(results[0].meta.make.as_deref(), Some("Canon"));
        assert_eq!(results[0].meta.iso, Some(800));
    }

    #[test]
    fn open_creates_parent_dirs() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("a/b/c/library.db");
        assert!(Database::open(&nested).is_ok());
    }
}
