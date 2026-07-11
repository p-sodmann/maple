//! maple-db — SQLite-backed image library database.
//!
//! Stores every image that has been imported into the library, keyed by its
//! file-system path and BLAKE3 content hash.  A background scanner
//! (`LibraryScanner`) periodically reconciles the on-disk state with the
//! database, marking records as `missing` when the file is deleted and
//! re-marking them `present` when it reappears.

mod embedding;
mod scanner;
mod schema;
mod semantic_db;
mod thumb_cache;
pub mod worker;

pub mod ai;
pub mod collections;
pub mod face_detector;
pub mod faces;
pub mod hasher;
pub mod metadata;
pub mod models;
pub mod query;
pub mod semantic;
pub mod stacker;

pub use ai::{spawn_ai_tagger, AiDescriber, AiTagger, LmStudioDescriber};
pub use metadata::rotate_image_file;
pub use collections::{Collection, CollectionWithRep};
pub use hasher::{load_onnx_embedder, spawn_hasher};
pub use stacker::{cluster_embeddings, update_stacks};
pub use face_detector::{spawn_face_tagger, DetectedFace, FaceDetector, FaceTagger};
pub use faces::{best_person_match, best_person_matches, cosine_similarity, FaceDetection, Person, PersonWithRep};
pub use metadata::{extract_all_exif_tags, extract_metadata, spawn_metadata_filler, ImageMetadata};
pub use query::SearchQuery;
pub use scanner::{set_scanner_paused, LibraryScanner};
pub use thumb_cache::ThumbnailCache;
pub use semantic::{spawn_sentence_embedder, split_sentences, SemanticEncoder, SentenceEmbedder};

use path_slash::{PathBufExt as _, PathExt as _};
use rusqlite::{params, Connection, OptionalExtension};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, MutexGuard};
use std::time::{SystemTime, UNIX_EPOCH};

// Normalise a path to forward-slash UTF-8 for SQLite storage.
// On Windows this converts the backslash separator to '/'; on Linux/macOS
// it is a no-op. Using a consistent separator makes databases portable
// across platforms.
pub(crate) fn path_to_db(p: &Path) -> String {
    p.to_slash_lossy().into_owned()
}

// Reconstruct a PathBuf from a forward-slash string read out of SQLite.
// On Windows, PathBuf::from_slash converts '/' back to '\' so the returned
// path works with the OS APIs. On Linux/macOS this is a no-op.
pub(crate) fn path_from_db(s: String) -> PathBuf {
    PathBuf::from_slash(s)
}

/// Lock the database mutex, recovering from poison.
///
/// A poisoned mutex means another thread panicked while holding the lock.
/// The SQLite connection itself remains valid, so we log and continue
/// rather than propagating the panic.
pub fn lock_db(db: &Mutex<Database>) -> MutexGuard<'_, Database> {
    match db.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            tracing::warn!("DB mutex was poisoned — recovering");
            poisoned.into_inner()
        }
    }
}

/// Register the sqlite-vec (`vec0`) extension as a SQLite auto-extension.
///
/// Auto-extensions are applied to every connection opened *after* this call,
/// so `Database::open` invokes it before `Connection::open`.  Guarded by a
/// `Once` so repeated `open` calls register it exactly once.
fn register_sqlite_vec() {
    use std::sync::Once;
    static REGISTER: Once = Once::new();
    REGISTER.call_once(|| {
        // SAFETY: `sqlite3_vec_init` has the signature expected by
        // `sqlite3_auto_extension`; the transmute matches the documented
        // sqlite-vec + rusqlite integration pattern.  The target fn-pointer
        // type lives in rusqlite's private ffi, so an inline annotation would
        // be brittle — hence the scoped allow.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            rusqlite::ffi::sqlite3_auto_extension(Some(std::mem::transmute(
                sqlite_vec::sqlite3_vec_init as *const (),
            )));
        }
    });
}

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

/// How an image matched the current search query.
#[derive(Debug, Clone)]
pub enum SearchHit {
    /// The image matched one or more keyword tokens in its metadata or description.
    /// `field` names the first field that matched ("filename", "camera", "lens",
    /// "description", or "person name").  `snippet` is the matching sentence from
    /// the AI description when the match was in a description field.
    Direct { field: String, snippet: Option<String> },
    /// The image was retrieved by vector similarity.
    /// `similarity` is the cosine similarity (0–1); `sentence` is the embedded
    /// sentence from the AI description that was nearest to the query.
    Semantic { similarity: f32, sentence: String },
}

/// Full record including EXIF metadata — returned by `Database::search_images`.
#[derive(Debug, Clone)]
pub struct LibraryImage {
    pub id: i64,
    pub path: PathBuf,
    /// Optional companion raw file path (e.g. the RAF for a JPG display file).
    pub raw_path: Option<PathBuf>,
    pub added_at: i64,
    pub status: ImageStatus,
    pub meta: ImageMetadata,
    /// BLAKE3 content hash — used as the thumbnail cache key.
    /// `None` only if the DB row has a corrupt/missing hash (should not happen).
    pub hash: Option<[u8; 32]>,
    /// Stack this image belongs to, if any.
    pub stack_id: Option<i64>,
    /// Number of images in the same stack (including this one).
    /// `None` for non-stacked images.
    pub stack_size: Option<usize>,
    /// How this image matched the active search query.  `None` for plain
    /// (unfiltered) listings and keyword-only results without semantic search.
    pub search_hit: Option<SearchHit>,
}

// ── Database ─────────────────────────────────────────────────────

/// `(image_id, path, content_hash)` — returned by [`Database::images_without_hash`].
type ImageHashCandidate = (i64, PathBuf, Option<[u8; 32]>);
/// `(image_id, hash_blob, stack_id)` — returned by [`Database::images_with_hash_and_stack`].
type ImageHashStackRow = (i64, Vec<u8>, Option<i64>);
/// `(path, status, content_hash)` — returned by [`Database::all_paths`].
type ImagePathStatusRow = (PathBuf, ImageStatus, Option<[u8; 32]>);

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
        // Register the sqlite-vec (`vec0`) extension before opening any
        // connection so the schema migration can create the vector table.
        register_sqlite_vec();
        let conn = Connection::open(path)?;
        conn.execute_batch("PRAGMA journal_mode=WAL;")?;
        schema::ensure_schema(&conn)?;
        Ok(Self { conn })
    }

    // ── Starred import paths ──────────────────────────────────────

    pub fn add_starred_path(&self, path: &str) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.conn.execute(
            "INSERT OR IGNORE INTO import_starred_paths (path, created_at) VALUES (?1, ?2)",
            params![path, now],
        )?;
        Ok(())
    }

    pub fn remove_starred_path(&self, path: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "DELETE FROM import_starred_paths WHERE path = ?1",
            params![path],
        )?;
        Ok(())
    }

    pub fn get_starred_paths(&self) -> anyhow::Result<Vec<String>> {
        let mut stmt = self.conn.prepare("SELECT path FROM import_starred_paths")?;
        let paths = stmt
            .query_map([], |r| r.get::<_, String>(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(paths)
    }

    // ── Write operations ─────────────────────────────────────────

    /// Insert an image record.  No-op if `path` already exists in the DB.
    ///
    /// The file basename is stored immediately as `filename` so that FTS-based
    /// filename search works before full EXIF extraction runs.
    ///
    /// `raw_path` is the optional companion raw file (e.g. the RAF when the
    /// display file is JPG).  Stored alongside the display path so the DB
    /// holds one row per *image*, not per file.
    pub fn insert_image(
        &self,
        path: &Path,
        hash: &[u8; 32],
        file_size: u64,
    ) -> anyhow::Result<()> {
        self.insert_image_with_raw(path, hash, file_size, None)
    }

    /// Insert an image with an optional raw companion path.
    pub fn insert_image_with_raw(
        &self,
        path: &Path,
        hash: &[u8; 32],
        file_size: u64,
        raw_path: Option<&Path>,
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

        let raw_str = raw_path.map(path_to_db);

        self.conn.execute(
            "INSERT OR IGNORE INTO images
                 (path, hash, file_size, added_at, status, filename, raw_path)
             VALUES (?1, ?2, ?3, ?4, 'present', ?5, ?6)",
            params![
                path_to_db(path),
                hash.as_slice(),
                file_size as i64,
                added_at,
                filename,
                raw_str,
            ],
        )?;
        Ok(())
    }

    // ── Image hash operations ─────────────────────────────────────

    /// Store a perceptual hash or embedding for `image_id` under `algorithm`.
    /// Overwrites any existing row for the same `(image_id, algorithm)` pair.
    pub fn insert_image_hash(
        &self,
        image_id: i64,
        algorithm: &str,
        hash_blob: &[u8],
    ) -> anyhow::Result<()> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.conn.execute(
            "INSERT OR REPLACE INTO image_hashes (image_id, algorithm, hash_blob, created_at)
             VALUES (?1, ?2, ?3, ?4)",
            params![image_id, algorithm, hash_blob, now],
        )?;
        Ok(())
    }

    /// Count images that have no hash for `algorithm`.
    pub fn count_images_without_hash(&self, algorithm: &str) -> anyhow::Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM images
             WHERE status = 'present'
               AND id NOT IN (
                   SELECT image_id FROM image_hashes WHERE algorithm = ?1
               )",
            params![algorithm],
            |r| r.get(0),
        )?;
        Ok(n as usize)
    }

    /// Return up to `limit` images that have no hash for `algorithm`.
    /// Used by the background hasher to find pending work.
    pub fn images_without_hash(
        &self,
        algorithm: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ImageHashCandidate>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, path, hash FROM images
             WHERE status = 'present'
               AND id NOT IN (
                   SELECT image_id FROM image_hashes WHERE algorithm = ?1
               )
             LIMIT ?2",
        )?;
        let rows = stmt
            .query_map(params![algorithm, limit as i64], |r| {
                let id: i64 = r.get(0)?;
                let path: String = r.get(1)?;
                let hash_bytes: Option<Vec<u8>> = r.get(2)?;
                Ok((id, path, hash_bytes))
            })?
            .filter_map(|r| {
                r.ok().map(|(id, path, hash_bytes)| {
                    let hash: Option<[u8; 32]> = hash_bytes.and_then(|b| {
                        b.try_into().ok()
                    });
                    (id, path_from_db(path), hash)
                })
            })
            .collect();
        Ok(rows)
    }

    /// Return all `(image_id, hash_blob)` rows for `algorithm`.
    /// Used by the stacker to compare all hashed images.
    pub fn images_with_hash(
        &self,
        algorithm: &str,
    ) -> anyhow::Result<Vec<(i64, Vec<u8>)>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT image_id, hash_blob FROM image_hashes WHERE algorithm = ?1",
        )?;
        let rows = stmt
            .query_map(params![algorithm], |r| {
                Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Return all `(image_id, hash_blob, stack_id)` rows for `algorithm`.
    /// Used by the stacker to seed existing cluster state.
    pub fn images_with_hash_and_stack(
        &self,
        algorithm: &str,
    ) -> anyhow::Result<Vec<ImageHashStackRow>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT ih.image_id, ih.hash_blob, i.stack_id
             FROM image_hashes ih
             JOIN images i ON i.id = ih.image_id
             WHERE ih.algorithm = ?1",
        )?;
        let rows = stmt
            .query_map(params![algorithm], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, Vec<u8>>(1)?,
                    r.get::<_, Option<i64>>(2)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    // ── Stack operations ─────────────────────────────────────────

    /// Create a new stack and return its id.
    pub fn create_stack(&self) -> anyhow::Result<i64> {
        let now = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.conn.execute(
            "INSERT INTO stacks (created_at) VALUES (?1)",
            params![now],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Assign `image_id` to `stack_id`.  Pass `None` to remove from any stack.
    pub fn set_image_stack(&self, image_id: i64, stack_id: Option<i64>) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET stack_id = ?1 WHERE id = ?2",
            params![stack_id, image_id],
        )?;
        Ok(())
    }

    /// Return all images belonging to `stack_id`, ordered by id (oldest first).
    pub fn images_in_stack(&self, stack_id: i64) -> anyhow::Result<Vec<LibraryImage>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT i.id, i.path, i.added_at, i.status,
                    i.filename, i.taken_at, i.make, i.model, i.lens,
                    i.focal_length, i.aperture, i.iso,
                    i.width, i.height, i.orientation, i.raw_path, i.hash,
                    i.stack_id,
                    (SELECT COUNT(*) FROM images sc
                     WHERE sc.stack_id = i.stack_id AND sc.status = 'present') AS stack_size
             FROM images i
             WHERE i.stack_id = ?1 AND i.status = 'present'
             ORDER BY i.id ASC",
        )?;
        let rows = stmt
            .query_map(params![stack_id], row_to_library_image)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Set the cover (favourite) image for a stack.
    /// The cover is used as the grid thumbnail for the stack.
    pub fn set_stack_cover(&self, stack_id: i64, image_id: i64) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE stacks SET cover_image_id = ?1 WHERE id = ?2",
            params![image_id, stack_id],
        )?;
        Ok(())
    }

    /// Remove `image_id` from its stack.
    ///
    /// If after removal the stack has fewer than 2 images, the remaining
    /// images are also unstacked and the stack row is deleted.
    pub fn remove_from_stack(&self, image_id: i64) -> anyhow::Result<()> {
        // Find the stack this image belongs to.
        let stack_id: Option<i64> = self
            .conn
            .query_row(
                "SELECT stack_id FROM images WHERE id = ?1",
                params![image_id],
                |r| r.get(0),
            )
            .optional()?
            .flatten();

        let Some(stack_id) = stack_id else {
            return Ok(()); // not stacked
        };

        // Remove this image from the stack.
        self.conn.execute(
            "UPDATE images SET stack_id = NULL WHERE id = ?1",
            params![image_id],
        )?;

        // If fewer than 2 images remain, disband the whole stack.
        let remaining: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM images WHERE stack_id = ?1 AND status = 'present'",
            params![stack_id],
            |r| r.get(0),
        )?;

        if remaining < 2 {
            self.conn.execute(
                "UPDATE images SET stack_id = NULL WHERE stack_id = ?1",
                params![stack_id],
            )?;
            self.conn.execute(
                "DELETE FROM stacks WHERE id = ?1",
                params![stack_id],
            )?;
        }

        Ok(())
    }

    /// Return the DB row id for an image by file path, or `None` if not found.
    pub fn image_id_for_path(&self, path: &Path) -> anyhow::Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare_cached("SELECT id FROM images WHERE path = ?1")?;
        let id = stmt
            .query_row(params![path_to_db(path)], |r| r.get(0))
            .optional()?;
        Ok(id)
    }

    /// Set the raw companion path on an existing image record.
    pub fn set_raw_path(&self, id: i64, raw_path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET raw_path = ?1 WHERE id = ?2",
            params![path_to_db(raw_path), id],
        )?;
        Ok(())
    }

    /// Update the on-disk location and display filename for an image record
    /// after a library restructure move (see `maple_import::restructure`).
    /// The `images_fts_au` trigger keeps `image_fts` in sync automatically.
    pub fn update_image_location(
        &self,
        id: i64,
        new_path: &Path,
        new_raw_path: Option<&Path>,
        new_filename: &str,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET path = ?1, raw_path = ?2, filename = ?3 WHERE id = ?4",
            params![
                path_to_db(new_path),
                new_raw_path.map(path_to_db),
                new_filename,
                id,
            ],
        )?;
        Ok(())
    }

    /// Populate / overwrite EXIF metadata for the record with `id`, and mark
    /// it as having had EXIF extraction run (see `records_needing_metadata`).
    pub fn update_metadata(&self, id: i64, meta: &ImageMetadata) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET
                 filename       = ?1,
                 taken_at       = ?2,
                 make           = ?3,
                 model          = ?4,
                 lens           = ?5,
                 focal_length   = ?6,
                 aperture       = ?7,
                 iso            = ?8,
                 width          = ?9,
                 height         = ?10,
                 orientation    = ?11,
                 exif_extracted = 1
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

    /// Replace the comprehensive EXIF tag set for `image_id` (delete +
    /// reinsert). Safe to call repeatedly, e.g. after re-extracting metadata.
    pub fn replace_exif_tags(&self, image_id: i64, tags: &[(String, String)]) -> anyhow::Result<()> {
        let tx = self.conn.unchecked_transaction()?;
        tx.execute("DELETE FROM image_exif_tags WHERE image_id = ?1", params![image_id])?;
        {
            let mut stmt = tx.prepare(
                "INSERT INTO image_exif_tags (image_id, tag, value) VALUES (?1, ?2, ?3)",
            )?;
            for (tag, value) in tags {
                stmt.execute(params![image_id, tag, value])?;
            }
        }
        tx.commit()?;
        Ok(())
    }

    /// All comprehensive EXIF tags for one image, alphabetical by tag name.
    pub fn exif_tags_for_image(&self, image_id: i64) -> anyhow::Result<Vec<(String, String)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT tag, value FROM image_exif_tags WHERE image_id = ?1 ORDER BY tag")?;
        let rows = stmt
            .query_map(params![image_id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Update the content hash and EXIF orientation for a record after a
    /// lossless in-place rotation.
    pub fn update_image_hash_and_orientation(
        &self,
        id: i64,
        hash: &[u8; 32],
        orientation: i64,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET hash = ?1, orientation = ?2 WHERE id = ?3",
            params![hash.as_slice(), orientation, id],
        )?;
        Ok(())
    }

    /// Mark a record as missing (file deleted from disk).
    pub fn mark_missing(&self, path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET status = 'missing' WHERE path = ?1",
            params![path_to_db(path)],
        )?;
        Ok(())
    }

    /// Mark a record as present (file has reappeared on disk).
    pub fn mark_present(&self, path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET status = 'present' WHERE path = ?1",
            params![path_to_db(path)],
        )?;
        Ok(())
    }

    // ── Read operations ──────────────────────────────────────────

    /// Search the library and return matching images with their metadata.
    ///
    /// With no text filter: returns all present images, newest-first.
    /// With a text filter: each whitespace-delimited token must match at
    ///   least one of: filename, make, model, lens, any AI description, or
    ///   any assigned person name.
    pub fn search_images(&self, query: &SearchQuery) -> anyhow::Result<Vec<LibraryImage>> {
        match (&query.text, &query.semantic_embedding) {
            // Hybrid: keyword + semantic vector results, merged.
            (Some(text), Some(embedding)) => {
                let k = if query.semantic_k == 0 { 200 } else { query.semantic_k };
                self.search_images_hybrid(
                    text,
                    embedding,
                    k,
                    query.limit,
                    query.offset,
                    query.collection_id,
                )
            }
            (Some(text), None) => {
                self.search_images_text(text, query.limit, query.offset, query.collection_id, query.person_id)
            }
            (None, _) => self.search_images_all(query.limit, query.offset, query.collection_id, query.person_id),
        }
    }

    /// Return all present images, newest-first.
    fn search_images_all(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        collection_id: Option<i64>,
        person_id: Option<i64>,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        use rusqlite::types::Value;

        let limit = limit.unwrap_or(500) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let coll_clause = if collection_id.is_some() {
            " AND i.id IN (SELECT image_id FROM collection_images WHERE collection_id = ?)"
        } else {
            ""
        };
        let person_clause = if person_id.is_some() {
            " AND i.id IN (SELECT image_id FROM face_detections WHERE person_id = ?)"
        } else {
            ""
        };

        let sql = format!(
            "WITH stack_covers AS (
                 SELECT s.id                                    AS stack_id,
                        COALESCE(s.cover_image_id, MIN(si.id)) AS cover_id,
                        COUNT(*)                                AS stack_size
                 FROM stacks s
                 JOIN images si ON si.stack_id = s.id AND si.status = 'present'
                 GROUP BY s.id
             )
             SELECT i.id, i.path, i.added_at, i.status,
                    i.filename, i.taken_at, i.make, i.model, i.lens,
                    i.focal_length, i.aperture, i.iso,
                    i.width, i.height, i.orientation, i.raw_path, i.hash,
                    i.stack_id,
                    sc.stack_size
             FROM images i
             LEFT JOIN stack_covers sc ON sc.stack_id = i.stack_id
             WHERE i.status = 'present'
               AND (
                 i.stack_id IS NULL
                 OR i.id = sc.cover_id
               ){coll_clause}{person_clause}
             ORDER BY i.added_at DESC
             LIMIT ? OFFSET ?"
        );

        let mut params: Vec<Value> = Vec::new();
        if let Some(cid) = collection_id {
            params.push(Value::Integer(cid));
        }
        if let Some(pid) = person_id {
            params.push(Value::Integer(pid));
        }
        params.push(Value::Integer(limit));
        params.push(Value::Integer(offset));

        let mut stmt = self.conn.prepare(&sql)?;
        let rows = stmt
            .query_map(rusqlite::params_from_iter(params), row_to_library_image)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Search present images by text tokens (AND logic).
    ///
    /// Each whitespace-delimited token must appear in at least one of:
    /// EXIF fields, AI descriptions, or assigned person names.
    fn search_images_text(
        &self,
        text: &str,
        limit: Option<usize>,
        offset: Option<usize>,
        collection_id: Option<i64>,
        person_id: Option<i64>,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        let limit = limit.unwrap_or(500) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let like_patterns: Vec<String> = text
            .split_whitespace()
            .map(|t| format!("%{}%", escape_like_token(t)))
            .collect();

        if like_patterns.is_empty() {
            return Ok(vec![]);
        }

        // Each token must match somewhere in the combined EXIF fields OR
        // in any AI description OR in any assigned person name OR in any
        // comprehensive EXIF tag value (shutter speed, GPS, flash, …).
        let exif_expr =
            "LOWER(COALESCE(i.filename,'') || ' ' || \
                   COALESCE(i.make,'')     || ' ' || \
                   COALESCE(i.model,'')    || ' ' || \
                   COALESCE(i.lens,''))";
        let ai_expr = "LOWER(COALESCE(ad.description,''))";
        let person_expr = "LOWER(COALESCE(p.name,''))";
        let exif_tags_expr = "EXISTS (SELECT 1 FROM image_exif_tags t \
                               WHERE t.image_id = i.id AND LOWER(t.value) LIKE ? ESCAPE '\\')";

        let token_conditions: String = like_patterns
            .iter()
            .map(|_| {
                format!(
                    "({exif_expr} LIKE ? ESCAPE '\\' \
                      OR {ai_expr} LIKE ? ESCAPE '\\' \
                      OR {person_expr} LIKE ? ESCAPE '\\' \
                      OR {exif_tags_expr})"
                )
            })
            .collect::<Vec<_>>()
            .join(" AND ");

        let coll_clause = if collection_id.is_some() {
            " AND i.id IN (SELECT image_id FROM collection_images WHERE collection_id = ?)"
        } else {
            ""
        };
        let person_clause = if person_id.is_some() {
            " AND i.id IN (SELECT image_id FROM face_detections WHERE person_id = ?)"
        } else {
            ""
        };

        let sql = format!(
            "WITH stack_covers AS (
                 SELECT s.id                                    AS stack_id,
                        COALESCE(s.cover_image_id, MIN(si.id)) AS cover_id,
                        COUNT(*)                                AS stack_size
                 FROM stacks s
                 JOIN images si ON si.stack_id = s.id AND si.status = 'present'
                 GROUP BY s.id
             )
             SELECT DISTINCT i.id, i.path, i.added_at, i.status,
                    i.filename, i.taken_at, i.make, i.model, i.lens,
                    i.focal_length, i.aperture, i.iso,
                    i.width, i.height, i.orientation, i.raw_path, i.hash,
                    i.stack_id,
                    sc.stack_size
             FROM images i
             LEFT JOIN stack_covers sc ON sc.stack_id = i.stack_id
             LEFT JOIN ai_descriptions ad ON ad.image_id = i.id
             LEFT JOIN face_detections fd ON fd.image_id = i.id
             LEFT JOIN persons p ON p.id = fd.person_id
             WHERE i.status = 'present'
               AND {token_conditions}
               AND (
                 i.stack_id IS NULL
                 OR i.id = sc.cover_id
               ){coll_clause}{person_clause}
             ORDER BY i.added_at DESC
             LIMIT ? OFFSET ?"
        );

        // Each token pattern appears four times: EXIF, AI desc, person name, EXIF tags.
        use rusqlite::types::Value;
        let mut params: Vec<Value> = like_patterns
            .into_iter()
            .flat_map(|p| {
                [
                    Value::Text(p.clone()),
                    Value::Text(p.clone()),
                    Value::Text(p.clone()),
                    Value::Text(p),
                ]
            })
            .collect();
        if let Some(cid) = collection_id {
            params.push(Value::Integer(cid));
        }
        if let Some(pid) = person_id {
            params.push(Value::Integer(pid));
        }
        let params: Vec<Value> = params
            .into_iter()
            .chain([Value::Integer(limit), Value::Integer(offset)])
            .collect();

        let tokens: Vec<String> = text
            .split_whitespace()
            .map(|t| t.to_lowercase())
            .collect();

        let mut stmt = self.conn.prepare(&sql)?;
        let mut rows: Vec<LibraryImage> = stmt
            .query_map(rusqlite::params_from_iter(params), row_to_library_image)?
            .filter_map(|r| r.ok())
            .collect();
        for img in &mut rows {
            let (field, snippet) = if let Some(f) = detect_exif_field(&self.conn, img, &tokens) {
                (f, None)
            } else {
                find_description_match(&self.conn, img.id, &tokens)
            };
            img.search_hit = Some(SearchHit::Direct { field, snippet });
        }
        Ok(rows)
    }

    // ── AI description operations ─────────────────────────────────

    /// Return `(id, path)` for all present images that have no description
    /// from `model_id` yet.  Used by `spawn_ai_tagger`.
    pub fn images_needing_ai_description(
        &self,
        model_id: &str,
    ) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.id, i.path
             FROM images i
             WHERE i.status = 'present'
               AND NOT EXISTS (
                   SELECT 1 FROM ai_descriptions ad
                   WHERE ad.image_id = i.id AND ad.model_id = ?1
               )",
        )?;
        let rows = stmt
            .query_map(params![model_id], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    path_from_db(row.get::<_, String>(1)?),
                ))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Store (or replace) an AI-generated description for one image/model pair.
    pub fn insert_ai_description(
        &self,
        image_id: i64,
        model_id: &str,
        description: &str,
    ) -> anyhow::Result<()> {
        let created_at = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;
        self.conn.execute(
            "INSERT INTO ai_descriptions(image_id, model_id, description, created_at)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(image_id, model_id) DO UPDATE SET
                 description = excluded.description,
                 created_at  = excluded.created_at",
            params![image_id, model_id, description, created_at],
        )?;
        Ok(())
    }

    /// Retrieve the AI description for a specific image/model pair, if any.
    pub fn ai_description_for_image(
        &self,
        image_id: i64,
        model_id: &str,
    ) -> anyhow::Result<Option<String>> {
        let mut stmt = self.conn.prepare(
            "SELECT description FROM ai_descriptions
             WHERE image_id = ?1 AND model_id = ?2",
        )?;
        let mut rows = stmt.query(params![image_id, model_id])?;
        Ok(rows.next()?.map(|r| r.get::<_, String>(0)).transpose()?)
    }

    /// Return all `(model_id, description)` pairs for `image_id`, ordered by
    /// `created_at` ascending.  Used by the detail window info popup.
    pub fn ai_descriptions_for_image(
        &self,
        image_id: i64,
    ) -> anyhow::Result<Vec<(String, String)>> {
        let mut stmt = self.conn.prepare(
            "SELECT model_id, description FROM ai_descriptions
             WHERE image_id = ?1
             ORDER BY created_at ASC",
        )?;
        let rows = stmt
            .query_map(params![image_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// Delete every row in `ai_descriptions`.  The FTS sync trigger and the
    /// sentence-embedding invalidation triggers cascade automatically.
    /// Returns the number of rows deleted.
    pub fn clear_all_ai_descriptions(&self) -> anyhow::Result<usize> {
        let n = self.conn.execute("DELETE FROM ai_descriptions", [])?;
        Ok(n)
    }

    /// Return `(id, path)` for all records where EXIF has not been extracted
    /// yet (`exif_extracted = 0`).  Used by `spawn_metadata_filler`.
    pub fn records_needing_metadata(&self) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id, path FROM images WHERE exif_extracted = 0 AND status = 'present'")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, i64>(0)?,
                    path_from_db(row.get::<_, String>(1)?),
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
                    path: path_from_db(path),
                    hash,
                    file_size: file_size as u64,
                    added_at,
                    status: ImageStatus::from_str(&status_str),
                })
            })
            .collect();
        Ok(records)
    }

    /// Return all `(path, status, hash)` triples — used by the scanner for
    /// reconciliation and thumbnail cache eviction.
    pub fn all_paths(&self) -> anyhow::Result<Vec<ImagePathStatusRow>> {
        let mut stmt = self.conn.prepare("SELECT path, status, hash FROM images")?;
        let rows = stmt
            .query_map([], |row| {
                let path: String = row.get(0)?;
                let status: String = row.get(1)?;
                let hash_bytes: Vec<u8> = row.get(2)?;
                Ok((path_from_db(path), ImageStatus::from_str(&status), hash_bytes))
            })?
            .filter_map(|r| r.ok())
            .map(|(p, s, h)| (p, s, h.try_into().ok()))
            .collect();
        Ok(rows)
    }

    /// All present images, packaged as restructure planning candidates (see
    /// `maple_import::restructure::plan_moves`). Metadata comes straight
    /// from already-extracted EXIF columns — no per-file disk re-read.
    pub fn restructure_candidates(&self) -> anyhow::Result<Vec<maple_import::RestructureCandidate>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, raw_path, taken_at, make, model FROM images WHERE status = 'present'",
        )?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, Option<String>>(2)?,
                    r.get::<_, Option<i64>>(3)?,
                    r.get::<_, Option<String>>(4)?,
                    r.get::<_, Option<String>>(5)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .map(|(id, path, raw_path, taken_at, make, model)| {
                let camera = match (make, model) {
                    (Some(make), Some(model)) => Some(format!("{make} {model}")),
                    (Some(v), None) | (None, Some(v)) => Some(v),
                    (None, None) => None,
                };
                maple_import::RestructureCandidate {
                    id,
                    current_path: path_from_db(path),
                    current_raw_path: raw_path.map(path_from_db),
                    datetime: taken_at.map(maple_import::ExifDateTime::from_unix_timestamp),
                    camera,
                }
            })
            .collect();
        Ok(rows)
    }

    /// Fetch a single image record by id.
    pub fn image_by_id(&self, id: i64) -> anyhow::Result<Option<LibraryImage>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.id, i.path, i.added_at, i.status,
                    i.filename, i.taken_at, i.make, i.model, i.lens,
                    i.focal_length, i.aperture, i.iso,
                    i.width, i.height, i.orientation, i.raw_path, i.hash,
                    i.stack_id,
                    CASE WHEN i.stack_id IS NOT NULL THEN
                        (SELECT COUNT(*) FROM images sc
                         WHERE sc.stack_id = i.stack_id AND sc.status = 'present')
                    ELSE NULL END AS stack_size
             FROM images i
             WHERE i.id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], row_to_library_image)?;
        Ok(rows.next().transpose()?)
    }

    /// Total number of records in the library.
    pub fn count(&self) -> anyhow::Result<u64> {
        let n: i64 = self
            .conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
        Ok(n as u64)
    }

    // ── Debug / similarity query ─────────────────────────────────

    /// Return the raw hash blob stored for `image_id` under `algorithm`, if any.
    pub fn hash_blob_for_image(
        &self,
        image_id: i64,
        algorithm: &str,
    ) -> anyhow::Result<Option<Vec<u8>>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT hash_blob FROM image_hashes WHERE image_id = ?1 AND algorithm = ?2",
        )?;
        let result = stmt
            .query_row(params![image_id, algorithm], |r| r.get(0))
            .optional()?;
        Ok(result)
    }

    /// Compute cosine similarity between two images' stored embeddings under
    /// `algorithm`.
    ///
    /// Returns `None` if either image has no stored embedding for that
    /// algorithm.
    pub fn similarity_for_images(
        &self,
        id_a: i64,
        id_b: i64,
        algorithm: &str,
    ) -> anyhow::Result<Option<f32>> {
        let blob_a = self.hash_blob_for_image(id_a, algorithm)?;
        let blob_b = self.hash_blob_for_image(id_b, algorithm)?;
        let (Some(a), Some(b)) = (blob_a, blob_b) else {
            return Ok(None);
        };
        let ea: Vec<f32> = a
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let eb: Vec<f32> = b
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        Ok(Some(crate::models::image_cosine_similarity(&ea, &eb)))
    }

    /// Return all distinct algorithm keys currently stored in `image_hashes`.
    pub fn stored_algorithms(&self) -> anyhow::Result<Vec<String>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT DISTINCT algorithm FROM image_hashes ORDER BY algorithm",
        )?;
        let rows = stmt
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Return the EXIF field name that first satisfies any of `tokens`, or `None`
/// when the match must have come from an AI description or a person name.
///
/// Checks the curated fields (filename, camera, lens) plus every
/// comprehensive EXIF tag stored for the image; for the latter, the
/// returned "field" is the tag's own name (e.g. `"ExposureTime"`).
pub(crate) fn detect_exif_field(
    conn: &Connection,
    img: &LibraryImage,
    tokens: &[String],
) -> Option<String> {
    let meta = &img.meta;
    let extra_tags: Vec<(String, String)> = conn
        .prepare("SELECT tag, value FROM image_exif_tags WHERE image_id = ?1")
        .and_then(|mut stmt| {
            stmt.query_map(params![img.id], |r| {
                Ok((r.get::<_, String>(0)?, r.get::<_, String>(1)?))
            })
            .map(|rows| rows.filter_map(|r| r.ok()).collect())
        })
        .unwrap_or_default();

    for token in tokens {
        if meta
            .filename
            .as_deref()
            .map(|s| s.to_lowercase().contains(token.as_str()))
            .unwrap_or(false)
        {
            return Some("filename".to_owned());
        }
        let camera = format!(
            "{} {}",
            meta.make.as_deref().unwrap_or(""),
            meta.model.as_deref().unwrap_or("")
        )
        .to_lowercase();
        if camera.trim().contains(token.as_str()) {
            return Some("camera".to_owned());
        }
        if meta
            .lens
            .as_deref()
            .map(|s| s.to_lowercase().contains(token.as_str()))
            .unwrap_or(false)
        {
            return Some("lens".to_owned());
        }
        if let Some((tag, _)) = extra_tags.iter().find(|(_, v)| v.to_lowercase().contains(token.as_str())) {
            return Some(tag.clone());
        }
    }
    None
}

/// Search AI descriptions for `image_id` to find the best matching sentence.
///
/// Prefers a sentence that contains ALL tokens (so "orange cat" shows a
/// sentence with both words).  Falls back to the first sentence that contains
/// any token when no all-match sentence exists.
/// Returns `("description", Some(sentence))` when found or
/// `("person name", None)` when no description text matches (meaning the
/// keyword hit came from a joined person name instead).
fn find_description_match(
    conn: &Connection,
    image_id: i64,
    tokens: &[String],
) -> (String, Option<String>) {
    let descriptions: Vec<String> = conn
        .prepare("SELECT description FROM ai_descriptions WHERE image_id = ?1")
        .ok()
        .and_then(|mut stmt| {
            stmt.query_map(params![image_id], |row| row.get::<_, String>(0))
                .ok()
                .map(|rows| rows.filter_map(|r| r.ok()).collect())
        })
        .unwrap_or_default();

    let mut any_match: Option<String> = None;

    for desc in &descriptions {
        for sentence in crate::semantic::split_sentences(desc) {
            let lower = sentence.to_lowercase();
            // Ideal: a sentence that satisfies every token.
            if tokens.iter().all(|t| lower.contains(t.as_str())) {
                return ("description".to_owned(), Some(sentence));
            }
            // Keep the first partial match as a fallback.
            if any_match.is_none() && tokens.iter().any(|t| lower.contains(t.as_str())) {
                any_match = Some(sentence);
            }
        }
    }

    if let Some(sentence) = any_match {
        return ("description".to_owned(), Some(sentence));
    }
    ("person name".to_owned(), None)
}

/// Escape SQL LIKE special characters in a search token.
fn escape_like_token(token: &str) -> String {
    token
        .to_lowercase()
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
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
    let raw_path: Option<String> = row.get(15)?;
    let hash_bytes: Vec<u8> = row.get(16)?;
    let hash: Option<[u8; 32]> = hash_bytes.try_into().ok();
    let stack_id: Option<i64> = row.get(17)?;
    let stack_size: Option<i64> = row.get(18)?;
    Ok(LibraryImage {
        id: row.get(0)?,
        path: path_from_db(row.get::<_, String>(1)?),
        raw_path: raw_path.map(path_from_db),
        added_at: row.get(2)?,
        status: ImageStatus::from_str(&status_str),
        meta,
        hash,
        stack_id,
        stack_size: stack_size.map(|n| n as usize),
        search_hit: None,
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
    fn exif_tags_round_trip_and_replace() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/e.jpg"), &fake_hash(20), 1024).unwrap();
        let id = db.search_images(&SearchQuery::default()).unwrap()[0].id;

        db.replace_exif_tags(
            id,
            &[
                ("ExposureTime".to_owned(), "1/250 sec.".to_owned()),
                ("Flash".to_owned(), "Flash did not fire".to_owned()),
            ],
        )
        .unwrap();
        let tags = db.exif_tags_for_image(id).unwrap();
        assert_eq!(
            tags,
            vec![
                ("ExposureTime".to_owned(), "1/250 sec.".to_owned()),
                ("Flash".to_owned(), "Flash did not fire".to_owned()),
            ]
        );

        // Replacing with a smaller set drops the old rows, not just appends.
        db.replace_exif_tags(id, &[("WhiteBalance".to_owned(), "Auto".to_owned())]).unwrap();
        let tags = db.exif_tags_for_image(id).unwrap();
        assert_eq!(tags, vec![("WhiteBalance".to_owned(), "Auto".to_owned())]);
    }

    #[test]
    fn search_matches_comprehensive_exif_tag() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/f.jpg"), &fake_hash(21), 1024).unwrap();
        db.insert_image(&PathBuf::from("/photos/g.jpg"), &fake_hash(22), 1024).unwrap();
        let ids: Vec<i64> = db
            .search_images(&SearchQuery::default())
            .unwrap()
            .iter()
            .map(|i| i.id)
            .collect();

        db.replace_exif_tags(ids[0], &[("WhiteBalance".to_owned(), "Manual".to_owned())]).unwrap();

        let q = SearchQuery::default().with_text("Manual");
        let results = db.search_images(&q).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].id, ids[0]);
        match &results[0].search_hit {
            Some(SearchHit::Direct { field, .. }) => assert_eq!(field, "WhiteBalance"),
            other => panic!("expected Direct hit, got {other:?}"),
        }
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

    #[test]
    fn records_needing_metadata_excludes_missing() {
        let (_dir, db) = tmp_db();
        let present = PathBuf::from("/photos/present.jpg");
        let missing = PathBuf::from("/photos/missing.jpg");

        // Freshly inserted images start with exif_extracted = 0 — no need to
        // simulate a "pre-EXIF" state separately.
        db.insert_image(&present, &fake_hash(10), 1024).unwrap();
        db.insert_image(&missing, &fake_hash(11), 1024).unwrap();
        db.mark_missing(&missing).unwrap();

        let needing = db.records_needing_metadata().unwrap();
        assert_eq!(needing.len(), 1, "missing records should not be returned");
        assert_eq!(needing[0].1, present);

        // update_metadata marks the record as extracted, so it drops out.
        db.update_metadata(needing[0].0, &ImageMetadata::default()).unwrap();
        assert!(db.records_needing_metadata().unwrap().is_empty());
    }

    // ── Stack cover / stack_size tests (exercises B5 fix) ─────────

    fn insert_and_get_id(db: &Database, path: &str, seed: u8) -> i64 {
        let p = PathBuf::from(path);
        db.insert_image(&p, &fake_hash(seed), 1024).unwrap();
        db.all_images()
            .unwrap()
            .into_iter()
            .find(|r| r.path == p)
            .unwrap()
            .id
    }

    #[test]
    fn search_all_shows_only_stack_cover() {
        let (_dir, db) = tmp_db();
        let id_a = insert_and_get_id(&db, "/photos/a.jpg", 1);
        let id_b = insert_and_get_id(&db, "/photos/b.jpg", 2);
        insert_and_get_id(&db, "/photos/c.jpg", 3); // unstacked

        let stack_id = db.create_stack().unwrap();
        db.set_image_stack(id_a, Some(stack_id)).unwrap();
        db.set_image_stack(id_b, Some(stack_id)).unwrap();

        let results = db.search_images(&SearchQuery::default()).unwrap();

        // Only 2 results: the stack cover (min id) + the unstacked image.
        assert_eq!(results.len(), 2);
        let cover = results.iter().find(|r| r.stack_id.is_some()).unwrap();
        assert_eq!(cover.id, id_a.min(id_b), "min-id image is the default cover");
        assert_eq!(cover.stack_size, Some(2));
    }

    #[test]
    fn search_all_respects_explicit_stack_cover() {
        let (_dir, db) = tmp_db();
        let id_a = insert_and_get_id(&db, "/photos/a.jpg", 1);
        let id_b = insert_and_get_id(&db, "/photos/b.jpg", 2);
        let id_c = insert_and_get_id(&db, "/photos/c.jpg", 3);

        let stack_id = db.create_stack().unwrap();
        db.set_image_stack(id_a, Some(stack_id)).unwrap();
        db.set_image_stack(id_b, Some(stack_id)).unwrap();
        db.set_image_stack(id_c, Some(stack_id)).unwrap();
        // Explicitly promote the last-inserted image as cover.
        db.set_stack_cover(stack_id, id_c).unwrap();

        let results = db.search_images(&SearchQuery::default()).unwrap();

        assert_eq!(results.len(), 1);
        let cover = results.first().unwrap();
        assert_eq!(cover.id, id_c, "explicit cover should be returned");
        assert_eq!(cover.stack_size, Some(3));
    }

    #[test]
    fn search_text_shows_only_stack_cover() {
        let (_dir, db) = tmp_db();
        let id_a = insert_and_get_id(&db, "/photos/alpha.jpg", 1);
        let id_b = insert_and_get_id(&db, "/photos/alpha2.jpg", 2);

        let stack_id = db.create_stack().unwrap();
        db.set_image_stack(id_a, Some(stack_id)).unwrap();
        db.set_image_stack(id_b, Some(stack_id)).unwrap();

        let q = SearchQuery::default().with_text("alpha");
        let results = db.search_images(&q).unwrap();

        // Both images match the query but only the cover should be returned.
        assert_eq!(results.len(), 1);
        let cover = results.first().unwrap();
        assert_eq!(cover.id, id_a.min(id_b));
        assert_eq!(cover.stack_size, Some(2));
    }

    // ── P3 path-normalisation tests ───────────────────────────────

    #[test]
    fn path_to_db_never_contains_backslashes() {
        // On every platform, forward-slash paths must survive unchanged.
        for s in &["/photos/vacation/img.jpg", "photos/img.jpg", "img.jpg"] {
            let p = PathBuf::from(s);
            let stored = path_to_db(&p);
            assert!(
                !stored.contains('\\'),
                "path_to_db produced a backslash for {:?}: {:?}",
                p,
                stored,
            );
        }
    }

    #[test]
    fn path_round_trip_through_db() {
        // Insert a path, read it back via image_id_for_path (which also
        // normalises before the lookup) and via all_images.
        let (_dir, db) = tmp_db();
        let path = PathBuf::from("/library/2024/vacation/DSC_0001.jpg");
        db.insert_image(&path, &fake_hash(50), 4096).unwrap();

        // Lookup by path must succeed.
        let id = db.image_id_for_path(&path).unwrap();
        assert!(id.is_some(), "path lookup failed after insert");

        // Path retrieved via all_images must equal the original.
        let records = db.all_images().unwrap();
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].path, path);
    }

    #[test]
    fn raw_path_round_trips_through_db() {
        let (_dir, db) = tmp_db();
        let jpg = PathBuf::from("/library/2024/DSC_0001.jpg");
        let raf = PathBuf::from("/library/2024/DSC_0001.RAF");
        db.insert_image_with_raw(&jpg, &fake_hash(60), 2048, Some(&raf))
            .unwrap();

        let results = db.search_images(&SearchQuery::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].raw_path.as_deref(), Some(raf.as_path()));
    }

    #[test]
    fn restructure_candidates_reflects_present_images_only() {
        let (_dir, db) = tmp_db();
        let present = PathBuf::from("/library/a.jpg");
        let missing = PathBuf::from("/library/b.jpg");
        db.insert_image(&present, &fake_hash(70), 1024).unwrap();
        db.insert_image(&missing, &fake_hash(71), 1024).unwrap();
        db.mark_missing(&missing).unwrap();

        let id = db.image_id_for_path(&present).unwrap().unwrap();
        db.update_metadata(
            id,
            &ImageMetadata {
                filename: Some("a.jpg".into()),
                taken_at: Some(1_710_513_045),
                make: Some("Fujifilm".into()),
                model: Some("X100V".into()),
                ..Default::default()
            },
        )
        .unwrap();

        let candidates = db.restructure_candidates().unwrap();
        assert_eq!(candidates.len(), 1);
        assert_eq!(candidates[0].current_path, present);
        assert_eq!(candidates[0].camera.as_deref(), Some("Fujifilm X100V"));
        assert!(candidates[0].datetime.is_some());
    }

    #[test]
    fn update_image_location_moves_row_and_resyncs_fts() {
        let (_dir, db) = tmp_db();
        let old = PathBuf::from("/library/old/a.jpg");
        db.insert_image(&old, &fake_hash(72), 1024).unwrap();
        let id = db.image_id_for_path(&old).unwrap().unwrap();

        let new_path = PathBuf::from("/library/2024/03/photo.jpg");
        db.update_image_location(id, &new_path, None, "photo.jpg").unwrap();

        let results = db.search_images(&SearchQuery::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].path, new_path);
        assert_eq!(results[0].meta.filename.as_deref(), Some("photo.jpg"));

        // The images_fts_au trigger should have resynced the FTS row to the
        // new filename.
        let hits = db.search_images(&SearchQuery::default().with_text("photo")).unwrap();
        assert_eq!(hits.len(), 1);
    }
}
