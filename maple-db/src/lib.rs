//! maple-db — SQLite-backed image library database.
//!
//! Stores every image that has been imported into the library, keyed by its
//! file-system path and BLAKE3 content hash.  A background scanner
//! (`LibraryScanner`) periodically reconciles the on-disk state with the
//! database, marking records as `missing` when the file is deleted and
//! re-marking them `present` when it reappears.

mod embedding;
#[cfg(test)]
mod listing_bench;
mod scanner;
mod schema;
mod semantic_db;
mod session_dino;
pub mod sync;
mod sync_identity;
mod sync_peers;
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
pub use session_dino::DinoEngine;
pub use face_detector::{spawn_face_tagger, DetectedFace, FaceDetector, FaceTagger};
pub use faces::{best_person_match, best_person_matches, cosine_similarity, FaceDetection, Person, PersonWithRep};
pub use metadata::{extract_all_exif_tags, extract_metadata, spawn_metadata_filler, ImageMetadata};
pub use query::{SearchOrder, SearchQuery};
pub use scanner::{set_scanner_paused, LibraryChanged, LibraryScanner};
pub use schema::SYNCED_TABLES;
pub use sync::{ApplyReport, MissingOriginal, SyncBatch, SyncRow, Tombstone};
pub use sync_identity::SyncIdentity;
pub use sync_peers::SyncPeer;
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

/// Whether this installation holds the photo's bytes, or only knows about it.
///
/// Machine-local, like [`ImageStatus`]: the same photo is `Local` on the
/// master and `Remote` on a relay servant, so the column is never replicated
/// (see V20 in `schema.rs`). A `Remote` row is `status = 'present'` — it is a
/// perfectly good library entry that appears in the grid; what makes it
/// remote is only where its pixels have to be fetched from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Locality {
    Local,
    Remote,
}

impl Locality {
    pub fn as_str(&self) -> &'static str {
        match self {
            Locality::Local => "local",
            Locality::Remote => "remote",
        }
    }

    /// Anything unrecognised reads as `Local`, matching the column default:
    /// treating an unknown value as remote would send the UI over the network
    /// for a file that is sitting on this disk.
    fn from_str(s: &str) -> Self {
        match s {
            "remote" => Locality::Remote,
            _ => Locality::Local,
        }
    }

    pub fn is_remote(&self) -> bool {
        matches!(self, Locality::Remote)
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
    /// Whether `path` can actually be opened here, or the bytes have to be
    /// fetched from `origin_device` (see [`Locality`]).
    pub locality: Locality,
    /// Device id this row arrived from, for a `Remote` row. `None` for
    /// anything this installation imported itself.
    pub origin_device: Option<String>,
}

// ── Database ─────────────────────────────────────────────────────

/// Replicated tables that `ON DELETE CASCADE` removes along with `parent`,
/// paired with the foreign-key column that points back at it.
///
/// Mirrors the `REFERENCES … ON DELETE CASCADE` clauses in `schema.rs`.
/// Relationships declared `ON DELETE SET NULL` — `face_detections.person_id`,
/// `images.stack_id` — are absent on purpose: those children survive the
/// delete and are stamped as ordinary edits instead.
fn cascade_children(parent: &str) -> &'static [(&'static str, &'static str)] {
    match parent {
        "images" => &[
            ("ai_descriptions", "image_id"),
            ("face_detections", "image_id"),
            ("collection_images", "image_id"),
        ],
        "collections" => &[("collection_images", "collection_id")],
        _ => &[],
    }
}

/// `(image_id, path, content_hash)` — returned by [`Database::images_without_hash`].
type ImageHashCandidate = (i64, PathBuf, Option<[u8; 32]>);
/// `(image_id, hash_blob, stack_id)` — returned by [`Database::images_with_hash_and_stack`].
type ImageHashStackRow = (i64, Vec<u8>, Option<i64>);
/// `(path, status, content_hash)` — returned by [`Database::all_paths`].
type ImagePathStatusRow = (PathBuf, ImageStatus, Option<[u8; 32]>);

pub struct Database {
    conn: Connection,
    /// This installation's device id and hybrid logical clock, used to stamp
    /// every write that replicates to a paired device.
    identity: SyncIdentity,
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
        // Foreign keys are load-bearing: deleting a collection, person or
        // image relies on `ON DELETE CASCADE` to take its children with it,
        // and sync relies on that in turn — a tombstone for a parent row
        // propagates as one delete, and each side's cascade cleans up its own
        // children rather than shipping a tombstone per child.
        //
        // rusqlite's bundled SQLite happens to be built with
        // SQLITE_DEFAULT_FOREIGN_KEYS, so this is already on today. Setting it
        // explicitly means the invariant survives a switch to a system SQLite
        // instead of silently turning every cascade into an orphaned row.
        conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON;")?;
        schema::ensure_schema(&conn)?;
        let identity = SyncIdentity::load(&conn)?;
        Ok(Self { conn, identity })
    }

    // ── Sync identity ────────────────────────────────────────────

    /// This installation's device id.
    pub fn device_id(&self) -> &str {
        self.identity.device_id()
    }

    /// Next `(rev, rev_dev)` stamp for a locally-originated write.
    ///
    /// Callers append `rev = ?, rev_dev = ?` to the statement's `SET` clause.
    /// Stamping happens here rather than in a trigger on purpose: V17 removed
    /// the last `AFTER UPDATE ON images` trigger because the library scanner
    /// and stacker issue bulk `UPDATE images SET status/stack_id`, and a
    /// trigger fires on every one of those rows. Explicit stamping keeps that
    /// cost off the writes that don't replicate.
    pub fn stamp(&self) -> anyhow::Result<(i64, String)> {
        self.identity.stamp(&self.conn)
    }

    /// Advance the local clock past a stamp received from a peer.
    pub fn observe_remote_rev(&self, rev: i64) {
        self.identity.observe(rev);
    }

    /// The schema version this library is at (`PRAGMA user_version`).
    ///
    /// Sync reports it in its handshake: two installations that differ here
    /// may disagree about what a replicated row even contains, and the
    /// honest response is to refuse the link rather than merge rows one side
    /// cannot represent.
    pub fn schema_version(&self) -> anyhow::Result<i64> {
        Ok(self.conn.query_row("PRAGMA user_version", [], |r| r.get(0))?)
    }

    /// Generate a fresh row guid.
    ///
    /// Matches the `lower(hex(randomblob(16)))` form the V18 backfill uses:
    /// 128 opaque random bits, no UUID crate needed, since nothing reads the
    /// RFC-4122 version or variant nibbles.
    pub fn new_guid(&self) -> anyhow::Result<String> {
        Ok(self
            .conn
            .query_row("SELECT lower(hex(randomblob(16)))", [], |r| r.get(0))?)
    }

    /// Draw `n` cryptographically random bytes.
    ///
    /// SQLite's `randomblob` rather than a `rand` dependency: it is seeded
    /// from the OS CSPRNG, and it is already the source behind `device_id`
    /// and every row guid, so sync has one place its randomness comes from
    /// rather than two. `maple-sync` takes this through its `RandomSource`
    /// trait, which is what lets its handshake tests replay a fixed stream.
    pub fn random_bytes(&self, n: usize) -> anyhow::Result<Vec<u8>> {
        // `randomblob(0)` returns *one* byte, not none — SQLite clamps N to
        // at least 1. Handled here so a caller sizing a buffer from a length
        // it computed cannot silently get one byte more than it asked for.
        if n == 0 {
            return Ok(Vec::new());
        }
        Ok(self
            .conn
            .query_row("SELECT randomblob(?1)", [n as i64], |r| r.get(0))?)
    }

    /// Record that the rows with `ids` in `table` were deleted locally,
    /// along with everything `ON DELETE CASCADE` will take with them.
    ///
    /// Deletion has to be represented explicitly: a row that is simply gone
    /// is indistinguishable from a row the peer has not sent yet, so without
    /// a tombstone the next sync would helpfully restore everything the user
    /// just deleted. The tombstone carries a stamp so it can lose to a later
    /// edit — a peer that modified the row *after* this delete resurrects it.
    ///
    /// # Why children are tombstoned too
    ///
    /// It is tempting to tombstone only the parent and let each device's own
    /// cascade clear its children — and for a plain delete that works. It
    /// breaks under **resurrection**: if the peer edited the parent after the
    /// delete, the parent comes back, but its children do not come back with
    /// it. The two devices then disagree about which children exist, and
    /// nothing re-sends them, because their stamps are older than the
    /// watermark. Tombstoning the children makes the delete symmetric, so
    /// both sides end up with the same (empty) set either way.
    ///
    /// Call this *before* the `DELETE`, while the guids are still readable.
    pub fn tombstone(&self, table: &str, ids: &[i64]) -> anyhow::Result<()> {
        if ids.is_empty() {
            return Ok(());
        }
        debug_assert!(
            schema::SYNCED_TABLES.contains(&table),
            "{table} is not replicated, so its deletes need no tombstone"
        );

        let (rev, rev_dev) = self.stamp()?;
        let tx = self.conn.unchecked_transaction()?;
        {
            let mut write = tx.prepare(
                "INSERT INTO sync_tombstones (guid, entity, rev, rev_dev)
                 VALUES (?1, ?2, ?3, ?4)
                 ON CONFLICT(guid) DO UPDATE SET rev = ?3, rev_dev = ?4",
            )?;
            let mut mark = |t: &str, guid: Option<String>| -> anyhow::Result<()> {
                // A row with no guid predates V18 and was never replicated,
                // so no peer can be holding a copy to resurrect.
                if let Some(guid) = guid {
                    write.execute(params![guid, t, rev, rev_dev])?;
                }
                Ok(())
            };

            let mut read = tx.prepare(&format!("SELECT guid FROM {table} WHERE id = ?1"))?;
            for id in ids {
                mark(
                    table,
                    read.query_row(params![id], |r| r.get(0)).optional()?.flatten(),
                )?;

                for (child, fk) in cascade_children(table) {
                    let mut kids = tx.prepare(&format!(
                        "SELECT guid FROM {child} WHERE {fk} = ?1 AND guid IS NOT NULL"
                    ))?;
                    let guids: Vec<String> = kids
                        .query_map(params![id], |r| r.get(0))?
                        .filter_map(|r| r.ok())
                        .collect();
                    for guid in guids {
                        mark(child, Some(guid))?;
                    }
                }
            }
        }
        tx.commit()?;
        Ok(())
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
    /// The file basename is stored immediately as `filename` so that filename
    /// search works before full EXIF extraction runs.
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
        let (rev, rev_dev) = self.stamp()?;
        let guid = self.new_guid()?;

        self.conn.execute(
            "INSERT OR IGNORE INTO images
                 (path, hash, file_size, added_at, status, filename, raw_path,
                  guid, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4, 'present', ?5, ?6, ?7, ?8, ?9)",
            params![
                path_to_db(path),
                hash.as_slice(),
                file_size as i64,
                added_at,
                filename,
                raw_str,
                guid,
                rev,
                rev_dev,
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
    ///
    /// Restricted to `locality = 'local'`: a relayed row's `path` names a file
    /// on the device that holds it, and this work opens the file.
    pub fn count_images_without_hash(&self, algorithm: &str) -> anyhow::Result<usize> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM images
             WHERE status = 'present' AND locality = 'local'
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
    ///
    /// Restricted to `locality = 'local'`: a relayed row's `path` names a file
    /// on the device that holds it, and this work opens the file.
    pub fn images_without_hash(
        &self,
        algorithm: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<ImageHashCandidate>> {
        let mut stmt = self.conn.prepare_cached(
            "SELECT id, path, hash FROM images
             WHERE status = 'present' AND locality = 'local'
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
        let (rev, rev_dev) = self.stamp()?;
        let guid = self.new_guid()?;
        self.conn.execute(
            "INSERT INTO stacks (created_at, guid, rev, rev_dev) VALUES (?1, ?2, ?3, ?4)",
            params![now, guid, rev, rev_dev],
        )?;
        Ok(self.conn.last_insert_rowid())
    }

    /// Assign `image_id` to `stack_id`.  Pass `None` to remove from any stack.
    pub fn set_image_stack(&self, image_id: i64, stack_id: Option<i64>) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE images SET stack_id = ?1, rev = ?2, rev_dev = ?3 WHERE id = ?4",
            params![stack_id, rev, rev_dev, image_id],
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
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE stacks SET cover_image_id = ?1, rev = ?2, rev_dev = ?3 WHERE id = ?4",
            params![image_id, rev, rev_dev, stack_id],
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
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE images SET stack_id = NULL, rev = ?1, rev_dev = ?2 WHERE id = ?3",
            params![rev, rev_dev, image_id],
        )?;

        // If fewer than 2 images remain, disband the whole stack.
        let remaining: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM images WHERE stack_id = ?1 AND status = 'present'",
            params![stack_id],
            |r| r.get(0),
        )?;

        if remaining < 2 {
            let (rev, rev_dev) = self.stamp()?;
            self.conn.execute(
                "UPDATE images SET stack_id = NULL, rev = ?1, rev_dev = ?2 WHERE stack_id = ?3",
                params![rev, rev_dev, stack_id],
            )?;
            self.tombstone("stacks", &[stack_id])?;
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
    ///
    /// Not stamped — see [`Database::mark_missing`]; `raw_path` is a local
    /// filesystem location, and each machine discovers its own.
    pub fn set_raw_path(&self, id: i64, raw_path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET raw_path = ?1 WHERE id = ?2",
            params![path_to_db(raw_path), id],
        )?;
        Ok(())
    }

    /// Update the on-disk location and display filename for an image record
    /// after a library restructure move (see `maple_import::restructure`).
    ///
    /// Not stamped — see [`Database::mark_missing`]. `filename` rides along
    /// unreplicated because it is derived from `path`: restructuring the
    /// workstation's library says nothing about where the laptop keeps its
    /// copy, and propagating either machine's paths would break the other's.
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
        let (rev, rev_dev) = self.stamp()?;
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
                 exif_extracted = 1,
                 rev            = ?12,
                 rev_dev        = ?13
             WHERE id = ?14",
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
                rev,
                rev_dev,
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
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            "UPDATE images SET hash = ?1, orientation = ?2, rev = ?3, rev_dev = ?4 WHERE id = ?5",
            params![hash.as_slice(), orientation, rev, rev_dev, id],
        )?;
        Ok(())
    }

    /// Mark a record as missing (file deleted from disk).
    ///
    /// Deliberately **not** stamped: `status` is a statement about this
    /// machine's disk, not about the photo. A file absent from the laptop is
    /// still present on the workstation, so replicating `'missing'` would let
    /// whichever machine holds fewer originals blank out the other's library.
    /// The same reasoning keeps `path`, `raw_path` and `file_size` local.
    pub fn mark_missing(&self, path: &Path) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE images SET status = 'missing' WHERE path = ?1",
            params![path_to_db(path)],
        )?;
        Ok(())
    }

    /// Mark a record as present (file has reappeared on disk).
    ///
    /// Not stamped — see [`Database::mark_missing`].
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
    /// With no text filter: returns all present images, ordered by
    /// `query.order`.
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
            (Some(text), None) => self.search_images_text(
                text,
                query.limit,
                query.offset,
                query.collection_id,
                query.person_id,
                query.order,
            ),
            (None, _) => self.search_images_all(
                query.limit,
                query.offset,
                query.collection_id,
                query.person_id,
                query.order,
            ),
        }
    }

    /// Total number of rows `query`'s filters match, ignoring its
    /// `limit`/`offset` — the "N photos" figure for a paged listing.
    ///
    /// `None` for hybrid (semantic) queries: their result set is a merge of
    /// a keyword page and a KNN result list, so there is no row count to be
    /// had without running the merge itself.
    pub fn count_images(&self, query: &SearchQuery) -> anyhow::Result<Option<usize>> {
        if query.semantic_embedding.is_some() && query.text.is_some() {
            return Ok(None);
        }

        let (from_where, params) = match &query.text {
            Some(text) => match text_from_where(
                text,
                Entry::Table,
                query.collection_id,
                query.person_id,
            ) {
                Some(parts) => parts,
                // No usable tokens — the row query returns nothing, so does this.
                None => return Ok(Some(0)),
            },
            None => all_from_where(Entry::Table, query.collection_id, query.person_id),
        };
        // The text query joins descriptions/faces and so can repeat a row per
        // match; `DISTINCT` mirrors its `SELECT DISTINCT`.
        let selector = if query.text.is_some() { "COUNT(DISTINCT i.id)" } else { "COUNT(*)" };
        let sql = format!("SELECT {selector} {from_where}");

        let mut stmt = self.conn.prepare(&sql)?;
        let n: i64 = stmt.query_row(rusqlite::params_from_iter(params), |r| r.get(0))?;
        Ok(Some(n.max(0) as usize))
    }

    /// Return all present images in `order`.
    fn search_images_all(
        &self,
        limit: Option<usize>,
        offset: Option<usize>,
        collection_id: Option<i64>,
        person_id: Option<i64>,
        order: SearchOrder,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        use rusqlite::types::Value;

        let limit = limit.unwrap_or(500) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let (from_where, mut params) = all_from_where(Entry::Index, collection_id, person_id);
        let order_by = crate::query::order_by_sql(order);
        let sql = format!(
            "SELECT {IMAGE_COLUMNS} {from_where} {order_by} LIMIT ? OFFSET ?"
        );

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
        order: SearchOrder,
    ) -> anyhow::Result<Vec<LibraryImage>> {
        use rusqlite::types::Value;

        let limit = limit.unwrap_or(500) as i64;
        let offset = offset.unwrap_or(0) as i64;

        let Some((from_where, mut params)) =
            text_from_where(text, Entry::Index, collection_id, person_id)
        else {
            return Ok(vec![]);
        };

        let order_by = crate::query::order_by_sql(order);
        let sql = format!(
            "SELECT DISTINCT {IMAGE_COLUMNS} {from_where} {order_by} LIMIT ? OFFSET ?"
        );

        params.push(Value::Integer(limit));
        params.push(Value::Integer(offset));

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
    ///
    /// Restricted to `locality = 'local'`: a relayed row's `path` names a file
    /// on the device that holds it, and this work opens the file.
    pub fn images_needing_ai_description(
        &self,
        model_id: &str,
    ) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.id, i.path
             FROM images i
             WHERE i.status = 'present' AND i.locality = 'local'
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
        let (rev, rev_dev) = self.stamp()?;
        let guid = self.new_guid()?;
        self.conn.execute(
            "INSERT INTO ai_descriptions
                 (image_id, model_id, description, created_at, guid, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(image_id, model_id) DO UPDATE SET
                 description = excluded.description,
                 created_at  = excluded.created_at,
                 rev         = excluded.rev,
                 rev_dev     = excluded.rev_dev",
            params![image_id, model_id, description, created_at, guid, rev, rev_dev],
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

    /// Delete every row in `ai_descriptions`.  The sentence-embedding
    /// invalidation triggers cascade automatically.  Returns the number of
    /// rows deleted.
    pub fn clear_all_ai_descriptions(&self) -> anyhow::Result<usize> {
        // A deliberate user action, so it has to propagate — otherwise the
        // next sync refills the table from a peer that still has them.
        let ids: Vec<i64> = self
            .conn
            .prepare("SELECT id FROM ai_descriptions")?
            .query_map([], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        self.tombstone("ai_descriptions", &ids)?;
        let n = self.conn.execute("DELETE FROM ai_descriptions", [])?;
        Ok(n)
    }

    /// Return `(id, path)` for all records where EXIF has not been extracted
    /// yet (`exif_extracted = 0`).  Used by `spawn_metadata_filler`.
    ///
    /// Restricted to `locality = 'local'`: a relayed row's `path` names a file
    /// on the device that holds it, and this work opens the file.
    pub fn records_needing_metadata(&self) -> anyhow::Result<Vec<(i64, PathBuf)>> {
        let mut stmt = self
            .conn
            .prepare(
                "SELECT id, path FROM images
                 WHERE exif_extracted = 0 AND status = 'present' AND locality = 'local'",
            )?;
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

    /// Return all `(path, status, hash)` triples for files this device
    /// actually holds — used by the scanner for reconciliation and thumbnail
    /// cache eviction.
    ///
    /// Remote rows are excluded, and that filter is load-bearing rather than
    /// tidy: their `path` is the *origin* device's, so the scanner would find
    /// nothing on disk for it, mark the row missing and evict its thumbnail
    /// on the very next 60-second pass — emptying a relay servant's grid a
    /// minute after it filled.
    ///
    /// Leaving them out of the caller's "already known" set is harmless: for
    /// a remote path to collide with a file under this machine's library dir
    /// the two installations would have to share a directory layout *and* a
    /// user, and `insert_image_with_raw` is `INSERT OR IGNORE`, so the worst
    /// case is a no-op.
    pub fn all_paths(&self) -> anyhow::Result<Vec<ImagePathStatusRow>> {
        let mut stmt = self
            .conn
            .prepare("SELECT path, status, hash FROM images WHERE locality = 'local'")?;
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
    ///
    /// Restricted to `locality = 'local'`: a relayed row's `path` names a file
    /// on the device that holds it, and this work opens the file.
    pub fn restructure_candidates(&self) -> anyhow::Result<Vec<maple_import::RestructureCandidate>> {
        let mut stmt = self.conn.prepare(
            "SELECT id, path, raw_path, taken_at, make, model
             FROM images WHERE status = 'present' AND locality = 'local'",
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
                    ELSE NULL END AS stack_size,
                    i.locality, i.origin_device
             FROM images i
             WHERE i.id = ?1",
        )?;
        let mut rows = stmt.query_map(params![id], row_to_library_image)?;
        Ok(rows.next().transpose()?)
    }

    /// The file a peer's `/blob/...` request should be served from.
    ///
    /// `raw` picks the companion raw file instead of the display image, which
    /// is the only thing `?raw=1` changes.
    ///
    /// `idx_images_hash` is deliberately **not** unique — a library may
    /// legitimately hold the same photo twice — so several rows can match.
    /// Any of them serves identical bytes by definition (the hash is of the
    /// content), so the lowest id is picked purely to make the choice
    /// deterministic. Rows that are remote or missing are skipped: their
    /// `path` names a file this machine cannot open.
    pub fn blob_path(&self, hash: &[u8; 32], raw: bool) -> anyhow::Result<Option<PathBuf>> {
        let column = if raw { "raw_path" } else { "path" };
        let found: Option<Option<String>> = self
            .conn
            .query_row(
                &format!(
                    "SELECT {column} FROM images
                     WHERE hash = ?1 AND locality = 'local' AND status = 'present'
                       AND {column} IS NOT NULL
                     ORDER BY id LIMIT 1"
                ),
                params![hash.as_slice()],
                |r| r.get(0),
            )
            .optional()?;
        Ok(found.flatten().map(path_from_db))
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

// ── Listing SQL fragments ────────────────────────────────────────
//
// The row listing and its `COUNT(*)` twin must apply exactly the same
// filters, or the "N photos" figure disagrees with what the grid can page
// through.  Both are assembled from the fragments below rather than being
// written out twice.

/// How a listing enters `images` — the one place the row query and its
/// `COUNT(*)` twin are allowed to differ.
///
/// The paged listing wants the V17 ordering index: `LIMIT` means it only
/// touches `limit + offset` rows, and the index hands them over already
/// sorted.  The count has no `LIMIT` and needs `stack_id` for every present
/// row, which those indexes do not carry — the planner walks one anyway and
/// then fetches each row from the table in index order, one random read per
/// row across the whole file.  Reading the table instead is ~7× faster, and
/// still leaves the collection/person filters free to drive the query
/// through `images`' INTEGER PRIMARY KEY, which `NOT INDEXED` does not
/// forbid.
#[derive(Clone, Copy)]
enum Entry {
    /// Through whichever ordering index matches the `ORDER BY`.
    Index,
    /// Straight down the table (unlimited aggregate — no ordering to serve).
    Table,
}

impl Entry {
    fn images(self) -> &'static str {
        match self {
            Entry::Index => "FROM images i",
            Entry::Table => "FROM images i NOT INDEXED",
        }
    }
}

/// A stack contributes exactly one row to the listing: the image the stack
/// names as its cover, or its lowest-id present member when it names none.
///
/// Correlated subqueries over `i` alone, rather than a join against a
/// materialised `stack_covers` CTE.  A cover test that reaches across a join
/// (`i.stack_id IS NULL OR i.id = sc.cover_id`) leaves SQLite unable to use
/// any index to satisfy the listing's `ORDER BY`, so it sorted the entire
/// table into a temp b-tree for every page the grid asked for — and adding
/// the V17 ordering indexes made that *worse*, because the CTE's own scan
/// then went through them too.  Written this way the ordering index drives
/// the scan and the cover test is a per-row filter.
///
/// A `stack_id` pointing at a row `stacks` no longer has yields NULL here,
/// which excludes the image — the same thing the outer join did.
const STACK_COVER_PREDICATE: &str = "(
             i.stack_id IS NULL
             OR i.id = (
                 SELECT COALESCE(
                            s.cover_image_id,
                            (SELECT MIN(m.id) FROM images m
                             WHERE m.stack_id = i.stack_id AND m.status = 'present'))
                 FROM stacks s WHERE s.id = i.stack_id
             )
           )";

/// Column list consumed by [`row_to_library_image`], in its expected order.
///
/// `stack_size` stays NULL for an unstacked image (`row_to_library_image`
/// hands it on as `Option`), which is what the outer join used to produce.
const IMAGE_COLUMNS: &str = "i.id, i.path, i.added_at, i.status,
            i.filename, i.taken_at, i.make, i.model, i.lens,
            i.focal_length, i.aperture, i.iso,
            i.width, i.height, i.orientation, i.raw_path, i.hash,
            i.stack_id,
            CASE WHEN i.stack_id IS NOT NULL THEN
                (SELECT COUNT(*) FROM images m
                 WHERE m.stack_id = i.stack_id AND m.status = 'present')
            ELSE NULL END,
            i.locality, i.origin_device";

/// `AND` clauses (and their bound ids) for the optional collection/person
/// filters, shared by the plain and text listings.
fn filter_clauses(
    collection_id: Option<i64>,
    person_id: Option<i64>,
) -> (String, Vec<rusqlite::types::Value>) {
    use rusqlite::types::Value;

    let mut sql = String::new();
    let mut params = Vec::new();
    if let Some(cid) = collection_id {
        sql.push_str(" AND i.id IN (SELECT image_id FROM collection_images WHERE collection_id = ?)");
        params.push(Value::Integer(cid));
    }
    if let Some(pid) = person_id {
        sql.push_str(" AND i.id IN (SELECT image_id FROM face_detections WHERE person_id = ?)");
        params.push(Value::Integer(pid));
    }
    (sql, params)
}

/// `FROM … WHERE …` for an unfiltered (no text) listing, plus its params.
fn all_from_where(
    entry: Entry,
    collection_id: Option<i64>,
    person_id: Option<i64>,
) -> (String, Vec<rusqlite::types::Value>) {
    let (extra, params) = filter_clauses(collection_id, person_id);
    let sql = format!(
        "{}
         WHERE i.status = 'present'
           AND {STACK_COVER_PREDICATE}{extra}",
        entry.images()
    );
    (sql, params)
}

/// `FROM … WHERE …` for a text listing, plus its params. `None` when `text`
/// holds no usable tokens (the caller returns an empty result).
fn text_from_where(
    text: &str,
    entry: Entry,
    collection_id: Option<i64>,
    person_id: Option<i64>,
) -> Option<(String, Vec<rusqlite::types::Value>)> {
    use rusqlite::types::Value;

    let like_patterns: Vec<String> = text
        .split_whitespace()
        .map(|t| format!("%{}%", escape_like_token(t)))
        .collect();
    if like_patterns.is_empty() {
        return None;
    }

    // Each token must match somewhere in the combined EXIF fields OR
    // in any AI description OR in any assigned person name OR in any
    // comprehensive EXIF tag value (shutter speed, GPS, flash, …).
    let exif_expr = "LOWER(COALESCE(i.filename,'') || ' ' || \
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

    let (extra, extra_params) = filter_clauses(collection_id, person_id);
    let sql = format!(
        "{}
         LEFT JOIN ai_descriptions ad ON ad.image_id = i.id
         LEFT JOIN face_detections fd ON fd.image_id = i.id
         LEFT JOIN persons p ON p.id = fd.person_id
         WHERE i.status = 'present'
           AND {token_conditions}
           AND {STACK_COVER_PREDICATE}{extra}",
        entry.images()
    );

    // Each token pattern appears four times: EXIF, AI desc, person name, EXIF tags.
    let params: Vec<Value> = like_patterns
        .into_iter()
        .flat_map(|p| {
            [
                Value::Text(p.clone()),
                Value::Text(p.clone()),
                Value::Text(p.clone()),
                Value::Text(p),
            ]
        })
        .chain(extra_params)
        .collect();

    Some((sql, params))
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
    let locality: String = row.get(19)?;
    let origin_device: Option<String> = row.get(20)?;
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
        locality: Locality::from_str(&locality),
        origin_device,
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

    // ── Sync stamping ────────────────────────────────────────────

    /// `(guid, rev, rev_dev)` for one row.
    fn stamp_of(db: &Database, table: &str, id: i64) -> (Option<String>, i64, Option<String>) {
        db.conn
            .query_row(
                &format!("SELECT guid, rev, rev_dev FROM {table} WHERE id = ?1"),
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .unwrap()
    }

    #[test]
    fn inserted_rows_carry_an_identity_and_a_stamp() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024)
            .unwrap();
        let id = db
            .image_id_for_path(&PathBuf::from("/photos/a.jpg"))
            .unwrap()
            .unwrap();

        let (guid, rev, rev_dev) = stamp_of(&db, "images", id);
        assert_eq!(guid.unwrap().len(), 32);
        assert!(rev > 0, "a fresh row must not sit at the zero watermark");
        assert_eq!(rev_dev.as_deref(), Some(db.device_id()));
    }

    #[test]
    fn editing_a_row_advances_its_stamp() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024)
            .unwrap();
        let id = db
            .image_id_for_path(&PathBuf::from("/photos/a.jpg"))
            .unwrap()
            .unwrap();
        let (guid_before, rev_before, _) = stamp_of(&db, "images", id);

        db.update_image_hash_and_orientation(id, &fake_hash(2), 6)
            .unwrap();

        let (guid_after, rev_after, _) = stamp_of(&db, "images", id);
        assert!(rev_after > rev_before, "the edit must be newer than the insert");
        assert_eq!(guid_before, guid_after, "identity is stable across edits");
    }

    #[test]
    fn local_only_columns_do_not_advance_the_stamp() {
        let (_dir, db) = tmp_db();
        let path = PathBuf::from("/photos/a.jpg");
        db.insert_image(&path, &fake_hash(1), 1024).unwrap();
        let id = db.image_id_for_path(&path).unwrap().unwrap();
        let (_, rev_before, _) = stamp_of(&db, "images", id);

        // `status` describes this machine's disk, not the photo. If these
        // bumped the stamp, a laptop that has fewer originals would win every
        // merge and mark the workstation's library missing.
        db.mark_missing(&path).unwrap();
        db.mark_present(&path).unwrap();
        db.set_raw_path(id, &PathBuf::from("/photos/a.raf")).unwrap();

        let (_, rev_after, _) = stamp_of(&db, "images", id);
        assert_eq!(rev_before, rev_after);
    }

    #[test]
    fn deleting_a_collection_leaves_a_tombstone() {
        let (_dir, db) = tmp_db();
        let cid = db.create_collection("Trip", "#3584e4", None).unwrap();
        let (guid, _, _) = stamp_of(&db, "collections", cid);
        let guid = guid.unwrap();

        db.delete_collection(cid).unwrap();

        let (entity, rev): (String, i64) = db
            .conn
            .query_row(
                "SELECT entity, rev FROM sync_tombstones WHERE guid = ?1",
                params![guid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .expect("a delete must be recorded, or the next sync restores it");
        assert_eq!(entity, "collections");
        assert!(rev > 0);
    }

    #[test]
    fn deleting_a_collection_cascades_to_its_memberships() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024)
            .unwrap();
        let iid = db
            .image_id_for_path(&PathBuf::from("/photos/a.jpg"))
            .unwrap()
            .unwrap();
        let cid = db.create_collection("Trip", "#3584e4", None).unwrap();
        db.add_image_to_collection(cid, iid).unwrap();

        db.delete_collection(cid).unwrap();

        // The cascade is what lets one parent tombstone stand in for the
        // whole subtree; SQLite rowids get reused, so a leftover membership
        // row would silently reattach itself to the next collection created.
        let orphans: i64 = db
            .conn
            .query_row(
                "SELECT COUNT(*) FROM collection_images WHERE collection_id = ?1",
                params![cid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(orphans, 0);
    }

    #[test]
    fn deleting_a_person_unassigns_and_stamps_their_faces() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024)
            .unwrap();
        let iid = db
            .image_id_for_path(&PathBuf::from("/photos/a.jpg"))
            .unwrap()
            .unwrap();
        let fid = db
            .insert_face_detection(iid, [0.1, 0.1, 0.2, 0.2], &[0.0; 512], 0.9)
            .unwrap();
        let pid = db.upsert_person("Ada").unwrap();
        db.assign_face_to_person(fid, Some(pid)).unwrap();
        let (_, rev_before, _) = stamp_of(&db, "face_detections", fid);

        db.delete_person(pid).unwrap();

        // The face survives with its box intact but loses the name, and that
        // un-assignment is a real edit — a peer still holding the old name
        // must lose to it.
        let (_, rev_after, _) = stamp_of(&db, "face_detections", fid);
        assert!(rev_after > rev_before);
        let person: Option<i64> = db
            .conn
            .query_row(
                "SELECT person_id FROM face_detections WHERE id = ?1",
                params![fid],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(person, None);
    }

    #[test]
    fn random_bytes_are_the_requested_length_and_not_constant() {
        let (_dir, db) = tmp_db();
        let a = db.random_bytes(32).unwrap();
        let b = db.random_bytes(32).unwrap();
        assert_eq!(a.len(), 32);
        assert_ne!(a, b, "randomblob must not return a constant");
        assert!(db.random_bytes(0).unwrap().is_empty());
    }

    #[test]
    fn stamps_increase_monotonically_across_tables() {
        let (_dir, db) = tmp_db();
        db.insert_image(&PathBuf::from("/photos/a.jpg"), &fake_hash(1), 1024)
            .unwrap();
        let iid = db
            .image_id_for_path(&PathBuf::from("/photos/a.jpg"))
            .unwrap()
            .unwrap();
        let cid = db.create_collection("Trip", "#3584e4", None).unwrap();
        db.add_image_to_collection(cid, iid).unwrap();
        let pid = db.upsert_person("Ada").unwrap();

        // One clock serves every table, so a single watermark per peer is
        // enough to ask "what changed since?" across the whole library.
        let img = stamp_of(&db, "images", iid).1;
        let coll = stamp_of(&db, "collections", cid).1;
        let person = stamp_of(&db, "persons", pid).1;
        assert!(img < coll, "images {img} should precede collections {coll}");
        assert!(coll < person, "collections {coll} should precede persons {person}");
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
    fn search_by_filename() {
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
    fn update_image_location_moves_row_and_updates_filename() {
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

        // Text search sees the new filename, not the old one.
        let hits = db.search_images(&SearchQuery::default().with_text("photo")).unwrap();
        assert_eq!(hits.len(), 1);
        let stale = db.search_images(&SearchQuery::default().with_text("a.jpg")).unwrap();
        assert!(stale.is_empty());
    }

    // ── Paged listings ───────────────────────────────────────────
    //
    // The library grid pages through these queries, so what matters is that
    // consecutive pages of the same query concatenate into exactly the
    // unpaged result — no repeats, no gaps, ordering preserved.

    /// Insert `n` images that all share one `added_at` second (the bulk
    /// import case), with `taken_at` running *backwards* relative to id so
    /// the two orderings are distinguishable. Returns their ids.
    fn insert_burst(db: &Database, n: usize) -> Vec<i64> {
        let mut ids = Vec::new();
        for k in 0..n {
            let path = PathBuf::from(format!("/photos/burst_{k:03}.jpg"));
            db.insert_image(&path, &fake_hash(k as u8), 1024).unwrap();
            let id = db.image_id_for_path(&path).unwrap().unwrap();
            db.update_metadata(
                id,
                &ImageMetadata {
                    filename: Some(format!("burst_{k:03}.jpg")),
                    taken_at: Some(1_700_000_000 - k as i64 * 3600),
                    ..Default::default()
                },
            )
            .unwrap();
            ids.push(id);
        }
        ids
    }

    fn page_ids(db: &Database, query: &SearchQuery, page: usize, size: usize) -> Vec<i64> {
        db.search_images(&query.clone().with_limit(size).with_offset(page * size))
            .unwrap()
            .iter()
            .map(|i| i.id)
            .collect()
    }

    #[test]
    fn pages_concatenate_into_the_unpaged_listing() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 25);

        let all: Vec<i64> = db
            .search_images(&SearchQuery::default())
            .unwrap()
            .iter()
            .map(|i| i.id)
            .collect();
        assert_eq!(all.len(), 25);

        let mut paged = Vec::new();
        for page in 0..3 {
            paged.extend(page_ids(&db, &SearchQuery::default(), page, 10));
        }
        assert_eq!(paged, all);
    }

    #[test]
    fn taken_order_pages_stay_in_date_order() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 25);

        let q = SearchQuery::default().with_order(SearchOrder::TakenDesc);
        let mut paged = Vec::new();
        for page in 0..3 {
            paged.extend(page_ids(&db, &q, page, 10));
        }

        // taken_at descends as the filename index ascends, so date order is
        // the insertion order — the opposite of the default added_at order,
        // which ties on the second and falls back to id DESC.
        let expected: Vec<i64> = insert_burst_ids(&db);
        assert_eq!(paged, expected);

        let added: Vec<i64> = page_ids(&db, &SearchQuery::default(), 0, 25);
        assert_ne!(added, expected);
    }

    /// Ids in ascending order — matches the `taken_at`-descending order
    /// produced by `insert_burst`.
    fn insert_burst_ids(db: &Database) -> Vec<i64> {
        let mut ids: Vec<i64> = db
            .search_images(&SearchQuery::default())
            .unwrap()
            .iter()
            .map(|i| i.id)
            .collect();
        ids.sort_unstable();
        ids
    }

    #[test]
    fn taken_order_falls_back_to_added_at_when_undated() {
        let (_dir, db) = tmp_db();
        // One dated far in the past, one with no EXIF date at all: the
        // undated one is inserted now and so must sort first.
        let dated = PathBuf::from("/photos/dated.jpg");
        let undated = PathBuf::from("/photos/undated.jpg");
        db.insert_image(&dated, &fake_hash(1), 1024).unwrap();
        db.insert_image(&undated, &fake_hash(2), 1024).unwrap();
        let dated_id = db.image_id_for_path(&dated).unwrap().unwrap();
        db.update_metadata(dated_id, &ImageMetadata { taken_at: Some(1), ..Default::default() })
            .unwrap();

        let rows = db
            .search_images(&SearchQuery::default().with_order(SearchOrder::TakenDesc))
            .unwrap();
        assert_eq!(rows[1].id, dated_id);
    }

    #[test]
    fn short_page_signals_the_end_of_the_listing() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 12);

        assert_eq!(page_ids(&db, &SearchQuery::default(), 0, 10).len(), 10);
        assert_eq!(page_ids(&db, &SearchQuery::default(), 1, 10).len(), 2);
        assert!(page_ids(&db, &SearchQuery::default(), 2, 10).is_empty());
    }

    #[test]
    fn text_search_pages_concatenate_and_keep_order() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 12);
        db.insert_image(&PathBuf::from("/photos/other.jpg"), &fake_hash(200), 1024).unwrap();

        let q = SearchQuery::default().with_text("burst").with_order(SearchOrder::TakenDesc);
        let all: Vec<i64> = db.search_images(&q).unwrap().iter().map(|i| i.id).collect();
        assert_eq!(all.len(), 12);

        let mut paged = page_ids(&db, &q, 0, 5);
        paged.extend(page_ids(&db, &q, 1, 5));
        paged.extend(page_ids(&db, &q, 2, 5));
        assert_eq!(paged, all);
    }

    #[test]
    fn count_images_ignores_limit_and_offset() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 12);

        let q = SearchQuery::default().with_limit(5).with_offset(5);
        assert_eq!(db.count_images(&q).unwrap(), Some(12));
    }

    #[test]
    fn count_images_matches_the_text_listing() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 12);
        db.insert_image(&PathBuf::from("/photos/other.jpg"), &fake_hash(200), 1024).unwrap();

        let q = SearchQuery::default().with_text("burst");
        assert_eq!(db.count_images(&q).unwrap(), Some(12));
        assert_eq!(db.count_images(&SearchQuery::default()).unwrap(), Some(13));
        // Blank-ish text has no usable tokens: the listing is empty, so is the count.
        let blank = SearchQuery { text: Some("   ".into()), ..Default::default() };
        assert_eq!(db.count_images(&blank).unwrap(), Some(0));
        assert!(db.search_images(&blank).unwrap().is_empty());
    }

    /// `EXPLAIN QUERY PLAN` for a listing, as one string.
    fn listing_plan(
        db: &Database,
        order: SearchOrder,
        collection_id: Option<i64>,
        person_id: Option<i64>,
    ) -> String {
        use rusqlite::types::Value;

        let (from_where, mut params) = all_from_where(Entry::Index, collection_id, person_id);
        let order_by = crate::query::order_by_sql(order);
        let sql = format!(
            "EXPLAIN QUERY PLAN \
             SELECT {IMAGE_COLUMNS} {from_where} {order_by} LIMIT ? OFFSET ?"
        );
        params.push(Value::Integer(10));
        params.push(Value::Integer(0));

        let mut stmt = db.conn.prepare(&sql).unwrap();
        stmt.query_map(rusqlite::params_from_iter(params), |r| r.get::<_, String>(3))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>()
            .join(" | ")
    }

    #[test]
    fn listing_orderings_are_served_by_an_index() {
        // The grid re-issues the listing on every scroll, so it must never
        // sort the whole table. SQLite matches `idx_images_listing_taken`
        // only when the query spells its indexed expression exactly the way
        // the DDL does, and it cannot use either index at all if the
        // stack-cover test reaches across a join — both regressions are
        // silent, showing up only as a grid that crawls on a big library.
        let (_dir, db) = tmp_db();
        insert_burst(&db, 25);
        let ids = insert_burst_ids(&db);
        let coll = db.create_collection("trip", "#fff", None).unwrap();
        db.add_image_to_collection(coll, ids[0]).unwrap();

        for order in [SearchOrder::AddedDesc, SearchOrder::TakenDesc] {
            for (label, cid, pid) in
                [("unfiltered", None, None), ("collection", Some(coll), None), ("person", None, Some(1))]
            {
                let plan = listing_plan(&db, order, cid, pid);
                assert!(
                    !plan.contains("TEMP B-TREE FOR ORDER BY"),
                    "{order:?}/{label} sorts the whole table: {plan}"
                );
                assert!(
                    plan.contains("idx_images_listing_"),
                    "{order:?}/{label} does not reach the listing index: {plan}"
                );
            }
        }
    }

    #[test]
    fn count_reads_the_table_not_an_ordering_index() {
        // The count has no ORDER BY to serve and needs `stack_id` for every
        // present row, which the ordering indexes do not carry — walking one
        // costs a random row fetch per row. See `Entry`.
        let (_dir, db) = tmp_db();
        insert_burst(&db, 25);

        let (from_where, params) = all_from_where(Entry::Table, None, None);
        let mut stmt = db
            .conn
            .prepare(&format!("EXPLAIN QUERY PLAN SELECT COUNT(*) {from_where}"))
            .unwrap();
        let plan = stmt
            .query_map(rusqlite::params_from_iter(params), |r| r.get::<_, String>(3))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect::<Vec<_>>()
            .join(" | ");

        assert!(
            !plan.contains("idx_images_listing_"),
            "count should not detour through an ordering index: {plan}"
        );
    }

    #[test]
    fn filtered_listings_page_without_gaps_or_repeats() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 12);
        let ids = insert_burst_ids(&db);

        let coll = db.create_collection("trip", "#fff", None).unwrap();
        for id in ids.iter().take(7) {
            db.add_image_to_collection(coll, *id).unwrap();
        }

        for order in [SearchOrder::AddedDesc, SearchOrder::TakenDesc] {
            let q = SearchQuery::default().with_collection(coll).with_order(order);
            let all: Vec<i64> =
                db.search_images(&q).unwrap().iter().map(|i| i.id).collect();
            assert_eq!(all.len(), 7);
            assert_eq!(db.count_images(&q).unwrap(), Some(7));

            let mut paged = page_ids(&db, &q, 0, 3);
            paged.extend(page_ids(&db, &q, 1, 3));
            paged.extend(page_ids(&db, &q, 2, 3));
            assert_eq!(paged, all, "{order:?} filtered pages must tile the listing");
        }
    }

    #[test]
    fn count_images_counts_a_stack_once() {
        let (_dir, db) = tmp_db();
        insert_burst(&db, 3);
        let ids = insert_burst_ids(&db);
        let stack_id = db.create_stack().unwrap();
        for id in &ids {
            db.set_image_stack(*id, Some(stack_id)).unwrap();
        }

        let listed = db.search_images(&SearchQuery::default()).unwrap().len();
        assert_eq!(listed, 1);
        assert_eq!(db.count_images(&SearchQuery::default()).unwrap(), Some(listed));
    }
}
