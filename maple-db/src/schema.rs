//! SQL schema constants and incremental migration runner.
//!
//! Versions are stored in `PRAGMA user_version`:
//!   0 → 1 : base `images` table + hash index
//!   1 → 2 : metadata columns + FTS5 index + sync triggers + backfill
//!   2 → 3 : ai_descriptions table (per-model AI-generated descriptions)
//!   3 → 4 : persons + face_detections tables (ONNX face recognition)
//!   4 → 5 : raw_path column for companion RAW files
//!   5 → 6 : collections + collection_images tables
//!   6 → 7 : sentence_embeddings + semantic_meta + vec_sentences (sqlite-vec)
//!   7 → 8 : skipped column on face_detections (remember skipped faces)
//!   8 → 9 : stacks table + stack_id column on images (burst/similar grouping)
//!   9 → 10: image_hashes table (algorithm-keyed perceptual/embedding hashes)

use rusqlite::Connection;

// ── V1: base schema ──────────────────────────────────────────────

const V1: &str = "
    CREATE TABLE IF NOT EXISTS images (
        id        INTEGER PRIMARY KEY,
        path      TEXT    NOT NULL UNIQUE,
        hash      BLOB    NOT NULL,
        file_size INTEGER NOT NULL,
        added_at  INTEGER NOT NULL,
        status    TEXT    NOT NULL DEFAULT 'present'
    );
    CREATE INDEX IF NOT EXISTS idx_images_hash ON images(hash);
";

// ── V2: EXIF metadata columns ────────────────────────────────────

/// One ALTER TABLE per column — SQLite does not support multi-column ALTER.
/// Errors from already-present columns are swallowed (idempotent).
const V2_COLUMNS: &[&str] = &[
    "ALTER TABLE images ADD COLUMN filename     TEXT",
    "ALTER TABLE images ADD COLUMN taken_at     INTEGER",
    "ALTER TABLE images ADD COLUMN make         TEXT",
    "ALTER TABLE images ADD COLUMN model        TEXT",
    "ALTER TABLE images ADD COLUMN lens         TEXT",
    "ALTER TABLE images ADD COLUMN focal_length REAL",
    "ALTER TABLE images ADD COLUMN aperture     REAL",
    "ALTER TABLE images ADD COLUMN iso          INTEGER",
    "ALTER TABLE images ADD COLUMN width        INTEGER",
    "ALTER TABLE images ADD COLUMN height       INTEGER",
    "ALTER TABLE images ADD COLUMN orientation  INTEGER",
];

// ── V2: FTS5 virtual table + sync triggers ───────────────────────

const V2_FTS: &str = "
    CREATE VIRTUAL TABLE IF NOT EXISTS image_fts USING fts5(
        filename, make, model, lens,
        tokenize='unicode61'
    );

    -- Insert trigger: keep FTS in sync when a row is added.
    CREATE TRIGGER IF NOT EXISTS images_fts_ai
        AFTER INSERT ON images BEGIN
            INSERT INTO image_fts(rowid, filename, make, model, lens)
            VALUES (new.id, new.filename, new.make, new.model, new.lens);
        END;

    -- Update trigger: replace the FTS entry when metadata changes.
    CREATE TRIGGER IF NOT EXISTS images_fts_au
        AFTER UPDATE ON images BEGIN
            DELETE FROM image_fts WHERE rowid = old.id;
            INSERT INTO image_fts(rowid, filename, make, model, lens)
            VALUES (new.id, new.filename, new.make, new.model, new.lens);
        END;

    -- Delete trigger: remove the FTS entry when a row is deleted.
    CREATE TRIGGER IF NOT EXISTS images_fts_ad
        AFTER DELETE ON images BEGIN
            DELETE FROM image_fts WHERE rowid = old.id;
        END;
";

/// Backfill the FTS index for rows that existed before the triggers were created.
///
/// V2 only runs once (guarded by `PRAGMA user_version`), so the FTS table is
/// guaranteed to be empty here — no delete step required.
const V2_FTS_BACKFILL: &str = "
    INSERT INTO image_fts(rowid, filename, make, model, lens)
        SELECT id, filename, make, model, lens FROM images;
";

// ── V3: AI-generated descriptions ────────────────────────────────

/// One row per (image, model) pair — allows tracking multiple AI models.
const V3: &str = "
    CREATE TABLE IF NOT EXISTS ai_descriptions (
        id          INTEGER PRIMARY KEY,
        image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        model_id    TEXT    NOT NULL,
        description TEXT    NOT NULL,
        created_at  INTEGER NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_desc_image_model
        ON ai_descriptions(image_id, model_id);
";

// ── V4: persons + face detections ────────────────────────────────

/// Named identities and per-face ONNX embeddings.
///
/// `face_detections.embedding` stores 512 × f32 little-endian (2048 bytes).
/// `face_detections.bbox_*` are normalized [0, 1] coordinates.
const V4: &str = "
    CREATE TABLE IF NOT EXISTS persons (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL UNIQUE,
        created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS face_detections (
        id         INTEGER PRIMARY KEY,
        image_id   INTEGER NOT NULL REFERENCES images(id)  ON DELETE CASCADE,
        bbox_x1    REAL    NOT NULL,
        bbox_y1    REAL    NOT NULL,
        bbox_x2    REAL    NOT NULL,
        bbox_y2    REAL    NOT NULL,
        embedding  BLOB    NOT NULL,
        person_id  INTEGER REFERENCES persons(id) ON DELETE SET NULL,
        confidence REAL    NOT NULL DEFAULT 1.0
    );

    CREATE INDEX IF NOT EXISTS idx_face_det_image  ON face_detections(image_id);
    CREATE INDEX IF NOT EXISTS idx_face_det_person ON face_detections(person_id);
";

// ── V5: raw companion path ─────────────────────────────────────────
//
// When a standard image (JPG) has a raw companion (RAF), we store the
// raw file path alongside the display path instead of creating two rows.
// The migration also deduplicates existing RAF+JPG pairs (done in Rust).

const V5_COLUMN: &str = "ALTER TABLE images ADD COLUMN raw_path TEXT";

// ── V6: collections ──────────────────────────────────────────────

/// Named, colour-coded collections and a many-to-many link table.
const V6: &str = "
    CREATE TABLE IF NOT EXISTS collections (
        id         INTEGER PRIMARY KEY,
        name       TEXT    NOT NULL UNIQUE,
        color      TEXT    NOT NULL DEFAULT '#3584e4',
        created_at INTEGER NOT NULL
    );

    CREATE TABLE IF NOT EXISTS collection_images (
        id            INTEGER PRIMARY KEY,
        collection_id INTEGER NOT NULL REFERENCES collections(id) ON DELETE CASCADE,
        image_id      INTEGER NOT NULL REFERENCES images(id)      ON DELETE CASCADE,
        added_at      INTEGER NOT NULL
    );

    CREATE UNIQUE INDEX IF NOT EXISTS idx_coll_img_unique
        ON collection_images(collection_id, image_id);
    CREATE INDEX IF NOT EXISTS idx_coll_img_collection
        ON collection_images(collection_id);
    CREATE INDEX IF NOT EXISTS idx_coll_img_image
        ON collection_images(image_id);
";

// ── V7: sentence embeddings (semantic search) ────────────────────
//
// Each sentence of an AI description is embedded into a dense vector and
// stored in the `vec_sentences` sqlite-vec virtual table for KNN distance
// search.  `sentence_embeddings` holds the metadata and links each vector
// (by shared rowid) back to its image and description.
//
// `semantic_meta` records the active encoder model + vector dimension so the
// index can be rebuilt when the model changes (see `Database::ensure_vec_table`).
//
// Invalidation is handled by triggers (robust regardless of the foreign_keys
// pragma, which is not enabled): editing or deleting a description drops its
// sentence rows, and deleting a sentence row drops its vector.

/// Default vector dimension used when the table is first created (the
/// `all-MiniLM-L6-v2` default model).  The table is recreated at the correct
/// dimension when a model is actually loaded (see `ensure_vec_table`).
const V7_DEFAULT_DIM: i64 = 384;

const V7: &str = "
    CREATE TABLE IF NOT EXISTS sentence_embeddings (
        id             INTEGER PRIMARY KEY,
        image_id       INTEGER NOT NULL,
        description_id INTEGER NOT NULL,
        encoder_model  TEXT    NOT NULL,
        sentence_index INTEGER NOT NULL,
        text           TEXT    NOT NULL,
        created_at     INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sent_emb_desc  ON sentence_embeddings(description_id);
    CREATE INDEX IF NOT EXISTS idx_sent_emb_image ON sentence_embeddings(image_id);

    CREATE TABLE IF NOT EXISTS semantic_meta (
        id            INTEGER PRIMARY KEY CHECK (id = 1),
        encoder_model TEXT    NOT NULL,
        dim           INTEGER NOT NULL
    );

    -- Invalidate a description's sentence rows when its text changes…
    CREATE TRIGGER IF NOT EXISTS sentence_emb_desc_au
        AFTER UPDATE OF description ON ai_descriptions
        WHEN new.description <> old.description BEGIN
            DELETE FROM sentence_embeddings WHERE description_id = old.id;
        END;

    -- …or when the description row is deleted.
    CREATE TRIGGER IF NOT EXISTS sentence_emb_desc_ad
        AFTER DELETE ON ai_descriptions BEGIN
            DELETE FROM sentence_embeddings WHERE description_id = old.id;
        END;

    -- Deleting a sentence row drops its vector (rowid == sentence id).
    CREATE TRIGGER IF NOT EXISTS sentence_emb_vec_ad
        AFTER DELETE ON sentence_embeddings BEGIN
            DELETE FROM vec_sentences WHERE rowid = old.id;
        END;
";

/// Build the `vec_sentences` vec0 virtual-table DDL for a given dimension.
pub(crate) fn vec_table_ddl(dim: i64) -> String {
    format!(
        "CREATE VIRTUAL TABLE IF NOT EXISTS vec_sentences \
         USING vec0(embedding float[{dim}] distance_metric=cosine);"
    )
}

// ── V8: skipped flag on face_detections ─────────────────────────
//
// When the user clicks "Skip" in the face-tagging wizard the face is marked
// `skipped = 1` so it is excluded from the default untagged queue.  A
// separate "Review Skipped" mode shows only these faces.  Assigning a name
// to any face clears the flag automatically.

const V8_COLUMN: &str =
    "ALTER TABLE face_detections ADD COLUMN skipped INTEGER NOT NULL DEFAULT 0";

// ── V9: stacks ───────────────────────────────────────────────────
//
// A stack groups near-identical or semantically similar shots (burst, bracketed
// exposures, etc.) so the library grid can display them as a single tile with a
// count badge.  Each row in `stacks` is a named group; images reference it via
// `stack_id`.  The relationship is one-to-many: an image belongs to at most one
// stack, but a stack can contain many images.

const V9: &str = "
    CREATE TABLE IF NOT EXISTS stacks (
        id         INTEGER PRIMARY KEY,
        created_at INTEGER NOT NULL
    );
";

const V9_COLUMN: &str =
    "ALTER TABLE images ADD COLUMN stack_id INTEGER REFERENCES stacks(id) ON DELETE SET NULL";

const V9_INDEX: &str =
    "CREATE INDEX IF NOT EXISTS idx_images_stack_id ON images(stack_id)";

// ── V10: image_hashes ────────────────────────────────────────────
//
// Stores a perceptual hash or dense embedding for each image, keyed by an
// algorithm string that encodes both the method and its parameters (e.g.
// "phash:8" or "onnx:facebook/dinov2-with-registers-base").
//
// When the active algorithm changes (settings.toml edit), the key changes and
// old rows are ignored.  The background hasher fills in the new rows
// automatically.  Old rows from unused algorithms are left in place (harmless)
// and may be cleaned up by a future migration.
//
// `hash_blob` encoding:
//   phash  — ImageHash bytes (image_hasher::ImageHash::as_bytes())
//   onnx   — Vec<f32> as little-endian bytes (same layout as face embeddings)

const V10: &str = "
    CREATE TABLE IF NOT EXISTS image_hashes (
        id          INTEGER PRIMARY KEY,
        image_id    INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        algorithm   TEXT    NOT NULL,
        hash_blob   BLOB    NOT NULL,
        created_at  INTEGER NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_image_hashes_image_alg
        ON image_hashes(image_id, algorithm);
    CREATE INDEX IF NOT EXISTS idx_image_hashes_algorithm
        ON image_hashes(algorithm);
";

// ── V11: stack cover ─────────────────────────────────────────────
//
// Adds a `cover_image_id` column to `stacks` so the user can pick which
// image in a stack is displayed as the grid thumbnail.  NULL means "use the
// image with the lowest id" (the default first-imported shot).

const V11: &str =
    "ALTER TABLE stacks ADD COLUMN cover_image_id INTEGER REFERENCES images(id) ON DELETE SET NULL";

// ── Migration runner ─────────────────────────────────────────────

/// Apply all pending schema migrations to `conn`.
///
/// Safe to call on any database version — already-applied migrations are
/// detected via `PRAGMA user_version` and skipped.
pub fn ensure_schema(conn: &Connection) -> anyhow::Result<()> {
    let version: i32 = conn.query_row("PRAGMA user_version", [], |r| r.get(0))?;

    if version < 1 {
        conn.execute_batch(V1)?;
        conn.execute_batch("PRAGMA user_version = 1")?;
    }

    if version < 2 {
        for sql in V2_COLUMNS {
            if let Err(e) = conn.execute_batch(sql) {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        conn.execute_batch(V2_FTS)?;
        conn.execute_batch(V2_FTS_BACKFILL)?;
        conn.execute_batch("PRAGMA user_version = 2")?;
    }

    if version < 3 {
        conn.execute_batch(V3)?;
        conn.execute_batch("PRAGMA user_version = 3")?;
    }

    if version < 4 {
        conn.execute_batch(V4)?;
        conn.execute_batch("PRAGMA user_version = 4")?;
    }

    if version < 5 {
        if let Err(e) = conn.execute_batch(V5_COLUMN) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        dedup_raw_companions(conn)?;
        conn.execute_batch("PRAGMA user_version = 5")?;
    }

    if version < 6 {
        conn.execute_batch(V6)?;
        conn.execute_batch("PRAGMA user_version = 6")?;
    }

    if version < 7 {
        // Create the vec0 table first so the triggers that reference it
        // resolve against an existing object.  (Requires the sqlite-vec
        // extension, registered in `Database::open` before the connection.)
        conn.execute_batch(&vec_table_ddl(V7_DEFAULT_DIM))?;
        conn.execute_batch(V7)?;
        conn.execute_batch(&format!(
            "INSERT OR IGNORE INTO semantic_meta(id, encoder_model, dim) \
             VALUES (1, '', {V7_DEFAULT_DIM});"
        ))?;
        conn.execute_batch("PRAGMA user_version = 7")?;
    }

    if version < 8 {
        if let Err(e) = conn.execute_batch(V8_COLUMN) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        conn.execute_batch("PRAGMA user_version = 8")?;
    }

    if version < 9 {
        conn.execute_batch(V9)?;
        if let Err(e) = conn.execute_batch(V9_COLUMN) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        conn.execute_batch(V9_INDEX)?;
        conn.execute_batch("PRAGMA user_version = 9")?;
    }

    if version < 10 {
        conn.execute_batch(V10)?;
        conn.execute_batch("PRAGMA user_version = 10")?;
    }

    if version < 11 {
        if let Err(e) = conn.execute_batch(V11) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        conn.execute_batch("PRAGMA user_version = 11")?;
    }

    Ok(())
}

/// One-shot V5 migration helper: find RAF rows whose stem matches a
/// JPG/PNG row in the same directory, move the RAF path into the JPG
/// row's `raw_path`, and delete the RAF row (cascading AI/face data).
fn dedup_raw_companions(conn: &Connection) -> anyhow::Result<()> {
    use std::collections::HashMap;
    use std::path::Path;

    // Collect all rows.
    let mut stmt = conn.prepare("SELECT id, path FROM images")?;
    let rows: Vec<(i64, String)> = stmt
        .query_map([], |r| Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?)))?
        .filter_map(|r| r.ok())
        .collect();

    // Group by (directory, lowercase stem).
    let mut groups: HashMap<(String, String), Vec<(i64, String)>> = HashMap::new();
    for (id, path_str) in &rows {
        let p = Path::new(path_str);
        let dir = p
            .parent()
            .map(|d| d.to_string_lossy().to_string())
            .unwrap_or_default();
        let stem = p
            .file_stem()
            .and_then(|s| s.to_str())
            .map(|s| s.to_ascii_lowercase())
            .unwrap_or_default();
        groups
            .entry((dir, stem))
            .or_default()
            .push((*id, path_str.clone()));
    }

    let tx = conn.unchecked_transaction()?;

    let mut merged = 0usize;
    for members in groups.values() {
        if members.len() < 2 {
            continue;
        }
        // Find the display file (non-raw) and raw file(s).
        let display = members
            .iter()
            .find(|(_, p)| !maple_import::is_raw_format(Path::new(p)));
        let raws: Vec<&(i64, String)> = members
            .iter()
            .filter(|(_, p)| maple_import::is_raw_format(Path::new(p)))
            .collect();

        if let (Some((display_id, _)), Some((raw_id, raw_path))) =
            (display, raws.first())
        {
            // Set raw_path on the display row.
            tx.execute(
                "UPDATE images SET raw_path = ?1 WHERE id = ?2",
                rusqlite::params![raw_path, display_id],
            )?;
            // Delete the raw row (ON DELETE CASCADE handles ai_descriptions
            // and face_detections).
            tx.execute(
                "DELETE FROM images WHERE id = ?1",
                rusqlite::params![raw_id],
            )?;
            merged += 1;
        }
    }

    tx.commit()?;

    if merged > 0 {
        tracing::info!("V5 migration: merged {merged} raw companion(s) into display rows");
    }
    Ok(())
}
