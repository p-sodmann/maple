//! SQL schema constants and incremental migration runner.
//!
//! Versions are stored in `PRAGMA user_version`:
//!   0 → 1 : base `images` table + hash index
//!   1 → 2 : metadata columns + FTS5 index + sync triggers + backfill
//!   2 → 3 : ai_descriptions table (per-model AI-generated descriptions)
//!   3 → 4 : persons + face_detections tables (ONNX face recognition)

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

    Ok(())
}
