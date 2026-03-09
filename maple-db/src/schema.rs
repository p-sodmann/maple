//! SQL schema constants and incremental migration runner.
//!
//! Versions are stored in `PRAGMA user_version`:
//!   0 → 1 : base `images` table + hash index
//!   1 → 2 : metadata columns + FTS5 index + sync triggers + backfill

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

    Ok(())
}
