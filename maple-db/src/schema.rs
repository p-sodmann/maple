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
//!   13 → 14: parent_id column on collections (hierarchy)
//!   14 → 15: image_exif_tags table (comprehensive per-image EXIF tag/value
//!            pairs) + exif_extracted column (explicit extraction-state gate)
//!   15 → 16: centroid_embedding + representative_image_id columns on
//!            collections (computed cover image, mirrors persons' V13)
//!   16 → 17: library-listing indexes on (status, added_at) and
//!            (status, COALESCE(taken_at, added_at)); drops the never-read
//!            FTS5 index and its write triggers
//!
//! Steps are append-only and replay history, so a fresh database still runs
//! V2's `image_fts` creation before V17 drops it again.

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
// Stores a dense DINOv2 image embedding for each image, keyed by an algorithm
// string that encodes the model repo (e.g. "onnx:onnx-community/dinov2-small").
//
// When `model_repo` changes (settings.toml edit), the key changes and old
// rows are ignored.  The background hasher fills in the new rows
// automatically.  Old rows from unused algorithms are left in place (harmless)
// and may be cleaned up by a future migration.
//
// `hash_blob` encoding: Vec<f32> as little-endian bytes (same layout as face
// embeddings).

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

// ── V12: starred import source paths ─────────────────────────────
//
// Persists which file-system paths the user has starred on the Import page.
// One row per starred path; starred = shown in "Favorites", unstarred = "Recent".

const V12: &str = "
    CREATE TABLE IF NOT EXISTS import_starred_paths (
        id         INTEGER PRIMARY KEY,
        path       TEXT    NOT NULL UNIQUE,
        created_at INTEGER NOT NULL
    );
";

// ── V13: person centroids + representative faces ──────────────────
//
// Each person row gains:
//   `centroid_embedding` — the mean of all its assigned face embeddings
//     (L2-normalised 512-dim f32, updated after every assignment).
//   `representative_face_id` — the face_detection whose embedding is closest
//     to the centroid; used as the avatar on the People page.
//
// Both are NULL until at least one face has been assigned and
// `Database::update_person_representative` is called.

const V13_CENTROID: &str =
    "ALTER TABLE persons ADD COLUMN centroid_embedding BLOB";
const V13_REP: &str =
    "ALTER TABLE persons ADD COLUMN representative_face_id INTEGER \
     REFERENCES face_detections(id) ON DELETE SET NULL";

// ── V15: comprehensive EXIF tags ──────────────────────────────────
//
// `images` only has dedicated columns for a curated subset of EXIF fields
// (make, model, lens, focal length, aperture, ISO, …). `image_exif_tags`
// stores every other standard EXIF tag the extractor can read — shutter
// speed, exposure program, flash, white balance, GPS coordinates, etc. —
// as human-readable name/value pairs, one row per (image, tag).
//
// `exif_extracted` replaces the old `filename IS NULL` gate that
// `records_needing_metadata` used to find rows the background metadata
// filler hadn't processed yet. That gate stopped working once
// `insert_image_with_raw` started setting `filename` immediately at insert
// time (so filename search works before EXIF is read) — it made the gate
// permanently false. This column tracks extraction state explicitly and is
// set by `update_metadata`.

const V15: &str = "
    CREATE TABLE IF NOT EXISTS image_exif_tags (
        id       INTEGER PRIMARY KEY,
        image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
        tag      TEXT    NOT NULL,
        value    TEXT    NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_exif_tags_image_tag ON image_exif_tags(image_id, tag);
";

const V15_COLUMN: &str =
    "ALTER TABLE images ADD COLUMN exif_extracted INTEGER NOT NULL DEFAULT 0";

// ── V16: collection centroids + representative (cover) images ──────
//
// Mirrors V13 for persons, over whole-image DINOv2 embeddings
// (`image_hashes`) instead of face embeddings:
//   `centroid_embedding` — the mean of the collection's member images'
//     embeddings (L2-normalised, updated after every membership change).
//   `representative_image_id` — the member image whose embedding is closest
//     to the centroid; used as the cover on the Collections gallery. Falls
//     back to the most-recently-added member when no member has an embedding
//     yet (background hasher still running) — see
//     `Database::update_collection_representative`.

const V16_CENTROID: &str =
    "ALTER TABLE collections ADD COLUMN centroid_embedding BLOB";
const V16_REP: &str =
    "ALTER TABLE collections ADD COLUMN representative_image_id INTEGER \
     REFERENCES images(id) ON DELETE SET NULL";

// ── V17: library listing indexes; retire the unread FTS5 index ────
//
// The grid pages through `images` filtered on `status` and ordered by one of
// two keys, and endless scrolling re-issues that query on every scroll.
// Neither key was indexed, so SQLite sorted the whole table into a temp
// b-tree for every page.
//
// `idx_images_listing_taken` indexes an *expression*.  SQLite only matches an
// expression index when the query spells the expression the same way, so this
// DDL and `query::order_by_sql` have to stay textually in step — change one
// and the planner silently goes back to the temp b-tree.
//
// The trailing `id DESC` makes each key total.  Without it `LIMIT`/`OFFSET`
// paging drops and repeats rows across page boundaries, because `added_at`
// and `taken_at` tie constantly within one bulk import.
//
// The index alone is not enough: the listing's stack-cover test used to reach
// across a `LEFT JOIN` and blocked the planner from using it either way —
// see `STACK_COVER_PREDICATE` in `lib.rs`, which is what makes these pay.
//
// Deliberately *not* widened to carry `stack_id`, which would let
// `count_images` read it without touching the table: that makes the index
// covering, and the planner then prefers it over `idx_images_stack_id` for
// the per-row stack-size subquery — a scan of every present row per result
// row.  `count_images` opts out of these indexes instead; see `Entry` in
// `lib.rs`.

const V17_INDEXES: &str = "
    CREATE INDEX IF NOT EXISTS idx_images_listing_added
        ON images(status, added_at DESC, id DESC);
    CREATE INDEX IF NOT EXISTS idx_images_listing_taken
        ON images(status, COALESCE(taken_at, added_at) DESC, id DESC);
";

// `image_fts` (V2) was never read: text search matches `LIKE '%token%'`
// against `images`, `ai_descriptions`, `persons` and `image_exif_tags`
// directly.  The three sync triggers made every insert into `images` about
// 7× more expensive, and fired on *every* UPDATE — including the bulk
// `status` and `stack_id` writes of the background scanner and stacker — to
// maintain an index nothing queried.
//
// Recreating it later is a matter of restoring the V2 DDL; note that it
// covers only filename/make/model/lens, so a text search built on it would
// have to index descriptions, person names and EXIF tag values too before it
// could replace the LIKE path without losing matches.
const V17_DROP_FTS: &str = "
    DROP TRIGGER IF EXISTS images_fts_ai;
    DROP TRIGGER IF EXISTS images_fts_au;
    DROP TRIGGER IF EXISTS images_fts_ad;
    DROP TABLE IF EXISTS image_fts;
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

    if version < 12 {
        conn.execute_batch(V12)?;
        conn.execute_batch("PRAGMA user_version = 12")?;
    }

    if version < 13 {
        for sql in &[V13_CENTROID, V13_REP] {
            if let Err(e) = conn.execute_batch(sql) {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        conn.execute_batch("PRAGMA user_version = 13")?;
    }

    if version < 14 {
        if let Err(e) = conn.execute_batch(
            "ALTER TABLE collections ADD COLUMN parent_id INTEGER REFERENCES collections(id) ON DELETE SET NULL"
        ) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        conn.execute_batch("PRAGMA user_version = 14")?;
    }

    if version < 15 {
        conn.execute_batch(V15)?;
        if let Err(e) = conn.execute_batch(V15_COLUMN) {
            if !e.to_string().to_lowercase().contains("duplicate column") {
                return Err(e.into());
            }
        }
        conn.execute_batch("PRAGMA user_version = 15")?;
    }

    if version < 16 {
        for sql in &[V16_CENTROID, V16_REP] {
            if let Err(e) = conn.execute_batch(sql) {
                if !e.to_string().to_lowercase().contains("duplicate column") {
                    return Err(e.into());
                }
            }
        }
        conn.execute_batch("PRAGMA user_version = 16")?;
    }

    if version < 17 {
        // Builds the two indexes over the existing rows — a few hundred
        // milliseconds for a library of a couple of hundred thousand photos,
        // paid once, on the first launch after the upgrade.
        conn.execute_batch(V17_INDEXES)?;
        conn.execute_batch(V17_DROP_FTS)?;
        conn.execute_batch("PRAGMA user_version = 17")?;
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

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Database, SearchOrder, SearchQuery};

    fn user_version(conn: &Connection) -> i32 {
        conn.query_row("PRAGMA user_version", [], |r| r.get(0)).expect("user_version")
    }

    fn exists(conn: &Connection, name: &str) -> bool {
        conn.query_row("SELECT COUNT(*) FROM sqlite_master WHERE name = ?1", [name], |r| {
            r.get::<_, i64>(0)
        })
        .map(|n| n > 0)
        .unwrap_or(false)
    }

    /// Wind a freshly migrated database back to the V16 state: the listing
    /// indexes gone, `image_fts` and its triggers restored.
    fn rewind_to_v16(conn: &Connection) {
        conn.execute_batch(
            "DROP INDEX IF EXISTS idx_images_listing_added;
             DROP INDEX IF EXISTS idx_images_listing_taken;",
        )
        .expect("drop v17 indexes");
        conn.execute_batch(V2_FTS).expect("restore fts");
        conn.execute_batch(V2_FTS_BACKFILL).expect("backfill fts");
        conn.execute_batch("PRAGMA user_version = 16").expect("set version");
    }

    #[test]
    fn v16_database_with_rows_migrates_to_v17() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open");
        let conn = &db.conn;
        rewind_to_v16(conn);
        assert_eq!(user_version(conn), 16);
        assert!(exists(conn, "image_fts"));

        // A populated library: an unstacked image, a two-image stack with an
        // explicit cover, an undated image, and a missing one.
        conn.execute_batch(
            "INSERT INTO stacks(id, created_at) VALUES (1, 0);
             INSERT INTO images(id, path, hash, file_size, added_at, status, filename,
                                taken_at, stack_id)
             VALUES (1, '/p/a.jpg', X'01', 1, 100, 'present', 'a.jpg', 90,   NULL),
                    (2, '/p/b.jpg', X'02', 1, 101, 'present', 'b.jpg', 91,   1),
                    (3, '/p/c.jpg', X'03', 1, 102, 'present', 'c.jpg', 92,   1),
                    (4, '/p/d.jpg', X'04', 1, 103, 'present', 'd.jpg', NULL, NULL),
                    (5, '/p/e.jpg', X'05', 1, 104, 'missing', 'e.jpg', 94,   NULL);
             UPDATE stacks SET cover_image_id = 3 WHERE id = 1;",
        )
        .expect("seed");

        ensure_schema(conn).expect("migrate 16 → 17");

        assert_eq!(user_version(conn), 17);
        assert!(exists(conn, "idx_images_listing_added"));
        assert!(exists(conn, "idx_images_listing_taken"));
        // Dropping the virtual table must take its shadow tables with it.
        let leftovers: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name LIKE 'image_fts%'",
                [],
                |r| r.get(0),
            )
            .expect("leftovers");
        assert_eq!(leftovers, 0, "the unread FTS index and its shadow tables are gone");
        for trigger in ["images_fts_ai", "images_fts_au", "images_fts_ad"] {
            assert!(!exists(conn, trigger), "{trigger} should be dropped");
        }

        // Rows survive, and the listing still collapses the stack to its
        // explicit cover and hides the missing image.
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))
            .expect("count");
        assert_eq!(n, 5);
        let ids: Vec<i64> = db
            .search_images(&SearchQuery::default().with_order(SearchOrder::AddedDesc))
            .expect("search")
            .iter()
            .map(|i| i.id)
            .collect();
        assert_eq!(ids, vec![4, 3, 1]);

        // Re-running is a no-op, not an error.
        ensure_schema(conn).expect("idempotent");
        assert_eq!(user_version(conn), 17);
    }

    #[test]
    fn fresh_database_lands_on_v17_without_the_fts_index() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open");
        assert_eq!(user_version(&db.conn), 17);
        assert!(exists(&db.conn, "idx_images_listing_added"));
        assert!(exists(&db.conn, "idx_images_listing_taken"));
        assert!(!exists(&db.conn, "image_fts"));
    }
}
