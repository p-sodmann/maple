//! Read locally-changed rows for shipping to a peer.
//!
//! A pull is "everything with `rev >` the watermark you last acknowledged".
//! Because `rev` comes from a monotonic clock shared by every table, one
//! watermark per peer covers the whole library.
//!
//! # Why batches end on a whole `rev` group
//!
//! Several writes stamp many rows with a *single* `rev`: the stacker's bulk
//! `UPDATE images SET stack_id = NULL`, and `clear_all_ai_descriptions`,
//! which can tombstone an entire table at one stamp. A naive `LIMIT n` would
//! cut such a group in half, and the receiver — having acknowledged the
//! watermark — would never ask for the remainder.
//!
//! So the batch boundary is chosen as a *rev value*, not a row count: take
//! the first `max_revs` distinct stamps above the watermark and ship every
//! row at or below the last of them. A batch can therefore exceed `max_revs`
//! rows, which is the point — a stamp group is atomic.

use std::collections::HashSet;

use rusqlite::{params, OptionalExtension, Row};

use super::wire::{
    AiDescriptionRow, CollectionImageRow, CollectionRow, Entity, FaceRow, GuidAlias, ImageRow,
    PersonRow, StackRow, Stamp, SyncBatch, SyncRow, Tombstone,
};
use crate::Database;

/// Default number of distinct stamps per batch.
pub const DEFAULT_MAX_REVS: usize = 500;

/// Upper bound on `max_revs`; see the clamp in [`Database::collect_changes`].
pub const MAX_REVS_LIMIT: usize = 1_000_000;

impl Database {
    /// Collect every replicated change with `rev > since`, up to a boundary
    /// that never splits a stamp group.
    ///
    /// The returned [`SyncBatch::next_rev`] is the peer's new watermark: the
    /// batch is complete for every stamp at or below it.
    pub fn collect_changes(&self, since: i64, max_revs: usize) -> anyhow::Result<SyncBatch> {
        // Clamped, not just floored: `max_revs` becomes a SQL `OFFSET`, and a
        // caller passing `usize::MAX` to mean "everything" would otherwise
        // wrap to a negative offset, which SQLite reads as zero — quietly
        // returning the *first* stamp instead of all of them.
        let ceiling = self.batch_ceiling(since, max_revs.clamp(1, MAX_REVS_LIMIT))?;
        if ceiling <= since {
            return Ok(SyncBatch {
                next_rev: since,
                ..Default::default()
            });
        }

        let mut rows = Vec::new();
        rows.extend(self.collect_images(since, ceiling)?);
        rows.extend(self.collect_stacks(since, ceiling)?);
        rows.extend(self.collect_persons(since, ceiling)?);
        rows.extend(self.collect_collections(since, ceiling)?);
        rows.extend(self.collect_ai_descriptions(since, ceiling)?);
        rows.extend(self.collect_faces(since, ceiling)?);
        rows.extend(self.collect_collection_images(since, ceiling)?);
        self.backfill_parents(&mut rows)?;

        Ok(SyncBatch {
            rows,
            tombstones: self.collect_tombstones(since, ceiling)?,
            aliases: self.collect_aliases(since, ceiling)?,
            next_rev: ceiling,
        })
    }

    /// Add the mandatory parents of any child row in `rows` that the stamp
    /// window did not already include.
    ///
    /// A batch has to stand on its own. A child row cannot be inserted
    /// without its parent, and the receiver defers one it cannot resolve —
    /// but the watermark still advances past the whole batch, so a deferred
    /// row is never re-sent. That is silent data loss.
    ///
    /// It is not a rare ordering fluke either: a child's stamp is routinely
    /// *older* than its parent's, because an identity merge re-stamps the
    /// image while the membership pointing at it keeps its original stamp.
    /// Whenever the two fall either side of a batch boundary, the child is
    /// dropped. Shipping the parents unconditionally removes the possibility.
    ///
    /// The extra rows cost nothing on the receiving side: applying a row
    /// whose stamp is not newer than the local copy is a no-op.
    fn backfill_parents(&self, rows: &mut Vec<SyncRow>) -> anyhow::Result<()> {
        let present: HashSet<String> = rows.iter().map(|r| r.guid().to_owned()).collect();
        let mut images: HashSet<String> = HashSet::new();
        let mut collections: HashSet<String> = HashSet::new();

        for row in rows.iter() {
            match row {
                SyncRow::AiDescription(r) => {
                    images.insert(r.image_guid.clone());
                }
                SyncRow::FaceDetection(r) => {
                    images.insert(r.image_guid.clone());
                }
                SyncRow::CollectionImage(r) => {
                    images.insert(r.image_guid.clone());
                    collections.insert(r.collection_guid.clone());
                }
                _ => {}
            }
        }

        for guid in images.difference(&present) {
            if let Some(row) = self.image_row_by_guid(guid)? {
                rows.push(row);
            }
        }
        for guid in collections.difference(&present) {
            if let Some(row) = self.collection_row_by_guid(guid)? {
                rows.push(row);
            }
        }
        Ok(())
    }

    fn image_row_by_guid(&self, guid: &str) -> anyhow::Result<Option<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.guid, i.rev, i.rev_dev, i.hash, i.orientation, i.taken_at,
                    i.make, i.model, i.lens, i.focal_length, i.aperture, i.iso,
                    i.width, i.height, s.guid, i.path, i.file_size
             FROM images i
             LEFT JOIN stacks s ON s.id = i.stack_id
             WHERE i.guid = ?1",
        )?;
        Ok(stmt.query_row(params![guid], image_row).optional()?)
    }

    fn collection_row_by_guid(&self, guid: &str) -> anyhow::Result<Option<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.guid, c.rev, c.rev_dev, c.name, c.color, c.created_at, p.guid
             FROM collections c
             LEFT JOIN collections p ON p.id = c.parent_id
             WHERE c.guid = ?1",
        )?;
        Ok(stmt.query_row(params![guid], collection_row).optional()?)
    }

    fn collect_aliases(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<GuidAlias>> {
        let mut stmt = self.conn.prepare(
            "SELECT alias, guid, rev, rev_dev FROM sync_guid_aliases
             WHERE rev > ?1 AND rev <= ?2 ORDER BY rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(GuidAlias {
                    alias: r.get(0)?,
                    guid: r.get(1)?,
                    stamp: Stamp::new(r.get::<_, i64>(2)?, r.get::<_, String>(3)?),
                })
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    /// The highest stamp this batch may include: the `max_revs`-th distinct
    /// stamp above `since`, or the global maximum when fewer exist.
    fn batch_ceiling(&self, since: i64, max_revs: usize) -> anyhow::Result<i64> {
        let union = SOURCES
            .iter()
            .map(|t| format!("SELECT DISTINCT rev FROM {t} WHERE rev > ?1"))
            .collect::<Vec<_>>()
            .join(" UNION ");

        // OFFSET max_revs - 1 lands on the last stamp we are willing to take.
        let nth: Option<i64> = self
            .conn
            .query_row(
                &format!("SELECT rev FROM ({union}) ORDER BY rev LIMIT 1 OFFSET ?2"),
                params![since, (max_revs - 1) as i64],
                |r| r.get(0),
            )
            .optional()?;

        if let Some(rev) = nth {
            return Ok(rev);
        }
        // Fewer than `max_revs` distinct stamps pending — take everything.
        let max: Option<i64> = self.conn.query_row(
            &format!("SELECT MAX(rev) FROM ({union})"),
            params![since],
            |r| r.get(0),
        )?;
        Ok(max.unwrap_or(since))
    }

    fn collect_images(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT i.guid, i.rev, i.rev_dev, i.hash, i.orientation, i.taken_at,
                    i.make, i.model, i.lens, i.focal_length, i.aperture, i.iso,
                    i.width, i.height, s.guid, i.path, i.file_size
             FROM images i
             LEFT JOIN stacks s ON s.id = i.stack_id
             WHERE i.rev > ?1 AND i.rev <= ?2 AND i.guid IS NOT NULL
             ORDER BY i.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], image_row)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_stacks(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT s.guid, s.rev, s.rev_dev, s.created_at, c.guid
             FROM stacks s
             LEFT JOIN images c ON c.id = s.cover_image_id
             WHERE s.rev > ?1 AND s.rev <= ?2 AND s.guid IS NOT NULL
             ORDER BY s.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(SyncRow::Stack(StackRow {
                    guid: r.get(0)?,
                    stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
                    created_at: r.get(3)?,
                    cover_image_guid: r.get(4)?,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_persons(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT guid, rev, rev_dev, name, created_at
             FROM persons
             WHERE rev > ?1 AND rev <= ?2 AND guid IS NOT NULL
             ORDER BY rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(SyncRow::Person(PersonRow {
                    guid: r.get(0)?,
                    stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
                    name: r.get(3)?,
                    created_at: r.get(4)?,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_collections(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT c.guid, c.rev, c.rev_dev, c.name, c.color, c.created_at, p.guid
             FROM collections c
             LEFT JOIN collections p ON p.id = c.parent_id
             WHERE c.rev > ?1 AND c.rev <= ?2 AND c.guid IS NOT NULL
             ORDER BY c.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], collection_row)?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_ai_descriptions(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        // An INNER join drops descriptions whose image has no guid; such a
        // row cannot be addressed on the wire, and the peer would reject it.
        let mut stmt = self.conn.prepare(
            "SELECT a.guid, a.rev, a.rev_dev, i.guid, a.model_id, a.description, a.created_at
             FROM ai_descriptions a
             JOIN images i ON i.id = a.image_id
             WHERE a.rev > ?1 AND a.rev <= ?2 AND a.guid IS NOT NULL AND i.guid IS NOT NULL
             ORDER BY a.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(SyncRow::AiDescription(AiDescriptionRow {
                    guid: r.get(0)?,
                    stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
                    image_guid: r.get(3)?,
                    model_id: r.get(4)?,
                    description: r.get(5)?,
                    created_at: r.get(6)?,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_faces(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT f.guid, f.rev, f.rev_dev, i.guid,
                    f.bbox_x1, f.bbox_y1, f.bbox_x2, f.bbox_y2,
                    f.embedding, f.confidence, f.skipped, p.guid
             FROM face_detections f
             JOIN images i ON i.id = f.image_id
             LEFT JOIN persons p ON p.id = f.person_id
             WHERE f.rev > ?1 AND f.rev <= ?2 AND f.guid IS NOT NULL AND i.guid IS NOT NULL
             ORDER BY f.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(SyncRow::FaceDetection(FaceRow {
                    guid: r.get(0)?,
                    stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
                    image_guid: r.get(3)?,
                    bbox: [r.get(4)?, r.get(5)?, r.get(6)?, r.get(7)?],
                    embedding: r.get(8)?,
                    confidence: r.get(9)?,
                    skipped: r.get::<_, i64>(10)? != 0,
                    person_guid: r.get(11)?,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_collection_images(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<SyncRow>> {
        let mut stmt = self.conn.prepare(
            "SELECT ci.guid, ci.rev, ci.rev_dev, c.guid, i.guid, ci.added_at
             FROM collection_images ci
             JOIN collections c ON c.id = ci.collection_id
             JOIN images i      ON i.id = ci.image_id
             WHERE ci.rev > ?1 AND ci.rev <= ?2 AND ci.guid IS NOT NULL
               AND c.guid IS NOT NULL AND i.guid IS NOT NULL
             ORDER BY ci.rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                Ok(SyncRow::CollectionImage(CollectionImageRow {
                    guid: r.get(0)?,
                    stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
                    collection_guid: r.get(3)?,
                    image_guid: r.get(4)?,
                    added_at: r.get(5)?,
                }))
            })?
            .filter_map(|r| r.ok())
            .collect();
        Ok(rows)
    }

    fn collect_tombstones(&self, since: i64, ceiling: i64) -> anyhow::Result<Vec<Tombstone>> {
        let mut stmt = self.conn.prepare(
            "SELECT guid, entity, rev, rev_dev FROM sync_tombstones
             WHERE rev > ?1 AND rev <= ?2 ORDER BY rev",
        )?;
        let rows = stmt
            .query_map(params![since, ceiling], |r| {
                let table: String = r.get(1)?;
                Ok((
                    r.get::<_, String>(0)?,
                    table,
                    r.get::<_, i64>(2)?,
                    r.get::<_, String>(3)?,
                ))
            })?
            .filter_map(|r| r.ok())
            .filter_map(|(guid, table, rev, rev_dev)| {
                Some(Tombstone {
                    guid,
                    entity: Entity::from_table(&table)?,
                    stamp: Stamp::new(rev, rev_dev),
                })
            })
            .collect();
        Ok(rows)
    }
}

/// Map one `images` row (joined to its stack) into a [`SyncRow`].
///
/// Shared by the stamp-window query and the parent backfill so the two can
/// never drift into selecting different columns.
fn image_row(r: &Row<'_>) -> rusqlite::Result<SyncRow> {
    let hash_blob: Vec<u8> = r.get(3)?;
    let mut hash = [0u8; 32];
    if hash_blob.len() == 32 {
        hash.copy_from_slice(&hash_blob);
    }
    Ok(SyncRow::Image(ImageRow {
        guid: r.get(0)?,
        stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
        hash,
        orientation: r.get(4)?,
        taken_at: r.get(5)?,
        make: r.get(6)?,
        model: r.get(7)?,
        lens: r.get(8)?,
        focal_length: r.get(9)?,
        aperture: r.get(10)?,
        iso: r.get(11)?,
        width: r.get(12)?,
        height: r.get(13)?,
        stack_guid: r.get(14)?,
        origin_path: r.get(15)?,
        file_size: r.get(16)?,
    }))
}

/// Map one `collections` row (joined to its parent) into a [`SyncRow`].
fn collection_row(r: &Row<'_>) -> rusqlite::Result<SyncRow> {
    Ok(SyncRow::Collection(CollectionRow {
        guid: r.get(0)?,
        stamp: Stamp::new(r.get::<_, i64>(1)?, r.get::<_, String>(2)?),
        name: r.get(3)?,
        color: r.get(4)?,
        created_at: r.get(5)?,
        parent_guid: r.get(6)?,
    }))
}

/// Every table carrying a `rev`, including tombstones — the inputs to the
/// batch-boundary calculation.
const SOURCES: &[&str] = &[
    "images",
    "ai_descriptions",
    "persons",
    "face_detections",
    "collections",
    "collection_images",
    "stacks",
    "sync_tombstones",
    "sync_guid_aliases",
];
