//! The last-write-wins merge engine.
//!
//! Applying a batch is deliberately *not* a straight loop over rows, because
//! replicated rows reference each other and the references form a cycle:
//! `images.stack_id → stacks.cover_image_id → images`, and
//! `collections.parent_id` points back into its own table. No ordering of a
//! single pass can satisfy all of them.
//!
//! Instead the work is split by what a row *needs* in order to exist:
//!
//! 1. **Tombstones.** Deletes go first so that a row later in the same batch
//!    can resurrect what a tombstone removed, rather than the reverse.
//! 2. **Parentless rows** — images, stacks, persons, collections. Their
//!    nullable foreign keys are left alone for now.
//! 3. **Rows with a mandatory parent** — descriptions, faces, memberships.
//!    Every parent they can reference exists by now; one that is still
//!    unknown means the peer has not sent it yet, so the row is deferred to a
//!    later sync rather than dropped.
//! 4. **Nullable foreign keys**, resolved guid → local rowid. This is where
//!    the cycles are broken, and it deliberately does not re-stamp: filling
//!    in a link is part of the same logical write, not a new edit.
//!
//! Applied rows keep the *originating* device's stamp verbatim. Re-stamping
//! them locally would mark every received row as locally modified and bounce
//! it straight back, so the two devices would trade the same rows forever.

use std::collections::HashMap;

use rusqlite::{params, OptionalExtension};

use super::wire::{Entity, Stamp, SyncBatch, SyncRow};
use crate::Database;

/// What an [`Database::apply_batch`] call did.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ApplyReport {
    pub inserted: usize,
    pub updated: usize,
    /// Incoming rows that lost to a newer local version.
    pub stale: usize,
    pub deleted: usize,
    /// Tombstones overruled by a newer local edit.
    pub resurrected: usize,
    /// Rows whose mandatory parent is not here yet; they arrive again next
    /// sync, because the watermark only advances past what was applied.
    pub deferred: usize,
    /// Images matched to an existing local row by content hash.
    pub unified: usize,
    /// Persons whose faces changed — the caller should refresh their
    /// centroid and representative face, which are derived and never synced.
    pub touched_persons: Vec<i64>,
    /// Collections whose membership changed, same reasoning.
    pub touched_collections: Vec<i64>,
}

impl ApplyReport {
    /// Whether anything at all changed locally.
    pub fn changed(&self) -> bool {
        self.inserted + self.updated + self.deleted + self.resurrected > 0
    }
}

impl Database {
    /// Merge a peer's batch into this library.
    ///
    /// The whole batch commits or none of it does, so a failure mid-apply
    /// cannot leave the library half-merged with the watermark advanced.
    pub fn apply_batch(&self, batch: &SyncBatch) -> anyhow::Result<ApplyReport> {
        // Advance the local clock past everything we are about to see, so a
        // subsequent local edit is unambiguously *after* these changes even
        // if the peer's wall clock runs ahead of ours.
        for row in &batch.rows {
            self.observe_remote_rev(row.stamp().rev);
        }
        for t in &batch.tombstones {
            self.observe_remote_rev(t.stamp.rev);
        }

        let tx = self.conn.unchecked_transaction()?;
        let mut report = ApplyReport::default();
        let mut ctx = ApplyCtx::default();
        for row in &batch.rows {
            if let SyncRow::Image(img) = row {
                *ctx.incoming_by_hash.entry(img.hash).or_insert(0) += 1;
            }
        }

        // Merge decisions first: a row arriving under a guid that has since
        // been aliased away must land on the surviving row, not create a
        // second one.
        for a in &batch.aliases {
            self.apply_alias(a, &mut report)?;
        }

        for t in &batch.tombstones {
            self.apply_tombstone(&t.entity, &t.guid, &t.stamp, &mut report)?;
        }

        // Pass 2 — rows that depend on nothing.
        for row in &batch.rows {
            match row {
                SyncRow::Image(_) | SyncRow::Stack(_) | SyncRow::Person(_)
                | SyncRow::Collection(_) => self.apply_row(row, &mut ctx, &mut report)?,
                _ => {}
            }
        }
        // Pass 3 — rows that cannot exist without a parent.
        for row in &batch.rows {
            match row {
                SyncRow::AiDescription(_) | SyncRow::FaceDetection(_)
                | SyncRow::CollectionImage(_) => self.apply_row(row, &mut ctx, &mut report)?,
                _ => {}
            }
        }
        // Pass 4 — the links that would have been circular.
        self.link_foreign_keys(batch, &mut ctx)?;
        // Only now do face rows have their `person_id`, so the derived-value
        // bookkeeping has to come after the links, not alongside the inserts.
        self.note_touched(batch, &mut ctx, &mut report)?;

        tx.commit()?;

        report.touched_persons.sort_unstable();
        report.touched_persons.dedup();
        report.touched_collections.sort_unstable();
        report.touched_collections.dedup();
        Ok(report)
    }

    // ── Identity merges ──────────────────────────────────────────

    /// Adopt a peer's decision that two guids name the same photo.
    ///
    /// Three shapes, depending on what this device happens to hold:
    /// only the losing row (rename it), both rows (fold one into the other),
    /// or neither/only the winner (just remember the alias).
    fn apply_alias(
        &self,
        alias: &super::wire::GuidAlias,
        report: &mut ApplyReport,
    ) -> anyhow::Result<()> {
        // The comparison is the *whole* stamp, not just `rev`. Both devices
        // can reach the same merge independently and stamp it on the same
        // millisecond; without the `rev_dev` tiebreak neither upsert fires,
        // each keeps its own attribution, and the two libraries disagree
        // forever over a row they actually agree about.
        self.conn.execute(
            "INSERT INTO sync_guid_aliases (alias, guid, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(alias) DO UPDATE SET guid = ?2, rev = ?3, rev_dev = ?4
             WHERE excluded.rev > sync_guid_aliases.rev
                OR (excluded.rev = sync_guid_aliases.rev
                    AND excluded.rev_dev > sync_guid_aliases.rev_dev)",
            params![alias.alias, alias.guid, alias.stamp.rev, alias.stamp.rev_dev],
        )?;

        let lost: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM images WHERE guid = ?1",
                params![alias.alias],
                |r| r.get(0),
            )
            .optional()?;
        let Some(lost) = lost else { return Ok(()) };

        let kept: Option<i64> = self
            .conn
            .query_row(
                "SELECT id FROM images WHERE guid = ?1",
                params![alias.guid],
                |r| r.get(0),
            )
            .optional()?;

        match kept {
            // Only the losing row is here — renaming preserves its children
            // and costs nothing.
            None => {
                self.conn.execute(
                    "UPDATE images SET guid = ?1 WHERE id = ?2",
                    params![alias.guid, lost],
                )?;
                // The renamed row now answers to a guid whose authoritative
                // version lives on the peer, carrying a stamp that is already
                // below the peer's watermark — so it will never be re-sent.
                // A fresh stamp makes *this* row the one that propagates,
                // which is what closes the loop.
                self.restamp("images", lost)?;
            }
            Some(kept) if kept != lost => {
                // Both rows exist, so one has to go. Move its children across
                // first — `OR IGNORE` drops any that would collide with an
                // equivalent row already on the survivor (same description
                // model, same collection), and the cascade clears those.
                for (child, fk) in [
                    ("ai_descriptions", "image_id"),
                    ("face_detections", "image_id"),
                    ("collection_images", "image_id"),
                ] {
                    self.conn.execute(
                        &format!("UPDATE OR IGNORE {child} SET {fk} = ?1 WHERE {fk} = ?2"),
                        params![kept, lost],
                    )?;
                }
                self.conn
                    .execute("DELETE FROM images WHERE id = ?1", params![lost])?;
                self.restamp("images", kept)?;
                report.unified += 1;
            }
            Some(_) => {}
        }
        Ok(())
    }

    // ── Tombstones ───────────────────────────────────────────────

    fn apply_tombstone(
        &self,
        entity: &Entity,
        guid: &str,
        stamp: &Stamp,
        report: &mut ApplyReport,
    ) -> anyhow::Result<()> {
        let table = entity.table();

        if let Some((id, local)) = self.row_stamp(table, guid)? {
            if local > *stamp {
                // Someone edited this row *after* the delete. The edit is the
                // later intent, so the row stays and the tombstone is dropped
                // — otherwise the delete would keep re-killing it every sync.
                self.conn
                    .execute("DELETE FROM sync_tombstones WHERE guid = ?1", params![guid])?;
                report.resurrected += 1;
                return Ok(());
            }

            if *entity == Entity::CollectionImage {
                if let Some(cid) = self.parent_of(table, id, "collection_id")? {
                    report.touched_collections.push(cid);
                }
            }
            if *entity == Entity::FaceDetection {
                if let Some(pid) = self.parent_of(table, id, "person_id")? {
                    report.touched_persons.push(pid);
                }
            }

            // Children go with it via ON DELETE CASCADE, which is why only
            // the parent row needs a tombstone on the wire.
            self.conn
                .execute(&format!("DELETE FROM {table} WHERE id = ?1"), params![id])?;
            report.deleted += 1;
        }

        // Record it locally even when the row was already absent: this device
        // may be relaying the delete on to a third one.
        //
        // Compares the full stamp — see the note in `apply_alias`; two
        // devices deleting the same row on the same millisecond would
        // otherwise keep disagreeing about who did it.
        self.conn.execute(
            "INSERT INTO sync_tombstones (guid, entity, rev, rev_dev)
             VALUES (?1, ?2, ?3, ?4)
             ON CONFLICT(guid) DO UPDATE SET rev = ?3, rev_dev = ?4
             WHERE excluded.rev > sync_tombstones.rev
                OR (excluded.rev = sync_tombstones.rev
                    AND excluded.rev_dev > sync_tombstones.rev_dev)",
            params![guid, table, stamp.rev, stamp.rev_dev],
        )?;
        Ok(())
    }

    // ── Rows ─────────────────────────────────────────────────────

    fn apply_row(
        &self,
        row: &SyncRow,
        ctx: &mut ApplyCtx,
        report: &mut ApplyReport,
    ) -> anyhow::Result<()> {
        let table = row.entity().table();
        let guid = row.guid();
        let stamp = row.stamp();

        // A delete that is newer than this version of the row wins.
        if let Some(t) = self.tombstone_stamp(guid)? {
            if t >= *stamp {
                report.stale += 1;
                return Ok(());
            }
            // The row is *newer* than the delete, so it is being resurrected
            // and the tombstone is spent. Leaving it behind is not cosmetic:
            // it keeps replicating, so the peer that resurrected the row
            // correctly would be told to delete it again on the next sync,
            // and the two devices trade the row back and forth forever.
            self.conn
                .execute("DELETE FROM sync_tombstones WHERE guid = ?1", params![guid])?;
            report.resurrected += 1;
        }

        let mut existing = self.row_stamp(table, guid)?;
        let mut merged = false;

        // Two devices that imported the same file independently minted
        // different guids for it. Reconcile them onto one identity instead of
        // letting the library grow a duplicate for every shared photo.
        //
        // Only when the match is unambiguous, though: `idx_images_hash` is
        // deliberately not unique, because a library legitimately holds the
        // same photo twice. With several candidates on either side there is
        // no way to pair them up that both devices would agree on, and
        // guessing makes the two libraries diverge permanently. Falling back
        // to "insert as its own row" is symmetric, so both sides converge on
        // the union — a visible duplicate is recoverable; divergence is not.
        if existing.is_none() {
            if let SyncRow::Image(img) = row {
                let unambiguous = ctx.incoming_by_hash.get(&img.hash).copied().unwrap_or(0) <= 1;
                if let Some(local_id) = self
                    .sole_image_id_by_hash(&img.hash)?
                    .filter(|_| unambiguous)
                {
                    let local_guid: Option<String> = self.conn.query_row(
                        "SELECT guid FROM images WHERE id = ?1",
                        params![local_id],
                        |r| r.get(0),
                    )?;
                    // Both sides run this and pick the same winner, so they
                    // converge without needing to agree who goes first.
                    if let Some(local_guid) = local_guid {
                        let (kept, lost) = if guid < local_guid.as_str() {
                            self.conn.execute(
                                "UPDATE images SET guid = ?1 WHERE id = ?2",
                                params![guid, local_id],
                            )?;
                            (guid, local_guid.as_str())
                        } else {
                            (local_guid.as_str(), guid)
                        };
                        // Rows referencing the losing guid were already
                        // shipped, and more will arrive naming it. Without
                        // this record they would resolve to nothing, be
                        // deferred, and never be re-sent — the watermark has
                        // moved past them.
                        //
                        // The alias is stamped so it replicates: the peer may
                        // hold two rows with this hash and therefore be
                        // unable to reach this conclusion on its own.
                        let (arev, adev) = self.stamp()?;
                        self.conn.execute(
                            "INSERT OR REPLACE INTO sync_guid_aliases (alias, guid, rev, rev_dev)
                             VALUES (?1, ?2, ?3, ?4)",
                            params![lost, kept, arev, adev],
                        )?;
                        ctx.guid_to_id.insert(guid.to_owned(), local_id);
                    }
                    existing = self.stamp_by_id(table, local_id)?.map(|s| (local_id, s));
                    merged = true;
                    report.unified += 1;
                }
            }
        }

        if let Some((id, local)) = &existing {
            if *local >= *stamp {
                report.stale += 1;
                let id = *id;
                ctx.guid_to_id.insert(guid.to_owned(), id);
                // Even when the incoming content loses, the *merge* is news
                // to the peer, so the surviving row still has to go back.
                if merged {
                    self.restamp(table, id)?;
                }
                return Ok(());
            }
        }

        let id = match existing {
            Some((id, _)) => {
                // A face moving between people invalidates *both* centroids,
                // so capture the outgoing one before the update overwrites it.
                if let SyncRow::FaceDetection(_) = row {
                    if let Some(pid) = self.parent_of("face_detections", id, "person_id")? {
                        report.touched_persons.push(pid);
                    }
                }
                self.update_row(row, id)?;
                report.updated += 1;
                id
            }
            None => match self.insert_row(row, ctx)? {
                Some(id) => {
                    report.inserted += 1;
                    id
                }
                None => {
                    report.deferred += 1;
                    return Ok(());
                }
            },
        };

        ctx.guid_to_id.insert(guid.to_owned(), id);
        // The merged row is this device's own conclusion, not something it
        // was told, so it needs a stamp that will carry it back to the peer.
        if merged {
            self.restamp(table, id)?;
        }
        Ok(())
    }

    /// Record which persons/collections need their derived columns refreshed.
    ///
    /// Runs after the linking pass: a face's `person_id` is a nullable
    /// foreign key resolved there, so reading it any earlier would always
    /// see `NULL` and silently report nothing to recompute.
    fn note_touched(
        &self,
        batch: &SyncBatch,
        ctx: &mut ApplyCtx,
        report: &mut ApplyReport,
    ) -> anyhow::Result<()> {
        for row in &batch.rows {
            let (table, column, guid) = match row {
                SyncRow::CollectionImage(r) => {
                    ("collection_images", "collection_id", &r.guid)
                }
                SyncRow::FaceDetection(r) => ("face_detections", "person_id", &r.guid),
                _ => continue,
            };
            let Some(id) = self.resolve(ctx, table, guid)? else {
                continue;
            };
            if let Some(parent) = self.parent_of(table, id, column)? {
                match row {
                    SyncRow::CollectionImage(_) => report.touched_collections.push(parent),
                    SyncRow::FaceDetection(_) => report.touched_persons.push(parent),
                    _ => {}
                }
            }
        }
        Ok(())
    }

    fn update_row(&self, row: &SyncRow, id: i64) -> anyhow::Result<()> {
        match row {
            SyncRow::Image(r) => {
                // `status`, `path`, `raw_path` and `filename` are absent by
                // design — they describe this machine's disk. `exif_extracted`
                // is set so the local metadata filler leaves the row alone.
                self.conn.execute(
                    "UPDATE images SET
                         hash = ?1, orientation = ?2, taken_at = ?3, make = ?4,
                         model = ?5, lens = ?6, focal_length = ?7, aperture = ?8,
                         iso = ?9, width = ?10, height = ?11,
                         exif_extracted = 1, rev = ?12, rev_dev = ?13
                     WHERE id = ?14",
                    params![
                        r.hash.as_slice(), r.orientation, r.taken_at, r.make, r.model,
                        r.lens, r.focal_length, r.aperture, r.iso, r.width, r.height,
                        r.stamp.rev, r.stamp.rev_dev, id
                    ],
                )?;
            }
            SyncRow::Stack(r) => {
                self.conn.execute(
                    "UPDATE stacks SET created_at = ?1, rev = ?2, rev_dev = ?3 WHERE id = ?4",
                    params![r.created_at, r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
            SyncRow::Person(r) => {
                self.conn.execute(
                    "UPDATE persons SET name = ?1, created_at = ?2, rev = ?3, rev_dev = ?4
                     WHERE id = ?5",
                    params![r.name, r.created_at, r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
            SyncRow::Collection(r) => {
                self.conn.execute(
                    "UPDATE collections SET name = ?1, color = ?2, created_at = ?3,
                         rev = ?4, rev_dev = ?5 WHERE id = ?6",
                    params![r.name, r.color, r.created_at, r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
            SyncRow::AiDescription(r) => {
                self.conn.execute(
                    "UPDATE ai_descriptions SET model_id = ?1, description = ?2,
                         created_at = ?3, rev = ?4, rev_dev = ?5 WHERE id = ?6",
                    params![r.model_id, r.description, r.created_at,
                            r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
            SyncRow::FaceDetection(r) => {
                self.conn.execute(
                    "UPDATE face_detections SET bbox_x1 = ?1, bbox_y1 = ?2, bbox_x2 = ?3,
                         bbox_y2 = ?4, embedding = ?5, confidence = ?6, skipped = ?7,
                         rev = ?8, rev_dev = ?9 WHERE id = ?10",
                    params![r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3], r.embedding,
                            r.confidence, r.skipped as i64, r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
            SyncRow::CollectionImage(r) => {
                self.conn.execute(
                    "UPDATE collection_images SET added_at = ?1, rev = ?2, rev_dev = ?3
                     WHERE id = ?4",
                    params![r.added_at, r.stamp.rev, r.stamp.rev_dev, id],
                )?;
            }
        }
        Ok(())
    }

    /// Insert a received row. `Ok(None)` means a mandatory parent is missing,
    /// so the row should be retried on a later sync.
    fn insert_row(&self, row: &SyncRow, ctx: &mut ApplyCtx) -> anyhow::Result<Option<i64>> {
        match row {
            SyncRow::Image(r) => {
                let name = r
                    .origin_path
                    .rsplit(['/', '\\'])
                    .next()
                    .unwrap_or("")
                    .to_owned();
                // `path` is UNIQUE and the peer's path may already name a
                // different local photo. A synthetic placeholder keeps the
                // row insertable; P6's `locality` column replaces this with a
                // first-class "this file lives on another device" marker.
                let path = if self.path_is_free(&r.origin_path)? {
                    r.origin_path.clone()
                } else {
                    format!("maple-remote://{}/{}", r.guid, name)
                };
                self.conn.execute(
                    "INSERT INTO images
                         (path, hash, file_size, added_at, status, filename, taken_at,
                          make, model, lens, focal_length, aperture, iso, width, height,
                          orientation, exif_extracted, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, 'missing', ?5, ?6, ?7, ?8, ?9, ?10, ?11,
                             ?12, ?13, ?14, ?15, 1, ?16, ?17, ?18)",
                    params![
                        path, r.hash.as_slice(), r.file_size, now_secs(), name, r.taken_at,
                        r.make, r.model, r.lens, r.focal_length, r.aperture, r.iso,
                        r.width, r.height, r.orientation, r.guid, r.stamp.rev, r.stamp.rev_dev
                    ],
                )?;
            }
            SyncRow::Stack(r) => {
                self.conn.execute(
                    "INSERT INTO stacks (created_at, guid, rev, rev_dev) VALUES (?1, ?2, ?3, ?4)",
                    params![r.created_at, r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
            SyncRow::Person(r) => {
                // `persons.name` is UNIQUE: if both devices independently
                // named someone "Ada", adopt the local row rather than
                // failing the whole batch on a constraint violation.
                if let Some(id) = self.person_id_by_name(&r.name)? {
                    self.conn.execute(
                        "UPDATE persons SET guid = ?1, created_at = ?2, rev = ?3, rev_dev = ?4
                         WHERE id = ?5",
                        params![r.guid, r.created_at, r.stamp.rev, r.stamp.rev_dev, id],
                    )?;
                    return Ok(Some(id));
                }
                self.conn.execute(
                    "INSERT INTO persons (name, created_at, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    params![r.name, r.created_at, r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
            SyncRow::Collection(r) => {
                if let Some(id) = self.collection_id_by_name(&r.name)? {
                    self.conn.execute(
                        "UPDATE collections SET guid = ?1, color = ?2, created_at = ?3,
                             rev = ?4, rev_dev = ?5 WHERE id = ?6",
                        params![r.guid, r.color, r.created_at, r.stamp.rev, r.stamp.rev_dev, id],
                    )?;
                    return Ok(Some(id));
                }
                self.conn.execute(
                    "INSERT INTO collections (name, color, created_at, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    params![r.name, r.color, r.created_at, r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
            SyncRow::AiDescription(r) => {
                let Some(image_id) = self.resolve(ctx, "images", &r.image_guid)? else {
                    return Ok(None);
                };
                self.conn.execute(
                    "INSERT INTO ai_descriptions
                         (image_id, model_id, description, created_at, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
                     ON CONFLICT(image_id, model_id) DO UPDATE SET
                         description = excluded.description,
                         created_at  = excluded.created_at,
                         guid        = excluded.guid,
                         rev         = excluded.rev,
                         rev_dev     = excluded.rev_dev",
                    params![image_id, r.model_id, r.description, r.created_at,
                            r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
            SyncRow::FaceDetection(r) => {
                let Some(image_id) = self.resolve(ctx, "images", &r.image_guid)? else {
                    return Ok(None);
                };
                self.conn.execute(
                    "INSERT INTO face_detections
                         (image_id, bbox_x1, bbox_y1, bbox_x2, bbox_y2, embedding,
                          confidence, skipped, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
                    params![image_id, r.bbox[0], r.bbox[1], r.bbox[2], r.bbox[3],
                            r.embedding, r.confidence, r.skipped as i64,
                            r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
            SyncRow::CollectionImage(r) => {
                let (Some(collection_id), Some(image_id)) = (
                    self.resolve(ctx, "collections", &r.collection_guid)?,
                    self.resolve(ctx, "images", &r.image_guid)?,
                ) else {
                    return Ok(None);
                };
                self.conn.execute(
                    "INSERT INTO collection_images
                         (collection_id, image_id, added_at, guid, rev, rev_dev)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)
                     ON CONFLICT(collection_id, image_id) DO UPDATE SET
                         added_at = excluded.added_at,
                         guid     = excluded.guid,
                         rev      = excluded.rev,
                         rev_dev  = excluded.rev_dev",
                    params![collection_id, image_id, r.added_at,
                            r.guid, r.stamp.rev, r.stamp.rev_dev],
                )?;
            }
        }
        Ok(Some(self.conn.last_insert_rowid()))
    }

    // ── Pass 4: nullable foreign keys ────────────────────────────

    fn link_foreign_keys(&self, batch: &SyncBatch, ctx: &mut ApplyCtx) -> anyhow::Result<()> {
        for row in &batch.rows {
            match row {
                SyncRow::Image(r) => {
                    self.link(ctx, "images", &r.guid, "stack_id", "stacks", r.stack_guid.as_deref())?
                }
                SyncRow::Stack(r) => self.link(
                    ctx, "stacks", &r.guid, "cover_image_id", "images",
                    r.cover_image_guid.as_deref(),
                )?,
                SyncRow::Collection(r) => self.link(
                    ctx, "collections", &r.guid, "parent_id", "collections",
                    r.parent_guid.as_deref(),
                )?,
                SyncRow::FaceDetection(r) => self.link(
                    ctx, "face_detections", &r.guid, "person_id", "persons",
                    r.person_guid.as_deref(),
                )?,
                _ => {}
            }
        }
        Ok(())
    }

    /// Point `table.column` at whatever `target_guid` resolves to locally.
    ///
    /// Does not touch `rev`: this completes the write that pass 2 or 3
    /// already stamped, and bumping it would send the row straight back.
    fn link(
        &self,
        ctx: &mut ApplyCtx,
        table: &str,
        guid: &str,
        column: &str,
        target_table: &str,
        target_guid: Option<&str>,
    ) -> anyhow::Result<()> {
        let Some(id) = self.resolve(ctx, table, guid)? else {
            return Ok(());
        };
        // An unresolvable target becomes NULL rather than being left at a
        // stale local value — the peer says this link is gone or points
        // somewhere we have not received yet, and a stale pointer is worse
        // than an absent one.
        let target = match target_guid {
            Some(g) => self.resolve(ctx, target_table, g)?,
            None => None,
        };
        self.conn.execute(
            &format!("UPDATE {table} SET {column} = ?1 WHERE id = ?2"),
            params![target, id],
        )?;
        Ok(())
    }

    // ── Lookups ──────────────────────────────────────────────────

    fn resolve(
        &self,
        ctx: &mut ApplyCtx,
        table: &str,
        guid: &str,
    ) -> anyhow::Result<Option<i64>> {
        if let Some(id) = ctx.guid_to_id.get(guid) {
            return Ok(Some(*id));
        }
        let mut id: Option<i64> = self
            .conn
            .query_row(
                &format!("SELECT id FROM {table} WHERE guid = ?1"),
                params![guid],
                |r| r.get(0),
            )
            .optional()?;

        // The guid may have lost an identity reconciliation; follow the
        // alias rather than treating the reference as dangling.
        if id.is_none() {
            let alias: Option<String> = self
                .conn
                .query_row(
                    "SELECT guid FROM sync_guid_aliases WHERE alias = ?1",
                    params![guid],
                    |r| r.get(0),
                )
                .optional()?;
            if let Some(alias) = alias {
                id = self
                    .conn
                    .query_row(
                        &format!("SELECT id FROM {table} WHERE guid = ?1"),
                        params![alias],
                        |r| r.get(0),
                    )
                    .optional()?;
            }
        }

        if let Some(id) = id {
            ctx.guid_to_id.insert(guid.to_owned(), id);
        }
        Ok(id)
    }

    fn row_stamp(&self, table: &str, guid: &str) -> anyhow::Result<Option<(i64, Stamp)>> {
        // Follow an alias before concluding the row is new. Skipping this
        // would insert a duplicate whenever a row arrives under a guid that
        // lost reconciliation *and* content-hash matching cannot save us —
        // an image edited since the merge no longer hashes the same.
        let guid = self.canonical_guid(guid)?;
        let row: Option<(i64, i64, Option<String>)> = self
            .conn
            .query_row(
                &format!("SELECT id, rev, rev_dev FROM {table} WHERE guid = ?1"),
                params![guid],
                |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)),
            )
            .optional()?;
        Ok(row.map(|(id, rev, dev)| (id, Stamp::new(rev, dev.unwrap_or_default()))))
    }

    /// Give `id` a fresh local stamp so the row re-propagates.
    ///
    /// Used only where this device has genuinely produced new information —
    /// an identity merge — and never for a row that was merely received.
    /// Re-stamping received rows would bounce every one of them straight
    /// back at the sender, forever.
    fn restamp(&self, table: &str, id: i64) -> anyhow::Result<()> {
        let (rev, rev_dev) = self.stamp()?;
        self.conn.execute(
            &format!("UPDATE {table} SET rev = ?1, rev_dev = ?2 WHERE id = ?3"),
            params![rev, rev_dev, id],
        )?;
        Ok(())
    }

    /// The guid actually in use for `guid`, following one alias hop.
    ///
    /// Aliases are written pointing at the winner, never chained onto another
    /// alias, so a single hop is always enough.
    fn canonical_guid(&self, guid: &str) -> anyhow::Result<String> {
        let alias: Option<String> = self
            .conn
            .query_row(
                "SELECT guid FROM sync_guid_aliases WHERE alias = ?1",
                params![guid],
                |r| r.get(0),
            )
            .optional()?;
        Ok(alias.unwrap_or_else(|| guid.to_owned()))
    }

    fn stamp_by_id(&self, table: &str, id: i64) -> anyhow::Result<Option<Stamp>> {
        let row: Option<(i64, Option<String>)> = self
            .conn
            .query_row(
                &format!("SELECT rev, rev_dev FROM {table} WHERE id = ?1"),
                params![id],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(row.map(|(rev, dev)| Stamp::new(rev, dev.unwrap_or_default())))
    }

    fn tombstone_stamp(&self, guid: &str) -> anyhow::Result<Option<Stamp>> {
        let row: Option<(i64, String)> = self
            .conn
            .query_row(
                "SELECT rev, rev_dev FROM sync_tombstones WHERE guid = ?1",
                params![guid],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .optional()?;
        Ok(row.map(|(rev, dev)| Stamp::new(rev, dev)))
    }

    /// The local image with this content hash, but only if it is the *only*
    /// one. Several matches mean the pairing is ambiguous; see the caller.
    fn sole_image_id_by_hash(&self, hash: &[u8; 32]) -> anyhow::Result<Option<i64>> {
        let mut stmt = self
            .conn
            .prepare("SELECT id FROM images WHERE hash = ?1 ORDER BY id LIMIT 2")?;
        let ids: Vec<i64> = stmt
            .query_map(params![hash.as_slice()], |r| r.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(match ids.len() {
            1 => Some(ids[0]),
            _ => None,
        })
    }

    fn person_id_by_name(&self, name: &str) -> anyhow::Result<Option<i64>> {
        Ok(self
            .conn
            .query_row("SELECT id FROM persons WHERE name = ?1", params![name], |r| {
                r.get(0)
            })
            .optional()?)
    }

    fn collection_id_by_name(&self, name: &str) -> anyhow::Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                "SELECT id FROM collections WHERE name = ?1",
                params![name],
                |r| r.get(0),
            )
            .optional()?)
    }

    fn path_is_free(&self, path: &str) -> anyhow::Result<bool> {
        let n: i64 = self.conn.query_row(
            "SELECT COUNT(*) FROM images WHERE path = ?1",
            params![path],
            |r| r.get(0),
        )?;
        Ok(n == 0)
    }

    fn parent_of(&self, table: &str, id: i64, column: &str) -> anyhow::Result<Option<i64>> {
        Ok(self
            .conn
            .query_row(
                &format!("SELECT {column} FROM {table} WHERE id = ?1"),
                params![id],
                |r| r.get(0),
            )
            .optional()?
            .flatten())
    }
}

/// Per-apply scratch state.
#[derive(Default)]
struct ApplyCtx {
    /// Guid → local rowid, memoised across the passes of one apply.
    guid_to_id: HashMap<String, i64>,
    /// How many images in this batch carry each content hash — the other
    /// half of the "is this match unambiguous?" test in identity
    /// reconciliation.
    incoming_by_hash: HashMap<[u8; 32], usize>,
}

fn now_secs() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}
