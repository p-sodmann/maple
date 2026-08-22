//! Applying a peer's batch, and the local bookkeeping that has to follow it.
//!
//! [`Database::apply_batch`](maple_db::Database::apply_batch) deliberately
//! *reports* what changed rather than acting on it: it is a transaction over
//! replicated rows, and recomputing a person's centroid inside that
//! transaction would drag ONNX-shaped work into a lock the whole UI waits on.
//!
//! But the recomputation is not optional. `persons.centroid_embedding`,
//! `representative_face_id`, `collections.centroid_embedding` and
//! `representative_image_id` are all derived from *local rowids* and are
//! never synced (§3.3). After a merge changes which faces belong to a person,
//! the stored centroid describes a set that no longer exists — face matching
//! would keep scoring against it until something else happened to touch that
//! person.
//!
//! So both directions — the master applying a push, the servant applying a
//! pull — funnel through [`apply_and_refresh`], and neither gets to forget.

use maple_db::sync::SyncBatch;
use maple_db::{ApplyReport, Database};

/// Merge `batch`, then rebuild every derived value the merge invalidated.
///
/// A failure to refresh is logged rather than propagated: the rows are
/// already committed, the derived values are a cache, and returning an error
/// here would make the caller re-send a batch that landed successfully.
pub fn apply_and_refresh(db: &Database, batch: &SyncBatch) -> anyhow::Result<ApplyReport> {
    let report = db.apply_batch(batch)?;

    for person_id in &report.touched_persons {
        if let Err(e) = db.update_person_representative(*person_id) {
            tracing::warn!("sync: could not refresh person {person_id} after merge: {e}");
        }
    }
    for collection_id in &report.touched_collections {
        if let Err(e) = db.update_collection_representative(*collection_id) {
            tracing::warn!("sync: could not refresh collection {collection_id} after merge: {e}");
        }
    }

    Ok(report)
}

/// Whether a watermark may advance past this batch.
///
/// Deferred rows are ones whose mandatory parent has not arrived yet. They
/// are re-sent on the next pass — but only if the sender still considers them
/// unsent. Advancing the watermark over a deferred row loses it silently and
/// permanently, which is the worst failure mode this protocol has, so the
/// rule gets a name rather than being an `if` buried in two call sites.
pub fn may_advance(report: &ApplyReport) -> bool {
    report.deferred == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_clean_apply_advances_the_watermark() {
        assert!(may_advance(&ApplyReport::default()));
    }

    #[test]
    fn a_deferred_row_pins_the_watermark() {
        let report = ApplyReport {
            inserted: 40,
            deferred: 1,
            ..ApplyReport::default()
        };
        assert!(
            !may_advance(&report),
            "one deferred row is enough — advancing past it loses it forever"
        );
    }
}
