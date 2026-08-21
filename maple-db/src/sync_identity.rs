//! This installation's sync identity and its hybrid logical clock.
//!
//! Every replicated row carries a `(rev, rev_dev)` stamp. `rev` comes from
//! the clock below; `rev_dev` is this device's id. Last-write-wins compares
//! the pair lexicographically, so both sides of a sync independently pick the
//! same winner without needing to agree on whose wall clock is right.
//!
//! # Why not a plain counter
//!
//! A per-row counter is not comparable across machines: "revision 4" on the
//! laptop says nothing about "revision 4" on the workstation. Raw wall-clock
//! time *is* comparable but goes backwards — NTP steps, DST-naive clocks, a
//! dead CMOS battery — and a clock that jumps back issues stamps that lose
//! every conflict until real time catches up.
//!
//! A hybrid logical clock takes `max(counter + 1, now_millis)`. It tracks
//! wall time closely enough that stamps stay human-meaningful, never goes
//! backwards, and advances past any remote stamp it observes, so causality is
//! preserved even between two machines whose clocks disagree.

use std::cell::Cell;
use std::time::{SystemTime, UNIX_EPOCH};

use rusqlite::Connection;

/// How many stamps to reserve per durable write of `sync_identity.clock`.
///
/// Persisting on every stamp would add a write to every insert — a 1000-photo
/// import would pay 1000 extra `UPDATE`s. Instead we reserve a block up front
/// and hand out stamps from memory until it's exhausted.
///
/// The cost of a crash is that the reserved-but-unused tail is skipped, which
/// is harmless: `rev` only ever needs to increase. The cost of *not*
/// reserving would be reissuing stamps after a crash — two different rows
/// sharing a `(rev, rev_dev)`, which silently corrupts merge decisions.
const CLOCK_RESERVATION: i64 = 10_000;

/// This installation's identity and clock.
pub struct SyncIdentity {
    device_id: String,
    /// Next stamp to hand out.
    next: Cell<i64>,
    /// Highest stamp covered by the durable reservation in `sync_identity`.
    /// Handing out a stamp above this requires reserving another block first.
    reserved: Cell<i64>,
}

impl SyncIdentity {
    /// Read this device's identity from `sync_identity`, resuming the clock
    /// above whatever the last run reserved.
    pub fn load(conn: &Connection) -> anyhow::Result<Self> {
        let (device_id, ceiling): (String, i64) = conn.query_row(
            "SELECT device_id, clock FROM sync_identity WHERE id = 1",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )?;
        Ok(Self {
            device_id,
            // Resume *above* the ceiling: everything up to it may already
            // have been handed out before the process exited.
            next: Cell::new(ceiling + 1),
            reserved: Cell::new(ceiling),
        })
    }

    pub fn device_id(&self) -> &str {
        &self.device_id
    }

    /// Produce the next `(rev, rev_dev)` stamp for a local write.
    ///
    /// Writes to `sync_identity.clock` only when the reservation is
    /// exhausted — roughly once per [`CLOCK_RESERVATION`] stamps, or once
    /// per 10 seconds of wall time when the clock is tracking real time.
    pub fn stamp(&self, conn: &Connection) -> anyhow::Result<(i64, String)> {
        let rev = self.next.get().max(now_millis());

        if rev > self.reserved.get() {
            let ceiling = rev + CLOCK_RESERVATION;
            conn.execute(
                "UPDATE sync_identity SET clock = ?1 WHERE id = 1",
                [ceiling],
            )?;
            self.reserved.set(ceiling);
        }

        self.next.set(rev + 1);
        Ok((rev, self.device_id.clone()))
    }

    /// Advance past a stamp observed from another device.
    ///
    /// This is what keeps causality intact across machines: a row we received
    /// at `rev` must be older than anything we write afterwards, even if the
    /// sender's clock runs ahead of ours.
    pub fn observe(&self, remote_rev: i64) {
        if remote_rev >= self.next.get() {
            self.next.set(remote_rev + 1);
        }
    }
}

/// Milliseconds since the Unix epoch, saturating at 0 for pre-1970 clocks.
fn now_millis() -> i64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema;

    fn db() -> Connection {
        let conn = Connection::open_in_memory().expect("open");
        schema::ensure_schema(&conn).expect("schema");
        conn
    }

    #[test]
    fn stamps_are_strictly_increasing() {
        let conn = db();
        let id = SyncIdentity::load(&conn).expect("load");

        let mut prev = 0;
        for _ in 0..5_000 {
            let (rev, dev) = id.stamp(&conn).expect("stamp");
            assert!(rev > prev, "rev {rev} did not exceed {prev}");
            assert_eq!(dev, id.device_id());
            prev = rev;
        }
    }

    #[test]
    fn stamps_track_wall_clock() {
        let conn = db();
        let id = SyncIdentity::load(&conn).expect("load");
        let (rev, _) = id.stamp(&conn).expect("stamp");

        // Close enough to now to be a meaningful timestamp, not a bare
        // counter — this is what makes `rev` readable in the tombstone table.
        let drift = (now_millis() - rev).abs();
        assert!(drift < 5_000, "stamp {rev} drifted {drift}ms from now");
    }

    #[test]
    fn observing_a_future_stamp_advances_the_clock() {
        let conn = db();
        let id = SyncIdentity::load(&conn).expect("load");

        // A peer whose clock runs a year fast.
        let far_future = now_millis() + 365 * 24 * 60 * 60 * 1000;
        id.observe(far_future);

        let (rev, _) = id.stamp(&conn).expect("stamp");
        assert!(
            rev > far_future,
            "stamp {rev} should exceed observed {far_future}"
        );
    }

    #[test]
    fn observing_an_old_stamp_does_not_rewind() {
        let conn = db();
        let id = SyncIdentity::load(&conn).expect("load");
        let (before, _) = id.stamp(&conn).expect("stamp");

        id.observe(1);

        let (after, _) = id.stamp(&conn).expect("stamp");
        assert!(after > before);
    }

    #[test]
    fn reopening_resumes_above_the_reservation() {
        let conn = db();
        let issued = {
            let id = SyncIdentity::load(&conn).expect("load");
            let (rev, _) = id.stamp(&conn).expect("stamp");
            rev
        };

        // Simulates a crash: the in-memory clock is gone, only the reserved
        // ceiling survived. Nothing may reissue a stamp at or below `issued`.
        let reopened = SyncIdentity::load(&conn).expect("reload");
        let (rev, _) = reopened.stamp(&conn).expect("stamp");
        assert!(
            rev > issued,
            "reopened clock reissued {rev}, at or below {issued}"
        );
    }

    #[test]
    fn device_id_is_stable_across_reopens() {
        let conn = db();
        let first = SyncIdentity::load(&conn).expect("load").device_id().to_owned();
        let second = SyncIdentity::load(&conn).expect("load").device_id().to_owned();
        assert_eq!(first, second);
        assert_eq!(first.len(), 32, "expected 16 random bytes as hex");
    }
}
