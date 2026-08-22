//! Accessors for the two V18 tables the UI reads and writes: this device's
//! role and name (`sync_identity`) and the list of paired devices
//! (`sync_peers`).
//!
//! # Why these three fields are not settings
//!
//! `settings.toml` would be the obvious home for "am I a master", "what am I
//! called" and "what mode is the laptop in", and the original plan put them
//! there. Two reasons they live here instead:
//!
//! * **They already exist here.** `sync_identity` was created in V18 with
//!   `role` and `device_name` columns, next to the device id and the hybrid
//!   logical clock; `sync_peers` was created with `mode`. Duplicating them
//!   into the settings file would create two sources of truth for values the
//!   merge engine consults, with no mechanism keeping them equal.
//! * **They must be transactional with the rows they describe.** Unpairing a
//!   device deletes its watermarks; changing a peer's mode changes which
//!   originals get fetched for rows that already exist. A settings write is
//!   a separate, non-atomic file rewrite that can land while a sync pass is
//!   half done.
//!
//! `settings.toml` keeps only `listen_addr` and `interval_secs` — machine
//! config with no row-level consequences. See `maple_state::sync`.
//!
//! Nothing here stamps a `rev`: these tables are not in
//! [`SYNCED_TABLES`](crate::SYNCED_TABLES) and are local bookkeeping by
//! design. A servant does not get to tell a master what mode it is in.

use maple_state::{PeerMode, SyncRole};
use rusqlite::{params, OptionalExtension};

use crate::Database;

/// One paired device, as the settings card renders it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SyncPeer {
    pub device_id: String,
    /// Display name the peer gave during pairing. `None` if it never sent one.
    pub name: Option<String>,
    pub mode: PeerMode,
    /// Highest `rev` pulled from this peer — the watermark the next pull asks
    /// above.
    pub last_pull_rev: i64,
    /// Highest local `rev` this peer has acknowledged.
    pub last_push_rev: i64,
    /// Unix milliseconds of the last successful contact, or `None` if never.
    pub last_seen_at: Option<i64>,
}

impl SyncPeer {
    /// The name to show, falling back to a short form of the device id for a
    /// peer that paired without sending one.
    pub fn display_name(&self) -> String {
        match self.name.as_deref() {
            Some(name) if !name.trim().is_empty() => name.to_owned(),
            _ => format!("Device {}", &self.device_id[..self.device_id.len().min(8)]),
        }
    }
}

impl Database {
    // ── This device ──────────────────────────────────────────────

    /// This installation's role in the sync topology.
    pub fn sync_role(&self) -> anyhow::Result<SyncRole> {
        let stored: String = self.conn.query_row(
            "SELECT role FROM sync_identity WHERE id = 1",
            [],
            |r| r.get(0),
        )?;
        Ok(SyncRole::parse(&stored))
    }

    /// Switch this installation's role.
    ///
    /// Master and servant are exclusive because the merge engine is built for
    /// a star topology — a machine that were both would relay its own writes
    /// back to itself through a peer. One column enforces that by
    /// construction; two booleans would not.
    pub fn set_sync_role(&self, role: SyncRole) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_identity SET role = ?1 WHERE id = 1",
            params![role.as_str()],
        )?;
        Ok(())
    }

    /// This installation's display name, as shown to peers during pairing.
    /// Empty until the user sets one.
    pub fn device_name(&self) -> anyhow::Result<String> {
        Ok(self.conn.query_row(
            "SELECT device_name FROM sync_identity WHERE id = 1",
            [],
            |r| r.get(0),
        )?)
    }

    /// Rename this installation. Trimmed, because a name with trailing
    /// whitespace renders as a ragged row in the peer's settings card and the
    /// user cannot see why.
    pub fn set_device_name(&self, name: &str) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_identity SET device_name = ?1 WHERE id = 1",
            params![name.trim()],
        )?;
        Ok(())
    }

    // ── Paired devices ───────────────────────────────────────────

    /// Every paired device.
    ///
    /// Ordered by name then device id, not by insertion or `ORDER BY
    /// RANDOM()`: the settings card re-reads this list on every repaint, and
    /// rows that swap places under the cursor make an `[Unpair]` button a
    /// hazard. `device_id` is a primary key, so the order is total.
    pub fn list_sync_peers(&self) -> anyhow::Result<Vec<SyncPeer>> {
        let mut stmt = self.conn.prepare(
            "SELECT device_id, name, mode, last_pull_rev, last_push_rev, last_seen_at
             FROM sync_peers
             ORDER BY coalesce(name, ''), device_id",
        )?;
        let rows = stmt.query_map([], |r| {
            Ok(SyncPeer {
                device_id: r.get(0)?,
                name: r.get(1)?,
                mode: PeerMode::parse(&r.get::<_, String>(2)?),
                last_pull_rev: r.get(3)?,
                last_push_rev: r.get(4)?,
                last_seen_at: r.get(5)?,
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    /// One paired device by id, or `None` if it is not paired.
    pub fn sync_peer(&self, device_id: &str) -> anyhow::Result<Option<SyncPeer>> {
        Ok(self
            .conn
            .query_row(
                "SELECT device_id, name, mode, last_pull_rev, last_push_rev, last_seen_at
                 FROM sync_peers WHERE device_id = ?1",
                params![device_id],
                |r| {
                    Ok(SyncPeer {
                        device_id: r.get(0)?,
                        name: r.get(1)?,
                        mode: PeerMode::parse(&r.get::<_, String>(2)?),
                        last_pull_rev: r.get(3)?,
                        last_push_rev: r.get(4)?,
                        last_seen_at: r.get(5)?,
                    })
                },
            )
            .optional()?)
    }

    /// Record a pairing, or update the name and mode of an existing one.
    ///
    /// Deliberately leaves the watermarks alone on a repeat: re-pairing an
    /// already-known device (the trust file was restored, the response was
    /// lost in flight) mints a new key but does not un-sync anything, and
    /// resetting `last_pull_rev` to zero would drag the entire library back
    /// across the wire for no reason.
    pub fn upsert_sync_peer(
        &self,
        device_id: &str,
        name: Option<&str>,
        mode: PeerMode,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "INSERT INTO sync_peers (device_id, name, mode)
             VALUES (?1, ?2, ?3)
             ON CONFLICT(device_id) DO UPDATE SET name = ?2, mode = ?3",
            params![device_id, name, mode.as_str()],
        )?;
        Ok(())
    }

    /// Change one peer's file mode without touching anything else.
    pub fn set_sync_peer_mode(&self, device_id: &str, mode: PeerMode) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_peers SET mode = ?1 WHERE device_id = ?2",
            params![mode.as_str(), device_id],
        )?;
        Ok(())
    }

    /// Note that this peer was reached. `now_ms` is a parameter rather than
    /// a `SystemTime::now()` inside, so the caller's tests stay off the wall
    /// clock — the same rule the rest of sync follows.
    pub fn touch_sync_peer(&self, device_id: &str, now_ms: i64) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_peers SET last_seen_at = ?1 WHERE device_id = ?2",
            params![now_ms, device_id],
        )?;
        Ok(())
    }

    /// Record that this peer holds everything we stamped at or below `rev`.
    ///
    /// Separate from [`Database::set_sync_peer_pull_rev`] because the two
    /// directions are learned at different moments and from different
    /// evidence — this one from the watermark a peer names when it pulls,
    /// the other from a batch it pushed. Writing both together would mean
    /// inventing a value for whichever half is not currently known.
    pub fn set_sync_peer_push_rev(&self, device_id: &str, rev: i64) -> anyhow::Result<()> {
        // `max` rather than assignment: watermarks only ever move forward,
        // and a reordered or replayed request must not walk one backwards
        // and re-ship everything in between.
        self.conn.execute(
            "UPDATE sync_peers SET last_push_rev = max(last_push_rev, ?1) WHERE device_id = ?2",
            params![rev, device_id],
        )?;
        Ok(())
    }

    /// Record the highest `rev` we have merged *from* this peer.
    pub fn set_sync_peer_pull_rev(&self, device_id: &str, rev: i64) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_peers SET last_pull_rev = max(last_pull_rev, ?1) WHERE device_id = ?2",
            params![rev, device_id],
        )?;
        Ok(())
    }

    /// Advance the watermarks after a successful pass.
    pub fn set_sync_peer_watermarks(
        &self,
        device_id: &str,
        last_pull_rev: i64,
        last_push_rev: i64,
    ) -> anyhow::Result<()> {
        self.conn.execute(
            "UPDATE sync_peers SET last_pull_rev = ?1, last_push_rev = ?2 WHERE device_id = ?3",
            params![last_pull_rev, last_push_rev, device_id],
        )?;
        Ok(())
    }

    /// Unpair a device. Returns whether it was paired in the first place.
    ///
    /// Only forgets the bookkeeping — the peer's *key* lives in
    /// `sync_trust.json`, and the caller must remove it there too. They are
    /// separate stores on purpose (secrets stay out of the database file that
    /// gets copied around), so unpairing is two writes and neither one alone
    /// is complete.
    pub fn remove_sync_peer(&self, device_id: &str) -> anyhow::Result<bool> {
        let removed = self.conn.execute(
            "DELETE FROM sync_peers WHERE device_id = ?1",
            params![device_id],
        )?;
        Ok(removed > 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn db() -> Database {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("library.db");
        // Leak the tempdir so the file outlives this call; the OS reclaims it.
        std::mem::forget(dir);
        Database::open(&path).expect("open")
    }

    #[test]
    fn role_defaults_to_off_and_round_trips() {
        let db = db();
        assert_eq!(db.sync_role().unwrap(), SyncRole::Off);

        for role in [SyncRole::Master, SyncRole::Servant, SyncRole::Off] {
            db.set_sync_role(role).unwrap();
            assert_eq!(db.sync_role().unwrap(), role);
        }
    }

    #[test]
    fn device_name_starts_empty_and_is_trimmed_on_write() {
        let db = db();
        assert_eq!(db.device_name().unwrap(), "");

        db.set_device_name("  Workstation  ").unwrap();
        assert_eq!(db.device_name().unwrap(), "Workstation");
    }

    #[test]
    fn upsert_replaces_rather_than_duplicating() {
        let db = db();
        db.upsert_sync_peer("aaaa1111", Some("Laptop"), PeerMode::Relay)
            .unwrap();
        db.upsert_sync_peer("aaaa1111", Some("Laptop Pro"), PeerMode::Full)
            .unwrap();

        let peers = db.list_sync_peers().unwrap();
        assert_eq!(peers.len(), 1, "a repeat pairing must not add a second row");
        assert_eq!(peers[0].name.as_deref(), Some("Laptop Pro"));
        assert_eq!(peers[0].mode, PeerMode::Full);
    }

    #[test]
    fn re_pairing_keeps_the_watermarks() {
        // Resetting these would re-pull the whole library after a lost
        // response, which is the case a repeat claim exists to survive.
        let db = db();
        db.upsert_sync_peer("bbbb2222", Some("Laptop"), PeerMode::Relay)
            .unwrap();
        db.set_sync_peer_watermarks("bbbb2222", 4_000, 5_000).unwrap();

        db.upsert_sync_peer("bbbb2222", Some("Laptop"), PeerMode::Relay)
            .unwrap();

        let peer = db.sync_peer("bbbb2222").unwrap().expect("still paired");
        assert_eq!(peer.last_pull_rev, 4_000);
        assert_eq!(peer.last_push_rev, 5_000);
    }

    #[test]
    fn remove_reports_whether_the_peer_existed() {
        let db = db();
        db.upsert_sync_peer("cccc3333", None, PeerMode::Partial)
            .unwrap();

        assert!(db.remove_sync_peer("cccc3333").unwrap());
        assert!(!db.remove_sync_peer("cccc3333").unwrap());
        assert!(db.list_sync_peers().unwrap().is_empty());
        assert!(db.sync_peer("cccc3333").unwrap().is_none());
    }

    #[test]
    fn listing_order_is_deterministic() {
        // A P2 test cost real debugging time to a nondeterministic ordering;
        // this list is re-read on every repaint next to an [Unpair] button,
        // so rows that move between reads are worse than untidy.
        let db = db();
        db.upsert_sync_peer("33", Some("Studio Mac"), PeerMode::Full)
            .unwrap();
        db.upsert_sync_peer("11", Some("Laptop"), PeerMode::Relay)
            .unwrap();
        db.upsert_sync_peer("22", Some("Laptop"), PeerMode::Partial)
            .unwrap();
        db.upsert_sync_peer("44", None, PeerMode::Relay).unwrap();

        let expected = vec!["44", "11", "22", "33"];
        for _ in 0..5 {
            let ids: Vec<String> = db
                .list_sync_peers()
                .unwrap()
                .into_iter()
                .map(|p| p.device_id)
                .collect();
            assert_eq!(ids, expected, "peer order must not vary between reads");
        }
    }

    #[test]
    fn mode_and_last_seen_update_independently() {
        let db = db();
        db.upsert_sync_peer("dddd4444", Some("Laptop"), PeerMode::Relay)
            .unwrap();

        db.set_sync_peer_mode("dddd4444", PeerMode::Partial).unwrap();
        db.touch_sync_peer("dddd4444", 1_700_000_000_000).unwrap();

        let peer = db.sync_peer("dddd4444").unwrap().unwrap();
        assert_eq!(peer.mode, PeerMode::Partial);
        assert_eq!(peer.last_seen_at, Some(1_700_000_000_000));
        assert_eq!(peer.name.as_deref(), Some("Laptop"), "mode change must not clear the name");
    }

    #[test]
    fn watermarks_only_move_forward() {
        // A replayed or reordered request must not walk a watermark back and
        // re-ship everything in between — worse, on the pull side it would
        // re-request rows the peer has already pruned tombstones for.
        let db = db();
        db.upsert_sync_peer("eeee5555", None, PeerMode::Relay).unwrap();

        db.set_sync_peer_pull_rev("eeee5555", 900).unwrap();
        db.set_sync_peer_push_rev("eeee5555", 800).unwrap();
        db.set_sync_peer_pull_rev("eeee5555", 100).unwrap();
        db.set_sync_peer_push_rev("eeee5555", 7).unwrap();

        let peer = db.sync_peer("eeee5555").unwrap().unwrap();
        assert_eq!(peer.last_pull_rev, 900);
        assert_eq!(peer.last_push_rev, 800);
    }

    #[test]
    fn the_two_watermark_directions_are_independent() {
        let db = db();
        db.upsert_sync_peer("ffff6666", None, PeerMode::Relay).unwrap();
        db.set_sync_peer_pull_rev("ffff6666", 500).unwrap();
        let peer = db.sync_peer("ffff6666").unwrap().unwrap();
        assert_eq!(peer.last_pull_rev, 500);
        assert_eq!(peer.last_push_rev, 0, "the push direction is learned separately");
    }

    #[test]
    fn a_nameless_peer_still_displays_as_something() {
        let db = db();
        db.upsert_sync_peer("0123456789abcdef", None, PeerMode::Relay)
            .unwrap();
        let peer = db.sync_peer("0123456789abcdef").unwrap().unwrap();
        assert_eq!(peer.display_name(), "Device 01234567");

        db.upsert_sync_peer("0123456789abcdef", Some("   "), PeerMode::Relay)
            .unwrap();
        let peer = db.sync_peer("0123456789abcdef").unwrap().unwrap();
        assert_eq!(peer.display_name(), "Device 01234567", "blank is not a name");
    }
}
