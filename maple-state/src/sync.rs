//! Sync vocabulary and the machine-local half of sync configuration.
//!
//! # Why these enums live here and not in `maple-db` or `maple-sync`
//!
//! `role` is a column of `sync_identity` and `mode` a column of `sync_peers`,
//! so `maple-db` is where they are read and written. But `maple-sync` needs
//! the role too — the status pill reads differently for a master than for a
//! servant — and it cannot borrow the type from `maple-db`, because P5 points
//! that dependency the other way: the worker and HTTP handlers in `maple-sync`
//! will call `maple_db`'s collect/apply engine. Defining the enums in the one
//! crate both already depend on keeps that direction free.
//!
//! # What is *not* here
//!
//! `role` and `device_name` are **not** settings. They live in
//! `sync_identity`, next to the device id and the hybrid logical clock, so
//! that a rename or a role change is transactional with the rows it affects
//! and cannot disagree with the database that replication actually uses.
//! Per-peer `mode` likewise lives in `sync_peers`. Only the two values below
//! — machine config a user may reasonably want to hand-edit — are settings.
//!
//! Keeping the settings file thin is deliberate for a second reason:
//! [`Settings::save`](crate::Settings::save) rewrites `settings.toml`
//! wholesale, discarding the comments the user (and `defaults.toml`) put
//! there. The sync card is the first UI that writes settings back at all, so
//! the less it owns, the less that write can destroy.

use serde::{Deserialize, Serialize};

/// What this installation is in the sync topology.
///
/// A star: one master, N servants. A machine that were both would sync its
/// own writes back to itself through a peer, so the two are exclusive rather
/// than a pair of independent toggles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SyncRole {
    #[default]
    Off,
    Master,
    Servant,
}

impl SyncRole {
    /// The value stored in `sync_identity.role`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Off => "off",
            Self::Master => "master",
            Self::Servant => "servant",
        }
    }

    /// Parse a stored value. Unknown text reads as [`SyncRole::Off`] rather
    /// than erroring: a database written by a future version that grew a
    /// fourth role should leave this one switched off, not refuse to open.
    pub fn parse(text: &str) -> Self {
        match text {
            "master" => Self::Master,
            "servant" => Self::Servant,
            _ => Self::Off,
        }
    }
}

impl std::fmt::Display for SyncRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Where a given peer's *photo files* live. Metadata is bidirectional in all
/// three; the mode governs bytes, not knowledge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeerMode {
    /// Every original exists on both sides.
    Full,
    /// The servant's originals are pushed to the master; master-only photos
    /// are visible on the servant but not stored.
    Partial,
    /// Nothing is stored on the servant; originals are fetched on demand.
    #[default]
    Relay,
}

impl PeerMode {
    /// The value stored in `sync_peers.mode`.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Full => "full",
            Self::Partial => "partial",
            Self::Relay => "relay",
        }
    }

    /// Parse a stored value, defaulting to [`PeerMode::Relay`] — the mode
    /// that stores nothing, which is the safe reading of an unrecognised
    /// value: it cannot fill a disk on the strength of a typo.
    pub fn parse(text: &str) -> Self {
        match text {
            "full" => Self::Full,
            "partial" => Self::Partial,
            _ => Self::Relay,
        }
    }

    /// Human label for the settings card.
    pub fn label(self) -> &'static str {
        match self {
            Self::Full => "Full",
            Self::Partial => "Partial",
            Self::Relay => "Relay",
        }
    }

    /// What this mode *does*, in the words of what the user will see.
    ///
    /// Deliberately not a description of disk usage. Relay is the pairing
    /// default on both sides, and its earlier line ("No originals stored
    /// here; loaded on demand") described only half of it: the half about
    /// this device. The other half is that nothing leaves this device either,
    /// so the master lists every photo here as a tile it can never open —
    /// a master runs no worker and has no route back to a servant, so there
    /// is nothing on that machine that could ever fill them in. A mode whose
    /// consequence is invisible reads as a bug in whichever machine shows it.
    pub fn explanation(self) -> &'static str {
        match self {
            Self::Full => {
                "Every photo is copied both ways. Both machines hold every original."
            }
            Self::Partial => {
                "This device's photos are copied to the master.                  The master's own stay remote here, loaded on demand."
            }
            Self::Relay => {
                "No photos are copied, either way. This device's photos stay only here,                  and the master lists them as tiles it cannot open."
            }
        }
    }

    /// Whether this mode moves any photo files at all.
    ///
    /// `false` for [`Relay`](Self::Relay), which is what makes a servant's
    /// pending uploads a permanent backlog rather than a queue — worth
    /// flagging rather than counting down.
    pub fn moves_originals(self) -> bool {
        !matches!(self, Self::Relay)
    }

    /// Next mode in the cycle, for a click-to-change control.
    pub fn next(self) -> Self {
        match self {
            Self::Full => Self::Partial,
            Self::Partial => Self::Relay,
            Self::Relay => Self::Full,
        }
    }
}

impl std::fmt::Display for PeerMode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// The `[sync]` section of `settings.toml` — machine config only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SyncSettings {
    /// Address the master's HTTP listener binds to (P5). `host:port`.
    #[serde(default = "SyncSettings::default_listen_addr")]
    pub listen_addr: String,
    /// Seconds between sync passes on a servant.
    #[serde(default = "SyncSettings::default_interval_secs")]
    pub interval_secs: u64,
}

impl SyncSettings {
    fn default_listen_addr() -> String {
        // 0.0.0.0 rather than a LAN address: the machine's address changes
        // with the network it is on, and a stored one goes stale after every
        // move. 7645 is unassigned by IANA.
        "0.0.0.0:7645".into()
    }

    fn default_interval_secs() -> u64 {
        300
    }
}

impl Default for SyncSettings {
    fn default() -> Self {
        Self {
            listen_addr: Self::default_listen_addr(),
            interval_secs: Self::default_interval_secs(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roles_round_trip_through_their_stored_form() {
        for role in [SyncRole::Off, SyncRole::Master, SyncRole::Servant] {
            assert_eq!(SyncRole::parse(role.as_str()), role);
        }
    }

    #[test]
    fn modes_round_trip_through_their_stored_form() {
        for mode in [PeerMode::Full, PeerMode::Partial, PeerMode::Relay] {
            assert_eq!(PeerMode::parse(mode.as_str()), mode);
        }
    }

    #[test]
    fn unknown_stored_values_fall_back_to_the_safe_reading() {
        // Off stores nothing and syncs nothing; relay stores nothing.
        assert_eq!(SyncRole::parse("overlord"), SyncRole::Off);
        assert_eq!(PeerMode::parse(""), PeerMode::Relay);
    }

    #[test]
    fn mode_cycle_visits_every_mode_and_returns() {
        let mut seen = vec![PeerMode::Full];
        let mut mode = PeerMode::Full;
        for _ in 0..3 {
            mode = mode.next();
            seen.push(mode);
        }
        assert_eq!(
            seen,
            vec![
                PeerMode::Full,
                PeerMode::Partial,
                PeerMode::Relay,
                PeerMode::Full
            ]
        );
    }
}
