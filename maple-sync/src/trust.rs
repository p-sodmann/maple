//! The trust file: `config_dir()/sync_trust.json`, mode `0600` on unix.
//!
//! One long-term key per paired device, plus that device's last-known
//! address. Nothing else — all the non-secret peer state (display name, sync
//! mode, watermarks, last-seen) lives in the `sync_peers` table, where it can
//! be updated in the same transaction as the rows it describes.
//!
//! # Why not `settings.toml`
//!
//! Two reasons, both concrete:
//!
//! * `Settings::save` serialises the whole struct and writes it back
//!   wholesale, which destroys the comments a user put in the file. A
//!   credential store gets rewritten on every pairing; the settings file
//!   should not be collateral damage.
//! * That write is also non-atomic — a crash mid-write leaves a truncated
//!   file. Losing settings means falling back to defaults; losing the trust
//!   file means every paired device needs re-pairing by hand.
//!
//! So this file is written atomically (temp file in the same directory, then
//! rename) and is not user-editable by design.

use std::io::Write;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// A long-term pairing key: 32 bytes shared with exactly one peer.
///
/// After pairing this value never crosses the wire again — it is only ever
/// used as a MAC key (see [`crate::auth`]).
#[derive(Clone, PartialEq, Eq)]
pub struct PeerKey([u8; 32]);

impl PeerKey {
    pub fn from_bytes(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }
}

/// Redacted on purpose: this type ends up inside structs that get `{:?}`'d
/// into `tracing` output, and a key in a log file is a key on disk in
/// plaintext, in a file nobody remembered to chmod.
impl std::fmt::Debug for PeerKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PeerKey(<redacted>)")
    }
}

impl Serialize for PeerKey {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use base64::Engine;
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(self.0))
    }
}

impl<'de> Deserialize<'de> for PeerKey {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        use base64::Engine;
        use serde::de::Error as _;
        let text = String::deserialize(d)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&text)
            .map_err(D::Error::custom)?;
        let bytes: [u8; 32] = bytes
            .try_into()
            .map_err(|_| D::Error::custom("peer key must be 32 bytes"))?;
        Ok(Self(bytes))
    }
}

/// One paired device.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustedPeer {
    pub device_id: String,
    pub key: PeerKey,
    /// Last address this peer was reached at, `host:port`. Advisory only —
    /// mDNS re-resolution (P8) overrides it, and a DHCP lease change makes it
    /// stale without making the pairing invalid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub address: Option<String>,
}

/// The whole file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrustFile {
    pub device_id: String,
    pub device_name: String,
    #[serde(default)]
    pub peers: Vec<TrustedPeer>,
}

impl TrustFile {
    /// Default location: alongside `settings.toml` and `session.json`.
    pub fn path() -> PathBuf {
        maple_state::config_dir().join("sync_trust.json")
    }

    pub fn new(device_id: impl Into<String>, device_name: impl Into<String>) -> Self {
        Self {
            device_id: device_id.into(),
            device_name: device_name.into(),
            peers: Vec::new(),
        }
    }

    /// Load from the default path. `Ok(None)` means "not paired yet".
    pub fn load() -> anyhow::Result<Option<Self>> {
        Self::load_from(&Self::path())
    }

    /// Load from a specific path.
    ///
    /// A missing file is `Ok(None)`, but a *malformed* one is an error rather
    /// than a silent default — unlike `Session::load`, which shrugs and
    /// starts fresh. Shrugging here would drop every stored key and present
    /// as "all your devices need re-pairing", hiding the real cause. Better
    /// to surface it and leave the file untouched for the user to inspect.
    pub fn load_from(path: &Path) -> anyhow::Result<Option<Self>> {
        let json = match std::fs::read_to_string(path) {
            Ok(json) => json,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
            Err(e) => return Err(anyhow::Error::from(e)),
        };
        let parsed = serde_json::from_str(&json)
            .map_err(|e| anyhow::anyhow!("{} is corrupt: {e}", path.display()))?;
        Ok(Some(parsed))
    }

    /// Persist to the default path.
    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&Self::path())
    }

    /// Persist to a specific path, atomically and (on unix) `0600`.
    ///
    /// The temp file is created in the *same directory* as the target, since
    /// `rename` is only atomic within a filesystem, and it is created with
    /// the restrictive mode from the start — creating it world-readable and
    /// chmod'ing afterwards leaves a window where the key is readable.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;

        let temp = path.with_extension("json.tmp");
        {
            let mut options = std::fs::OpenOptions::new();
            options.write(true).create(true).truncate(true);
            #[cfg(unix)]
            {
                use std::os::unix::fs::OpenOptionsExt as _;
                options.mode(0o600);
            }
            let mut file = options.open(&temp)?;
            file.write_all(json.as_bytes())?;
            // Flush before the rename: otherwise a crash can publish an empty
            // file over a good one, which is exactly what the temp-and-rename
            // dance is supposed to prevent.
            file.sync_all()?;
        }

        // An existing target may predate the mode-on-create above, or have
        // been written by a build without it; rename preserves the *source*
        // file's mode, so this is already covered — but a pre-existing loose
        // mode on the temp path is not, hence create+truncate above.
        std::fs::rename(&temp, path)?;
        Ok(())
    }

    pub fn peer(&self, device_id: &str) -> Option<&TrustedPeer> {
        self.peers.iter().find(|p| p.device_id == device_id)
    }

    /// Add a peer, or replace the entry for a device that re-paired.
    ///
    /// Re-pairing mints a fresh key, so the old one must go rather than
    /// accumulate — two entries for one device would make verification
    /// depend on iteration order.
    pub fn upsert_peer(&mut self, peer: TrustedPeer) {
        match self.peers.iter_mut().find(|p| p.device_id == peer.device_id) {
            Some(existing) => *existing = peer,
            None => self.peers.push(peer),
        }
    }

    /// Forget a peer. Returns whether it was there.
    pub fn remove_peer(&mut self, device_id: &str) -> bool {
        let before = self.peers.len();
        self.peers.retain(|p| p.device_id != device_id);
        self.peers.len() != before
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> TrustFile {
        let mut file = TrustFile::new("dev-a", "Workstation");
        file.upsert_peer(TrustedPeer {
            device_id: "dev-b".into(),
            key: PeerKey::from_bytes([7u8; 32]),
            address: Some("192.168.1.31:7645".into()),
        });
        file
    }

    #[test]
    fn round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync_trust.json");
        sample().save_to(&path).unwrap();

        let loaded = TrustFile::load_from(&path).unwrap().expect("file exists");
        assert_eq!(loaded.device_id, "dev-a");
        assert_eq!(loaded.device_name, "Workstation");
        let peer = loaded.peer("dev-b").expect("peer survived the round trip");
        assert_eq!(peer.key.as_bytes(), &[7u8; 32]);
        assert_eq!(peer.address.as_deref(), Some("192.168.1.31:7645"));
    }

    #[cfg(unix)]
    #[test]
    fn is_written_owner_only() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync_trust.json");
        sample().save_to(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "trust file must not be group- or world-readable");
    }

    #[cfg(unix)]
    #[test]
    fn overwrite_keeps_owner_only_mode() {
        use std::os::unix::fs::PermissionsExt as _;

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync_trust.json");
        sample().save_to(&path).unwrap();
        // Re-pairing rewrites the file; the rename must not import a looser
        // mode from anywhere.
        sample().save_to(&path).unwrap();

        let mode = std::fs::metadata(&path).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }

    #[test]
    fn missing_file_is_not_an_error() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("nope.json");
        assert!(TrustFile::load_from(&missing).unwrap().is_none());
    }

    #[test]
    fn corrupt_file_is_an_error_not_a_silent_reset() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync_trust.json");
        std::fs::write(&path, "{ not json").unwrap();
        assert!(
            TrustFile::load_from(&path).is_err(),
            "a corrupt trust file must surface, not present as 'never paired'"
        );
    }

    #[test]
    fn re_pairing_replaces_the_key_rather_than_duplicating_the_peer() {
        let mut file = sample();
        file.upsert_peer(TrustedPeer {
            device_id: "dev-b".into(),
            key: PeerKey::from_bytes([9u8; 32]),
            address: None,
        });
        assert_eq!(file.peers.len(), 1);
        assert_eq!(file.peer("dev-b").unwrap().key.as_bytes(), &[9u8; 32]);
    }

    #[test]
    fn removing_a_peer_reports_whether_it_existed() {
        let mut file = sample();
        assert!(file.remove_peer("dev-b"));
        assert!(!file.remove_peer("dev-b"));
        assert!(file.peer("dev-b").is_none());
    }

    #[test]
    fn key_is_redacted_in_debug_output() {
        let rendered = format!("{:?}", sample());
        assert!(!rendered.contains("BwcH"), "base64 of the key leaked into Debug");
        assert!(rendered.contains("<redacted>"));
    }

    #[test]
    fn a_wrong_length_key_is_rejected_on_load() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sync_trust.json");
        std::fs::write(
            &path,
            r#"{"device_id":"a","device_name":"A",
                "peers":[{"device_id":"b","key":"c2hvcnQ="}]}"#,
        )
        .unwrap();
        assert!(TrustFile::load_from(&path).is_err());
    }
}
