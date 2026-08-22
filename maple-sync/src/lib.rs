//! maple-sync — transport, pairing and authentication for library sync.
//!
//! This crate owns everything *around* replication: how two installations
//! establish trust ([`pairing`]), where the resulting credential is stored
//! ([`trust`]), and how every later request proves who sent it ([`auth`]).
//! The merge engine itself lives in `maple_db::sync`, because merging needs
//! transactional SQL against the schema `maple-db` owns.
//!
//! # Threat model — read this before trusting it with anything
//!
//! Sync traffic is **plain HTTP with no TLS**. Request *contents* — metadata,
//! and later photo bytes — are readable by anyone who can see the LAN, and
//! nothing here changes that. What the primitives in this crate do provide:
//!
//! * **Impersonation resistance.** Only a device that completed a pairing
//!   handshake with a human present holds the long-term key, and every
//!   request is MAC'd with it. An unpaired machine cannot issue a request
//!   that verifies, and cannot forge a *response* either — the client
//!   verifies the master's proof during pairing, which is the half that stops
//!   a fake master on the same network from harvesting a real one's library.
//! * **Replay resistance.** Requests carry a timestamp and a nonce, and are
//!   rejected outside a ±5 minute window or on a nonce already seen.
//! * **Tamper evidence.** The MAC covers the method, the path and a hash of
//!   the body, so a man-in-the-middle cannot rewrite a request in flight.
//!
//! What it does **not** provide: confidentiality, forward secrecy, or any
//! protection once an attacker holds the key file. This is a trusted-home-LAN
//! feature; it needs revisiting before any remote-access story.
//!
//! # Determinism
//!
//! Nothing here samples the clock or the system RNG on its own. Timestamps
//! arrive as `now_ms` arguments and random bytes come from an injected
//! [`RandomSource`], so every handshake and every signature is reproducible
//! in a test. Production callers pass [`now_ms()`] and a source backed by
//! SQLite's `randomblob` (`maple_db::Database::random_bytes`), which is how
//! device ids and row guids are already minted.

pub mod auth;
pub mod backoff;
pub mod client;
pub mod merge;
pub mod pairing;
pub mod protocol;
pub mod random;
pub mod server;
pub mod status;
pub mod trust;
pub mod worker;

pub use auth::{now_ms, NonceRing, SignedRequest};
pub use backoff::{Backoff, FailureKind, Retry};
pub use pairing::{
    pair_secret, ClaimRequest, ClaimResponse, Initiator, PairCode, PairingSlot, PairingWindow,
};
pub use protocol::{ErrorBody, ErrorCode, Hello, PullRequest, PushResponse, PROTOCOL_VERSION};
pub use random::{FnRandom, RandomSource, SharedRandom};
pub use client::{SyncClient, SyncFailure};
pub use server::{Clock, SyncServer};
pub use worker::{SyncWorker, WorkerConfig};
pub use status::{
    relative_time, StatusCell, StatusDisplay, StatusTone, SyncRole, SyncState, SyncStatus,
};
pub use trust::{PeerKey, TrustFile, TrustStore, TrustedPeer};
