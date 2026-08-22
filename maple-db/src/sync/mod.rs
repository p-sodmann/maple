//! Replication between a master installation and its servants.
//!
//! This module owns the *data* half of sync: what a replicated row looks like
//! on the wire ([`wire`]), how local changes are gathered for a peer
//! ([`collect`]), and how a peer's changes are merged in ([`apply`]).
//! Transport, pairing and authentication live in the `maple-sync` crate.
//!
//! The engine lives here rather than in `maple-sync` because merging needs
//! transactional SQL against the schema `maple-db` owns; the alternative was
//! making `Connection` public, which would hand every caller the ability to
//! write unstamped rows.
//!
//! # The invariant everything rests on
//!
//! Each replicated row carries `(guid, rev, rev_dev)`. `guid` is a stable
//! identity that survives crossing between devices, where a local rowid would
//! not. `(rev, rev_dev)` is a totally-ordered stamp, so two devices merging
//! the same pair of conflicting edits independently choose the same winner —
//! no coordination, no round trip, and no dependence on whose clock is right.

pub mod apply;
pub mod collect;
pub mod wire;

pub use apply::ApplyReport;
pub use collect::DEFAULT_MAX_REVS;
pub use wire::{
    AiDescriptionRow, CollectionImageRow, CollectionRow, Entity, FaceRow, GuidAlias, ImageRow,
    PersonRow, StackRow, Stamp, SyncBatch, SyncRow, Tombstone,
};

#[cfg(test)]
mod tests;
