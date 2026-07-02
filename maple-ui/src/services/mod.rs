//! Service layer — DB queries + data-shape composition, no Slint `Weak`/
//! `ModelRc`/callback types. Window controllers call into these instead of
//! locking the DB inline; see the phased extraction plan in project memory.

pub mod collections;
