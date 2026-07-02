//! Service layer — DB queries + data-shape composition, no Slint `Weak`/
//! `ModelRc`/callback types. Window controllers call into these instead of
//! locking the DB inline; see the phased extraction plan in project memory.

pub mod collections;
pub mod faces;
pub mod images;
pub mod import;
pub mod people;
pub mod settings;
