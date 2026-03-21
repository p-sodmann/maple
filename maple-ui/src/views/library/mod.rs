//! Library browser — searchable grid view of imported images.

pub(super) mod face_shared;
pub(super) mod image_loader;
pub(crate) mod collection_manager;
mod detail_window;
mod face_tagging;
pub mod grid;
mod page;
mod search_bar;

pub use face_tagging::build_face_tagging_page;
pub use page::build_library_page;
