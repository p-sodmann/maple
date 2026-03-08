//! maple-import — Import engine.
//!
//! Scans source directories for images and handles file import.

mod copy;
mod hash;
mod scan;

pub use copy::{copy_images, CopyResult, CopySummary};
pub use hash::content_hash;
pub use scan::{scan_images, ImageFile};

pub struct ImportEngine;
