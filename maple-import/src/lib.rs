//! maple-import — Import engine.
//!
//! Scans source directories for images and handles file import.

mod copy;
mod hash;
pub mod image_source;
pub mod raw;
mod scan;

pub use copy::{copy_images, CopyResult, CopySummary};
pub use hash::content_hash;
pub use image_source::{is_raw_format, loadable_image_bytes};
pub use scan::{scan_grouped, scan_images, CopyMode, ImageFile, ImageGroup};

pub struct ImportEngine;
