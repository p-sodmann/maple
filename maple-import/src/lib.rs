//! maple-import — Import engine.
//!
//! Scans source directories for images and handles file import.

mod copy;
pub mod exif_date;
mod hash;
pub mod image_source;
pub mod loader;
pub mod phash;
pub mod raw;
mod scan;

pub use copy::{copy_images, CopyResult, CopySummary};
pub use exif_date::ExifDateTime;
pub use hash::content_hash;
pub use image_source::{is_raw_format, loadable_image_bytes, raw_preview_supported};
pub use loader::{apply_orientation, decode_image, decode_image_bytes};
pub use phash::{compute_phash, phash_similarity, ImageHash};
pub use scan::{scan_grouped, scan_grouped_excluding, scan_images, CopyMode, ImageFile, ImageGroup};

pub struct ImportEngine;
