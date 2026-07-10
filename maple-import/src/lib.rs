//! maple-import — Import engine.
//!
//! Scans source directories for images and handles file import.

mod copy;
mod embed_cache;
pub mod exif_date;
mod hash;
pub mod image_source;
pub mod loader;
pub mod path_template;
pub mod raw;
pub mod restructure;
mod scan;
pub mod sharpness;

pub use copy::{copy_images, CopyResult, CopySummary};
pub use embed_cache::EmbeddingCache;
pub use exif_date::ExifDateTime;
pub use hash::content_hash;
pub use image_source::{is_raw_format, loadable_image_bytes, raw_preview_supported};
pub use loader::{apply_orientation, decode_image, decode_image_bytes};
pub use path_template::{render_filename_stem, render_folder, TemplateContext};
pub use restructure::{execute_moves, plan_moves, MoveResult, PlannedMove, RestructureCandidate, RestructureSummary};
pub use scan::{scan_grouped, scan_grouped_excluding, scan_images, CopyMode, ImageFile, ImageGroup};
pub use sharpness::laplacian_variance;

pub struct ImportEngine;
