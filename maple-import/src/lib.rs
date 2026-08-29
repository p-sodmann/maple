//! maple-import — Import engine.
//!
//! Scans source directories for images and handles file import.

mod copy;
mod embed_cache;
pub mod exif_date;
pub mod exif_read;
mod hash;
pub mod image_source;
pub mod loader;
pub mod path_template;
pub mod preview;
mod preview_cache;
pub mod raw;
pub mod restructure;
mod scan;
pub mod session;
pub mod sharpness;

pub use copy::{copy_images, place_file, place_pair, CopyResult, CopySummary};
pub use embed_cache::EmbeddingCache;
pub use exif_date::ExifDateTime;
pub use exif_read::ExifContext;
pub use hash::{content_hash, hash_bytes};
pub use image_source::{
    is_raw_format, loadable_image_bytes, loadable_image_bytes_named, raw_preview_supported,
};
pub use loader::{apply_orientation, decode_image, decode_image_bytes};
pub use path_template::{render_filename_stem, render_folder, TemplateContext};
pub use preview_cache::{CachedPreview, PreviewCache, PreviewKey, PREVIEW_CACHE_FILE};
pub use restructure::{execute_moves, plan_moves, MoveResult, PlannedMove, RestructureCandidate, RestructureSummary};
pub use session::{
    segment, BlockTileEngine, ColorKmeansEngine, CutReason, EnsembleEngine, Frame,
    GridHistogramEngine, Link, SegmentParams, Segmentation, Session, SessionEngine, Signature,
    TimeGapEngine,
};
pub use scan::{scan_grouped, scan_grouped_excluding, scan_images, CopyMode, ImageFile, ImageGroup};
pub use sharpness::laplacian_variance;

pub struct ImportEngine;
