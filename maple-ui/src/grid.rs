//! Library thumbnail grid controller (Slint).
//!
//! Mirrors the former GTK `LibraryGrid` (views/library/grid.rs):
//!   1. A background thread queries the DB and sends `Records`.
//!   2. Placeholder tiles fill the grid immediately.
//!   3. Parallel thumbnail workers send `Thumb` messages; tiles are filled
//!      in-place as decoded RGB arrives.
//!
//! Each `load()` increments a generation counter; the `slint::Timer` poller
//! discards messages from superseded loads, so rapid search changes never
//! produce stale or interleaved grid content. This is the Slint analogue of the
//! old `glib::timeout_add_local` poller — all background work still runs on
//! `std::thread` + `std::sync::mpsc`, and only the UI-thread delivery changes.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use slint::{
    Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel,
};

use maple_db::{LibraryImage, SearchHit, SearchQuery, ThumbnailCache};
use maple_import::raw_preview_supported;

use crate::thumbnail;
use crate::ThumbItem;

const POLL_MS: u64 = 32;

// ── Worker messages ──────────────────────────────────────────────

enum GridMsg {
    /// Initial batch of DB results (establishes grid size).
    Records(Vec<LibraryImage>),
    /// One thumbnail finished — carries decoded RGB pixels.
    Thumb {
        index: usize,
        rgb: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Format recognised but preview extraction not yet implemented.
    Unsupported { index: usize },
    /// All thumbnails have been generated.
    Done,
}

// ── Public interface ─────────────────────────────────────────────

/// Thumbnail grid that reloads from the DB on demand.
///
/// Cheap to clone — all internal state is reference-counted, so a clone shares
/// the same backing model and record list.
#[derive(Clone)]
pub struct LibraryGrid {
    model: Rc<VecModel<ThumbItem>>,
    records: Rc<RefCell<Vec<LibraryImage>>>,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<ThumbnailCache>,
    quality: u8,
    thumb_px: Rc<Cell<u32>>,
    generation: Rc<Cell<u64>>,
    /// Current poller; replaced (and thereby stopped) on each `load()`.
    timer: Rc<RefCell<Option<Timer>>>,
}

impl LibraryGrid {
    pub fn new(
        db: Arc<Mutex<maple_db::Database>>,
        cache: Arc<ThumbnailCache>,
        quality: u8,
        thumb_px: u32,
    ) -> Self {
        Self {
            model: Rc::new(VecModel::default()),
            records: Rc::new(RefCell::new(Vec::new())),
            db,
            cache,
            quality,
            thumb_px: Rc::new(Cell::new(thumb_px)),
            generation: Rc::new(Cell::new(0)),
            timer: Rc::new(RefCell::new(None)),
        }
    }

    /// The backing model — bind to the `library-items` window property.
    pub fn model(&self) -> ModelRc<ThumbItem> {
        ModelRc::from(self.model.clone())
    }

    /// Snapshot of the currently loaded records (for the activate handler).
    pub fn records(&self) -> Rc<RefCell<Vec<LibraryImage>>> {
        self.records.clone()
    }

    /// Update the thumbnail render size. Takes effect on the next `load()`.
    /// Wired to the settings window in Phase 7.
    #[allow(dead_code)]
    pub fn set_thumb_size(&self, px: u32) {
        self.thumb_px.set(px);
    }

    /// Reload the grid from the database using `query`.
    ///
    /// Clears the grid immediately and cancels any in-progress previous load.
    pub fn load(&self, query: SearchQuery) {
        let gen = self.generation.get() + 1;
        self.generation.set(gen);

        // Drop the previous poller (stops it) and clear the grid.
        *self.timer.borrow_mut() = None;
        self.model.set_vec(Vec::<ThumbItem>::new());

        let db = self.db.clone();
        let cache = self.cache.clone();
        let quality = self.quality;
        let thumb_px = self.thumb_px.get();
        let (tx, rx) = mpsc::channel::<GridMsg>();

        // ── Worker thread (unchanged threading model) ─────────────
        std::thread::spawn(move || {
            let records = match db.lock() {
                Ok(d) => d.search_images(&query).unwrap_or_default(),
                Err(_) => return,
            };

            let _ = tx.send(GridMsg::Records(records.clone()));

            if records.is_empty() {
                let _ = tx.send(GridMsg::Done);
                return;
            }

            let parallelism = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4);
            let chunk_size = (records.len() / parallelism).max(1);

            std::thread::scope(|scope| {
                for (chunk_start, chunk) in records.chunks(chunk_size).enumerate() {
                    let tx = tx.clone();
                    let cache = cache.clone();
                    scope.spawn(move || {
                        for (i, rec) in chunk.iter().enumerate() {
                            let index = chunk_start * chunk_size + i;
                            match load_thumbnail(rec, thumb_px, quality, &cache) {
                                Ok((rgb, width, height)) => {
                                    let _ = tx.send(GridMsg::Thumb { index, rgb, width, height });
                                }
                                Err(e) => {
                                    tracing::warn!("Thumbnail failed for {}: {e}", rec.path.display());
                                    if !raw_preview_supported(&rec.path) {
                                        let _ = tx.send(GridMsg::Unsupported { index });
                                    }
                                }
                            }
                        }
                    });
                }
            });

            let _ = tx.send(GridMsg::Done);
        });

        // ── UI-thread poller (slint::Timer) ───────────────────────
        let timer = Timer::default();
        let slot = self.timer.clone();
        let model = self.model.clone();
        let records_ref = self.records.clone();
        let generation = self.generation.clone();

        timer.start(TimerMode::Repeated, Duration::from_millis(POLL_MS), move || {
            // Superseded by a newer load → stop self.
            if generation.get() != gen {
                if let Some(t) = slot.borrow().as_ref() {
                    t.stop();
                }
                return;
            }

            while let Ok(msg) = rx.try_recv() {
                match msg {
                    GridMsg::Records(records) => {
                        let placeholders: Vec<ThumbItem> =
                            records.iter().map(placeholder_item).collect();
                        *records_ref.borrow_mut() = records;
                        model.set_vec(placeholders);
                    }
                    GridMsg::Thumb { index, rgb, width, height } => {
                        if let Some(mut item) = model.row_data(index) {
                            item.image = rgb_to_image(&rgb, width, height);
                            item.loaded = true;
                            model.set_row_data(index, item);
                        }
                    }
                    GridMsg::Unsupported { index } => {
                        if let Some(mut item) = model.row_data(index) {
                            item.unsupported = true;
                            model.set_row_data(index, item);
                        }
                    }
                    GridMsg::Done => {
                        if let Some(t) = slot.borrow().as_ref() {
                            t.stop();
                        }
                        return;
                    }
                }
            }
        });

        *self.timer.borrow_mut() = Some(timer);
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Build the initial placeholder tile for a record (image filled in later).
fn placeholder_item(rec: &LibraryImage) -> ThumbItem {
    ThumbItem {
        image: Image::default(),
        name: SharedString::from(rec.meta.filename.as_deref().unwrap_or("…")),
        loaded: false,
        unsupported: false,
        stack_size: rec.stack_size.unwrap_or(0) as i32,
        score: SharedString::from(score_caption(rec.search_hit.as_ref())),
    }
}

/// Caption shown under a tile during search (empty when not a search hit).
fn score_caption(hit: Option<&SearchHit>) -> String {
    match hit {
        Some(SearchHit::Direct { .. }) => "direct".to_owned(),
        Some(SearchHit::Semantic { similarity, .. }) => {
            let pct = (similarity * 100.0).clamp(0.0, 100.0);
            format!("{pct:.0}% match")
        }
        None => String::new(),
    }
}

/// Wrap a tight RGB buffer into a Slint `Image`. Returns an empty image on a
/// size mismatch rather than panicking.
fn rgb_to_image(rgb: &[u8], width: u32, height: u32) -> Image {
    if rgb.len() != (width as usize * height as usize * 3) {
        return Image::default();
    }
    let mut buf = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
    buf.make_mut_bytes().copy_from_slice(rgb);
    Image::from_rgb8(buf)
}

/// Load a thumbnail for `rec`, using the thumbnail cache when possible.
///
/// Cache hit: decode stored WebP to RGB. Cache miss: render from disk, encode
/// WebP, store in cache, return RGB.
fn load_thumbnail(
    rec: &LibraryImage,
    max_size: u32,
    quality: u8,
    cache: &ThumbnailCache,
) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    if let Some(hash) = rec.hash {
        if let Some(webp) = cache.get(&hash) {
            return thumbnail::decode_webp_rgb(&webp);
        }
    }

    let (rgb, w, h) = thumbnail::render_to_rgb(&rec.path, max_size)?;

    if let Some(hash) = rec.hash {
        let webp = thumbnail::encode_webp_rgb(&rgb, w, h, quality);
        if let Err(e) = cache.insert(&hash, &webp) {
            tracing::warn!("Thumbnail cache write failed for {}: {e}", rec.path.display());
        }
    }

    Ok((rgb, w, h))
}
