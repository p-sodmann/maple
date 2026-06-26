//! Library thumbnail grid.
//!
//! `LibraryGrid` wraps a `gtk4::FlowBox` and manages asynchronous loading:
//!   1. A background thread queries the DB and sends `Records`.
//!   2. Placeholder spinners fill the grid immediately.
//!   3. Parallel thumbnail workers send `Thumb` messages; placeholders are
//!      replaced in-place as they arrive.
//!
//! Each `load()` call increments an internal generation counter.  The
//! glib poller discards messages from superseded loads, so rapid search
//! changes never produce stale or interleaved grid content.
//!
//! # Thumbnail cache
//!
//! Workers check the `ThumbnailCache` (LMDB) before decoding.  On a cache
//! hit the WebP bytes are decoded in the worker thread to packed RGB.  On a
//! miss the image is decoded, Lanczos3-resized, and encoded as WebP before
//! being written to the cache.  The UI thread always receives plain RGB pixels
//! and constructs a `gdk::Texture` from them without requiring any gdk-pixbuf
//! format loader for WebP.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use gtk4::gdk;
use gtk4::gdk_pixbuf;
use gtk4::glib;
use gtk4::prelude::*;

use maple_db::{LibraryImage, SearchQuery, ThumbnailCache};

use crate::thumbnail;

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
    /// All thumbnails have been generated.
    Done,
}

// ── Public interface ─────────────────────────────────────────────

/// Thumbnail grid that reloads from the DB on demand.
///
/// Cheap to clone — all internal state is reference-counted, so a clone
/// shares the same underlying grid widget and record list.
#[derive(Clone)]
pub struct LibraryGrid {
    widget: gtk4::FlowBox,
    db: Arc<Mutex<maple_db::Database>>,
    records: Rc<RefCell<Vec<LibraryImage>>>,
    generation: Rc<Cell<u64>>,
    cache: Arc<ThumbnailCache>,
    thumb_quality: u8,
    /// Longest-edge pixel size for generated and displayed thumbnails.
    thumb_px: Rc<Cell<u32>>,
}

impl LibraryGrid {
    /// Create the grid.
    ///
    /// `on_activate` is called on the GTK main thread whenever the user clicks
    /// a cell.  It receives the index of the activated image, a snapshot of all
    /// currently loaded records, and the root `gtk4::Window`.
    pub fn new(
        db: Arc<Mutex<maple_db::Database>>,
        cache: Arc<ThumbnailCache>,
        thumb_quality: u8,
        thumb_px: u32,
        on_activate: impl Fn(usize, Vec<LibraryImage>, gtk4::Window) + 'static,
    ) -> Self {
        let flow_box = gtk4::FlowBox::builder()
            .valign(gtk4::Align::Start)
            .max_children_per_line(30)
            .min_children_per_line(2)
            .selection_mode(gtk4::SelectionMode::Single)
            .homogeneous(true)
            .row_spacing(8)
            .column_spacing(8)
            .margin_start(12)
            .margin_end(12)
            .margin_top(12)
            .margin_bottom(12)
            .css_classes(["maple-grid"])
            .build();

        let records: Rc<RefCell<Vec<LibraryImage>>> = Rc::new(RefCell::new(Vec::new()));

        flow_box.connect_child_activated({
            let records = records.clone();
            move |fb, child| {
                let idx = child.index() as usize;
                let snap = records.borrow().clone();
                if snap.get(idx).is_some() {
                    if let Some(window) = fb.root().and_downcast::<gtk4::Window>() {
                        on_activate(idx, snap, window);
                    }
                }
            }
        });

        Self {
            widget: flow_box,
            db,
            records,
            generation: Rc::new(Cell::new(0)),
            cache,
            thumb_quality,
            thumb_px: Rc::new(Cell::new(thumb_px)),
        }
    }

    /// The underlying widget — embed inside a `gtk4::ScrolledWindow`.
    pub fn widget(&self) -> &gtk4::FlowBox {
        &self.widget
    }

    /// Update the thumbnail render size.  Takes effect on the next `load()`.
    pub fn set_thumb_size(&self, px: u32) {
        self.thumb_px.set(px);
    }

    /// Reload the grid from the database using `query`.
    ///
    /// Clears the grid immediately and cancels any in-progress previous load.
    pub fn load(&self, query: SearchQuery) {
        let gen = self.generation.get() + 1;
        self.generation.set(gen);

        while let Some(child) = self.widget.first_child() {
            self.widget.remove(&child);
        }

        let db = self.db.clone();
        let cache = self.cache.clone();
        let quality = self.thumb_quality;
        let thumb_px = self.thumb_px.get();
        let (tx, rx) = mpsc::channel::<GridMsg>();

        // ── Worker thread ─────────────────────────────────────────
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
                                    tracing::warn!(
                                        "Thumbnail failed for {}: {e}",
                                        rec.path.display()
                                    );
                                }
                            }
                        }
                    });
                }
            });

            let _ = tx.send(GridMsg::Done);
        });

        // ── UI-thread poller ──────────────────────────────────────
        let flow_box = self.widget.clone();
        let records_ref = self.records.clone();
        let generation = self.generation.clone();
        // Snapshot size so placeholder and cells are consistent within one load.
        let px = thumb_px;

        glib::timeout_add_local(Duration::from_millis(POLL_MS), move || {
            if generation.get() != gen {
                return glib::ControlFlow::Break;
            }

            while let Ok(msg) = rx.try_recv() {
                match msg {
                    GridMsg::Records(records) => {
                        *records_ref.borrow_mut() = records.clone();
                        for rec in &records {
                            let child = gtk4::FlowBoxChild::new();
                            let name = rec.meta.filename.as_deref().unwrap_or("…");
                            child.set_child(Some(&build_placeholder(name, rec.similarity, px)));
                            flow_box.append(&child);
                        }
                    }

                    GridMsg::Thumb { index, rgb, width, height } => {
                        if let Some(child) = flow_box.child_at_index(index as i32) {
                            let records = records_ref.borrow();
                            if let Some(rec) = records.get(index) {
                                let bytes = glib::Bytes::from(&rgb);
                                let pixbuf = gdk_pixbuf::Pixbuf::from_bytes(
                                    &bytes,
                                    gdk_pixbuf::Colorspace::Rgb,
                                    false,
                                    8,
                                    width as i32,
                                    height as i32,
                                    (width * 3) as i32,
                                );
                                let texture = gdk::Texture::for_pixbuf(&pixbuf);
                                let name = rec.meta.filename.as_deref().unwrap_or("?");
                                child.set_child(Some(&build_cell(
                                    &texture,
                                    name,
                                    rec.similarity,
                                    px,
                                )));
                            }
                        }
                    }

                    GridMsg::Done => return glib::ControlFlow::Break,
                }
            }

            glib::ControlFlow::Continue
        });
    }
}

// ── Thumbnail loading with cache ─────────────────────────────────

/// Load a thumbnail for `rec`, using the LMDB cache when possible.
///
/// Cache hit: decode stored WebP to RGB.
/// Cache miss: render from disk, encode WebP, store in cache, return RGB.
fn load_thumbnail(
    rec: &LibraryImage,
    max_size: u32,
    quality: u8,
    cache: &ThumbnailCache,
) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    // Try the cache first if we have a content hash.
    if let Some(hash) = rec.hash {
        if let Some(webp) = cache.get(&hash) {
            return thumbnail::decode_webp_rgb(&webp);
        }
    }

    // Cache miss — decode and resize from disk.
    let (rgb, w, h) = thumbnail::render_to_rgb(&rec.path, max_size)?;

    // Write to cache (best-effort — a write failure just means a cold miss
    // next time, not a hard error).
    if let Some(hash) = rec.hash {
        let webp = thumbnail::encode_webp_rgb(&rgb, w, h, quality);
        if let Err(e) = cache.insert(&hash, &webp) {
            tracing::warn!("Thumbnail cache write failed for {}: {e}", rec.path.display());
        }
    }

    Ok((rgb, w, h))
}

// ── Cell widgets ─────────────────────────────────────────────────

fn build_placeholder(name: &str, similarity: Option<f32>, px: u32) -> gtk4::Box {
    let spinner = gtk4::Spinner::builder()
        .spinning(true)
        .width_request(32)
        .height_request(32)
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .hexpand(true)
        .vexpand(true)
        .build();
    spinner.add_css_class("maple-slow-spinner");

    let frame = gtk4::Box::builder()
        .width_request(px as i32)
        .height_request(px as i32)
        .hexpand(true)
        .vexpand(true)
        .css_classes(["maple-placeholder"])
        .build();
    frame.append(&spinner);

    labeled_cell(&frame, name, similarity)
}

fn build_cell(texture: &gdk::Texture, name: &str, similarity: Option<f32>, px: u32) -> gtk4::Box {
    let picture = gtk4::Picture::for_paintable(texture);
    picture.set_size_request(px as i32, px as i32);
    picture.set_content_fit(gtk4::ContentFit::Cover);
    picture.set_overflow(gtk4::Overflow::Hidden);
    picture.add_css_class("maple-thumb");

    labeled_cell(&picture, name, similarity)
}

fn labeled_cell(
    content: &impl IsA<gtk4::Widget>,
    name: &str,
    similarity: Option<f32>,
) -> gtk4::Box {
    let label = gtk4::Label::new(Some(name));
    label.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
    label.set_max_width_chars(20);
    label.add_css_class("caption");
    label.add_css_class("dim-label");

    let cell = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .css_classes(["maple-card"])
        .build();
    cell.append(content);
    cell.append(&label);

    if let Some(sim) = similarity {
        let pct = (sim * 100.0).clamp(0.0, 100.0);
        let score = gtk4::Label::new(Some(&format!("{pct:.0}% match")));
        score.add_css_class("caption");
        score.add_css_class("maple-score");
        cell.append(&score);
    }

    cell
}
