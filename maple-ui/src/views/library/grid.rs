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

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;

use maple_db::{LibraryImage, SearchQuery};

use crate::thumbnail::generate_thumbnail;

const THUMB_PX: u32 = 200;
const POLL_MS: u64 = 32;

// ── Worker messages ──────────────────────────────────────────────

enum GridMsg {
    /// Initial batch of DB results (establishes grid size).
    Records(Vec<LibraryImage>),
    /// One thumbnail finished generating.
    Thumb { index: usize, png_bytes: Vec<u8> },
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
    /// Active records, shared with the child-activated signal handler.
    records: Rc<RefCell<Vec<LibraryImage>>>,
    /// Monotonically increasing; stale poller closures detect mismatches.
    generation: Rc<Cell<u64>>,
}

impl LibraryGrid {
    /// Create the grid.
    ///
    /// `on_activate` is called on the GTK main thread whenever the user clicks
    /// a cell.  It receives the selected `LibraryImage` and the root
    /// `gtk4::Window` (so callers can open a transient detail window).
    pub fn new(
        db: Arc<Mutex<maple_db::Database>>,
        on_activate: impl Fn(LibraryImage, gtk4::Window) + 'static,
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
            .build();

        let records: Rc<RefCell<Vec<LibraryImage>>> = Rc::new(RefCell::new(Vec::new()));

        // Wire the activation signal once — it always reads from the current
        // `records` snapshot, so it stays valid across reloads.
        flow_box.connect_child_activated({
            let records = records.clone();
            move |fb, child| {
                let idx = child.index() as usize;
                if let Some(rec) = records.borrow().get(idx).cloned() {
                    if let Some(window) = fb.root().and_downcast::<gtk4::Window>() {
                        on_activate(rec, window);
                    }
                }
            }
        });

        Self {
            widget: flow_box,
            db,
            records,
            generation: Rc::new(Cell::new(0)),
        }
    }

    /// The underlying widget — embed inside a `gtk4::ScrolledWindow`.
    pub fn widget(&self) -> &gtk4::FlowBox {
        &self.widget
    }

    /// Reload the grid from the database using `query`.
    ///
    /// Clears the grid immediately and cancels any in-progress previous load.
    pub fn load(&self, query: SearchQuery) {
        // Bump generation so any running poller stops picking up old messages.
        let gen = self.generation.get() + 1;
        self.generation.set(gen);

        // Clear existing cells right away.
        while let Some(child) = self.widget.first_child() {
            self.widget.remove(&child);
        }

        let db = self.db.clone();
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
                    scope.spawn(move || {
                        for (i, rec) in chunk.iter().enumerate() {
                            let index = chunk_start * chunk_size + i;
                            match generate_thumbnail(&rec.path, THUMB_PX) {
                                Ok(bytes) => {
                                    let _ = tx.send(GridMsg::Thumb {
                                        index,
                                        png_bytes: bytes,
                                    });
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

        glib::timeout_add_local(Duration::from_millis(POLL_MS), move || {
            if generation.get() != gen {
                return glib::ControlFlow::Break; // superseded load — stop polling
            }

            while let Ok(msg) = rx.try_recv() {
                match msg {
                    GridMsg::Records(records) => {
                        *records_ref.borrow_mut() = records.clone();
                        for rec in &records {
                            let child = gtk4::FlowBoxChild::new();
                            let name = rec.meta.filename.as_deref().unwrap_or("…");
                            child.set_child(Some(&build_placeholder(name)));
                            flow_box.append(&child);
                        }
                    }

                    GridMsg::Thumb { index, png_bytes } => {
                        if let Some(child) = flow_box.child_at_index(index as i32) {
                            let records = records_ref.borrow();
                            if let Some(rec) = records.get(index) {
                                let bytes = glib::Bytes::from(&png_bytes);
                                if let Ok(texture) = gdk::Texture::from_bytes(&bytes) {
                                    let name = rec.meta.filename.as_deref().unwrap_or("?");
                                    child.set_child(Some(&build_cell(&texture, name)));
                                }
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

// ── Cell widgets ─────────────────────────────────────────────────

fn build_placeholder(name: &str) -> gtk4::Box {
    let spinner = gtk4::Spinner::builder()
        .spinning(true)
        .width_request(32)
        .height_request(32)
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .hexpand(true)
        .vexpand(true)
        .build();

    let frame = gtk4::Box::builder()
        .width_request(THUMB_PX as i32)
        .height_request(THUMB_PX as i32)
        .hexpand(true)
        .vexpand(true)
        .build();
    frame.append(&spinner);

    labeled_cell(&frame, name)
}

fn build_cell(texture: &gdk::Texture, name: &str) -> gtk4::Box {
    let picture = gtk4::Picture::for_paintable(texture);
    picture.set_size_request(THUMB_PX as i32, THUMB_PX as i32);
    picture.set_content_fit(gtk4::ContentFit::Cover);

    labeled_cell(&picture, name)
}

/// Wrap any widget with a caption label beneath it.
fn labeled_cell(content: &impl IsA<gtk4::Widget>, name: &str) -> gtk4::Box {
    let label = gtk4::Label::new(Some(name));
    label.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
    label.set_max_width_chars(20);
    label.add_css_class("caption");

    let cell = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .build();
    cell.append(content);
    cell.append(&label);
    cell
}
