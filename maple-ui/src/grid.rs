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
use std::cmp::Reverse;
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use slint::{
    Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel,
};

use maple_db::{LibraryImage, SearchQuery, ThumbnailCache};
use maple_import::raw_preview_supported;

use crate::services::images::search_library;
use crate::thumbnail;
use crate::transforms::{build_date_groups, score_caption};
use crate::{DateGroup, ThumbItem};

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
    date_groups: Rc<VecModel<DateGroup>>,
    records: Rc<RefCell<Vec<LibraryImage>>>,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<ThumbnailCache>,
    quality: u8,
    thumb_px: Rc<Cell<u32>>,
    date_view: Rc<Cell<bool>>,
    generation: Rc<Cell<u64>>,
    /// Current poller; replaced (and thereby stopped) on each `load()`.
    timer: Rc<RefCell<Option<Timer>>>,
    /// When set, placeholder tiles are built pre-selected according to this
    /// membership set as the in-flight `load()`'s results stream in —
    /// closes the race where `apply_membership()` runs before the
    /// background query has delivered its first batch (nothing to mark yet)
    /// by applying the same intent to whatever arrives afterward. See
    /// `apply_membership`/`clear_selection`.
    pending_membership: Rc<RefCell<Option<HashSet<i64>>>>,
}

type ReloadHook = Rc<dyn Fn()>;

thread_local! {
    /// The app's single `LibraryGrid` + its last-used query, registered once
    /// from `run()`. Lets any other window (rotation, library restructure, …)
    /// ask the library to refresh after an out-of-band change, without
    /// `LibraryGrid`/`SearchQuery` having to be threaded through that
    /// window's own constructor.
    ///
    /// The third element is a "same-context refresh" hook — re-applies the
    /// active select-mode target's membership (if any) after the reload,
    /// since a fresh `load()` always starts every tile unselected and has
    /// no way to know about `select_target` on its own (see `lib.rs`'s
    /// `resync_selection`). Called by both `request_reload` and the
    /// date-view toggle, the two places that reload with the *same* query
    /// rather than switching to a new context.
    static REFRESH_HANDLE: RefCell<Option<(LibraryGrid, Rc<RefCell<SearchQuery>>, ReloadHook)>> =
        const { RefCell::new(None) };
}

/// Register the app's `LibraryGrid`, its current-query cell, and the
/// select-mode resync hook. Called once from `run()` right after all three
/// are constructed.
pub fn register(grid: LibraryGrid, current_query: Rc<RefCell<SearchQuery>>, on_reloaded: ReloadHook) {
    REFRESH_HANDLE.with(|cell| *cell.borrow_mut() = Some((grid, current_query, on_reloaded)));
}

/// Reload the library grid with whatever query it was last showing, then
/// re-apply any active select-mode membership sync (see `REFRESH_HANDLE`).
///
/// Call this after a change made from outside the grid's own callbacks
/// leaves its cached records/thumbnails stale — e.g. a library restructure
/// moving files, or an in-place image rotation changing a thumbnail's hash.
/// A no-op before `register` has run.
pub fn request_reload() {
    REFRESH_HANDLE.with(|cell| {
        if let Some((grid, current_query, on_reloaded)) = cell.borrow().as_ref() {
            grid.load(current_query.borrow().clone());
            on_reloaded();
        }
    });
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
            date_groups: Rc::new(VecModel::default()),
            records: Rc::new(RefCell::new(Vec::new())),
            db,
            cache,
            quality,
            thumb_px: Rc::new(Cell::new(thumb_px)),
            date_view: Rc::new(Cell::new(false)),
            generation: Rc::new(Cell::new(0)),
            timer: Rc::new(RefCell::new(None)),
            pending_membership: Rc::new(RefCell::new(None)),
        }
    }

    /// The backing model — bind to the `library-items` window property.
    pub fn model(&self) -> ModelRc<ThumbItem> {
        ModelRc::from(self.model.clone())
    }

    /// Day-grouped headers for the current items — bind to the
    /// `library-date-groups` window property. Only meaningful (contiguous)
    /// when the grid was last loaded with date-view sorting enabled.
    pub fn date_groups_model(&self) -> ModelRc<DateGroup> {
        ModelRc::from(self.date_groups.clone())
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

    /// Sort by photo-taken date (grouped into contiguous days) instead of
    /// library-insertion order. Takes effect on the next `load()`.
    pub fn set_date_view(&self, on: bool) {
        self.date_view.set(on);
    }

    /// Reload the grid from the database using `query`.
    ///
    /// Clears the grid immediately and cancels any in-progress previous load.
    pub fn load(&self, query: SearchQuery) {
        let gen = self.generation.get() + 1;
        self.generation.set(gen);

        // Drop the previous poller (stops it) and clear the grid. A fresh
        // load starts with no pre-selection intent — callers that want it
        // call `apply_membership()` afterward, which re-arms
        // `pending_membership` for whatever batch arrives.
        *self.timer.borrow_mut() = None;
        *self.pending_membership.borrow_mut() = None;
        self.model.set_vec(Vec::<ThumbItem>::new());

        let db = self.db.clone();
        let cache = self.cache.clone();
        let quality = self.quality;
        let thumb_px = self.thumb_px.get();
        let date_view = self.date_view.get();
        let (tx, rx) = mpsc::channel::<GridMsg>();

        // ── Worker thread (unchanged threading model) ─────────────
        std::thread::spawn(move || {
            let mut records = search_library(&db, &query);

            // Re-sort into contiguous day groups (newest day first) so the
            // date-grouped view can slice `start..start+count` per day.
            if date_view {
                records.sort_by_key(|r| Reverse(r.meta.taken_at.unwrap_or(r.added_at)));
            }

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
        let date_groups = self.date_groups.clone();
        let records_ref = self.records.clone();
        let generation = self.generation.clone();
        let pending_membership = self.pending_membership.clone();

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
                        // Reads the *current* value, not a snapshot from when
                        // `load()` was called — if `apply_membership()` armed
                        // this after the background query had already
                        // started, this still-arriving first batch comes in
                        // with the right tiles pre-selected.
                        let pending = pending_membership.borrow();
                        let placeholders: Vec<ThumbItem> = records
                            .iter()
                            .map(|r| {
                                let selected = pending.as_ref().is_some_and(|set| set.contains(&r.id));
                                placeholder_item(r, selected)
                            })
                            .collect();
                        drop(pending);
                        date_groups.set_vec(build_date_groups(&records));
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

    /// Apply a select-mode gesture reported by `SelectOverlay` (library.slint)
    /// and return the new total selected count.
    ///
    /// `base < 0` means the flat grid (page padding applies, `count` is the
    /// full item count); `base >= 0` means one date-group's tile block
    /// (`base` is the group's start index into the model, `count` is the
    /// group's own size, no page padding — tiles there are positioned
    /// relative to the group's own container). A near-zero-movement gesture
    /// is treated as a tap (toggles the single covered tile); anything
    /// larger sets every covered tile selected (drag never deselects).
    ///
    /// `pitch_x`/`pitch_y`/`pad` mirror the geometry constants in
    /// `library.slint` (`Theme.gap` = 14px, `Theme.pad` = 22px, tile label
    /// height = 28px) — the render cell size is already shared via
    /// `thumb_px`, since it's the same settings value the UI binds as
    /// `library-cell-size`.
    ///
    /// `sync_collection` — when the grid is currently showing exactly one
    /// collection's photos (the Library was filtered via a Collections
    /// gallery card), every tile is a member by construction, so a tap's
    /// selection flip is mirrored live into DB membership: deselecting
    /// removes the image from the collection, re-selecting adds it back.
    /// `None` outside that context (general browsing / search results),
    /// where select-mode is purely "build a set to add to some collection".
    pub fn apply_marquee(
        &self,
        base: i32,
        count: i32,
        rect: (f32, f32, f32, f32),
        columns: f32,
        sync_collection: Option<i64>,
    ) -> i32 {
        const GAP: f32 = 14.0;
        const PAD: f32 = 22.0;
        const LABEL_H: f32 = 28.0;

        let (x0, y0, x1, y1) = rect;
        let columns = (columns.round() as i32).max(1);
        let cell = self.thumb_px.get() as f32;
        let pitch_x = cell + GAP;
        let pitch_y = cell + LABEL_H + GAP;

        let flat = base < 0;
        let offset = if flat { PAD } else { 0.0 };

        let (xa, xb) = (x0.min(x1), x0.max(x1));
        let (ya, yb) = (y0.min(y1), y0.max(y1));
        let is_tap = (xb - xa) < 6.0 && (yb - ya) < 6.0;

        let col0 = (((xa - offset) / pitch_x).floor() as i32).clamp(0, columns - 1);
        let col1 = (((xb - offset) / pitch_x).floor() as i32).clamp(0, columns - 1);
        let row0 = (((ya - offset) / pitch_y).floor() as i32).max(0);
        let row1 = (((yb - offset) / pitch_y).floor() as i32).max(0);

        let base = if flat { 0 } else { base };
        let total = self.model.row_count() as i32;

        let records = self.records.borrow();
        let mut membership_changed = false;

        for r in row0..=row1 {
            for c in col0..=col1 {
                let local = r * columns + c;
                if local < 0 || local >= count {
                    continue;
                }
                let idx = base + local;
                if idx < 0 || idx >= total {
                    continue;
                }
                let idx = idx as usize;
                if let Some(mut item) = self.model.row_data(idx) {
                    let new_selected = if is_tap { !item.selected } else { true };
                    if new_selected != item.selected {
                        if let (Some(coll_id), Some(rec)) = (sync_collection, records.get(idx)) {
                            if let Ok(g) = self.db.lock() {
                                let res = if new_selected {
                                    g.add_image_to_collection(coll_id, rec.id)
                                } else {
                                    g.remove_image_from_collection(coll_id, rec.id)
                                };
                                if let Err(e) = res {
                                    tracing::warn!(
                                        "collection select-sync image={} collection={coll_id}: {e}",
                                        rec.id
                                    );
                                }
                            }
                            membership_changed = true;
                        }
                        item.selected = new_selected;
                        self.model.set_row_data(idx, item);
                    }
                }
            }
        }
        drop(records);

        if membership_changed {
            if let Some(coll_id) = sync_collection {
                if let Ok(g) = self.db.lock() {
                    if let Err(e) = g.update_collection_representative(coll_id) {
                        tracing::warn!("update_collection_representative {coll_id}: {e}");
                    }
                }
            }
        }

        self.selected_count()
    }

    /// Image ids of every currently-selected tile, in model order.
    pub fn selected_ids(&self) -> Vec<i64> {
        let records = self.records.borrow();
        (0..self.model.row_count())
            .filter_map(|i| {
                let item = self.model.row_data(i)?;
                if !item.selected {
                    return None;
                }
                records.get(i).map(|r| r.id)
            })
            .collect()
    }

    /// Sync every currently loaded tile's checkbox to real membership in
    /// `member_ids`, and keep applying it to tiles from this same in-flight
    /// `load()` as they arrive (see `pending_membership` on the struct).
    /// Used when a sidebar collection dot becomes the "editing target"
    /// while in select-mode: in a general/unfiltered view most tiles won't
    /// be members, so this shows exactly which ones already are; in a
    /// gallery-card-filtered view every tile is a member, so this is
    /// equivalent to selecting all of them. Subsequent taps then live-sync
    /// against the same target via `apply_marquee`'s `sync_collection`.
    /// Returns the new selected count.
    pub fn apply_membership(&self, member_ids: &HashSet<i64>) -> i32 {
        *self.pending_membership.borrow_mut() = Some(member_ids.clone());
        let records = self.records.borrow();
        for i in 0..self.model.row_count() {
            if let Some(mut item) = self.model.row_data(i) {
                let is_member = records.get(i).is_some_and(|r| member_ids.contains(&r.id));
                if item.selected != is_member {
                    item.selected = is_member;
                    self.model.set_row_data(i, item);
                }
            }
        }
        drop(records);
        self.selected_count()
    }

    /// Clear every tile's selection flag and stop applying any pending
    /// membership sync to newly arriving records (does not exit
    /// select-mode itself — callers own that toggle).
    pub fn clear_selection(&self) {
        *self.pending_membership.borrow_mut() = None;
        for i in 0..self.model.row_count() {
            if let Some(mut item) = self.model.row_data(i) {
                if item.selected {
                    item.selected = false;
                    self.model.set_row_data(i, item);
                }
            }
        }
    }

    fn selected_count(&self) -> i32 {
        (0..self.model.row_count())
            .filter(|&i| self.model.row_data(i).is_some_and(|item| item.selected))
            .count() as i32
    }
}

// ── Helpers ──────────────────────────────────────────────────────

/// Build the initial placeholder tile for a record (image filled in later).
fn placeholder_item(rec: &LibraryImage, selected: bool) -> ThumbItem {
    ThumbItem {
        id: rec.id as i32,
        image: Image::default(),
        name: SharedString::from(rec.meta.filename.as_deref().unwrap_or("…")),
        loaded: false,
        unsupported: false,
        stack_size: rec.stack_size.unwrap_or(0) as i32,
        score: SharedString::from(score_caption(rec.search_hit.as_ref())),
        selected,
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
