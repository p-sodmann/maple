//! Library thumbnail grid controller (Slint).
//!
//! Mirrors the former GTK `LibraryGrid` (views/library/grid.rs), with the
//! library loaded one page at a time instead of all at once:
//!   1. A background thread queries one page of the DB and sends `Page`.
//!   2. Placeholder tiles for that page are appended to the grid immediately.
//!   3. Parallel thumbnail workers send `Thumb` messages; tiles are filled
//!      in-place as decoded RGB arrives.
//!   4. The view reports how far the user has scrolled (`request_more`), and
//!      the next page is fetched before the viewport reaches the end.
//!
//! Pages only ever *append*: `records` and the tile model grow together at
//! the tail, so a model row index is always the matching record's index —
//! the invariant `apply_marquee`, `selected_ids`, `apply_membership` and the
//! `activated(idx)` handler in `lib.rs` all rely on. Nothing is ever
//! recycled or windowed out from under them.
//!
//! Each `load()` increments a generation counter and resets paging to page 0;
//! the `slint::Timer` poller discards messages from superseded loads, so
//! rapid search changes never produce stale or interleaved grid content. An
//! append is *not* a new load and never bumps the generation. This is the
//! Slint analogue of the old `glib::timeout_add_local` poller — all
//! background work still runs on `std::thread` + `std::sync::mpsc`, and only
//! the UI-thread delivery changes.

use std::cell::{Cell, RefCell};
use std::collections::HashSet;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use slint::{
    Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel,
};

use maple_db::{LibraryImage, SearchOrder, SearchQuery, ThumbnailCache};
use maple_import::raw_preview_supported;

use crate::paging::{PageCursor, PAGE_SIZE};
use crate::services::images::{count_library, search_library};
use crate::thumbnail;
use crate::transforms::{append_date_groups, score_caption};
use crate::{DateGroup, ThumbItem};

const POLL_MS: u64 = 32;

// ── Worker messages ──────────────────────────────────────────────

enum GridMsg {
    /// One page of DB results, to be appended at `offset` (which always
    /// equals the current record count — see `PageCursor`).
    Page {
        offset: usize,
        records: Vec<LibraryImage>,
    },
    /// Row count of the whole (unpaged) result set, sent once per load.
    /// `None` when the query has no countable total (hybrid search).
    Total(Option<usize>),
    /// One thumbnail finished — carries decoded RGB pixels.
    Thumb {
        index: usize,
        rgb: Vec<u8>,
        width: u32,
        height: u32,
    },
    /// Format recognised but preview extraction not yet implemented.
    Unsupported { index: usize },
    /// Every thumbnail of one page has been generated.
    PageDone,
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
    /// Query of the current load, with its ordering already applied. Each
    /// page re-uses it with a different `limit`/`offset`.
    query: Rc<RefCell<SearchQuery>>,
    /// Paging state for the current load — reset by `load()`, advanced by
    /// `request_more()` and by each arriving page.
    cursor: Rc<RefCell<PageCursor>>,
    /// Sender for the current generation's pages. Workers from a superseded
    /// load keep the old one, whose receiver `load()` has already dropped.
    tx: Rc<RefCell<Option<mpsc::Sender<GridMsg>>>>,
    rx: Rc<RefCell<Option<mpsc::Receiver<GridMsg>>>>,
    /// Page workers that have not signalled `PageDone` yet. The poller runs
    /// only while this is non-zero, so an idle grid costs no timer wakeups.
    active_pages: Rc<Cell<usize>>,
    polling: Rc<Cell<bool>>,
    /// Notified with the unpaged row count of the current query (`None`
    /// while unknown) — the header's photo count, which can no longer be
    /// read off the loaded item count.
    total_hook: TotalHook,
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
/// Sink for the current query's total row count (see `on_total_count`).
type TotalHook = Rc<RefCell<Option<Rc<dyn Fn(Option<usize>)>>>>;
/// What `register` stores for `request_reload` to act on — see `REFRESH_HANDLE`.
type RefreshHandle = (LibraryGrid, Rc<RefCell<SearchQuery>>, ReloadHook);

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
    static REFRESH_HANDLE: RefCell<Option<RefreshHandle>> = const { RefCell::new(None) };
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
            query: Rc::new(RefCell::new(SearchQuery::default())),
            cursor: Rc::new(RefCell::new(PageCursor::default())),
            tx: Rc::new(RefCell::new(None)),
            rx: Rc::new(RefCell::new(None)),
            active_pages: Rc::new(Cell::new(0)),
            polling: Rc::new(Cell::new(false)),
            total_hook: Rc::new(RefCell::new(None)),
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

    /// Register the sink for the current query's total row count. Called
    /// once, from `run()`; fires on every `load()` (with `None`, the count
    /// not being known yet) and again when the count query returns.
    pub fn on_total_count(&self, hook: impl Fn(Option<usize>) + 'static) {
        *self.total_hook.borrow_mut() = Some(Rc::new(hook));
    }

    /// Reload the grid from the database using `query`.
    ///
    /// Clears the grid immediately, cancels any in-progress previous load,
    /// resets paging to page 0, and fetches the first page.
    pub fn load(&self, query: SearchQuery) {
        let gen = self.generation.get() + 1;
        self.generation.set(gen);

        // Drop the previous poller (stops it) and clear the grid. A fresh
        // load starts with no pre-selection intent — callers that want it
        // call `apply_membership()` afterward, which re-arms
        // `pending_membership` for whatever batch arrives.
        *self.timer.borrow_mut() = None;
        self.polling.set(false);
        self.active_pages.set(0);
        *self.pending_membership.borrow_mut() = None;
        self.model.set_vec(Vec::<ThumbItem>::new());
        self.date_groups.set_vec(Vec::<DateGroup>::new());
        self.records.borrow_mut().clear();
        self.cursor.borrow_mut().reset();
        self.report_total(None);

        // The ordering has to come from SQL. Sorting here would only ever
        // sort one page within itself, and page 2 would then interleave
        // wrongly with page 1 instead of continuing it.
        let order =
            if self.date_view.get() { SearchOrder::TakenDesc } else { SearchOrder::AddedDesc };
        *self.query.borrow_mut() = query.with_order(order);

        // A fresh channel per generation: pages still in flight from the
        // superseded load keep sending into the old one, whose receiver is
        // dropped right here, so their rows can never reach this grid.
        let (tx, rx) = mpsc::channel::<GridMsg>();
        *self.tx.borrow_mut() = Some(tx);
        *self.rx.borrow_mut() = Some(rx);

        self.fetch_next_page();
    }

    /// Ask the grid to have at least `rows` items loaded.
    ///
    /// Called from the view as it scrolls, with the item index the viewport
    /// is approaching plus a prefetch lead (see `library.slint`). Cheap and
    /// idempotent — it only raises a high-water mark and starts a fetch if
    /// one is actually due.
    pub fn request_more(&self, rows: i32) {
        if rows <= 0 {
            return;
        }
        self.cursor.borrow_mut().want(rows as usize);
        self.fetch_next_page();
    }

    /// Spawn the worker for the next page, if one is due (nothing already in
    /// flight, listing not exhausted, view wants more than is loaded).
    fn fetch_next_page(&self) {
        let Some(offset) = self.cursor.borrow_mut().take_next_offset() else {
            return;
        };
        let Some(tx) = self.tx.borrow().clone() else {
            return;
        };

        let gen = self.generation.get();
        let query = self.query.borrow().clone().with_limit(PAGE_SIZE).with_offset(offset);
        let db = self.db.clone();
        let cache = self.cache.clone();
        let quality = self.quality;
        let thumb_px = self.thumb_px.get();
        // The row count is a property of the query, not of the page.
        let with_total = offset == 0;

        self.active_pages.set(self.active_pages.get() + 1);
        tracing::debug!("library page: offset {offset}, {PAGE_SIZE} rows");

        // ── Worker thread (unchanged threading model) ─────────────
        std::thread::spawn(move || {
            let records = search_library(&db, &query);

            if with_total {
                let _ = tx.send(GridMsg::Total(count_library(&db, &query)));
            }
            let page_len = records.len();
            let _ = tx.send(GridMsg::Page { offset, records: records.clone() });

            if page_len > 0 {
                let parallelism =
                    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
                let chunk_size = (page_len / parallelism).max(1);

                std::thread::scope(|scope| {
                    for (chunk_index, chunk) in records.chunks(chunk_size).enumerate() {
                        let tx = tx.clone();
                        let cache = cache.clone();
                        scope.spawn(move || {
                            for (i, rec) in chunk.iter().enumerate() {
                                // Absolute row index: this page starts at
                                // `offset` in the accumulated grid.
                                let index = offset + chunk_index * chunk_size + i;
                                match load_thumbnail(rec, thumb_px, quality, &cache) {
                                    Ok((rgb, width, height)) => {
                                        let _ =
                                            tx.send(GridMsg::Thumb { index, rgb, width, height });
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Thumbnail failed for {}: {e}",
                                            rec.path.display()
                                        );
                                        if !raw_preview_supported(&rec.path) {
                                            let _ = tx.send(GridMsg::Unsupported { index });
                                        }
                                    }
                                }
                            }
                        });
                    }
                });
            }

            let _ = tx.send(GridMsg::PageDone);
        });

        self.start_poller(gen);
    }

    /// Start the UI-thread poller for generation `gen`, unless one is
    /// already running. It stops itself once every page worker has finished,
    /// and is restarted by the next `fetch_next_page()`.
    fn start_poller(&self, gen: u64) {
        if self.polling.get() {
            return;
        }
        self.polling.set(true);

        let timer = Timer::default();
        let slot = self.timer.clone();
        let grid = self.clone();

        timer.start(TimerMode::Repeated, Duration::from_millis(POLL_MS), move || {
            // Superseded by a newer load → do nothing, ever. `load()` drops
            // this timer (which stops it) before starting the replacement,
            // so this is belt and braces; it deliberately does *not* stop
            // `slot`, which by now holds the *new* load's poller.
            if grid.generation.get() != gen {
                return;
            }

            loop {
                let Some(msg) = grid.rx.borrow().as_ref().and_then(|rx| rx.try_recv().ok()) else {
                    return;
                };
                match msg {
                    GridMsg::Page { offset, records } => grid.append_page(offset, records),
                    GridMsg::Total(total) => grid.report_total(total),
                    GridMsg::Thumb { index, rgb, width, height } => {
                        if let Some(mut item) = grid.model.row_data(index) {
                            item.image = rgb_to_image(&rgb, width, height);
                            item.loaded = true;
                            grid.model.set_row_data(index, item);
                        }
                    }
                    GridMsg::Unsupported { index } => {
                        if let Some(mut item) = grid.model.row_data(index) {
                            item.unsupported = true;
                            grid.model.set_row_data(index, item);
                        }
                    }
                    GridMsg::PageDone => {
                        grid.active_pages.set(grid.active_pages.get().saturating_sub(1));
                        if grid.active_pages.get() == 0 {
                            if let Some(t) = slot.borrow().as_ref() {
                                t.stop();
                            }
                            grid.polling.set(false);
                            return;
                        }
                    }
                }
            }
        });

        *self.timer.borrow_mut() = Some(timer);
    }

    /// Append one page's records and tiles at the tail, in lockstep.
    fn append_page(&self, offset: usize, page: Vec<LibraryImage>) {
        let mut records = self.records.borrow_mut();
        if offset != records.len() {
            // Unreachable: only one page is ever in flight, and the next is
            // requested only once the previous has landed. Refuse to append
            // rather than mis-map this page's thumbnails onto other tiles.
            drop(records);
            tracing::warn!(
                "library page at offset {offset} does not continue the loaded rows — paging stopped"
            );
            self.cursor.borrow_mut().abandon();
            return;
        }

        // Reads the *current* value, not a snapshot from when `load()` was
        // called — if `apply_membership()` armed this after the background
        // query had already started, this still-arriving page comes in with
        // the right tiles pre-selected.
        let pending = self.pending_membership.borrow();
        let placeholders: Vec<ThumbItem> = page
            .iter()
            .map(|r| {
                let selected = pending.as_ref().is_some_and(|set| set.contains(&r.id));
                placeholder_item(r, selected)
            })
            .collect();
        drop(pending);

        let page_len = page.len();
        records.extend(page);
        // Day groups are recomputed from the accumulated records, not from
        // the page alone: a page can continue the day the previous one
        // ended on, and that group has to stay one contiguous run.
        let mut groups: Vec<DateGroup> = self.date_groups.iter().collect();
        append_date_groups(&mut groups, &records, offset);
        drop(records);

        self.model.extend(placeholders);
        sync_date_groups(&self.date_groups, &groups);

        self.cursor.borrow_mut().page_arrived(page_len);
        // One `request_more` can span several pages (a long jump of the
        // scrollbar, or a viewport taller than one page).
        self.fetch_next_page();
    }

    fn report_total(&self, total: Option<usize>) {
        let hook = self.total_hook.borrow().clone();
        if let Some(hook) = hook {
            hook(total);
        }
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
                            let g = maple_db::lock_db(&self.db);
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
                            // Released before the Slint model write below.
                            drop(g);
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
                if let Err(e) = maple_db::lock_db(&self.db).update_collection_representative(coll_id)
                {
                    tracing::warn!("update_collection_representative {coll_id}: {e}");
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

/// Bring the date-group model in line with `new`, touching as few rows as
/// possible. Appending a page only ever grows the last existing group and
/// adds groups after it; replacing the whole model would instead tear down
/// and rebuild every tile in the date-grouped view on every page.
fn sync_date_groups(model: &VecModel<DateGroup>, new: &[DateGroup]) {
    for (i, group) in new.iter().enumerate() {
        match model.row_data(i) {
            Some(old) if old == *group => {}
            Some(_) => model.set_row_data(i, group.clone()),
            None => model.push(group.clone()),
        }
    }
    while model.row_count() > new.len() {
        model.remove(model.row_count() - 1);
    }
}

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
