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
//! A `refresh()` is the same query again, and deliberately *not* a `load()`:
//! it re-reads exactly the rows already on screen and patches them in place,
//! keeping the scroll position, the selection, and every thumbnail that has
//! already been decoded. Everything that changes the library out of band
//! goes through it — the 60-second scanner, a sync pass, a rotation — and
//! any of those reloading from page 0 would throw the user back to the top
//! of their library, in the scanner's case once a minute.
//!
//! Each `load()` increments a generation counter and resets paging to page 0;
//! the `slint::Timer` poller discards messages from superseded loads, so
//! rapid search changes never produce stale or interleaved grid content. An
//! append is *not* a new load and never bumps the generation. This is the
//! Slint analogue of the old `glib::timeout_add_local` poller — all
//! background work still runs on `std::thread` + `std::sync::mpsc`, and only
//! the UI-thread delivery changes.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use slint::{
    Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, Timer, TimerMode, VecModel,
};

use maple_db::{LibraryImage, SearchOrder, SearchQuery, ThumbnailCache};
use maple_import::raw_preview_supported;

use crate::paging::{PageCursor, PAGE_SIZE};
use crate::remote::RemoteBlobs;
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
    /// The rows already on screen, read again. Replaces them in place rather
    /// than appending — see [`LibraryGrid::refresh`].
    ///
    /// `asked_for` is the limit the query ran with: fewer rows than that
    /// means the listing ended, which is the only way a refresh can learn
    /// that photos were deleted off the end.
    Refresh {
        records: Vec<LibraryImage>,
        asked_for: usize,
    },
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

/// Re-read the library grid without disturbing it, then re-apply any active
/// select-mode membership sync (see `REFRESH_HANDLE`).
///
/// Call this after a change made from outside the grid's own callbacks
/// leaves its cached records or thumbnails stale — a library restructure
/// moving files, an in-place rotation changing a thumbnail's hash, a sync
/// pass merging a peer's photos, the 60-second scanner finding new ones.
/// A no-op before `register` has run.
///
/// This is [`LibraryGrid::refresh`], not `load`: callers here are *events*,
/// not navigation, and none of them is a reason to send the user back to
/// the top of their library. The scanner alone would do it every minute.
pub fn request_reload() {
    REFRESH_HANDLE.with(|cell| {
        if let Some((grid, ..)) = cell.borrow().as_ref() {
            grid.refresh();
        }
    });
}

/// Run the select-mode resync hook, once a refresh has actually landed.
///
/// Separate from [`request_reload`] because a refresh is asynchronous: the
/// rows arrive from a worker thread, and re-applying membership before they
/// do would apply it to the tiles that are about to be replaced.
fn notify_reloaded() {
    let hook = REFRESH_HANDLE
        .with(|cell| cell.borrow().as_ref().map(|(_, _, on_reloaded)| on_reloaded.clone()));
    if let Some(hook) = hook {
        hook();
    }
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

    /// Re-read the rows already on screen and patch them in place.
    ///
    /// The same query as the last `load`, limited to what is currently
    /// loaded, with the result reconciled against what is displayed instead
    /// of replacing it. Three things survive that a `load` would destroy:
    /// the scroll position, every already-decoded thumbnail, and the
    /// selection.
    ///
    /// The scroll survives because the model is replaced in one `set_vec`
    /// and never passes through empty: `viewport-height` is a function of
    /// `items.length` (`library.slint`), so a model that empties for even an
    /// instant collapses the viewport and Slint clamps `viewport-y` to zero.
    /// Measured, not assumed — a swap of equal length leaves the offset
    /// untouched, and a shorter list clamps to its new end, which is the only
    /// thing it could do.
    ///
    /// An empty grid has none of that to protect and takes the simple path —
    /// which is also the case that matters most, since a library that came
    /// up empty is exactly the one an import is about to fill.
    pub fn refresh(&self) {
        let loaded = self.records.borrow().len();
        if loaded == 0 {
            let query = self.query.borrow().clone();
            self.load(query);
            notify_reloaded();
            return;
        }

        // A new generation, for the same reason `load` takes one: any page
        // still in flight assumed a row count this is about to change, and
        // `append_page` would refuse it and stop paging. Orphaning it costs
        // one re-fetch — `want` is untouched, so `fetch_next_page` at the end
        // of `apply_refresh` asks for it again.
        let gen = self.generation.get() + 1;
        self.generation.set(gen);
        *self.timer.borrow_mut() = None;
        self.polling.set(false);
        self.active_pages.set(0);

        let (tx, rx) = mpsc::channel::<GridMsg>();
        *self.tx.borrow_mut() = Some(tx.clone());
        *self.rx.borrow_mut() = Some(rx);

        let query = self.query.borrow().clone().with_limit(loaded).with_offset(0);
        let db = self.db.clone();
        self.active_pages.set(1);
        tracing::debug!("library refresh: re-reading {loaded} rows");

        std::thread::spawn(move || {
            let records = search_library(&db, &query);
            let _ = tx.send(GridMsg::Total(count_library(&db, &query)));
            let _ = tx.send(GridMsg::Refresh { records, asked_for: loaded });
            let _ = tx.send(GridMsg::PageDone);
        });

        self.start_poller(gen);
    }

    /// Reconcile freshly-read rows against what is on screen.
    fn apply_refresh(&self, fresh: Vec<LibraryImage>, asked_for: usize) {
        if !self.rows_differ(&fresh) {
            // The overwhelmingly common case — a scan that changed something
            // elsewhere in the library, or a sync pass that merged metadata
            // for rows below the fold. Touch nothing: no model reset, no
            // re-decode, no repaint.
            self.cursor.borrow_mut().refreshed(fresh.len(), asked_for);
            self.fetch_next_page();
            return;
        }

        // Index every tile that is already decoded by the row it belongs to,
        // so a photo that has not changed keeps its pixels. Keyed on the
        // content hash as well as the id: a rotation keeps the id and mints a
        // new hash, and reusing the tile there would show the old orientation
        // until the next navigation.
        let mut decoded: HashMap<(i64, Option<[u8; 32]>), ThumbItem> = HashMap::new();
        for (index, old) in self.records.borrow().iter().enumerate() {
            if let Some(item) = self.model.row_data(index) {
                decoded.insert((old.id, old.hash), item);
            }
        }

        let pending = self.pending_membership.borrow().clone();
        let mut items = Vec::with_capacity(fresh.len());
        let mut undecoded: Vec<(usize, LibraryImage)> = Vec::new();
        for (index, rec) in fresh.iter().enumerate() {
            let selected = pending
                .as_ref()
                .map(|set| set.contains(&rec.id))
                .unwrap_or_else(|| {
                    decoded.get(&(rec.id, rec.hash)).is_some_and(|item| item.selected)
                });
            match decoded.remove(&(rec.id, rec.hash)) {
                // The caption, stack size and search score come from the row
                // and may well have changed; only the pixels are carried.
                Some(old) if old.loaded => {
                    let mut item = placeholder_item(rec, selected);
                    item.image = old.image;
                    item.loaded = true;
                    items.push(item);
                }
                Some(old) => {
                    let mut item = placeholder_item(rec, selected);
                    item.unsupported = old.unsupported;
                    if !item.unsupported {
                        undecoded.push((index, rec.clone()));
                    }
                    items.push(item);
                }
                None => {
                    items.push(placeholder_item(rec, selected));
                    undecoded.push((index, rec.clone()));
                }
            }
        }
        drop(pending);

        let mut groups: Vec<DateGroup> = Vec::new();
        append_date_groups(&mut groups, &fresh, 0);

        let len = fresh.len();
        *self.records.borrow_mut() = fresh;
        // One `set_vec`, never a clear followed by a fill: an intermediate
        // empty model collapses the view's viewport and loses the scroll.
        self.model.set_vec(items);
        sync_date_groups(&self.date_groups, &groups);
        self.cursor.borrow_mut().refreshed(len, asked_for);

        self.spawn_thumbs(undecoded);
        self.fetch_next_page();
        notify_reloaded();
    }

    /// Whether the freshly-read rows differ from what is displayed in any way
    /// the grid renders.
    ///
    /// Compared field by field rather than by row count alone: a photo
    /// deleted and another imported between two scans leaves the count
    /// identical and every tile wrong.
    fn rows_differ(&self, fresh: &[LibraryImage]) -> bool {
        let records = self.records.borrow();
        records.len() != fresh.len()
            || records.iter().zip(fresh).any(|(old, new)| !same_tile(old, new))
    }

    /// Decode thumbnails for rows that have none, at their new indices.
    fn spawn_thumbs(&self, work: Vec<(usize, LibraryImage)>) {
        if work.is_empty() {
            return;
        }
        let Some(tx) = self.tx.borrow().clone() else {
            return;
        };
        let gen = self.generation.get();
        let cache = self.cache.clone();
        let quality = self.quality;
        let thumb_px = self.thumb_px.get();

        self.active_pages.set(self.active_pages.get() + 1);
        std::thread::spawn(move || {
            decode_thumbs(&tx, &work, thumb_px, quality, &cache);
            let _ = tx.send(GridMsg::PageDone);
        });
        self.start_poller(gen);
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
            let _ = tx.send(GridMsg::Page { offset, records: records.clone() });

            // A page is contiguous, so its absolute row indices are just the
            // offset plus the position. A refresh's are not, which is why
            // `decode_thumbs` takes them explicitly.
            let work: Vec<(usize, LibraryImage)> = records
                .into_iter()
                .enumerate()
                .map(|(i, rec)| (offset + i, rec))
                .collect();
            decode_thumbs(&tx, &work, thumb_px, quality, &cache);

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
                    GridMsg::Refresh { records, asked_for } => {
                        grid.apply_refresh(records, asked_for)
                    }
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
/// Decode thumbnails for `work` — `(absolute row index, record)` — in
/// parallel, streaming each one back as it finishes.
///
/// Shared by page loading and in-place refresh. They differ only in whether
/// the indices are contiguous, so the indices are carried rather than
/// derived; letting each caller compute them separately is how a refresh
/// would end up painting a photo onto the wrong tile.
fn decode_thumbs(
    tx: &mpsc::Sender<GridMsg>,
    work: &[(usize, LibraryImage)],
    thumb_px: u32,
    quality: u8,
    cache: &Arc<ThumbnailCache>,
) {
    if work.is_empty() {
        return;
    }
    let parallelism = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4);
    let chunk_size = (work.len() / parallelism).max(1);

    let blobs = crate::remote::blobs();
    std::thread::scope(|scope| {
        for chunk in work.chunks(chunk_size) {
            let tx = tx.clone();
            let cache = cache.clone();
            let blobs = blobs.clone();
            scope.spawn(move || {
                for (index, rec) in chunk {
                    let index = *index;
                    match load_thumbnail(rec, thumb_px, quality, &cache, &blobs) {
                        Ok((rgb, width, height)) => {
                            let _ = tx.send(GridMsg::Thumb { index, rgb, width, height });
                        }
                        Err(e) => {
                            tracing::warn!("Thumbnail failed for {}: {e}", rec.path.display());
                            // A remote miss is transient — the master may be
                            // asleep, or the hash may have moved under a
                            // rotation — so the tile keeps its placeholder and
                            // the next load tries again. "Unsupported" is
                            // permanent, and claiming it here would be a lie
                            // the user cannot clear.
                            if !rec.locality.is_remote() && !raw_preview_supported(&rec.path) {
                                let _ = tx.send(GridMsg::Unsupported { index });
                            }
                        }
                    }
                }
            });
        }
    });
}

/// Whether two readings of the same row would draw the same tile.
///
/// Everything the tile shows, and nothing else: a `taken_at` that changed
/// without moving the row, or a peer's `rev` bump, is not a repaint. The
/// hash is in because it keys the thumbnail — a rotation changes it and the
/// old pixels are then wrong.
fn same_tile(old: &LibraryImage, new: &LibraryImage) -> bool {
    old.id == new.id
        && old.hash == new.hash
        && old.status == new.status
        && old.locality == new.locality
        && old.meta.filename == new.meta.filename
        && old.stack_size == new.stack_size
}

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
/// ```text
/// cache hit                    → decode WebP
/// miss, and the file is here   → render from disk, cache, return RGB
/// miss, and the file is remote → GET /blob/thumb/{hash}, cache, decode
/// ```
///
/// The remote branch caches too, and deliberately so: §3.6's "loads on demand
/// without saving" is about *originals*. A thumbnail is ~10 KB and is the
/// difference between a grid that scrolls and one that re-fetches the whole
/// viewport every time the user moves.
fn load_thumbnail(
    rec: &LibraryImage,
    max_size: u32,
    quality: u8,
    cache: &ThumbnailCache,
    blobs: &RemoteBlobs,
) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    if let Some(hash) = rec.hash {
        if let Some(webp) = cache.get(&hash) {
            return thumbnail::decode_webp_rgb(&webp);
        }
    }

    if rec.locality.is_remote() {
        // No hash means no blob key — `rec.path` names a file on another
        // machine, so there is nothing local to fall back to either.
        let hash = rec
            .hash
            .ok_or_else(|| anyhow::anyhow!("remote image {} has no content hash", rec.id))?;
        let webp = blobs.thumb(&hash)?;
        // Cached under the master's encoding, not this device's thumbnail
        // settings: the bytes are whatever the master rendered, and pretending
        // otherwise by re-encoding would cost a decode round-trip to change
        // nothing the user can see.
        if let Err(e) = cache.insert(&hash, &webp) {
            tracing::warn!("Thumbnail cache write failed for remote image {}: {e}", rec.id);
        }
        return thumbnail::decode_webp_rgb(&webp);
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

#[cfg(test)]
mod tests {
    use super::*;
    use maple_db::{ImageStatus, Locality};

    fn row(id: i64, name: &str) -> LibraryImage {
        LibraryImage {
            id,
            path: format!("/photos/{name}").into(),
            raw_path: None,
            added_at: 0,
            status: ImageStatus::Present,
            meta: maple_db::ImageMetadata {
                filename: Some(name.to_owned()),
                ..Default::default()
            },
            hash: Some([1u8; 32]),
            stack_id: None,
            stack_size: None,
            search_hit: None,
            locality: Locality::Local,
            origin_device: None,
        }
    }

    #[test]
    fn a_row_that_did_not_change_keeps_its_tile() {
        // The common case by far: the scanner runs every minute and finds
        // nothing. Repainting the grid on that schedule would be worse than
        // never refreshing it at all.
        let before = row(1, "a.jpg");
        let mut after = row(1, "a.jpg");
        // Things the tile does not show may move freely.
        after.added_at = 99;
        after.meta.taken_at = Some(1_700_000_000);
        assert!(same_tile(&before, &after));
    }

    #[test]
    fn a_rotation_invalidates_the_thumbnail_it_had() {
        // Same row, new content hash — the decoded pixels show the old
        // orientation and must not be carried across.
        let before = row(1, "a.jpg");
        let mut after = row(1, "a.jpg");
        after.hash = Some([2u8; 32]);
        assert!(!same_tile(&before, &after));
    }

    #[test]
    fn a_photo_that_went_missing_or_remote_redraws() {
        let before = row(1, "a.jpg");

        let mut gone = row(1, "a.jpg");
        gone.status = ImageStatus::Missing;
        assert!(!same_tile(&before, &gone));

        let mut relayed = row(1, "a.jpg");
        relayed.locality = Locality::Remote;
        assert!(
            !same_tile(&before, &relayed),
            "its pixels come from somewhere else now"
        );
    }

    #[test]
    fn a_renamed_photo_redraws_its_caption() {
        let before = row(1, "a.jpg");
        let after = row(1, "20240315_a.jpg");
        assert!(!same_tile(&before, &after));
    }

    #[test]
    fn a_swap_that_keeps_the_count_is_still_a_change() {
        // One photo deleted and another imported between two scans leaves the
        // row count identical and every tile below it wrong, which is why the
        // comparison is row by row rather than a length check.
        let before = [row(1, "a.jpg"), row(2, "b.jpg")];
        let after = [row(1, "a.jpg"), row(3, "c.jpg")];
        assert!(before
            .iter()
            .zip(after.iter())
            .any(|(old, new)| !same_tile(old, new)));
    }
}
