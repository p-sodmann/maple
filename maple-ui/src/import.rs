//! Import browser controller (Slint port of views/source_picker.rs +
//! views/image_browser/).
//!
//! Opens a separate top-level [`ImportWindow`] held as a `thread_local!`
//! singleton. The window drives two phases: folder picking and then browsing
//! scan results + copying selected images.
//!
//! Background workers use `std::thread` + `mpsc`. A `slint::Timer` running on
//! the Slint main thread drains the channels.

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use image::{DynamicImage, RgbImage};
use maple_import::Signature;
use slint::{ComponentHandle, Model, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::import_previews::{PhotoRef, PreviewMsg, PreviewService, Retained};
use crate::services::import::{insert_imported_images, ImportEntry};
use crate::thumbnail;
use crate::{ImportItem, ImportWindow};

/// Name of the embedding cache file written to the root of a scanned source
/// directory (e.g. an SD card). Dotfile-prefixed so the scanner's existing
/// hidden-file filtering already ignores it.
const EMBED_CACHE_FILE: &str = ".maple_embed_cache.bin";

/// How many newly-computed embeddings accumulate before the SD-card cache is
/// flushed to disk (mirrors `maple-db::hasher`'s `BATCH_SIZE` convention).
const EMBED_CACHE_FLUSH_EVERY: usize = 20;

/// Rows to decode before the strip has reported a real viewport.
///
/// It reports one as soon as it has a height, but a scan that finished
/// first would otherwise sit there with a single decoded tile.
const INITIAL_WINDOW: usize = 15;

/// How far the reader may run ahead of the decode pool, as a multiple of
/// the pool size.
///
/// Each queued job holds one file's bytes, so this bounds what the reader
/// can pull into RAM. Deep enough to keep every decoder fed through an
/// uneven patch, shallow enough that a big card cannot be read faster than
/// it is consumed.
const READ_AHEAD_PER_DECODER: usize = 3;

/// How many rendered photos may queue up in front of the embedder.
///
/// The bound is what keeps the two stages from turning into a memory leak:
/// each job carries a 256×256 RGB buffer, so an unbounded queue on a
/// 5 000-photo card would be about a gigabyte of thumbnails waiting for
/// ONNX. At this depth the render workers block instead, by which point the
/// user already has 64 tiles on screen.
const EMBED_QUEUE_DEPTH: usize = 64;

/// Rendered frames allowed to queue up for the session engine.
///
/// Deeper than the embedder's because the stage behind it is ~0.2 ms a
/// photo against the card's ~100 ms, so this queue is almost always empty;
/// the depth is there for the burst that arrives when several decoders
/// finish at once, not to smooth out a slow consumer.
const SIGNATURE_QUEUE_DEPTH: usize = 128;

/// Previews allowed to queue up for the medium's own cache.
///
/// Deep, because the stage behind it writes to the card in batches and
/// must never make a decoder wait on one: the whole point of the cache is
/// to spend less time on the card, not more.
const PREVIEW_CACHE_QUEUE_DEPTH: usize = 256;

/// Previews written back to the medium per batch.
///
/// The cache file is append-only, so a flush costs only the new records —
/// about 1 MB at this size. Small enough that a card pulled mid-scan loses
/// little, large enough that the scan is not writing after every photo.
const PREVIEW_CACHE_FLUSH_EVERY: usize = 64;

/// How long the scan drain may spend applying messages in one tick.
///
/// A *time* budget, not a message count. The work per message is nowhere
/// near uniform — a `Thumb` patches a model row and repaints, a `Signature`
/// writes one field and shows nothing — and a fixed count sized on a small
/// folder turns a real card into a trickle: 954 photos are 1,910 messages,
/// and at ten per 30 ms tick that is six seconds of draining before
/// `finish_scan` (which runs on the *last* message) can segment anything.
/// The `f` grid spends all of it saying "no sessions detected", which is
/// how this was found.
const SCAN_DRAIN_BUDGET: Duration = Duration::from_millis(8);

thread_local! {
    static IMPORT: RefCell<Option<Import>> = const { RefCell::new(None) };
}

// ── Background worker messages ────────────────────────────────────

enum ScanMsg {
    /// Every photo the scan found, before a single byte has been read.
    /// `scan_grouped` knows the whole list up front, so the strip can show
    /// the true count and every filename immediately — and the preview
    /// service has the paths it needs to decode on demand.
    Found(Vec<PhotoRef>),
    Thumb(ScanThumb),
    /// A session-detection signature, arriving separately and later than
    /// the photo's tile — see [`spawn_scan_worker`].
    Signature { index: usize, signature: Signature },
    /// A DINOv2 embedding, when `[stacks] enabled` turned the embedder on.
    /// It no longer groups anything here — it is carried so the copy can
    /// store it, sparing the library's own stacker a second inference pass.
    Embedding { index: usize, embedding: Vec<f32> },
    Done,
    Error(String),
}

/// Why a photo has no preview — kept apart because the two mean very
/// different things when you go looking for the file afterwards.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum Readable {
    #[default]
    Yes,
    /// The display file would not decode, but another file in the group
    /// did — a corrupt JPEG next to an intact raw. The photo is fine and
    /// the tile is real; it just came from the other half of the pair.
    FromCompanion,
    /// The bytes came back but decoded to nothing, and nothing in the group
    /// could stand in: corrupt, truncated, or a format the decoder does not
    /// actually handle.
    NoPreview,
    /// The read never returned inside the configured budget and the scan
    /// walked away. Usually the card or the cable, not the file.
    TimedOut,
}

impl Readable {
    /// Whether there is a picture to show. A companion preview counts.
    fn ok(self) -> bool {
        matches!(self, Self::Yes | Self::FromCompanion)
    }

    /// Short reason for the log summary at the end of a scan.
    fn reason(self) -> &'static str {
        match self {
            Self::Yes => "ok",
            Self::FromCompanion => "display file unreadable, preview from companion",
            Self::NoPreview => "could not be decoded",
            Self::TimedOut => "timed out",
        }
    }
}

/// What earlier sessions already decided about the photos on a medium.
///
/// Read-only for the whole scan, so it travels as a bare `Arc` — nothing
/// mutates it and there is nothing for a lock to protect.
struct PriorDecisions {
    imported: maple_state::SeenSet,
    skipped: maple_state::SeenSet,
}

/// One scanned photo, handed from the scan worker to the UI thread.
struct ScanThumb {
    index: usize,
    path: PathBuf,
    companions: Vec<PathBuf>,
    content_hash: [u8; 32],
    imported: bool,
    /// An earlier session moved past this photo without marking it.
    skipped_before: bool,
    /// Whether a preview came back at all, and if not, why. The row still
    /// exists either way and can still be selected and copied; it just has
    /// no picture to show.
    readable: Readable,
    /// Variance-of-Laplacian sharpness score, computed whenever a
    /// signature is — it is what picks the keeper out of a session.
    sharpness: Option<f32>,
    /// Capture instant, fractional seconds since the epoch. Parsed from the
    /// bytes already in hand, so it costs no extra card I/O.
    taken: Option<f64>,
    /// The canonical preview, made from this photo's bytes or taken
    /// straight out of the medium's cache. `None` only when nothing about
    /// the file could be decoded.
    preview: Option<Vec<u8>>,
}

enum CopyMsg {
    Progress {
        done: usize,
        total: usize,
    },
    Done {
        copied: usize,
        failed: usize,
        /// Where each copied source file actually landed. The library DB must
        /// record destination paths — inserting the source path would store an
        /// SD-card path that vanishes when the card is ejected.
        dest_by_source: HashMap<PathBuf, PathBuf>,
    },
    Error(String),
}

enum RotateMsg {
    Done {
        content_hash: [u8; 32],
        thumb: (Vec<u8>, u32, u32),
        preview: (Vec<u8>, u32, u32),
    },
    Error(String),
}

// ── Per-entry state (UI thread) ───────────────────────────────────

struct Entry {
    path: PathBuf,
    companions: Vec<PathBuf>,
    content_hash: [u8; 32],
    is_imported: bool,
    /// The user has moved off this photo without marking it for import —
    /// a decision, not an absence, and what paints the red ✗. Set on every
    /// departure, marked or not, so that un-marking a photo later turns it
    /// into a skip rather than back into "never looked at". Pre-set for
    /// photos an earlier session already skipped.
    passed: bool,
    /// An earlier session already decided about this photo: imported it, or
    /// passed over it. The predicate behind "Hide old images".
    decided_before: bool,
    /// Whether the scan got a preview out of this file, and if not, why.
    /// The row is still here and still copyable; only its picture is
    /// missing.
    readable: Readable,
    /// Decoded thumbnail, present only while this photo is in the retained
    /// tier. `None` means never decoded, or evicted — `webp` says which.
    thumb: Option<slint::Image>,
    /// The canonical preview (see [`maple_import::preview`]) — ~15 KB
    /// against the decoded frame's ~196 KB, and the *only* pixel
    /// representation this photo keeps.
    ///
    /// Everything reads it: the tile inflates from it, and the sharpness
    /// score and session signature below were computed from the frame it
    /// decodes to. Present for every photo the scan read (or found in the
    /// medium's cache), so eviction drops the pixels down to this and
    /// scrolling back never returns to the card.
    webp: Option<Vec<u8>>,
    /// What this photo looks like to the session engine (`None` if session
    /// detection is off, or the photo never decoded). Session detection is
    /// sequential, so this is only ever read in scan order.
    signature: Option<Signature>,
    /// DINOv2 embedding, present only when `[stacks] enabled` is on. Not
    /// used for grouping here — stored with the photo on copy so the
    /// library's stacker does not have to compute it again.
    embedding: Option<Vec<f32>>,
    /// Capture instant, fractional seconds since the epoch, from EXIF.
    /// `None` when the photo carries no usable timestamp — which the
    /// segmentation reads as "no gap information", not as "no gap".
    taken: Option<f64>,
    /// Variance-of-Laplacian sharpness score, used to pick the keeper out
    /// of a detected session.
    sharpness: Option<f32>,
}

/// Which entries the filmstrip is currently showing, and where.
///
/// "Hide old images" filters the model but not the entry list: the scan
/// index stays every photo's identity — selection, groups, navigation and
/// the preview all address entries by it — and this is the only place that
/// knows a model row is a different number.
#[derive(Default)]
struct Visible {
    /// Model row → entry index.
    rows: Vec<usize>,
    /// Entry index → model row; `None` while the entry is filtered out.
    row_of: Vec<Option<usize>>,
}

impl Visible {
    fn rebuild(&mut self, entries: &[Entry], hide_old: bool) {
        self.rows.clear();
        self.row_of.clear();
        self.row_of.resize(entries.len(), None);
        for (i, e) in entries.iter().enumerate() {
            if hide_old && e.decided_before {
                continue;
            }
            self.row_of[i] = Some(self.rows.len());
            self.rows.push(i);
        }
    }

    /// The model row showing entry `i`, if it is on screen at all.
    fn row(&self, i: usize) -> Option<usize> {
        self.row_of.get(i).copied().flatten()
    }

    fn shows(&self, i: usize) -> bool {
        self.row(i).is_some()
    }
}

/// The strip row showing entry `idx`, or -1 when the filter is hiding it.
///
/// -1 reads as "leave the scroll where it is": a photo the strip does not
/// contain has no row to park it in, and row 0 would be a lie that jumps the
/// user to the top of the card.
fn strip_row(visible: &Visible, idx: usize) -> i32 {
    visible.row(idx).map_or(-1, |r| r as i32)
}

// ── Controller struct ─────────────────────────────────────────────

/// Shared state for one import window, passed to each `wire_*` function.
///
/// Every field is a cheap clone target, and the window is held only as a
/// [`slint::Weak`] — cloning fields out of an `ImportCtx` inside a callback
/// therefore can't capture a strong `ImportWindow` and leak the window.
#[derive(Clone)]
struct ImportCtx {
    window: slint::Weak<ImportWindow>,
    db: Arc<Mutex<maple_db::Database>>,
    entries: Rc<RefCell<Vec<Entry>>>,
    selected: Rc<RefCell<HashSet<usize>>>,
    current: Rc<Cell<usize>>,
    /// Row ↔ entry mapping for the filmstrip; identity when nothing is
    /// hidden.
    visible: Rc<RefCell<Visible>>,
    /// Whether photos an earlier session already decided on are filtered
    /// out of the strip.
    hide_old: Rc<Cell<bool>>,
    /// Decodes previews on demand, prioritised by where the user is
    /// looking. Replaced at the start of every scan.
    previews: Rc<RefCell<Option<PreviewService>>>,
    /// Which previews are currently decoded, least-recently-seen first out.
    retained: Rc<RefCell<Retained>>,
    /// The model rows the strip last reported as on screen (inclusive), and
    /// the row it is centred on. Read whenever the visible set changes
    /// under us — a filter toggle, or the end of a scan.
    preview_window: Rc<Cell<(usize, usize, usize)>>,
    /// Drains decoded previews for as long as the browser is open, which is
    /// longer than the scan itself lasts.
    preview_timer: Rc<RefCell<Option<Timer>>>,
    /// The medium the entries on screen actually came off.
    ///
    /// Not the same thing as `source`, which is wherever the folder picker
    /// currently points: picking a new source and closing the window
    /// without re-scanning would otherwise write the *previous* card's
    /// verdicts into the new one's record.
    scanned_source: Rc<RefCell<PathBuf>>,
    /// Persistent model, mutated in place via `set_row_data` for single-row
    /// changes (selection toggle, thumb arriving, rotate, …) instead of being
    /// replaced wholesale. Swapping in a brand-new `VecModel` on every click
    /// forces Slint to tear down and recreate every tile's `TouchArea`; a
    /// second click landing mid-rebuild would then hit a fresh TouchArea that
    /// never saw the press, silently dropping the click. Reserve full
    /// `set_vec` resets for genuinely bulk changes (new scan, all counts known).
    model: Rc<VecModel<ImportItem>>,
    /// Which entry's big preview is actually on screen right now — `None`
    /// until something has genuinely been rendered into it. `current`
    /// defaults to 0 before anything is ever previewed, so comparing against
    /// `current` alone can't tell "index 0 is already showing" apart from
    /// "nothing has been shown yet, and this happens to be index 0".
    preview_shown_idx: Rc<Cell<Option<usize>>>,
    /// Count of thumbnails processed so far during the current scan — drives
    /// the progress bar shown while `scanning` is true.
    scanned_count: Rc<Cell<usize>>,
    source: Rc<RefCell<PathBuf>>,
    dest: Rc<RefCell<PathBuf>>,
    /// Every session the last scan detected, **tiling the whole sequence**
    /// — solo photos included, because the `f` grid has to be able to drag
    /// a boundary onto any photo. `groups` is the `len >= 2` subset.
    sessions: Rc<RefCell<Vec<maple_import::Session>>>,
    /// Whether the `f` grid is open. While it is, "Hide old images" stands
    /// down: session boundaries are about capture *sequence*, and a strip
    /// with photos missing from the middle of a session would draw a band
    /// that lies about what it contains.
    grid_open: Rc<Cell<bool>>,
    /// The photo a grid click opened a session on, waiting for the click
    /// that closes it. `None` when no edit is half-finished.
    ///
    /// It is UI-only state and deliberately not part of the session list:
    /// an open edit describes an intention, and until the second click
    /// there is no boundary to record.
    pending_cut: Rc<Cell<Option<usize>>>,
    /// Detected session groups from the last scan — each a sorted list of
    /// flat `entries` indices. Rebuilt from `sessions` whenever a boundary
    /// moves.
    groups: Rc<RefCell<Vec<Vec<usize>>>>,
    /// The A-vs-B pass over those groups, while it is running.
    ///
    /// Rebuilt rather than resumed whenever the grouping changes — see
    /// [`crate::import_tournament`], where skipping the already-settled is
    /// what makes a rebuild cost the user nothing.
    tournament: Rc<RefCell<Option<crate::import_tournament::Tournament>>>,
    /// The shared zoom and pan — *one* of each, for both panes. Two panes
    /// each keeping their own would have to be told to follow each other,
    /// and would drift the first time a render was dropped.
    pair_view: Rc<RefCell<PairView>>,
    /// Renders the two panes; alive only while the tournament is on, since
    /// it holds two full decodes.
    pair_renderer: Rc<RefCell<Option<crate::import_tournament::PairRenderer>>>,
    pair_timer: Rc<RefCell<Option<Timer>>>,
    /// The tags every photo marked from *now on* will carry.
    ///
    /// A brush, not a batch setting: it survives a copy, changes mid-pass,
    /// and is cleared with `c`. Which is exactly why it needs the floating
    /// panel — invisible state that silently changes what an import writes
    /// would be a trap.
    brush: Rc<RefCell<Vec<Tag>>>,
    /// Entry index → the collection ids that were on the brush at the moment
    /// that photo was marked.
    ///
    /// Recorded at mark time rather than read at copy time, so changing the
    /// brush half way through a pass tags the two halves differently.
    /// Unmarking a photo drops its record, so this can never outlive the
    /// selection it belongs to.
    brushed: Rc<RefCell<HashMap<usize, Vec<i64>>>>,
    /// Timer slots for the in-flight background jobs. Each holds its poller
    /// alive for as long as the job it drains can still report back.
    scan_timer: Rc<RefCell<Option<Timer>>>,
    copy_timer: Rc<RefCell<Option<Timer>>>,
    copy_done_timer: Rc<RefCell<Option<Timer>>>,
    rotate_timer: Rc<RefCell<Option<Timer>>>,
}

struct Import {
    window: ImportWindow,
    ctx: ImportCtx,
}

/// Find the burst group (if any) that `idx` belongs to.
fn find_group(groups: &[Vec<usize>], idx: usize) -> Option<&[usize]> {
    groups.iter().find(|g| g.contains(&idx)).map(|g| g.as_slice())
}

/// Open (or reuse) the import window (legacy entry point).
#[allow(dead_code)]
pub fn open(db: Arc<Mutex<maple_db::Database>>, is_dark: bool) {
    open_with_source(db, std::path::PathBuf::new(), is_dark);
}

/// Open the import browser window pre-seeded with `source_path`, syncing
/// dark-mode state.
///
/// Called when the user clicks "Start Scan" on the embedded ImportPage.
/// If `source_path` is empty the window opens on the picker phase as before.
pub fn open_with_source(db: Arc<Mutex<maple_db::Database>>, source_path: std::path::PathBuf, is_dark: bool) {
    if IMPORT.with(|i| i.borrow().is_none()) {
        match build(db) {
            Ok(imp) => IMPORT.with(|cell| *cell.borrow_mut() = Some(imp)),
            Err(e) => {
                tracing::error!("Failed to build import window: {e}");
                return;
            }
        }
    }
    IMPORT.with(|cell| {
        let guard = cell.borrow();
        if let Some(imp) = guard.as_ref() {
            imp.window.set_dark(is_dark);
            // Pre-set the source path then trigger a scan if one was provided.
            if !source_path.as_os_str().is_empty() {
                let s = source_path.to_string_lossy().into_owned();
                imp.window.set_source_path(SharedString::from(s));
                *imp.ctx.source.borrow_mut() = source_path;
                imp.window.invoke_start_scan();
            }
            if let Err(e) = imp.window.show() {
                tracing::error!("Failed to show import window: {e}");
            }
        }
    });
}

/// Propagate a theme change to the import window while it is open.
pub fn set_dark(dark: bool) {
    IMPORT.with(|i| {
        let guard = i.borrow();
        if let Some(imp) = guard.as_ref() {
            imp.window.set_dark(dark);
        }
    });
}

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<Import, slint::PlatformError> {
    let window = ImportWindow::new()?;

    let model: Rc<VecModel<ImportItem>> = Rc::new(VecModel::default());
    window.set_items(ModelRc::from(model.clone()));

    // The embedded sidebar ImportPage only lets the user pick a *source*
    // folder — there is no destination step in that flow anymore, so default
    // to the configured library directory (the same place the scanner,
    // thumbcache, etc. already treat as "the library"). Without this, `dest`
    // stays empty forever and "Copy Selected" silently no-ops.
    let dest: Rc<RefCell<PathBuf>> =
        Rc::new(RefCell::new(maple_state::Settings::load().library_dir));
    window.set_dest_path(SharedString::from(dest.borrow().to_string_lossy().into_owned()));

    let ctx = ImportCtx {
        window: window.as_weak(),
        db,
        entries: Rc::new(RefCell::new(Vec::new())),
        selected: Rc::new(RefCell::new(HashSet::new())),
        current: Rc::new(Cell::new(0)),
        visible: Rc::new(RefCell::new(Visible::default())),
        // Photos an earlier session already decided on are hidden by
        // default: the point of a re-scan is what is *new* on the card.
        hide_old: Rc::new(Cell::new(true)),
        previews: Rc::new(RefCell::new(None)),
        retained: Rc::new(RefCell::new(Retained::new(
            maple_state::Settings::load().import.retained_previews(),
        ))),
        preview_window: Rc::new(Cell::new((0, 0, 0))),
        preview_timer: Rc::new(RefCell::new(None)),
        scanned_source: Rc::new(RefCell::new(PathBuf::new())),
        model,
        preview_shown_idx: Rc::new(Cell::new(None)),
        scanned_count: Rc::new(Cell::new(0)),
        source: Rc::new(RefCell::new(PathBuf::new())),
        dest,
        sessions: Rc::new(RefCell::new(Vec::new())),
        grid_open: Rc::new(Cell::new(false)),
        pending_cut: Rc::new(Cell::new(None)),
        groups: Rc::new(RefCell::new(Vec::new())),
        tournament: Rc::new(RefCell::new(None)),
        pair_view: Rc::new(RefCell::new(PairView::default())),
        pair_renderer: Rc::new(RefCell::new(None)),
        pair_timer: Rc::new(RefCell::new(None)),
        brush: Rc::new(RefCell::new(Vec::new())),
        brushed: Rc::new(RefCell::new(HashMap::new())),
        scan_timer: Rc::new(RefCell::new(None)),
        copy_timer: Rc::new(RefCell::new(None)),
        copy_done_timer: Rc::new(RefCell::new(None)),
        rotate_timer: Rc::new(RefCell::new(None)),
    };

    wire_chrome(&window, &ctx);
    wire_scan(&window, &ctx);
    wire_browse(&window, &ctx);
    wire_copy(&window, &ctx);
    wire_rotate(&window, &ctx);
    wire_tags(&window, &ctx);
    wire_tournament(&window, &ctx);

    Ok(Import { window, ctx })
}

// ── Close / pickers ───────────────────────────────────────────────

/// Wire the window chrome: closing, the two folder pickers, and the
/// file-naming template editor.
fn wire_chrome(window: &ImportWindow, ctx: &ImportCtx) {
    // ── Close ─────────────────────────────────────────────────────
    window.on_close_requested({
        let ctx = ctx.clone();
        move || {
            commit_skips(&ctx);
            if let Some(w) = ctx.window.upgrade() {
                let _ = w.hide();
            }
        }
    });

    // The platform's own close button does not route through the callback
    // above, and losing a session's triage to it would be the easiest way
    // to lose it.
    window.window().on_close_requested({
        let ctx = ctx.clone();
        move || {
            commit_skips(&ctx);
            slint::CloseRequestResponse::HideWindow
        }
    });

    // ── Pick source ───────────────────────────────────────────────
    window.on_pick_source({
        let w = ctx.window.clone();
        let source = ctx.source.clone();
        move || {
            let picked = rfd::FileDialog::new()
                .set_title("Choose source folder")
                .pick_folder();
            if let Some(path) = picked {
                let s = path.to_string_lossy().into_owned();
                *source.borrow_mut() = path;
                if let Some(w) = w.upgrade() {
                    w.set_source_path(SharedString::from(s));
                }
            }
        }
    });

    // ── Pick destination ──────────────────────────────────────────
    window.on_pick_dest({
        let w = ctx.window.clone();
        let dest = ctx.dest.clone();
        move || {
            let picked = rfd::FileDialog::new()
                .set_title("Choose destination folder")
                .pick_folder();
            if let Some(path) = picked {
                let s = path.to_string_lossy().into_owned();
                *dest.borrow_mut() = path;
                if let Some(w) = w.upgrade() {
                    w.set_dest_path(SharedString::from(s));
                }
            }
        }
    });

    // ── Configure file naming ────────────────────────────────────
    window.on_open_path_template({
        let w = ctx.window.clone();
        let db = ctx.db.clone();
        move || {
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            crate::path_template_window::open(db.clone(), is_dark);
        }
    });
}

// ── Start scan ────────────────────────────────────────────────────

fn wire_scan(window: &ImportWindow, ctx: &ImportCtx) {
    window.on_start_scan({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let src = ctx.source.borrow().clone();
            if src.as_os_str().is_empty() {
                return;
            }

            // Reset state.
            ctx.entries.borrow_mut().clear();
            ctx.selected.borrow_mut().clear();
            ctx.groups.borrow_mut().clear();
            ctx.current.set(0);
            ctx.preview_shown_idx.set(None);
            ctx.scanned_count.set(0);
            ctx.model.set_vec(Vec::new());
            w.set_selected_count(0);
            w.set_copy_done(false);
            w.set_total_count(0);
            w.set_scanned_count(0);
            w.set_scanning(true);
            w.set_preview_photo(slint::Image::default());
            w.set_preview_filename(SharedString::default());
            w.set_status_text("Scanning…".into());
            w.set_in_browser(true);

            let (tx, rx) = mpsc::channel::<ScanMsg>();

            let settings = maple_state::Settings::load();
            let library_dir = settings.library_dir.clone();
            let stack_settings = settings.stacks.clone();
            let session_settings = settings.sessions.clone();

            // Both records live on the medium being scanned, so a card
            // plugged into a second machine still knows what it has already
            // given up and what was already turned down. The library copies
            // are only the fallback for a source that carries none (see
            // `SeenSet::load_for_source`).
            let prior = Arc::new(PriorDecisions {
                imported: maple_state::SeenSet::load_for_source(
                    &src, &library_dir, maple_state::Record::Imported,
                ),
                skipped: maple_state::SeenSet::load_for_source(
                    &src, &library_dir, maple_state::Record::Skipped,
                ),
            });
            *ctx.scanned_source.borrow_mut() = src.clone();
            // A fresh medium: nothing decoded, nothing remembered, and the
            // "new only" filter back on.
            ctx.previews.borrow_mut().take();
            ctx.retained.borrow_mut().clear();
            ctx.preview_window.set((0, INITIAL_WINDOW, 0));
            ctx.hide_old.set(true);
            w.set_hide_old(true);
            ctx.sessions.borrow_mut().clear();
            // A new card is a new set of comparisons, and the old verdicts
            // were about photos that are no longer on screen. Dropping the
            // tournament drops its record with it — there is no second
            // copy to forget about.
            ctx.tournament.borrow_mut().take();
            w.set_tourney_available(false);
            w.set_tourney_on(false);
            stop_pair_renderer(&ctx);
            w.set_old_count(0);
            w.set_preview_state(0);

            spawn_scan_worker(
                src,
                stack_settings,
                session_settings,
                settings.import.clone(),
                prior,
                tx.clone(),
            );

            // Timer to drain scan results.
            let ctx2 = ctx.clone();
            let timer = Timer::default();
            timer.start(
                TimerMode::Repeated,
                Duration::from_millis(30),
                move || {
                    let Some(w) = ctx2.window.upgrade() else { return };

                    let deadline = std::time::Instant::now() + SCAN_DRAIN_BUDGET;
                    loop {
                        // Only messages that touch the UI are charged
                        // against the budget. A signature backlog is pure
                        // data — draining it costs a field write each and
                        // paying a whole tick for ten of them is what made
                        // a big card crawl.
                        let painted = match rx.try_recv() {
                            Ok(ScanMsg::Found(photos)) => {
                                apply_scan_listing(&w, &ctx2, photos);
                                true
                            }
                            Ok(ScanMsg::Thumb(thumb)) => {
                                apply_scan_thumb(&w, &ctx2, thumb);
                                true
                            }
                            Ok(ScanMsg::Embedding { index, embedding }) => {
                                if let Some(e) = ctx2.entries.borrow_mut().get_mut(index) {
                                    e.embedding = Some(embedding);
                                }
                                false
                            }
                            Ok(ScanMsg::Signature { index, signature }) => {
                                // Arrives after the photo's tile and needs
                                // no repaint — only `finish_scan`'s
                                // segmentation reads it.
                                if let Some(e) = ctx2.entries.borrow_mut().get_mut(index) {
                                    e.signature = Some(signature);
                                }
                                false
                            }
                            Ok(ScanMsg::Done) => {
                                finish_scan(&w, &ctx2);
                                return;
                            }
                            Ok(ScanMsg::Error(e)) => {
                                w.set_scanning(false);
                                w.set_status_text(format!("Scan error: {e}").into());
                                return;
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => return,
                        };
                        if painted && std::time::Instant::now() >= deadline {
                            break;
                        }
                    }
                },
            );
            *ctx.scan_timer.borrow_mut() = Some(timer);
        }
    });
}

/// Scan `src`, streaming one [`ScanThumb`] per photo.
///
/// Three stages, split along where the work actually blocks.
///
/// **Reading is serial, on this thread.** A camera card is one bus: twelve
/// threads pulling twelve files off it at once each get a twelfth of the
/// bandwidth and all finish late *together*, so the grid sits empty for
/// minutes and then fills in a burst. One reader at full bandwidth hands
/// over the first photo almost immediately and the rest at a steady rate.
/// It also reads each file **once** — hashing and decoding used to open it
/// separately, doubling the traffic over the slowest link in the whole
/// pipeline.
///
/// **Decoding fans out.** It is pure CPU over bytes already in memory, so
/// it parallelises cleanly and cannot stall on the card.
///
/// **Signing is serial again**, behind everything else. Session detection
/// walks the card in order and its engine is one `&mut` — and for the time
/// member that is load-bearing rather than incidental: [`TimeGapEngine`]
/// anchors its epoch on the first frame it sees, so one engine per decode
/// thread would give each its own origin and make their signatures
/// incomparable. Results arrive later as [`ScanMsg::Signature`], long after
/// the tile is on screen.
///
/// The optional DINOv2 embed stage still runs beside it when `[stacks]
/// enabled` is on, but only to *store* embeddings for the library's own
/// stacking — it no longer decides any grouping here. It costs 26 ms/photo
/// against the session engines' ~0.2 ms, and its bounded queue backpressures
/// the decoders and thence the reader, which is why it is off by default.
///
/// A file the card never returns from costs one read timeout and is then
/// abandoned — before, it stalled the scan forever. Both the pool size and
/// that timeout come from `[import]` in settings.toml.
///
/// If the embedder fails to load (e.g. no network for a first-time model
/// fetch), or the session engine spec does not name a real engine, log once
/// and let the scan finish without it — both are enrichment, never a hard
/// requirement to finish scanning.
fn spawn_scan_worker(
    src: PathBuf,
    stack_settings: maple_state::StackSettings,
    session_settings: maple_state::SessionSettings,
    tuning: maple_state::ImportSettings,
    prior: Arc<PriorDecisions>,
    tx: mpsc::Sender<ScanMsg>,
) {
    std::thread::spawn(move || {
        let scanned_groups = match maple_import::scan_grouped(&src) {
            Ok(g) => g,
            Err(e) => {
                let _ = tx.send(ScanMsg::Error(e.to_string()));
                return;
            }
        };
        let total = scanned_groups.len();
        let listing = scanned_groups
            .iter()
            .map(|g| PhotoRef {
                display: g.display.path.clone(),
                companions: g.companions.iter().map(|c| c.path.clone()).collect(),
            })
            .collect();
        if tx.send(ScanMsg::Found(listing)).is_err() {
            return;
        }

        let decoders = tuning.decoders();
        let budget = tuning.read_timeout();
        // One line naming the settings actually in force, so a slow scan
        // can be reasoned about from the log alone.
        tracing::info!(
            "Import scan: {total} photos from {}, 1 reader + {decoders} decoders, {:?} read timeout",
            src.display(),
            budget
        );
        // Build the engine once, up front: a spec naming nothing real is a
        // settings mistake worth one clear line in the log, not one warning
        // per photo.
        let mut engine = match session_settings.enabled {
            false => None,
            true => match maple_import::session::engine_from_spec(&session_settings.engine) {
                Ok(engine) => {
                    tracing::info!("Import scan: sessions by {}", engine.describe());
                    Some(engine)
                }
                Err(err) => {
                    tracing::warn!(
                        "Import scan: [sessions] engine = {:?} is unusable, \
                         scanning without session detection: {err}",
                        session_settings.engine
                    );
                    None
                }
            },
        };

        let (jobs_tx, jobs_rx) =
            mpsc::sync_channel::<DecodeJob>(decoders * READ_AHEAD_PER_DECODER);
        let jobs_rx = Arc::new(Mutex::new(jobs_rx));
        // What this medium already told us about itself. Every hit below
        // is a file that never gets opened.
        let preview_cache = Arc::new(Mutex::new(maple_import::PreviewCache::load_from(&src)));
        let known = preview_cache.lock().map(|c| c.len()).unwrap_or(0);
        if known > 0 {
            tracing::info!("Import scan: the medium already has {known} previews");
        }

        let (embed_tx, embed_rx) = mpsc::sync_channel::<EmbedJob>(EMBED_QUEUE_DEPTH);
        let (sig_tx, sig_rx) = mpsc::sync_channel::<SignatureJob>(SIGNATURE_QUEUE_DEPTH);
        let (cache_tx, cache_rx) = mpsc::sync_channel::<CacheJob>(PREVIEW_CACHE_QUEUE_DEPTH);
        let sessions_on = engine.is_some();

        std::thread::scope(|scope| {
            if stack_settings.enabled {
                let tx = tx.clone();
                let src = src.clone();
                let settings = stack_settings.clone();
                scope.spawn(move || embed_stage(&src, &settings, embed_rx, &tx));
            }
            if let Some(engine) = engine.as_mut() {
                let tx = tx.clone();
                scope.spawn(move || signature_stage(engine.as_mut(), sig_rx, &tx));
            }
            {
                let cache = preview_cache.clone();
                scope.spawn(move || preview_cache_stage(&cache, cache_rx));
            }

            for _ in 0..decoders {
                let tx = tx.clone();
                let jobs_rx = jobs_rx.clone();
                let embed_tx = stack_settings.enabled.then(|| embed_tx.clone());
                let sig_tx = sessions_on.then(|| sig_tx.clone());
                let cache_tx = cache_tx.clone();
                scope.spawn(move || {
                    loop {
                        // Take the job and release the lock before decoding:
                        // holding it across the decode would serialise the
                        // pool back down to one worker.
                        let job = match jobs_rx.lock() {
                            Ok(rx) => rx.recv(),
                            Err(_) => break,
                        };
                        match job {
                            Ok(job) => decode_one(
                                job,
                                embed_tx.as_ref(),
                                sig_tx.as_ref(),
                                Some(&cache_tx),
                                &tx,
                            ),
                            Err(_) => break,
                        }
                    }
                });
            }

            // The reader itself. Sequential, in scan order, on this thread.
            for (index, group) in scanned_groups.iter().enumerate() {
                let job = read_one(index, group, budget, &prior, &src, &preview_cache);
                if jobs_tx.send(job).is_err() {
                    break;
                }
            }

            // Every stage ends when its last sender is gone, so these
            // originals have to go before the scope joins — otherwise the
            // pipeline waits on itself forever.
            drop(jobs_tx);
            drop(embed_tx);
            drop(sig_tx);
            drop(cache_tx);
        });

        let _ = tx.send(ScanMsg::Done);
    });
}

/// One photo waiting for a free core, and where its pixels came from.
struct DecodeJob {
    index: usize,
    path: PathBuf,
    companions: Vec<PathBuf>,
    hash: [u8; 32],
    imported: bool,
    skipped_before: bool,
    source: Source,
    readable: Readable,
    /// How the medium's cache identifies this file, for writing the
    /// preview back. `None` when the file's metadata could not be read, in
    /// which case nothing is cached for it.
    key: Option<maple_import::PreviewKey>,
}

/// Where one photo's pixels come from.
enum Source {
    /// Bytes read off the medium this run — the file itself for an
    /// ordinary image, a raw's embedded preview for a raw. `None` when the
    /// read failed or ran out of time. The canonical preview still has to
    /// be made, and is then written back to the medium.
    Fresh(Option<Vec<u8>>),
    /// The medium's own cache already held this photo's preview, so the
    /// file was **never opened**. That is the point of the cache: a rescan
    /// of an unchanged card costs a `stat` per photo instead of a ~100 ms
    /// read, and the capture time rides along because parsing it would
    /// have meant opening the file after all.
    Cached { webp: Vec<u8>, taken: Option<f64> },
}

/// One photo's worth of work for the session engine, queued behind the
/// decoders.
struct SignatureJob {
    index: usize,
    /// The frame the canonical preview decodes to — see
    /// [`maple_import::preview`] for why nothing computes on anything else.
    frame: RgbImage,
    /// Capture instant. The engine may vote on it — `time-gap` votes on
    /// nothing else — so it travels with the pixels rather than being
    /// looked up later.
    taken: Option<f64>,
}

/// One photo's worth of work for the embedder, queued behind the decoders.
struct EmbedJob {
    index: usize,
    hash: [u8; 32],
    frame: RgbImage,
}

/// One photo's preview on its way back to the medium it came from.
struct CacheJob {
    key: maple_import::PreviewKey,
    preview: maple_import::CachedPreview,
}

/// Work out everything about one photo that needs the medium, and look up
/// what earlier sessions decided about it. Runs on the single reader
/// thread.
///
/// **The file is opened only if it has to be.** The medium carries a cache
/// of the previews already made from it, keyed by what a directory entry
/// alone reveals — path, size and mtime — so a photo the last scan already
/// described costs one `stat` here instead of reading 25 MB off a card.
/// The content hash rides in the cache for the same reason: computing it
/// *is* the read being skipped.
fn read_one(
    index: usize,
    group: &maple_import::ImageGroup,
    budget: Duration,
    prior: &PriorDecisions,
    root: &Path,
    cache: &Mutex<maple_import::PreviewCache>,
) -> DecodeJob {
    let path = group.display.path.clone();
    let companions: Vec<PathBuf> = group.companions.iter().map(|c| c.path.clone()).collect();
    let key = maple_import::PreviewKey::for_file(root, &path);

    // Cloned out from under the lock rather than borrowed: the cache is
    // also being written to behind us, and a 15 KB memcpy is nothing
    // against the read it saves.
    let cached = key
        .as_ref()
        .and_then(|k| cache.lock().ok().and_then(|c| c.get(k).cloned()));

    if let Some(maple_import::CachedPreview { content_hash, taken, webp }) = cached {
        return DecodeJob {
            index,
            path,
            companions,
            hash: content_hash,
            imported: prior.imported.contains(&content_hash),
            skipped_before: prior.skipped.contains(&content_hash),
            source: Source::Cached { webp, taken },
            readable: Readable::Yes,
            key,
        };
    }

    let Read { hash, bytes, readable } = read_with_budget(&path, budget);

    // A photo we could not hash has no history to look up, and `SeenSet`
    // refuses to store the all-zero placeholder either way.
    let (imported, skipped_before) = match hash {
        Some(h) => (prior.imported.contains(&h), prior.skipped.contains(&h)),
        None => (false, false),
    };

    DecodeJob {
        index,
        path,
        companions,
        hash: hash.unwrap_or([0u8; 32]),
        imported,
        skipped_before,
        source: Source::Fresh(bytes),
        readable,
        key,
    }
}

/// What one read off the medium produces.
struct Read {
    /// The file's BLAKE3 content hash — `None` if it could not be read in
    /// time. Always over the *file*, never over a raw's preview, so it
    /// stays the same identifier the library and `SeenSet` use.
    hash: Option<[u8; 32]>,
    /// Bytes for the decoder, or `None` if there is nothing to decode.
    bytes: Option<Vec<u8>>,
    readable: Readable,
}

/// Read `path`, giving up after `budget`.
///
/// The read runs on its own thread so that abandoning it is possible at
/// all: a thread blocked on a card that has stopped answering cannot be
/// cancelled, only outlived. A timed-out thread is left to finish on its
/// own and its result discarded.
///
/// The budget is a parameter rather than read from settings here so a test
/// can prove the walk-away actually happens without waiting the configured
/// timeout to do it.
fn read_with_budget(path: &Path, budget: Duration) -> Read {
    let started = std::time::Instant::now();
    let outcome = within_budget(budget, {
        let path = path.to_path_buf();
        move || {
            // For an ordinary image these are the file's own bytes, so the
            // same read serves both the hash and the decode. Only a raw
            // needs the file opened twice — its preview is not the file.
            let bytes = maple_import::loadable_image_bytes(&path).ok();
            let hash = if maple_import::is_raw_format(&path) {
                maple_import::content_hash(&path).ok()
            } else {
                bytes.as_deref().map(maple_import::hash_bytes)
            };
            (hash, bytes)
        }
    });

    match outcome {
        Some((hash, Some(bytes))) => {
            Read { hash, bytes: Some(bytes), readable: Readable::Yes }
        }
        Some((hash, None)) => {
            tracing::warn!(
                target: "maple::import::unreadable",
                "unreadable after {:?}: {}",
                started.elapsed(),
                path.display()
            );
            Read { hash, bytes: None, readable: Readable::NoPreview }
        }
        None => {
            // The one the user goes hunting for: full path, not just the
            // filename, because two cards can hold the same DSCF0042.RAF.
            tracing::warn!(
                target: "maple::import::unreadable",
                "TIMEOUT after {:?} (budget {:?}), moving on without it: {}",
                started.elapsed(),
                budget,
                path.display()
            );
            Read { hash: None, bytes: None, readable: Readable::TimedOut }
        }
    }
}

/// Read the bytes to decode for `path`, giving up after `budget`.
///
/// The file itself for an ordinary image, a raw's embedded preview for a
/// raw. Shared with `import_previews`, which re-reads a photo whose preview
/// was evicted and whose WebP copy is gone too.
pub(crate) fn read_preview_bytes(path: &Path, budget: Duration) -> Option<Vec<u8>> {
    within_budget(budget, {
        let path = path.to_path_buf();
        move || maple_import::loadable_image_bytes(&path).ok()
    })
    .flatten()
}

/// Run `work` on its own thread and stop waiting for it after `budget`.
///
/// A thread blocked on a card that has stopped answering cannot be
/// cancelled, only outlived: on a timeout the thread is left to finish on
/// its own and its result discarded. The same trade `image_loader.rs` makes
/// for the detail view.
fn within_budget<T: Send + 'static>(
    budget: Duration,
    work: impl FnOnce() -> T + Send + 'static,
) -> Option<T> {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let _ = tx.send(work());
    });
    rx.recv_timeout(budget).ok()
}

/// Turn one photo into its canonical preview, the metadata row the UI
/// lists, and the frame every check runs on.
///
/// **One representation, and everything reads it.** The preview is encoded
/// here (or arrived from the medium's cache), decoded back once, and that
/// frame — not the pristine decode, which is thrown away — is what the
/// sharpness score, the session signature and the embedder all see. The
/// point is reproducibility: the only pixels the pipeline keeps are the
/// only pixels it computed on, so re-deriving any of it later from the
/// stored preview gives the scan's own answer back. See
/// [`maple_import::preview`].
///
/// Encoding here rather than on the reader is the same argument the whole
/// pipeline is built on: the reader is the one serial stage and the
/// slowest link, and this is pure CPU over bytes already in hand.
fn decode_one(
    job: DecodeJob,
    embed_tx: Option<&mpsc::SyncSender<EmbedJob>>,
    sig_tx: Option<&mpsc::SyncSender<SignatureJob>>,
    cache_tx: Option<&mpsc::SyncSender<CacheJob>>,
    tx: &mpsc::Sender<ScanMsg>,
) {
    let DecodeJob {
        index, path, companions, hash, imported, skipped_before, source, readable, key,
    } = job;

    // For a raw the bytes are the extracted preview, which carries the
    // same EXIF block — so the capture time costs no extra card I/O
    // either way.
    let (preview, taken, fresh) = match source {
        Source::Cached { webp, taken } => (Some(webp), taken, false),
        Source::Fresh(bytes) => {
            let taken = bytes
                .as_deref()
                .and_then(|b| maple_import::exif_read::read_bytes(b).capture_secs());
            let webp = bytes.as_deref().and_then(|b| match maple_import::preview::encode(b) {
                Ok(webp) => Some(webp),
                Err(err) => {
                    // The row stays listed and copyable with no picture,
                    // and the on-demand service will try the group's
                    // companions when the strip reaches it.
                    tracing::warn!(
                        target: "maple::import::unreadable",
                        "no preview for {}: {err}",
                        path.display()
                    );
                    None
                }
            });
            (webp, taken, true)
        }
    };

    let frame = preview
        .as_deref()
        .and_then(|webp| match maple_import::preview::decode(webp) {
            Ok(frame) => Some(frame),
            Err(err) => {
                tracing::warn!("Import scan: unreadable preview for {}: {err}", path.display());
                None
            }
        });

    let sharpness = frame
        .as_ref()
        .map(|f| maple_import::laplacian_variance(f.as_raw(), f.width(), f.height()));

    // Each `send` blocks once its stage is a queue-depth behind, which is
    // the throttle keeping every queue bounded — and the reason the
    // embedder slows the whole scan down while the signature stage, at
    // ~0.2 ms a photo, never catches up enough to matter. The frame is
    // cloned only when *both* want it, i.e. only when someone turned the
    // embedder on; the default path moves it.
    if let Some(frame) = frame {
        match (sig_tx, embed_tx) {
            (Some(sig_tx), Some(embed_tx)) => {
                let _ = sig_tx.send(SignatureJob { index, frame: frame.clone(), taken });
                let _ = embed_tx.send(EmbedJob { index, hash, frame });
            }
            (Some(sig_tx), None) => {
                let _ = sig_tx.send(SignatureJob { index, frame, taken });
            }
            (None, Some(embed_tx)) => {
                let _ = embed_tx.send(EmbedJob { index, hash, frame });
            }
            (None, None) => {}
        }
    }

    // Give the medium its copy — but only for a photo actually read this
    // run (a cached one is already there) and actually hashed: the
    // all-zero placeholder is not an identity, and storing it would let
    // the next scan serve one unreadable photo's preview for another.
    if let (true, Some(key), Some(webp), Some(cache_tx)) =
        (fresh, key, preview.as_ref(), cache_tx)
    {
        if hash != [0u8; 32] {
            let _ = cache_tx.send(CacheJob {
                key,
                preview: maple_import::CachedPreview {
                    content_hash: hash,
                    taken,
                    webp: webp.clone(),
                },
            });
        }
    }

    let _ = tx.send(ScanMsg::Thumb(ScanThumb {
        index,
        path,
        companions,
        content_hash: hash,
        imported,
        skipped_before,
        readable,
        sharpness,
        taken,
        preview,
    }));
}

/// Serial half of the scan: turn canonical previews into session
/// signatures.
///
/// Owns the one `&mut dyn SessionEngine`, which is what makes it serial.
/// That is not merely a borrow-checker artefact: [`TimeGapEngine`] anchors
/// its epoch on the first frame it sees (capture times near 1.8e9 quantise
/// to ~128-second steps in an `f32`, so it stores time relative to an
/// origin), and one engine per decode thread would hand each its own
/// origin and make their signatures incomparable.
///
/// Frames arrive in *decode-completion* order, not scan order, which is
/// fine — a signature describes one photo and nothing else. Order is
/// restored on the UI thread, where each lands in its own entry.
///
/// A photo the engine refuses simply produces no signature.
/// `segment_with_holes` then makes it its own session rather than
/// asserting a continuity nobody measured.
fn signature_stage(
    engine: &mut dyn maple_import::SessionEngine,
    jobs: mpsc::Receiver<SignatureJob>,
    tx: &mpsc::Sender<ScanMsg>,
) {
    for job in jobs {
        match engine.signature(&maple_import::Frame::new(&job.frame, job.taken)) {
            Ok(signature) => {
                let _ = tx.send(ScanMsg::Signature { index: job.index, signature });
            }
            Err(err) => {
                tracing::warn!("Import scan: signature failed for photo {}: {err}", job.index);
            }
        }
    }
}

/// Serial half of the scan: write canonical previews back to the medium.
///
/// One thread for the same reason there is one reader — a card is one bus,
/// and several decoders appending to it independently would contend with
/// the reads the scan is still doing. Writes go out in batches of
/// [`PREVIEW_CACHE_FLUSH_EVERY`]; the file is append-only, so a batch costs
/// only the photos new since the last one.
///
/// The lock is held across each flush, which briefly stalls the reader's
/// own lookups. That is the right way round: a scan that is *writing* the
/// cache is one whose lookups are missing anyway.
fn preview_cache_stage(cache: &Mutex<maple_import::PreviewCache>, jobs: mpsc::Receiver<CacheJob>) {
    let mut since_flush = 0usize;
    for job in jobs {
        let Ok(mut cache) = cache.lock() else { return };
        cache.insert(job.key, job.preview);
        since_flush += 1;
        if since_flush >= PREVIEW_CACHE_FLUSH_EVERY {
            flush_preview_cache(&mut cache);
            since_flush = 0;
        }
    }
    if let Ok(mut cache) = cache.lock() {
        flush_preview_cache(&mut cache);
    }
}

/// Best-effort by nature: a write-protected or full card must cost the user
/// a slower next scan and nothing else.
fn flush_preview_cache(cache: &mut maple_import::PreviewCache) {
    let pending = cache.pending();
    if let Err(err) = cache.flush() {
        tracing::warn!("Import scan: failed to write {pending} previews to the medium: {err}");
    }
}

/// Serial half of the scan: turn rendered photos into DINOv2 embeddings.
///
/// Owns the single ONNX session and the SD-card embedding cache, which is
/// why it cannot be one of the parallel workers. Ends when every render
/// worker has dropped its sender.
fn embed_stage(
    src: &Path,
    stack_settings: &maple_state::StackSettings,
    jobs: mpsc::Receiver<EmbedJob>,
    tx: &mpsc::Sender<ScanMsg>,
) {
    let algorithm_key = stack_settings.algorithm_key();
    let cache_path = src.join(EMBED_CACHE_FILE);
    let mut cache = maple_import::EmbeddingCache::load_from(&cache_path, &algorithm_key);
    let mut embedder = match maple_db::load_onnx_embedder(stack_settings) {
        Ok(e) => Some(e),
        Err(err) => {
            tracing::warn!(
                "Import scan: failed to load image embedder, skipping burst detection: {err}"
            );
            None
        }
    };
    let mut unflushed = 0usize;

    // Keep draining even with no embedder: cache hits are still real
    // embeddings, and leaving the channel unread would block every render
    // worker as soon as the queue filled.
    for job in jobs {
        let embedding = match cache.get(&job.hash) {
            Some(cached) => Some(cached.to_vec()),
            None => embedder.as_mut().and_then(|embedder| {
                match embedder.embed(&DynamicImage::ImageRgb8(job.frame)) {
                    Ok(v) => {
                        cache.insert(job.hash, v.clone());
                        unflushed += 1;
                        Some(v)
                    }
                    Err(err) => {
                        tracing::warn!("Import scan: embedding failed for photo {}: {err}", job.index);
                        None
                    }
                }
            }),
        };

        if let Some(embedding) = embedding {
            let _ = tx.send(ScanMsg::Embedding { index: job.index, embedding });
        }

        if unflushed >= EMBED_CACHE_FLUSH_EVERY {
            save_embed_cache(&cache, &cache_path);
            unflushed = 0;
        }
    }

    if unflushed > 0 {
        save_embed_cache(&cache, &cache_path);
    }
}

fn save_embed_cache(cache: &maple_import::EmbeddingCache, path: &Path) {
    if let Err(err) = cache.save_to(path) {
        tracing::warn!("Import scan: failed to write embedding cache: {err}");
    }
}

/// The scan's listing arrived — size the entry list, fill in every path,
/// and start decoding whatever the strip is showing.
///
/// The listing comes before any file has been read, so the count and the
/// filenames are right from the first frame; hashes and "already seen"
/// flags land later, per photo, as the reader gets to them.
fn apply_scan_listing(w: &ImportWindow, ctx: &ImportCtx, photos: Vec<PhotoRef>) {
    let n = photos.len();
    {
        let mut ents = ctx.entries.borrow_mut();
        ents.clear();
        ents.reserve(n);
        for photo in &photos {
            ents.push(Entry {
                path: photo.display.clone(),
                companions: photo.companions.clone(),
                content_hash: [0; 32],
                is_imported: false,
                passed: false,
                decided_before: false,
                readable: Readable::Yes,
                thumb: None,
                webp: None,
                signature: None,
                embedding: None,
                taken: None,
                sharpness: None,
            });
        }
    }
    w.set_total_count(n as i32);

    let tuning = maple_state::Settings::load().import;
    let (tx, rx) = mpsc::channel::<PreviewMsg>();
    *ctx.previews.borrow_mut() = Some(PreviewService::start(
        Arc::new(photos),
        tuning.read_timeout(),
        tuning.decoders(),
        tx,
    ));
    start_preview_drain(ctx, rx);

    // Bulk reset — once per scan, not on every click.
    refilter(w, ctx);
}

/// Drain decoded previews for as long as the browser is open.
///
/// A separate timer from the scan's: the user goes on scrolling long after
/// the index pass has finished, and every scroll is a new decode request.
fn start_preview_drain(ctx: &ImportCtx, rx: mpsc::Receiver<PreviewMsg>) {
    let drain_ctx = ctx.clone();
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(30), move || {
        let Some(w) = drain_ctx.window.upgrade() else { return };
        // A handful per tick: each one decodes a WebP into a frame, and
        // starving the event loop is exactly what this whole change is
        // avoiding. The *encode* happens on a worker, not here.
        for _ in 0..6 {
            match rx.try_recv() {
                Ok(msg) => apply_preview(&w, &drain_ctx, msg),
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
    });
    *ctx.preview_timer.borrow_mut() = Some(timer);
}

/// One decoded preview arrived (or failed to).
fn apply_preview(w: &ImportWindow, ctx: &ImportCtx, msg: PreviewMsg) {
    let index = match msg {
        PreviewMsg::Ready { index, webp, from_companion } => {
            if let Some(e) = ctx.entries.borrow_mut().get_mut(index) {
                e.webp = Some(webp);
                if from_companion {
                    e.readable = Readable::FromCompanion;
                }
            }
            // Inflating goes through the same path an evicted preview comes
            // back through: one place turns the kept representation into
            // pixels, and it already handles retention and the redraw.
            restore_preview(ctx, index);
            index
        }
        PreviewMsg::Failed { index } => {
            if let Some(e) = ctx.entries.borrow_mut().get_mut(index) {
                if e.readable.ok() {
                    e.readable = Readable::NoPreview;
                }
            }
            index
        }
    };
    redraw(ctx, index);
    if ctx.preview_shown_idx.get() == Some(index) {
        set_preview_state(w, ctx, index);
    }
}

/// Drop a preview's pixels, keeping its WebP so it can come back without
/// touching the medium.
fn evict_preview(ctx: &ImportCtx, index: usize) {
    if let Some(e) = ctx.entries.borrow_mut().get_mut(index) {
        e.thumb = None;
    }
    redraw(ctx, index);
}

/// Re-inflate an evicted preview from its WebP copy. Cheap enough to do on
/// the UI thread — it is a 256 px image — and it saves a card read.
fn restore_preview(ctx: &ImportCtx, index: usize) -> bool {
    let webp = match ctx.entries.borrow().get(index) {
        Some(e) if e.thumb.is_none() => e.webp.clone(),
        _ => None,
    };
    let Some(webp) = webp else { return false };
    let Ok((rgb, width, height)) = crate::thumbnail::decode_webp_rgb(&webp) else {
        return false;
    };
    let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(&rgb, width, height);
    if let Some(e) = ctx.entries.borrow_mut().get_mut(index) {
        e.thumb = Some(slint::Image::from_rgb8(buf));
    }
    let dropped = ctx.retained.borrow_mut().insert(index);
    for index in dropped {
        evict_preview(ctx, index);
    }
    redraw(ctx, index);
    true
}

/// Repaint one entry's tile, if it is on screen at all.
fn redraw(ctx: &ImportCtx, index: usize) {
    update_row(
        &ctx.model,
        &ctx.entries.borrow(),
        &ctx.selected.borrow(),
        &ctx.sessions.borrow(),
        &ctx.visible.borrow(),
        index,
    );
}

/// The strip reported which rows are on screen — decode those, and let go
/// of what has been out of sight longest.
///
/// `first`/`last` are model rows and inclusive; the strip already adds its
/// own prefetch margin either side of the viewport.
fn set_preview_window(ctx: &ImportCtx, first: usize, last: usize, focus: usize) {
    ctx.preview_window.set((first, last, focus));
    request_previews(ctx);
}

/// Is the strip showing this entry right now?
///
/// The preview window is in model *rows*, not entry indices — "Hide old
/// images" renumbers one without moving the other — so this has to go
/// through `visible` rather than compare the index directly.
fn in_preview_window(ctx: &ImportCtx, index: usize) -> bool {
    let (first, last, _) = ctx.preview_window.get();
    ctx.visible
        .borrow()
        .row(index)
        .is_some_and(|row| row >= first && row <= last)
}

/// Ask the preview service for everything on screen that is not already
/// decoded, nearest the viewport first.
fn request_previews(ctx: &ImportCtx) {
    let (first, last, focus) = ctx.preview_window.get();
    let rows: Vec<usize> = {
        let visible = ctx.visible.borrow();
        if visible.rows.is_empty() {
            return;
        }
        // Never ask for more than can be held. A window wider than the
        // retention cap is self-defeating: the tail would evict the head
        // and the strip would decode the same photos over and over. It
        // also means a bad geometry reading cannot turn into a request to
        // decode the entire card.
        let ceiling = ctx.retained.borrow().capacity();
        let last = last
            .min(visible.rows.len().saturating_sub(1))
            .min(first.saturating_add(ceiling.saturating_sub(1)));
        if first > last {
            return;
        }
        visible.rows[first..=last].to_vec()
    };
    let focus_entry = {
        let visible = ctx.visible.borrow();
        visible.rows.get(focus).copied().unwrap_or(0)
    };

    // Seeing a photo is what keeps its preview alive, so touch every row on
    // screen before anything is allowed to be evicted.
    {
        let mut retained = ctx.retained.borrow_mut();
        for &index in &rows {
            retained.touch(index);
        }
    }

    let mut wanted = Vec::new();
    for index in rows {
        let has_pixels = ctx
            .entries
            .borrow()
            .get(index)
            .is_some_and(|e| e.thumb.is_some());
        if has_pixels || restore_preview(ctx, index) {
            continue;
        }
        wanted.push(index);
    }

    if let Some(service) = ctx.previews.borrow().as_ref() {
        service.want(wanted, focus_entry);
    }
}

/// One scanned photo arrived — fill its entry in and refresh its row.
fn apply_scan_thumb(w: &ImportWindow, ctx: &ImportCtx, msg: ScanThumb) {
    let ScanThumb {
        index, path, companions, content_hash, imported, skipped_before, readable, sharpness,
        taken, preview,
    } = msg;
    let has_preview = preview.is_some();
    {
        let mut ents = ctx.entries.borrow_mut();
        if let Some(e) = ents.get_mut(index) {
            e.path = path.clone();
            e.companions = companions;
            e.content_hash = content_hash;
            e.is_imported = imported;
            // A photo an earlier session passed over shows its red ✗ right
            // away — that verdict is exactly what `passed` means.
            e.passed = skipped_before;
            e.decided_before = imported || skipped_before;
            e.readable = readable;
            e.sharpness = sharpness;
            e.taken = taken;
            // The scan makes a preview for every photo it reads, so the
            // strip almost never has to go back to the card. A photo the
            // scan could not decode keeps whatever the on-demand service
            // recovered for it.
            if let Some(webp) = preview {
                e.webp = Some(webp);
            }
        }
    }
    if has_preview && in_preview_window(ctx, index) {
        // Nothing else would: the preview window is only re-requested when
        // the viewport *moves*, and a user watching a scan fill in is not
        // scrolling. Without this the tiles on screen stay blank until the
        // next scroll even though their pixels are already in memory.
        restore_preview(ctx, index);
    }
    if imported || skipped_before {
        // Count it now rather than waiting for `refilter`. The *filter* can
        // only be applied in bulk once every verdict is in, but "Hide old
        // images" is gated on this count being non-zero, and a card of a few
        // thousand photos takes minutes to read — so deferring the count
        // hides the button for the whole scan, exactly when the user is
        // looking for it. Each index arrives once, so incrementing is exact,
        // and `refilter` recomputes it from scratch at the end regardless.
        w.set_old_count(w.get_old_count() + 1);
    }
    update_row(
        &ctx.model,
        &ctx.entries.borrow(),
        &ctx.selected.borrow(),
        &ctx.sessions.borrow(),
        &ctx.visible.borrow(),
        index,
    );
    ctx.scanned_count.set(ctx.scanned_count.get() + 1);
    w.set_scanned_count(ctx.scanned_count.get() as i32);
    // Name the first photo straight away. Its picture arrives separately,
    // through the preview service, once the strip says it is on screen.
    if index == 0 && ctx.current.get() == 0 {
        w.set_preview_filename(
            path.file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
                .into(),
        );
        show_preview(w, &ctx.entries, 0);
    }
}

/// The scan finished — segment the card into sessions.
///
/// Nothing is marked. The importer used to auto-select the sharpest photo
/// of every session, which is a guess dressed as a decision: sharpness
/// ranks a crisp badly-framed frame above the one where the subject is
/// looking at the camera, and once it is marked nobody looks again. The
/// keeper is chosen in the tournament instead (see
/// [`crate::import_tournament`]), where the two photos are actually put
/// side by side.
///
/// Segmentation, not clustering: the sequence is walked once and each photo
/// is asked whether the scene changed *here*. That is why it can run at
/// the end over signatures gathered in any order, and why a session always
/// comes out as a contiguous range — which is what makes the `f` grid able
/// to show one as a band with two ends the user can drag.
fn finish_scan(w: &ImportWindow, ctx: &ImportCtx) {
    let n = ctx.entries.borrow().len();

    let sessions = detect_sessions(&ctx.entries.borrow(), &maple_state::Settings::load().sessions);
    *ctx.sessions.borrow_mut() = sessions;
    let resolved_groups = groups_from_sessions(&ctx.sessions.borrow());

    *ctx.groups.borrow_mut() = resolved_groups;
    // A fresh segmentation is a fresh set of comparisons.
    rebuild_tournament(w, ctx);

    w.set_scanning(false);
    let quality = report_scan_quality(&ctx.entries.borrow());
    w.set_status_text(scan_status_text(n, quality).into());
    // Bulk reset — happens once when the scan finishes, not on every
    // click. This is also the point at which every entry's `decided_before`
    // is finally known, so it is where "Hide old images" can first take
    // effect (and where its count stops moving).
    refilter(w, ctx);
}

/// Segment the scanned photos into sessions.
///
/// Returns sessions **tiling the whole sequence**, solo photos included —
/// the `f` grid needs every photo to belong somewhere so a boundary can be
/// dragged onto it. [`groups_from_sessions`] is what narrows that to the
/// real groups.
///
/// Empty when session detection is off, when the engine spec names nothing
/// real, or when not one photo produced a signature (a card of files that
/// would not decode). Never an error the user has to clear: grouping is
/// enrichment on top of a scan that already worked.
fn detect_sessions(
    entries: &[Entry],
    settings: &maple_state::SessionSettings,
) -> Vec<maple_import::Session> {
    if !settings.enabled || entries.is_empty() {
        return Vec::new();
    }
    let signatures: Vec<Option<Signature>> = entries.iter().map(|e| e.signature.clone()).collect();
    if signatures.iter().all(Option::is_none) {
        return Vec::new();
    }
    let Ok(engine) = maple_import::session::engine_from_spec(&settings.engine) else {
        // The scan worker already logged why; saying it twice adds nothing.
        return Vec::new();
    };
    let times: Vec<Option<f64>> = entries.iter().map(|e| e.taken).collect();
    let params = segment_params(engine.as_ref(), settings);
    maple_import::session::segment_with_holes(engine.as_ref(), &signatures, &times, &params)
        .sessions
}

/// Turn the settings into [`SegmentParams`], leaving anything the user did
/// not set at what the engine itself considers right.
///
/// `cut` is the one that must not be copied blindly: engine distances are
/// not comparable to each other, so `0` means "ask the engine" rather than
/// "no threshold".
fn segment_params(
    engine: &dyn maple_import::SessionEngine,
    settings: &maple_state::SessionSettings,
) -> maple_import::SegmentParams {
    let mut params = maple_import::SegmentParams::for_spec(engine, &settings.engine);
    if settings.cut > 0.0 {
        params.cut = settings.cut;
    }
    params.hard_gap_secs = settings.hard_gap_secs;
    params.max_outliers = settings.max_outliers;
    params.anchor_factor = settings.anchor_factor;
    params
}

/// The sessions worth calling groups: two photos or more.
///
/// A one-photo session is a real answer — "nothing else belongs with this"
/// — but it is not a group, and auto-picking a keeper out of a group of one
/// would mark the whole card.
fn groups_from_sessions(sessions: &[maple_import::Session]) -> Vec<Vec<usize>> {
    sessions
        .iter()
        .filter(|s| s.len() >= 2)
        .map(|s| (s.start..s.end).collect())
        .collect()
}

/// Toggle the session boundary immediately before photo `at`.
///
/// The whole edit vocabulary, because a boundary is the only thing a
/// session *is*: sessions tile the sequence, so inserting a boundary splits
/// one in two and removing it merges two into one. "Set the start here" and
/// "set the stop here" are the same operation one photo apart, which is why
/// the `f` grid needs no separate merge key and no drag state.
///
/// `at == 0` and `at >= len` are no-ops: the sequence already begins and
/// ends there, and a boundary outside it would describe nothing.
fn toggle_boundary(sessions: &mut Vec<maple_import::Session>, at: usize) {
    let Some(last) = sessions.last().map(|s| s.end) else { return };
    if at == 0 || at >= last {
        return;
    }
    match sessions.iter().position(|s| s.start == at) {
        // Already a boundary: merge this session into the one before it.
        Some(i) if i > 0 => {
            sessions[i - 1].end = sessions[i].end;
            sessions.remove(i);
        }
        Some(_) => {}
        // No boundary yet: split whichever session spans this photo.
        None => {
            if let Some(i) = sessions.iter().position(|s| s.contains(at)) {
                let end = sessions[i].end;
                sessions[i].end = at;
                sessions.insert(i + 1, maple_import::Session { start: at, end });
            }
        }
    }
}

/// Put a session boundary immediately before photo `at`, if there is not
/// one already.
///
/// The click vocabulary, as against [`toggle_boundary`]'s key vocabulary.
/// A click that says "the session starts here" about a photo that already
/// starts one has got what it asked for; toggling would remove the
/// boundary, which is the opposite of what was asked. Returns whether
/// anything actually moved, so a click that changes nothing costs no
/// rebuild.
fn ensure_boundary(sessions: &mut Vec<maple_import::Session>, at: usize) -> bool {
    let Some(last) = sessions.last().map(|s| s.end) else { return false };
    if at == 0 || at >= last || sessions.iter().any(|s| s.start == at) {
        return false;
    }
    let Some(i) = sessions.iter().position(|s| s.contains(at)) else { return false };
    let end = sessions[i].end;
    sessions[i].end = at;
    sessions.insert(i + 1, maple_import::Session { start: at, end });
    true
}

/// How many photos in a finished scan needed help, or never rendered.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
struct ScanQuality {
    /// Display file was unreadable; the preview came off a companion.
    recovered: usize,
    /// Nothing in the group produced a picture.
    unreadable: usize,
}

fn scan_status_text(n: usize, quality: ScanQuality) -> String {
    // Say it on screen too. The per-file warnings go to stderr, which is
    // nowhere at all when the app was launched from a bundle.
    let mut out = format!("{n} photo{} found", if n == 1 { "" } else { "s" });
    if quality.recovered > 0 {
        out.push_str(&format!(" · {} previewed from RAW", quality.recovered));
    }
    if quality.unreadable > 0 {
        out.push_str(&format!(" · {} without a preview", quality.unreadable));
    }
    if quality.recovered > 0 || quality.unreadable > 0 {
        out.push_str(" (see the log for paths)");
    }
    out
}

/// Log every photo whose display file let it down, with full paths, and
/// report the counts.
///
/// One block at the end rather than only the per-file lines, because those
/// are scattered through a scan's worth of output and the path is exactly
/// what you need in order to go look at the file. Photos recovered from a
/// companion are listed too: the picture is fine, but the display file next
/// to it is corrupt and the user is the only one who can decide about that.
fn report_scan_quality(entries: &[Entry]) -> ScanQuality {
    let notable: Vec<&Entry> = entries
        .iter()
        .filter(|e| e.readable != Readable::Yes)
        .collect();
    let quality = ScanQuality {
        recovered: notable
            .iter()
            .filter(|e| e.readable == Readable::FromCompanion)
            .count(),
        unreadable: notable.iter().filter(|e| !e.readable.ok()).count(),
    };
    if notable.is_empty() {
        return quality;
    }
    let list = notable
        .iter()
        .map(|e| format!("  {}: {}", e.readable.reason(), e.path.display()))
        .collect::<Vec<_>>()
        .join("\n");
    tracing::warn!(
        target: "maple::import::unreadable",
        "{} of {} photos had an unreadable display file ({} recovered from a companion, {} with no preview at all):\n{list}",
        notable.len(),
        entries.len(),
        quality.recovered,
        quality.unreadable
    );
    quality
}

// ── Browsing (click + navigation) ─────────────────────────────────

fn wire_browse(window: &ImportWindow, ctx: &ImportCtx) {
    // ── Item clicked (toggle-select + update preview) ──────────────
    window.on_item_clicked({
        let ctx = ctx.clone();
        move |idx| {
            let Some(w) = ctx.window.upgrade() else { return };
            // `idx` is the entry's scan index, not its row: the strip emits
            // `item.index` so that hiding old photos cannot make a click
            // land on the wrong photo.
            let idx = idx as usize;
            if idx >= ctx.entries.borrow().len() {
                return;
            }
            // Only skip the reload if this exact photo is *already* the one
            // on screen — `current` alone isn't enough, since it defaults to
            // 0 before anything has ever been previewed.
            let already_shown = ctx.preview_shown_idx.get() == Some(idx);
            let marked = {
                let mut sel = ctx.selected.borrow_mut();
                if sel.contains(&idx) {
                    sel.remove(&idx);
                    false
                } else {
                    sel.insert(idx);
                    true
                }
            };
            record_brush(
                &mut ctx.brushed.borrow_mut(),
                idx,
                marked,
                &ctx.brush.borrow(),
            );
            w.set_copy_done(false);
            w.set_selected_count(ctx.selected.borrow().len() as i32);
            // A click only ever changes this one row's selection state —
            // update it in place rather than rebuilding the whole model.
            update_row(&ctx.model, &ctx.entries.borrow(), &ctx.selected.borrow(),
                       &ctx.sessions.borrow(), &ctx.visible.borrow(), idx);

            // Clicking the already-open photo just toggles its selection —
            // don't re-decode and re-render the big preview for no reason.
            if already_shown {
                set_preview_state(&w, &ctx, idx);
                return;
            }
            // Clicking a different tile leaves the open photo behind, which
            // is the same decision as stepping past it with the arrows.
            if let Some(prev) = ctx.preview_shown_idx.get() {
                leave(&ctx, prev);
            }
            set_current(&w, &ctx, idx);
            ctx.preview_shown_idx.set(Some(idx));

            show_preview(&w, &ctx.entries, idx);
            set_preview_state(&w, &ctx, idx);
        }
    });

    // ── Navigation ────────────────────────────────────────────────
    let make_nav = |delta: i32| {
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let len = ctx.entries.borrow().len();
            if len == 0 {
                return;
            }
            let cur = ctx.current.get();

            let new_idx = nav_visible_target(
                &ctx.groups.borrow(),
                &ctx.visible.borrow(),
                cur,
                len,
                delta,
            );

            if new_idx == cur {
                return;
            }
            leave(&ctx, cur);
            set_current(&w, &ctx, new_idx);
            ctx.preview_shown_idx.set(Some(new_idx));

            show_preview(&w, &ctx.entries, new_idx);
            set_preview_state(&w, &ctx, new_idx);
        }
    };
    window.on_preview_window({
        let ctx = ctx.clone();
        move |first, last, focus| {
            set_preview_window(
                &ctx,
                first.max(0) as usize,
                last.max(0) as usize,
                focus.max(0) as usize,
            );
        }
    });

    window.on_nav_prev(make_nav(-1));
    window.on_nav_next(make_nav(1));

    // ── The `f` grid: see the sessions, and correct them ──────────
    window.on_toggle_grid({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let open = !ctx.grid_open.get();
            ctx.grid_open.set(open);
            w.set_grid_open(open);
            // An edit half-made does not survive leaving the view it was
            // made in: the second click has nowhere to land.
            ctx.pending_cut.set(None);
            w.set_pending_session_start(-1);
            if open {
                // Opening it *during* a scan lands on the scan's own
                // frontier rather than wherever the cursor was left. The
                // reason to open the grid mid-scan is to watch the card
                // come in, and that happens at the far end of what has
                // been read, not at photo 0.
                let n = ctx.entries.borrow().len();
                if w.get_scanning() && n > 0 {
                    let frontier = ctx.scanned_count.get().saturating_sub(1).min(n - 1);
                    ctx.current.set(frontier);
                }
            }
            // The filter is suspended while the grid is up, so the visible
            // set really does change — this is a `refilter`, not a repaint.
            // It re-parks the current photo, which is what scrolls the grid
            // to it.
            refilter(&w, &ctx);
        }
    });

    // "Set the start here" and "set the stop here" are one operation a
    // photo apart: a boundary before `at`, or before the one after it.
    let cut_at = {
        let ctx = ctx.clone();
        move |at: usize, ensure: bool| {
            let Some(w) = ctx.window.upgrade() else { return };
            {
                let mut sessions = ctx.sessions.borrow_mut();
                if ensure {
                    ensure_boundary(&mut sessions, at);
                } else {
                    toggle_boundary(&mut sessions, at);
                }
            }
            *ctx.groups.borrow_mut() = groups_from_sessions(&ctx.sessions.borrow());
            // A boundary has a session on each side, so moving one changes
            // two ranges and every comparison drawn from them. The pass is
            // rebuilt rather than patched — and because it is rebuilt from
            // what is still *undecided*, correcting the grouping re-asks
            // about nothing the user has already settled.
            rebuild_tournament(&w, &ctx);
            w.set_selected_count(ctx.selected.borrow().len() as i32);
            // Every tile from here on carries a different session id, and
            // the marks moved — cheaper to think about than to patch row
            // by row, and it only happens on a keypress.
            refilter(&w, &ctx);
        }
    };
    window.on_cut_before({
        let cut_at = cut_at.clone();
        move |at| cut_at(at.max(0) as usize, false)
    });
    window.on_cut_after({
        let cut_at = cut_at.clone();
        move |at| cut_at(at.max(0) as usize + 1, false)
    });

    // A click in the grid draws a session by its two ends: the first opens
    // one on that photo, the next closes it there. Two clicks are the same
    // pair of boundaries `[` and `]` set, which is the point — there is
    // still only one edit vocabulary, and the mouse speaks it too.
    //
    // The click also moves the cursor, so opening a session does not cost
    // the ability to navigate by clicking; it just does both.
    window.on_grid_clicked({
        let ctx = ctx.clone();
        let cut_at = cut_at.clone();
        move |at| {
            let Some(w) = ctx.window.upgrade() else { return };
            let at = at.max(0) as usize;
            let n = ctx.entries.borrow().len();
            if at >= n {
                return;
            }
            set_current(&w, &ctx, at);
            match ctx.pending_cut.get() {
                // Closing one: the stop goes *after* this photo, which is
                // a boundary before the next — the same photo-apart
                // identity `[` and `]` are built on.
                Some(start) if at >= start => {
                    ctx.pending_cut.set(None);
                    w.set_pending_session_start(-1);
                    cut_at(at + 1, true);
                }
                // Clicking back before the open start is a change of mind
                // about where it begins, not a backwards session.
                _ => {
                    ctx.pending_cut.set(Some(at));
                    w.set_pending_session_start(at as i32);
                    cut_at(at, true);
                }
            }
        }
    });

    // ── Hide / show photos an earlier session already decided on ──
    window.on_toggle_hide_old({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let hide = !ctx.hide_old.get();
            ctx.hide_old.set(hide);
            w.set_hide_old(hide);
            // A view filter only: nothing is deselected, so a burst member
            // auto-picked out of an old group is still copied. Hiding
            // changes what you look at, never what happens.
            refilter(&w, &ctx);
        }
    });
}

/// Step from `cur`, then keep stepping until the landing spot is a photo
/// the strip is actually showing.
///
/// [`nav_target`] knows about burst groups; this only skips over what "Hide
/// old images" has filtered out. Returns `cur` when there is nothing
/// visible left in that direction, which is what makes the arrow keys stop
/// at the ends instead of jumping to a hidden photo.
fn nav_visible_target(
    groups: &[Vec<usize>],
    visible: &Visible,
    cur: usize,
    len: usize,
    delta: i32,
) -> usize {
    let mut at = nav_target(groups, cur, len, delta);
    // Bounded by the entry count: `nav_target` clamps at both ends, so the
    // walk always terminates, but a group layout should never be able to
    // turn a mis-step into a spin either.
    for _ in 0..len {
        if at == cur || visible.shows(at) {
            return at;
        }
        let next = nav_target(groups, at, len, delta);
        if next == at {
            return cur;
        }
        at = next;
    }
    cur
}

/// Where a left/right step from `cur` lands.
///
/// If the current photo belongs to a detected burst group, left/right jumps
/// directly to the prev/next member of that group (skipping over unrelated
/// interleaved photos). Moving past the first/last member of the group falls
/// through to ordinary flat navigation — arrow keys never trap the user
/// inside a burst; they page through all of its members, then continue into
/// the rest of the scan. Solo entries behave exactly as a plain flat clamp,
/// same as before this feature existed.
fn nav_target(groups: &[Vec<usize>], cur: usize, len: usize, delta: i32) -> usize {
    match find_group(groups, cur) {
        Some(members) => {
            let pos = members.iter().position(|&m| m == cur).unwrap_or(0);
            let next_pos = pos as i64 + delta as i64;
            if next_pos >= 0 && (next_pos as usize) < members.len() {
                members[next_pos as usize]
            } else {
                // Fell off the group's boundary — continue past it
                // with ordinary flat navigation from here.
                (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize
            }
        }
        None => (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize,
    }
}

/// Show entry `idx` in the big preview: filename and the already-decoded
/// grid thumb immediately, then a higher-res render from a worker thread.
fn show_preview(w: &ImportWindow, entries: &Rc<RefCell<Vec<Entry>>>, idx: usize) {
    let ents = entries.borrow();
    if let Some(e) = ents.get(idx) {
        let filename = e
            .path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        w.set_preview_filename(filename.into());
        // Show the existing thumb as a quick preview.
        if let Some(thumb) = &e.thumb {
            w.set_preview_photo(thumb.clone());
        }
        // Kick off a higher-res preview load.
        let path = e.path.clone();
        let w_weak = w.as_weak();
        w.set_preview_loading(true);
        std::thread::spawn(move || {
            let result = thumbnail::render_to_rgb(&path, 1200);
            let _ = slint::invoke_from_event_loop(move || {
                let Some(w) = w_weak.upgrade() else { return };
                if let Ok((rgb, pw, ph)) = result {
                    let buf =
                        slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
                            &rgb, pw, ph,
                        );
                    w.set_preview_photo(slint::Image::from_rgb8(buf));
                }
                w.set_preview_loading(false);
            });
        });
    }
}

// ── The tournament ────────────────────────────────────────────────
//
// Nothing on a card is preselected any more. The keeper of a session is
// chosen by looking at it beside its rival, which is the one thing a
// sharpness number cannot do — see [`crate::import_tournament`] for why
// the pass is shaped the way it is. Everything here is the wiring: the
// verdict keys, the shared zoom, and the renderer that keeps two decodes
// alive so a crop comes off the original rather than a 256 px preview.

use crate::import_tournament::{self, PaneMsg, PaneRequest, Side, Tournament, Verdict, MAX_ZOOM};

/// The shared view state of the two panes.
struct PairView {
    /// Size of one pane in pixels. Rust renders the crop at exactly this
    /// size rather than letting Slint scale it up — a zoomed pixel that
    /// had been resampled twice would be a worse picture than the one the
    /// user is trying to judge.
    view_w: u32,
    view_h: u32,
    /// 1.0 is "the whole photo fits the pane".
    zoom: f32,
    /// Wanted centre, in normalised source coordinates. Normalised is what
    /// lets a portrait and a landscape frame stay paired.
    cx: f32,
    cy: f32,
    /// Bumped on every change; a render carrying an older token is dropped
    /// rather than painted over a newer one.
    token: u64,
    /// Dimensions of the left photo's decode. The pan clamp reads one
    /// image, not both — reconciling two would let the shorter frame's
    /// edge drag the other off whatever was being compared.
    left_src: (u32, u32),
    /// Which photo each pane is currently showing.
    ///
    /// Only so a pane whose photo did *not* change is left alone. `1` —
    /// the most common keystroke — keeps the left contestant, and dropping
    /// it back to its blurry placeholder only to redraw the same crop a
    /// moment later would make the one thing being compared flicker on
    /// every press.
    shown: (Option<usize>, Option<usize>),
}

impl Default for PairView {
    fn default() -> Self {
        PairView {
            view_w: 0,
            view_h: 0,
            zoom: 1.0,
            cx: 0.5,
            cy: 0.5,
            token: 0,
            left_src: (0, 0),
            shown: (None, None),
        }
    }
}

impl PairView {
    /// Back to showing both photos whole. Called whenever the pair changes:
    /// a zoom is a question about *these* two frames, and carrying it to
    /// the next pair would open on a corner of something the user has not
    /// seen yet.
    fn fit(&mut self) {
        self.zoom = 1.0;
        self.cx = 0.5;
        self.cy = 0.5;
    }
}

fn wire_tournament(window: &ImportWindow, ctx: &ImportCtx) {
    window.on_toggle_tournament({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let on = !w.get_tourney_on();
            w.set_tourney_on(on);
            if on {
                start_pair_renderer(&ctx);
                rebuild_tournament(&w, &ctx);
            } else {
                stop_pair_renderer(&ctx);
                // Verdicts landed on rows the strip was not showing, and
                // some of those photos are now marked or passed over.
                refilter(&w, &ctx);
            }
        }
    });

    window.on_tourney_verdict({
        let ctx = ctx.clone();
        move |key| {
            let Some(w) = ctx.window.upgrade() else { return };
            let Some(v) = Verdict::from_key(key) else { return };
            apply_verdict(&w, &ctx, v);
        }
    });

    window.on_tourney_undo({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            undo_verdict(&w, &ctx);
        }
    });

    // The pane's size changed — a window resize, or the panes appearing
    // for the first time. Everything on screen is rendered *to* that size,
    // so nothing can be drawn before it is known.
    window.on_tourney_view({
        let ctx = ctx.clone();
        move |pw, ph| {
            let (pw, ph) = (pw.max(0) as u32, ph.max(0) as u32);
            {
                let mut v = ctx.pair_view.borrow_mut();
                if (v.view_w, v.view_h) == (pw, ph) || pw == 0 || ph == 0 {
                    return;
                }
                v.view_w = pw;
                v.view_h = ph;
            }
            request_pair(&ctx);
        }
    });

    window.on_tourney_pan({
        let ctx = ctx.clone();
        move |dx, dy| {
            {
                let mut v = ctx.pair_view.borrow_mut();
                let (sw, sh) = v.left_src;
                // At fit there is nothing to pan, and no photo means no
                // scale to convert pane pixels into source pixels with.
                if v.zoom <= 1.0 || sw == 0 || sh == 0 || v.view_w == 0 {
                    return;
                }
                let fit = (v.view_w as f32 / sw as f32).min(v.view_h as f32 / sh as f32);
                let scale = fit * v.zoom;
                // Dragging right moves the picture right, so the window
                // into it moves left.
                let (cx, cy) = (v.cx - dx / (scale * sw as f32), v.cy - dy / (scale * sh as f32));
                let (cx, cy) = import_tournament::clamp_center(
                    sw, sh, v.view_w, v.view_h, v.zoom, cx, cy,
                );
                if (cx, cy) == (v.cx, v.cy) {
                    return;
                }
                v.cx = cx;
                v.cy = cy;
            }
            request_pair(&ctx);
        }
    });

    window.on_tourney_zoom({
        let ctx = ctx.clone();
        move |delta, fx, fy| {
            {
                let mut v = ctx.pair_view.borrow_mut();
                // A trackpad reports many small deltas and a wheel a few
                // large ones; clamping the notch count is what stops one
                // flick of a wheel jumping from fit to maximum.
                let notches = (delta / 50.0).clamp(-1.5, 1.5);
                let next = (v.zoom * 1.35f32.powf(notches)).clamp(1.0, MAX_ZOOM);
                if (next - v.zoom).abs() < 0.001 {
                    return;
                }
                let (sw, sh) = v.left_src;
                let (cx, cy) = if sw == 0 || sh == 0 || v.view_w == 0 {
                    (0.5, 0.5)
                } else {
                    import_tournament::zoom_at(
                        sw, sh, v.view_w, v.view_h, v.zoom, v.cx, v.cy, next, fx, fy,
                    )
                };
                v.zoom = next;
                v.cx = cx;
                v.cy = cy;
            }
            publish_zoom(&ctx);
            request_pair(&ctx);
        }
    });

    window.on_tourney_reset_zoom({
        let ctx = ctx.clone();
        move || {
            if ctx.pair_view.borrow().zoom <= 1.0 {
                return;
            }
            ctx.pair_view.borrow_mut().fit();
            publish_zoom(&ctx);
            request_pair(&ctx);
        }
    });
}

fn publish_zoom(ctx: &ImportCtx) {
    if let Some(w) = ctx.window.upgrade() {
        w.set_tourney_zoom_level(ctx.pair_view.borrow().zoom);
    }
}

/// Start the pane renderer and the timer that drains it.
///
/// Both live only while the tournament is on: the renderer holds two full
/// decodes, which is tens of megabytes to keep alive for a mode nobody is
/// in.
fn start_pair_renderer(ctx: &ImportCtx) {
    let (tx, rx) = mpsc::channel::<PaneMsg>();
    *ctx.pair_renderer.borrow_mut() = Some(import_tournament::PairRenderer::spawn(tx));
    ctx.pair_view.borrow_mut().fit();

    let ctx2 = ctx.clone();
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, Duration::from_millis(16), move || {
        let Some(w) = ctx2.window.upgrade() else { return };
        while let Ok(msg) = rx.try_recv() {
            apply_pane(&w, &ctx2, msg);
        }
    });
    *ctx.pair_timer.borrow_mut() = Some(timer);
}

fn stop_pair_renderer(ctx: &ImportCtx) {
    ctx.pair_renderer.borrow_mut().take();
    ctx.pair_timer.borrow_mut().take();
}

fn apply_pane(w: &ImportWindow, ctx: &ImportCtx, msg: PaneMsg) {
    match msg {
        PaneMsg::Ready { side, token, rgb, w: pw, h: ph, src_w, src_h } => {
            // A render of a viewport the user has already scrolled past
            // must not land on top of a newer one.
            if token != ctx.pair_view.borrow().token {
                return;
            }
            let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(&rgb, pw, ph);
            let img = slint::Image::from_rgb8(buf);
            match side {
                Side::Left => {
                    ctx.pair_view.borrow_mut().left_src = (src_w, src_h);
                    w.set_tourney_left(img);
                    w.set_tourney_left_shown(true);
                }
                Side::Right => {
                    w.set_tourney_right(img);
                    w.set_tourney_right_shown(true);
                }
            }
        }
        // The card's copy of this photo would not decode. The pane keeps
        // whatever the canonical preview gave it — which is the whole
        // reason that placeholder is put up first.
        PaneMsg::Failed { side, token } => {
            tracing::warn!(target: "maple::import::tournament", "pane {side:?} token {token} failed");
        }
    }
}

/// Build the pass over the groups as they now stand.
///
/// Called when the tournament is switched on, when a scan finishes, and
/// when a session boundary moves. Photos already settled are skipped, so a
/// rebuild never re-asks a question the user has answered.
fn rebuild_tournament(w: &ImportWindow, ctx: &ImportCtx) {
    // A rebuild can put a different photo in a pane without the index
    // changing hands, so nothing may be assumed still on screen.
    ctx.pair_view.borrow_mut().shown = (None, None);
    let has_groups = ctx.groups.borrow().iter().any(|g| g.len() >= 2);
    // Stays offered once the pass is over — hiding the switch would take
    // away the only way back out of the mode.
    w.set_tourney_available(has_groups);

    // Every verdict already given rides across the rebuild, so correcting
    // a boundary re-groups what is *left* and re-asks nothing.
    let carried = ctx.tournament.borrow().as_ref().map(Tournament::carry).unwrap_or_default();
    let t = {
        let entries = ctx.entries.borrow();
        let groups = ctx.groups.borrow();
        Tournament::build(&groups, carried, |i| {
            // A photo already in the library cannot be the answer to
            // "which of these do I import", and one that never decoded
            // cannot be judged at all — both stay out of the pass and
            // remain for the ordinary triage.
            entries.get(i).is_none_or(|e| e.is_imported || !e.readable.ok())
        })
    };
    *ctx.tournament.borrow_mut() = Some(t);
    ctx.pair_view.borrow_mut().fit();
    publish_zoom(ctx);
    publish_tournament(w, ctx);
}

/// Push the current comparison into the window, and ask for its pixels.
fn publish_tournament(w: &ImportWindow, ctx: &ImportCtx) {
    let (pair, done, total, round, rounds, note) = {
        let guard = ctx.tournament.borrow();
        let Some(t) = guard.as_ref() else { return };
        let (done, total) = t.progress();
        let (round, rounds) = t.round();
        w.set_tourney_can_undo(t.can_undo());
        // "No pair on screen" and "the pass is over" are the same state
        // here, but only one of them says why.
        w.set_tourney_idle(t.finished());
        (t.pair(), done, total, round, rounds, tournament_note(t))
    };
    w.set_tourney_done(done as i32);
    w.set_tourney_total(total as i32);
    w.set_tourney_round(round as i32);
    w.set_tourney_rounds(rounds as i32);

    let Some((left, right)) = pair else {
        w.set_tourney_note(note.into());
        return;
    };

    // The canonical preview goes up immediately and the crop off the
    // original replaces it. Clearing the panes instead would flash an
    // empty frame on every verdict; leaving the *previous* photo up would
    // be worse still — for a moment the user would be comparing the wrong
    // pair.
    {
        let ents = ctx.entries.borrow();
        let was = ctx.pair_view.borrow().shown;
        for (side, idx, before) in [(Side::Left, left, was.0), (Side::Right, right, was.1)] {
            let Some(e) = ents.get(idx) else { continue };
            let name: SharedString = e
                .path
                .file_name()
                .map(|n| n.to_string_lossy().into_owned())
                .unwrap_or_default()
                .into();
            let meta: SharedString = pane_meta(e, idx).into();
            // A pane still showing the same photo keeps the crop it has.
            let fresh = before != Some(idx);
            let placeholder = fresh.then(|| pane_placeholder(e)).flatten();
            match side {
                Side::Left => {
                    w.set_tourney_left_name(name);
                    w.set_tourney_left_meta(meta);
                    if fresh {
                        w.set_tourney_left_shown(placeholder.is_some());
                        w.set_tourney_left(placeholder.unwrap_or_default());
                    }
                }
                Side::Right => {
                    w.set_tourney_right_name(name);
                    w.set_tourney_right_meta(meta);
                    if fresh {
                        w.set_tourney_right_shown(placeholder.is_some());
                        w.set_tourney_right(placeholder.unwrap_or_default());
                    }
                }
            }
        }
        ctx.pair_view.borrow_mut().shown = (Some(left), Some(right));
    }
    request_pair(ctx);
}

/// What to say when there is no comparison on screen.
///
/// "Nothing to run" and "nothing left to run" are different answers, and
/// only one of them means the user is finished.
fn tournament_note(t: &Tournament) -> String {
    let (kept, passed) = t.tally();
    if t.is_empty() && kept + passed == 0 {
        return "No session has two photos to compare — nothing to run a tournament on.".into();
    }
    format!("Tournament complete — {kept} kept, {passed} passed over.")
}

/// The pixel size to render one pane at.
///
/// The pane reports its own size and that is the number that is actually
/// right. But everything on screen is rendered *to* that number, so a
/// report that never arrives leaves both panes blank forever — and a Slint
/// `changed` handler that quietly does not fire is a trap this file has
/// already been caught by twice (`changed items` against `set_vec`, and a
/// freshly-created grid never firing `changed current-grid-row`). So when
/// nothing has been reported yet the size is derived from the window
/// instead: half its width, less the header and the two caption bars. That
/// is wrong by a few pixels and costs a slightly soft render until the
/// real number lands, which is a better failure than a feature that does
/// not draw.
fn pane_size(window: (u32, u32), reported_w: u32, reported_h: u32) -> (u32, u32) {
    if reported_w > 0 && reported_h > 0 {
        return (reported_w, reported_h);
    }
    const CHROME_H: u32 = 46 + 30;
    ((window.0 / 2).max(1), window.1.saturating_sub(CHROME_H).max(1))
}

/// Something to look at while the crop off the original is being made.
///
/// The high-resolution render needs a decode — up to half a second for a
/// raw — and putting an empty frame up for that long on every verdict
/// would make the pass feel broken. The canonical preview is ~15 KB and
/// already in memory for every photo the scan read, so inflating it here
/// costs a millisecond and shows the right photo immediately, at the wrong
/// resolution, which is the correct order to be wrong in.
fn pane_placeholder(e: &Entry) -> Option<slint::Image> {
    if let Some(thumb) = &e.thumb {
        return Some(thumb.clone());
    }
    let frame = maple_import::preview::decode(e.webp.as_deref()?).ok()?;
    let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
        frame.as_raw(),
        frame.width(),
        frame.height(),
    );
    Some(slint::Image::from_rgb8(buf))
}

/// The line under a contestant: where it sits on the card, and the number
/// that used to decide this on the user's behalf.
///
/// The sharpness score is shown rather than hidden. It is still measured
/// (from the canonical preview, like everything else), the user is being
/// asked to judge exactly the thing it measures, and withholding a number
/// the system already has would be arbitrary. It is a hint at the bottom
/// of the frame, not a verdict — which is the whole difference from what
/// it used to be.
fn pane_meta(e: &Entry, idx: usize) -> String {
    match e.sharpness {
        Some(s) => format!("#{}  ·  sharpness {s:.0}", idx + 1),
        None => format!("#{}", idx + 1),
    }
}

/// Ask the renderer for both panes at the current view state.
///
/// Both sides every time, even when only the challenger changed: the
/// incumbent's decode is still cached, so re-rendering it is a resize
/// rather than a read, and one code path is worth more than the saving.
fn request_pair(ctx: &ImportCtx) {
    let Some((left, right)) = ctx.tournament.borrow().as_ref().and_then(Tournament::pair) else {
        return;
    };
    let paths = {
        let ents = ctx.entries.borrow();
        match (ents.get(left), ents.get(right)) {
            (Some(l), Some(r)) => (l.path.clone(), r.path.clone()),
            _ => return,
        }
    };
    let Some(w) = ctx.window.upgrade() else { return };
    let size = w.window().size();
    let scale = w.window().scale_factor().max(0.1);
    let logical = (
        (size.width as f32 / scale) as u32,
        (size.height as f32 / scale) as u32,
    );
    let (token, view_w, view_h, zoom, cx, cy) = {
        let mut v = ctx.pair_view.borrow_mut();
        let (view_w, view_h) = pane_size(logical, v.view_w, v.view_h);
        v.token += 1;
        (v.token, view_w, view_h, v.zoom, v.cx, v.cy)
    };
    let guard = ctx.pair_renderer.borrow();
    let Some(renderer) = guard.as_ref() else { return };
    for (side, path) in [(Side::Left, paths.0), (Side::Right, paths.1)] {
        renderer.request(PaneRequest { side, token, path, view_w, view_h, zoom, cx, cy });
    }
}

/// Record one verdict: the photos it settled, the marks they carry, and
/// the next comparison.
fn apply_verdict(w: &ImportWindow, ctx: &ImportCtx, v: Verdict) {
    let (settled, was) = {
        let mut guard = ctx.tournament.borrow_mut();
        let Some(t) = guard.as_mut() else { return };
        let was = t.round();
        (t.decide(v), was)
    };
    if settled.is_empty() {
        return;
    }
    settle_marks(
        &mut ctx.entries.borrow_mut(),
        &mut ctx.selected.borrow_mut(),
        &mut ctx.brushed.borrow_mut(),
        &ctx.brush.borrow(),
        &settled,
    );
    finish_verdict(w, ctx, was, settled.iter().map(|&(i, _)| i));
}

/// Take back the last verdict, withdrawing every mark it made.
fn undo_verdict(w: &ImportWindow, ctx: &ImportCtx) {
    let (undone, was) = {
        let mut guard = ctx.tournament.borrow_mut();
        let Some(t) = guard.as_mut() else { return };
        let was = t.round();
        (t.undo(), was)
    };
    if undone.is_empty() {
        return;
    }
    withdraw_marks(
        &mut ctx.entries.borrow_mut(),
        &mut ctx.selected.borrow_mut(),
        &mut ctx.brushed.borrow_mut(),
        &undone,
    );
    finish_verdict(w, ctx, was, undone.into_iter());
}

/// Turn one verdict's settlements into marks.
///
/// A photo that wins is marked for import; a photo that loses is
/// `passed` — the same decision the red ✗ records when the user steps past
/// one by hand, and the same one `commit_skips` writes to the medium so a
/// re-scan does not offer it again. Losing a comparison is not "no answer
/// yet", and recording it as one would hand the whole card back next time.
fn settle_marks(
    entries: &mut [Entry],
    selected: &mut HashSet<usize>,
    brushed: &mut HashMap<usize, Vec<i64>>,
    brush: &[Tag],
    settled: &[(usize, bool)],
) {
    for &(idx, kept) in settled {
        if kept {
            selected.insert(idx);
        } else {
            selected.remove(&idx);
        }
        // Same rule as marking by hand: the brush is read at the moment
        // the photo is marked, so changing tags half way through a pass
        // tags the two halves differently.
        record_brush(brushed, idx, kept, brush);
        if let Some(e) = entries.get_mut(idx) {
            e.passed = !kept;
        }
    }
}

/// Put the photos an undone verdict decided back to undecided.
fn withdraw_marks(
    entries: &mut [Entry],
    selected: &mut HashSet<usize>,
    brushed: &mut HashMap<usize, Vec<i64>>,
    undone: &[usize],
) {
    for &idx in undone {
        selected.remove(&idx);
        record_brush(brushed, idx, false, &[]);
        if let Some(e) = entries.get_mut(idx) {
            // `decided_before` is deliberately untouched: that is an
            // *earlier* session's verdict, not this pass's to withdraw.
            e.passed = false;
        }
    }
}

/// The half both a verdict and its undo share: repaint the rows that
/// moved, then show the next comparison.
///
/// `was` is the round the pass was on beforehand, which is what decides
/// whether the zoom survives — see below.
fn finish_verdict(
    w: &ImportWindow,
    ctx: &ImportCtx,
    was: (usize, usize),
    touched: impl Iterator<Item = usize>,
) {
    w.set_copy_done(false);
    w.set_selected_count(ctx.selected.borrow().len() as i32);
    for idx in touched {
        update_row(
            &ctx.model,
            &ctx.entries.borrow(),
            &ctx.selected.borrow(),
            &ctx.sessions.borrow(),
            &ctx.visible.borrow(),
            idx,
        );
    }
    // The zoom survives inside a session and is dropped between them.
    // Twenty frames of one child in one room are framed alike, so having
    // to re-zoom onto the eyes nineteen times is exactly the tedium this
    // is meant to remove; the next *session* is a different scene, and
    // opening it on a corner of something the user has not seen yet would
    // be disorienting.
    let now = ctx.tournament.borrow().as_ref().map(Tournament::round).unwrap_or(was);
    if now.0 != was.0 {
        ctx.pair_view.borrow_mut().fit();
    }
    publish_zoom(ctx);
    publish_tournament(w, ctx);
}

// ── Import tags ───────────────────────────────────────────────────
//
// Maple has one labelling system, `collections`, so an import tag *is* a
// collection — one you happened to assign while triaging rather than
// afterwards. Nothing new is stored: the tag a photo picks up here is
// visible in the Collections window, filterable in the library, and
// replicated by sync, all for free.

/// One label the brush can apply, denormalised out of a `collections` row
/// so the picker does not hold the database lock while it is open.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Tag {
    id: i64,
    name: String,
    /// Hex, as stored in `collections.color`.
    color: String,
}

/// Colours a newly minted import tag can take.
///
/// Chosen *from the name* rather than at random or by position, so two
/// devices that both create "Holiday" before they ever sync agree on what
/// it looks like — and so re-creating a deleted tag brings its colour back.
const TAG_PALETTE: [&str; 8] = [
    "#B5543E", "#3388FF", "#33B073", "#D79A1B", "#9B59B6", "#1ABC9C", "#E4572E", "#5B7DB1",
];

fn tag_color_for(name: &str) -> &'static str {
    let sum: usize = name.bytes().map(usize::from).sum();
    TAG_PALETTE[sum % TAG_PALETTE.len()]
}

/// Find the collection called `name`, creating it if there isn't one.
///
/// An existing name wins rather than erroring: `collections.name` is UNIQUE,
/// and a user typing a name that already exists means "that one". This is
/// also the path that lets the picker's text field double as a search box.
fn ensure_tag(db: &Arc<Mutex<maple_db::Database>>, name: &str) -> Option<Tag> {
    let name = name.trim();
    if name.is_empty() {
        return None;
    }
    let guard = maple_db::lock_db(db);
    match guard.all_collections() {
        Ok(all) => {
            if let Some(c) = all.into_iter().find(|c| c.name == name) {
                return Some(Tag { id: c.id, name: c.name, color: c.color });
            }
        }
        Err(err) => {
            // Fall through and try to create it: a failed read must not stop
            // the user tagging, and a name collision is caught below anyway.
            tracing::warn!("ensure_tag: all_collections failed: {err}");
        }
    }
    let color = tag_color_for(name);
    match guard.create_collection(name, color, None) {
        Ok(id) => Some(Tag { id, name: name.to_string(), color: color.to_string() }),
        Err(err) => {
            tracing::warn!("ensure_tag: could not create tag {name:?}: {err}");
            None
        }
    }
}

/// Every tag in the library, with `chosen` set for the ones on the brush.
fn tag_choices(db: &Arc<Mutex<maple_db::Database>>, brush: &[Tag]) -> Vec<Tag> {
    match maple_db::lock_db(db).all_collections() {
        Ok(all) => all
            .into_iter()
            .map(|c| Tag { id: c.id, name: c.name, color: c.color })
            .collect(),
        Err(err) => {
            // Better a picker holding only what is already on the brush than
            // no picker at all — the name field still works.
            tracing::warn!("tag_choices: all_collections failed: {err}");
            brush.to_vec()
        }
    }
}

/// Stamp (or clear) the brush on one photo as it is marked.
///
/// Reading the brush *here*, at mark time, rather than at copy time is what
/// makes it a brush: change tags half way through a pass and the two halves
/// are tagged differently. Unmarking drops the record rather than keeping it
/// around, so a photo re-marked under a different brush cannot pick the old
/// tags back up.
fn record_brush(
    brushed: &mut HashMap<usize, Vec<i64>>,
    idx: usize,
    marked: bool,
    brush: &[Tag],
) {
    if marked {
        brushed.insert(idx, brush.iter().map(|t| t.id).collect());
    } else {
        brushed.remove(&idx);
    }
}

fn to_ui_tag(tag: &Tag, chosen: bool) -> crate::ImportTag {
    crate::ImportTag {
        id: tag.id as i32,
        name: SharedString::from(tag.name.as_str()),
        color: crate::transforms::hex_to_color(&tag.color),
        chosen,
    }
}

/// Push the brush into the floating panel, and — while the picker is open —
/// into its list too.
///
/// One function because the two views must agree: a tag ticked in the picker
/// and missing from the panel would leave the user unsure what an import is
/// about to write.
fn publish_tags(w: &ImportWindow, ctx: &ImportCtx) {
    let brush = ctx.brush.borrow();
    let chips: Vec<crate::ImportTag> = brush.iter().map(|t| to_ui_tag(t, true)).collect();
    w.set_tags(ModelRc::new(VecModel::from(chips)));

    if !w.get_tag_picker_open() {
        return;
    }
    let choices: Vec<crate::ImportTag> = tag_choices(&ctx.db, &brush)
        .iter()
        .map(|t| to_ui_tag(t, brush.iter().any(|b| b.id == t.id)))
        .collect();
    w.set_tag_choices(ModelRc::new(VecModel::from(choices)));
}

fn wire_tags(window: &ImportWindow, ctx: &ImportCtx) {
    window.on_open_tag_picker({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            // Set the flag first: `publish_tags` only fills the choice list
            // when the picker is actually open.
            w.set_tag_picker_open(true);
            publish_tags(&w, &ctx);
        }
    });

    window.on_close_tag_picker({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            w.set_tag_picker_open(false);
            // The picker owned the keyboard while it was open; hand it back
            // or the triage hotkeys stay dead until the user clicks.
            w.invoke_refocus_keys();
        }
    });

    window.on_tag_toggled({
        let ctx = ctx.clone();
        move |id| {
            let Some(w) = ctx.window.upgrade() else { return };
            let id = id as i64;
            let held = ctx.brush.borrow().iter().position(|t| t.id == id);
            match held {
                Some(at) => {
                    ctx.brush.borrow_mut().remove(at);
                }
                None => {
                    // The name is not on the brush, so it has to come from
                    // the library — and looking it up by id keeps the picker
                    // honest if a collection was renamed under it.
                    let found = maple_db::lock_db(&ctx.db)
                        .collection_by_id(id)
                        .ok()
                        .flatten()
                        .map(|c| Tag { id: c.id, name: c.name, color: c.color });
                    let Some(tag) = found else {
                        tracing::warn!("tag_toggled: no collection {id}");
                        return;
                    };
                    ctx.brush.borrow_mut().push(tag);
                }
            }
            publish_tags(&w, &ctx);
        }
    });

    window.on_tag_created({
        let ctx = ctx.clone();
        move |name| {
            let Some(w) = ctx.window.upgrade() else { return };
            let Some(tag) = ensure_tag(&ctx.db, name.as_str()) else { return };
            // Typing a name already on the brush is a no-op rather than a
            // duplicate chip.
            if !ctx.brush.borrow().iter().any(|t| t.id == tag.id) {
                ctx.brush.borrow_mut().push(tag);
            }
            publish_tags(&w, &ctx);
        }
    });

    window.on_clear_tags({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            // Only the brush. Photos already marked keep what they were
            // marked with — `c` means "stop tagging", not "untag".
            ctx.brush.borrow_mut().clear();
            publish_tags(&w, &ctx);
        }
    });

    publish_tags(window, ctx);
}

// ── Copy selected ─────────────────────────────────────────────────

/// Per-run values the copy drain needs alongside the shared [`ImportCtx`].
struct CopyRun {
    /// The selected entries, ascending — the rows to mark imported.
    sel_indices: Vec<usize>,
    /// Root of the medium these photos came off — where the imported
    /// record is written back.
    source: PathBuf,
    library_dir: PathBuf,
    algorithm_key: String,
}

fn wire_copy(window: &ImportWindow, ctx: &ImportCtx) {
    window.on_copy_selected({
        let ctx = ctx.clone();
        move || {
            let Some(w) = ctx.window.upgrade() else { return };
            let sel = ctx.selected.borrow().clone();
            if sel.is_empty() {
                return;
            }
            let dst = ctx.dest.borrow().clone();
            if dst.as_os_str().is_empty() {
                w.set_status_text("No destination folder set.".into());
                return;
            }
            let src = ctx.scanned_source.borrow().clone();

            w.set_copying(true);
            w.set_copy_done(false);
            w.set_status_text("Copying…".into());

            let settings = maple_state::Settings::load();
            let folder_template = settings.path_template.folder.clone();
            let filename_template = settings.path_template.filename.clone();

            let copy_mode = copy_mode_from_index(w.get_copy_mode());

            let mut sel_indices: Vec<usize> = sel.iter().copied().collect();
            sel_indices.sort_unstable();
            let sources = copy_sources(&ctx.entries.borrow(), &sel_indices, copy_mode);

            let run = CopyRun {
                sel_indices,
                source: src,
                library_dir: settings.library_dir.clone(),
                algorithm_key: settings.stacks.algorithm_key(),
            };

            let (tx, rx) = mpsc::channel::<CopyMsg>();
            let dst2 = dst.clone();
            std::thread::spawn(move || {
                let result = maple_import::copy_images(
                    &sources,
                    &dst2,
                    &folder_template,
                    &filename_template,
                    |done, total| {
                        let _ = tx.send(CopyMsg::Progress { done, total });
                    },
                );
                match result {
                    Ok(summary) => {
                        let _ = tx.send(CopyMsg::Done {
                            copied: summary.copied,
                            failed: summary.failed,
                            dest_by_source: summary.destination_map(),
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(CopyMsg::Error(e.to_string()));
                    }
                }
            });

            let ctx2 = ctx.clone();
            let timer = Timer::default();
            timer.start(
                TimerMode::Repeated,
                Duration::from_millis(30),
                move || {
                    let Some(w) = ctx2.window.upgrade() else { return };
                    loop {
                        match rx.try_recv() {
                            Ok(CopyMsg::Progress { done, total }) => {
                                if total > 0 {
                                    w.set_status_text(
                                        format!("Copying… {done} / {total}").into(),
                                    );
                                }
                            }
                            Ok(CopyMsg::Done {
                                copied,
                                failed,
                                dest_by_source,
                            }) => {
                                finish_copy(&w, &ctx2, &run, copied, failed, &dest_by_source);
                                return;
                            }
                            Ok(CopyMsg::Error(e)) => {
                                w.set_copying(false);
                                w.set_status_text(format!("Copy error: {e}").into());
                                return;
                            }
                            Err(mpsc::TryRecvError::Empty) => break,
                            Err(mpsc::TryRecvError::Disconnected) => {
                                w.set_copying(false);
                                return;
                            }
                        }
                    }
                },
            );
            *ctx.copy_timer.borrow_mut() = Some(timer);
        }
    });
}

/// Which files of each selected group the copy-mode dropdown asks for.
fn copy_mode_from_index(mode: i32) -> maple_import::CopyMode {
    match mode {
        0 => maple_import::CopyMode::DisplayOnly,
        2 => maple_import::CopyMode::RawOnly,
        _ => maple_import::CopyMode::All,
    }
}

/// Write the photos this session turned down to the medium's skipped
/// record.
///
/// Called when a copy finishes and when the import window closes — the two
/// ends of a session. A card triaged across several sittings must not
/// present the same rejects again in the next one, and triage that ends
/// without a copy is still triage.
///
/// Photos already in the library are left out: they are in the *imported*
/// record, which already answers "decided before" for them, and listing one
/// in both would only blur what each record means.
fn commit_skips(ctx: &ImportCtx) {
    let source = ctx.scanned_source.borrow().clone();
    if source.as_os_str().is_empty() {
        return;
    }
    let mut skipped = maple_state::SeenSet::new();
    {
        let entries = ctx.entries.borrow();
        let selected = ctx.selected.borrow();
        for (i, e) in entries.iter().enumerate() {
            if e.passed && !e.is_imported && !selected.contains(&i) {
                skipped.insert(&e.content_hash);
            }
        }
    }
    if skipped.is_empty() {
        return;
    }
    let library_dir = maple_state::Settings::load().library_dir;
    record_on_medium(&skipped, &source, &library_dir, maple_state::Record::Skipped);
}

/// Fold `set` into one of the medium's records, logging what happened.
///
/// A source that cannot be written to is not a failure — a read-only card
/// is ordinary, and the library replica still carries the decision — so it
/// is reported at `info`. Losing the record entirely is the `warn`.
fn record_on_medium(
    set: &maple_state::SeenSet,
    source: &std::path::Path,
    library_dir: &std::path::Path,
    record: maple_state::Record,
) {
    match set.merge_save_to_source(source, library_dir, record) {
        Ok(true) => {}
        Ok(false) => tracing::info!(
            "Import: {} is not writable; {} was recorded in the library only",
            source.display(),
            record.on_medium()
        ),
        Err(err) => tracing::warn!("Import: failed to write {}: {err}", record.on_medium()),
    }
}

/// Flatten the selected entries into the file list to hand `copy_images`.
fn copy_sources(
    entries: &[Entry],
    sel_indices: &[usize],
    copy_mode: maple_import::CopyMode,
) -> Vec<PathBuf> {
    let mut sources: Vec<PathBuf> = Vec::new();
    for &i in sel_indices {
        if let Some(e) = entries.get(i) {
            let group = maple_import::ImageGroup {
                display: maple_import::ImageFile {
                    path: e.path.clone(),
                    size: 0,
                },
                companions: e
                    .companions
                    .iter()
                    .map(|p| maple_import::ImageFile { path: p.clone(), size: 0 })
                    .collect(),
            };
            for p in group.paths_for_copy(copy_mode) {
                sources.push(p);
            }
        }
    }
    sources
}

/// The copy finished — record the imported photos and flash the button.
///
/// `dest_by_source` maps each source file to where it landed; entries whose
/// display file is absent from it were either not copied (a `RawOnly` run) or
/// failed, and are left for the library scanner to discover on its next pass.
fn finish_copy(
    w: &ImportWindow,
    ctx: &ImportCtx,
    run: &CopyRun,
    copied: usize,
    failed: usize,
    dest_by_source: &HashMap<PathBuf, PathBuf>,
) {
    // Record the import on the medium it came off.
    //
    // Only the hashes copied *this run* go into `fresh`; the union with
    // what is already on disk happens inside `merge_save_to_source`. That
    // is what makes two importers running at once combine rather than
    // clobber — the old load-modify-write here silently lost one of them.
    {
        let mut fresh = maple_state::SeenSet::new();
        let mut ents = ctx.entries.borrow_mut();
        for &i in &run.sel_indices {
            if let Some(e) = ents.get_mut(i) {
                e.is_imported = true;
                fresh.insert(&e.content_hash);
            }
        }
        drop(ents);
        record_on_medium(
            &fresh,
            &run.source,
            &run.library_dir,
            maple_state::Record::Imported,
        );
    }
    // A copy ends a session's worth of triage, so the skips go down with it.
    commit_skips(ctx);
    // Insert display files into library DB, under the path they were copied
    // to. `Entry::path` is the *source* path — an SD-card path that stops
    // existing the moment the card is ejected.
    let to_insert: Vec<ImportEntry> = {
        let ents = ctx.entries.borrow();
        let brushed = ctx.brushed.borrow();
        run.sel_indices
            .iter()
            .filter_map(|&i| ents.get(i).map(|e| (i, e)))
            .filter_map(|(i, e)| {
                // No destination for the display file means it wasn't copied:
                // either the copy failed, or this was a `RawOnly` run whose
                // raw file the scanner will pick up and hash for itself.
                // `content_hash` is the *display* file's hash, so pinning it
                // to a raw file here would poison the thumbnail cache key.
                let path = dest_by_source.get(&e.path)?.clone();
                let raw_path = e
                    .companions
                    .iter()
                    .find(|c| maple_import::is_raw_format(c))
                    .and_then(|c| dest_by_source.get(c))
                    .cloned();
                Some(ImportEntry {
                    path,
                    raw_path,
                    content_hash: e.content_hash,
                    embedding: e.embedding.clone(),
                    collections: brushed.get(&i).cloned().unwrap_or_default(),
                })
            })
            .collect()
    };
    insert_imported_images(&ctx.db, &to_insert, &run.algorithm_key);
    // Backfill EXIF for the records just inserted, and refresh the library
    // grid once it has: an import is the one moment the user is *expecting*
    // new photos to appear, and until now they only did on the next scan.
    maple_db::spawn_metadata_filler(
        ctx.db.clone(),
        Some(Arc::new(|| {
            let _ = slint::invoke_from_event_loop(crate::grid::request_reload);
        })),
    );
    ctx.selected.borrow_mut().clear();
    // These belong to the selection that was just consumed. The *brush*
    // deliberately survives: "subsequent imports have the tag added" means
    // the next pass keeps tagging until the user says otherwise.
    ctx.brushed.borrow_mut().clear();
    w.set_selected_count(0);
    w.set_copying(false);
    w.set_status_text(copy_status_text(copied, failed).into());
    // Only the copied rows' selected/imported flags
    // changed — update those in place.
    {
        let ents = ctx.entries.borrow();
        let sel = ctx.selected.borrow();
        let ses = ctx.sessions.borrow();
        let vis = ctx.visible.borrow();
        for &i in &run.sel_indices {
            update_row(&ctx.model, &ents, &sel, &ses, &vis, i);
        }
    }

    // Flash the button green, then revert to the
    // normal "Copy Selected" state.
    w.set_copy_done(true);
    let w_weak = w.as_weak();
    let done_timer = Timer::default();
    done_timer.start(
        TimerMode::SingleShot,
        Duration::from_millis(2500),
        move || {
            if let Some(w) = w_weak.upgrade() {
                w.set_copy_done(false);
            }
        },
    );
    *ctx.copy_done_timer.borrow_mut() = Some(done_timer);
}

fn copy_status_text(copied: usize, failed: usize) -> String {
    if failed == 0 {
        format!("Copied {copied} photo{}", if copied == 1 { "" } else { "s" })
    } else {
        format!("Copied {copied}, {failed} failed")
    }
}

// ── Rotate current photo ─────────────────────────────────────────────

/// Wire the rotate buttons.
///
/// Patches the EXIF Orientation tag on the *source* file in place
/// (mirrors detail.rs's rotate_current), then re-renders the grid thumb
/// and the big preview so the change is visible immediately. The file's
/// bytes (and therefore its content hash) change, so the in-memory
/// Entry's content_hash is updated too — it's what gets recorded in the
/// SeenSet / DB on copy.
fn wire_rotate(window: &ImportWindow, ctx: &ImportCtx) {
    window.on_rotate({
        let ctx = ctx.clone();
        move |clockwise| {
            let Some(w) = ctx.window.upgrade() else { return };
            if w.get_rotating() {
                return;
            }
            let idx = ctx.current.get();
            let path = {
                let ents = ctx.entries.borrow();
                match ents.get(idx) {
                    Some(e) => e.path.clone(),
                    None => return,
                }
            };

            w.set_rotating(true);

            let (tx, rx) = mpsc::channel::<RotateMsg>();
            std::thread::spawn(move || {
                let msg = match maple_db::rotate_image_file(&path, clockwise) {
                    Ok((_, content_hash)) => {
                        let thumb = thumbnail::render_to_rgb(&path, 256);
                        let preview = thumbnail::render_to_rgb(&path, 1200);
                        match (thumb, preview) {
                            (Ok(thumb), Ok(preview)) => RotateMsg::Done { content_hash, thumb, preview },
                            (Err(e), _) | (_, Err(e)) => RotateMsg::Error(e.to_string()),
                        }
                    }
                    Err(e) => RotateMsg::Error(e.to_string()),
                };
                let _ = tx.send(msg);
            });

            let ctx2 = ctx.clone();
            let timer = Timer::default();
            timer.start(TimerMode::Repeated, Duration::from_millis(32), move || {
                let Some(w) = ctx2.window.upgrade() else { return };
                let outcome = match rx.try_recv() {
                    Ok(m) => m,
                    Err(mpsc::TryRecvError::Empty) => return,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        RotateMsg::Error("Rotation worker vanished".to_owned())
                    }
                };
                if let Some(t) = ctx2.rotate_timer.borrow().as_ref() {
                    t.stop();
                }
                match outcome {
                    RotateMsg::Done { content_hash, thumb, preview } => {
                        apply_rotation(&w, &ctx2, idx, content_hash, thumb, preview);
                    }
                    RotateMsg::Error(msg) => {
                        w.set_status_text(format!("Rotate failed: {msg}").into());
                        w.set_rotating(false);
                    }
                }
            });
            *ctx.rotate_timer.borrow_mut() = Some(timer);
        }
    });
}

/// A rotation landed — swap in the re-rendered thumb and preview.
fn apply_rotation(
    w: &ImportWindow,
    ctx: &ImportCtx,
    idx: usize,
    content_hash: [u8; 32],
    thumb: (Vec<u8>, u32, u32),
    preview: (Vec<u8>, u32, u32),
) {
    if let Some(e) = ctx.entries.borrow_mut().get_mut(idx) {
        e.content_hash = content_hash;
        let (rgb, tw, th) = thumb;
        let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
            &rgb, tw, th,
        );
        e.thumb = Some(slint::Image::from_rgb8(buf));
    }
    let (rgb, pw, ph) = preview;
    let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
        &rgb, pw, ph,
    );
    w.set_preview_photo(slint::Image::from_rgb8(buf));
    update_row(&ctx.model, &ctx.entries.borrow(), &ctx.selected.borrow(),
               &ctx.sessions.borrow(), &ctx.visible.borrow(), idx);
    w.set_rotating(false);
}

// ── Model rows ────────────────────────────────────────────────────

/// Build the [`ImportItem`] for a single entry.
fn make_item(
    entries: &[Entry],
    selected: &HashSet<usize>,
    sessions: &[maple_import::Session],
    i: usize,
) -> ImportItem {
    let e = &entries[i];
    let display_is_raw = maple_import::is_raw_format(&e.path);
    let has_jpg =
        !display_is_raw || e.companions.iter().any(|c| !maple_import::is_raw_format(c));
    let has_raw =
        display_is_raw || e.companions.iter().any(|c| maple_import::is_raw_format(c));
    let is_selected = selected.contains(&i);
    let ordinal = sessions.iter().position(|s| s.contains(i));
    let session = ordinal.map(|n| &sessions[n]);
    ImportItem {
        index: i as i32,
        filename: e
            .path
            .file_name()
            .map(|n| SharedString::from(n.to_string_lossy().as_ref()))
            .unwrap_or_default(),
        thumb: e.thumb.clone().unwrap_or_default(),
        // "Has a picture", not "has been scanned": previews are decoded
        // around the viewport and dropped again when they fall behind.
        loaded: e.thumb.is_some(),
        is_selected,
        is_imported: e.is_imported,
        // One verdict per tile, decided here rather than half here and half
        // in the markup. Marking a photo outranks having walked past it,
        // and a photo already in the library is never a "skip" — it is
        // wearing the ✓ scrim, which says more.
        is_skipped: e.passed && !is_selected && !e.is_imported,
        is_unreadable: !e.readable.ok(),
        // The picture is real, but the file the library will point at is
        // not — worth a mark the user can notice while triaging.
        from_companion: e.readable == Readable::FromCompanion,
        is_old: e.decided_before,
        // A session of one is a real answer but not a group, and a tile
        // that said "1" would be noise on most of a card.
        stack_size: session.filter(|s| s.len() >= 2).map(|s| s.len() as i32).unwrap_or(0),
        // Where this photo sits in its session. Both true at once is a
        // session of one; the grid uses them to close the band on itself.
        session_start: session.is_some_and(|s| s.start == i),
        session_end: session.is_some_and(|s| s.end == i + 1),
        // Identity for the highlight, parity for the alternating tint.
        session_id: ordinal.map(|n| n as i32).unwrap_or(-1),
        has_jpg,
        has_raw,
    }
}

/// Build the full [`ImportItem`] vec from current state, for bulk resets
/// (new scan, count/size known, scan finished). Not for per-click updates —
/// use [`update_row`] for those so a full model swap doesn't tear down and
/// recreate every tile's `TouchArea` (which can drop a click landing
/// mid-rebuild).
fn build_items(
    entries: &[Entry],
    selected: &HashSet<usize>,
    sessions: &[maple_import::Session],
    visible: &Visible,
) -> Vec<ImportItem> {
    visible
        .rows
        .iter()
        .map(|&i| make_item(entries, selected, sessions, i))
        .collect()
}

/// Recompute which entries the strip shows and rebuild the model.
///
/// A full `set_vec`, which tears down and rebuilds every tile, so it is
/// only called when the visible set can actually have changed: the toggle,
/// and the two bulk points of a scan. Rows arriving one at a time go
/// through [`update_row`] instead.
fn refilter(w: &ImportWindow, ctx: &ImportCtx) {
    let items = {
        let entries = ctx.entries.borrow();
        let mut visible = ctx.visible.borrow_mut();
        // The `f` grid suspends the filter rather than working around it:
        // one model serves both views, and a session band drawn over a
        // sequence with photos missing from the middle would lie.
        visible.rebuild(&entries, ctx.hide_old.get() && !ctx.grid_open.get());
        w.set_old_count(entries.iter().filter(|e| e.decided_before).count() as i32);
        w.set_session_count(ctx.sessions.borrow().len() as i32);
        // The number that means something. Half the sessions on a real card
        // are solo — 954 photos segment into 329 sessions but only 165
        // groups — so the raw count alone reads as noise.
        w.set_group_count(
            ctx.sessions.borrow().iter().filter(|s| s.len() >= 2).count() as i32,
        );
        build_items(&entries, &ctx.selected.borrow(), &ctx.sessions.borrow(), &visible)
    };
    ctx.model.set_vec(items);

    // Turning the filter on can hide the photo the user is looking at.
    // Land them on the first one still showing rather than leaving the
    // preview on something the strip no longer contains.
    let landing = {
        let visible = ctx.visible.borrow();
        if visible.shows(ctx.current.get()) {
            None
        } else {
            visible.rows.first().copied()
        }
    };
    if let Some(idx) = landing {
        set_current(w, ctx, idx);
        ctx.preview_shown_idx.set(Some(idx));
        show_preview(w, &ctx.entries, idx);
    } else {
        // The photo the user is on survived the filter, but the rows around
        // it were renumbered wholesale — re-park it, or the strip would go
        // on showing whatever now happens to sit at the old scroll offset.
        set_current(w, ctx, ctx.current.get());
    }
    set_preview_state(w, ctx, ctx.current.get());
    // The rows now mean different entries, so what is worth decoding has
    // changed even though the viewport has not moved.
    {
        let visible = ctx.visible.borrow();
        let (first, last, _) = ctx.preview_window.get();
        let row = visible.row(ctx.current.get()).unwrap_or(0);
        if let Some(win) = preview_window_for(visible.rows.len(), last.saturating_sub(first), row) {
            drop(visible);
            ctx.preview_window.set(win);
        }
    }
    request_previews(ctx);
}

/// Where the preview window belongs when `row` is the current photo and the
/// strip is `rows` long, keeping a span of `span` rows. `None` for an empty
/// strip.
///
/// The strip only reports its window when the *viewport* moves, and toggling
/// "Hide old images" does not move it — it renumbers every row underneath
/// it. So the window Rust is holding afterwards names different photos than
/// it did, and after a filter that shrank the strip it can name rows that no
/// longer exist at all, which makes [`request_previews`] bail out and leaves
/// every visible tile blank. Re-deriving it from where the current photo
/// actually landed is what keeps them filled; the strip corrects it on the
/// next scroll.
fn preview_window_for(rows: usize, span: usize, row: usize) -> Option<(usize, usize, usize)> {
    if rows == 0 {
        return None;
    }
    let span = span.max(INITIAL_WINDOW);
    let last_row = rows - 1;
    // Centre on the current photo, then let both ends fall out of the
    // clamp: near the top `first` is 0, near the bottom `last` is the end.
    let first = row.saturating_sub(span / 2).min(last_row);
    let last = first.saturating_add(span).min(last_row);
    Some((first, last, row.min(last_row)))
}

/// Point the window at entry `idx`: the highlighted tile, and the row the
/// filmstrip parks it in.
///
/// One function because the two must not drift. The strip scrolls when the
/// *row* changes, so setting the index alone would highlight a tile that is
/// off screen and leave it there — which is exactly what stepping with the
/// arrow keys used to do once the selection walked past the viewport.
fn set_current(w: &ImportWindow, ctx: &ImportCtx, idx: usize) {
    ctx.current.set(idx);
    w.set_current_index(idx as i32);
    w.set_current_row(strip_row(&ctx.visible.borrow(), idx));
    // Which band the `f` grid should light up. Follows the cursor, so
    // walking the card walks the highlight, and `[`/`]` always act on the
    // session the user is looking at.
    w.set_current_session(
        ctx.sessions
            .borrow()
            .iter()
            .position(|s| s.contains(idx))
            .map(|n| n as i32)
            .unwrap_or(-1),
    );
}

/// Mirror the current photo's verdict into the preview bar's chip, so the
/// state is legible where the user is actually looking.
///
/// 0 undecided, 1 marked for import, 2 moved past, 3 already in the
/// library — the same precedence the tile badge uses.
fn set_preview_state(w: &ImportWindow, ctx: &ImportCtx, idx: usize) {
    let entries = ctx.entries.borrow();
    // Marked first: in a triage pass the chip's job is to say what *this*
    // session decided, and "already in the library" is a fact the tile's ✓
    // is already carrying.
    let state = match entries.get(idx) {
        None => 0,
        Some(_) if ctx.selected.borrow().contains(&idx) => 1,
        Some(e) if e.is_imported => 3,
        Some(e) if e.passed => 2,
        Some(_) => 0,
    };
    w.set_preview_state(state);
}

/// Moving off a photo is a decision: whatever is left unmarked earns the
/// red ✗ from that moment on.
///
/// Marked photos are flagged too even though nothing shows for them, so
/// that un-marking one later reads as a skip rather than reverting to
/// "never looked at" — the user did look at it.
fn leave(ctx: &ImportCtx, idx: usize) {
    let newly_passed = match ctx.entries.borrow_mut().get_mut(idx) {
        Some(e) if !e.passed => {
            e.passed = true;
            true
        }
        _ => false,
    };
    if newly_passed {
        update_row(
            &ctx.model,
            &ctx.entries.borrow(),
            &ctx.selected.borrow(),
            &ctx.sessions.borrow(),
            &ctx.visible.borrow(),
            idx,
        );
    }
}

/// Update the row showing entry `i` in place. A no-op when that entry is
/// currently filtered out of the strip — it will be rebuilt with the right
/// state whenever the filter next changes.
fn update_row(
    model: &VecModel<ImportItem>,
    entries: &[Entry],
    selected: &HashSet<usize>,
    sessions: &[maple_import::Session],
    visible: &Visible,
    i: usize,
) {
    if i >= entries.len() {
        return;
    }
    if let Some(row) = visible.row(i) {
        model.set_row_data(row, make_item(entries, selected, sessions, i));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(path: &str, sharpness: Option<f32>) -> Entry {
        Entry {
            path: PathBuf::from(path),
            companions: vec![],
            content_hash: [0; 32],
            is_imported: false,
            passed: false,
            decided_before: false,
            readable: Readable::Yes,
            thumb: None,
            webp: None,
            signature: None,
            embedding: None,
            taken: None,
            sharpness,
        }
    }

    /// The mapping the strip has when nothing is filtered out.
    fn all_visible(entries: &[Entry]) -> Visible {
        let mut v = Visible::default();
        v.rebuild(entries, false);
        v
    }

    // ── find_group / nav_target ───────────────────────────────────

    #[test]
    fn find_group_returns_the_group_holding_the_index() {
        let groups = vec![vec![0, 2], vec![3, 4, 5]];
        assert_eq!(find_group(&groups, 2), Some([0, 2].as_slice()));
        assert_eq!(find_group(&groups, 4), Some([3, 4, 5].as_slice()));
        assert_eq!(find_group(&groups, 1), None);
        assert_eq!(find_group(&[], 0), None);
    }

    #[test]
    fn nav_target_walks_the_burst_group_before_the_flat_list() {
        // Entries 1, 3 and 4 form one burst; 0 and 2 are unrelated shots
        // interleaved between them.
        let groups = vec![vec![1, 3, 4]];
        assert_eq!(nav_target(&groups, 1, 5, 1), 3);
        assert_eq!(nav_target(&groups, 3, 5, 1), 4);
        assert_eq!(nav_target(&groups, 4, 5, -1), 3);
    }

    #[test]
    fn nav_target_falls_through_to_flat_steps_at_a_group_boundary() {
        let groups = vec![vec![1, 3, 4]];
        // Past the last member — continue flat rather than trapping the user.
        assert_eq!(nav_target(&groups, 4, 6, 1), 5);
        // Before the first member — likewise.
        assert_eq!(nav_target(&groups, 1, 6, -1), 0);
    }

    #[test]
    fn nav_target_clamps_at_the_ends_without_groups() {
        assert_eq!(nav_target(&[], 0, 3, -1), 0);
        assert_eq!(nav_target(&[], 2, 3, 1), 2);
        assert_eq!(nav_target(&[], 1, 3, 1), 2);
    }

    // ── Burst resolution ──────────────────────────────────────────


    // ── Session boundary editing (the `f` grid) ───────────────────

    fn sess(spans: &[(usize, usize)]) -> Vec<maple_import::Session> {
        spans.iter().map(|&(start, end)| maple_import::Session { start, end }).collect()
    }

    fn spans(s: &[maple_import::Session]) -> Vec<(usize, usize)> {
        s.iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn a_boundary_splits_the_session_that_spans_it() {
        let mut s = sess(&[(0, 10)]);
        toggle_boundary(&mut s, 4);
        assert_eq!(spans(&s), vec![(0, 4), (4, 10)]);
    }

    #[test]
    fn toggling_the_same_boundary_again_merges_it_back() {
        // Split and merge are one operation, which is why the grid needs no
        // separate merge key.
        let mut s = sess(&[(0, 10)]);
        toggle_boundary(&mut s, 4);
        toggle_boundary(&mut s, 4);
        assert_eq!(spans(&s), vec![(0, 10)]);
    }

    #[test]
    fn sessions_still_tile_the_sequence_after_any_edit() {
        // The invariant everything else leans on: every photo belongs to
        // exactly one session, so the grid can always draw a band under it
        // and `session_of` never returns None.
        let mut s = sess(&[(0, 12)]);
        for at in [7, 3, 9, 3, 11, 1, 9] {
            toggle_boundary(&mut s, at);
            let mut expect = 0;
            for session in &s {
                assert_eq!(session.start, expect, "gap or overlap: {:?}", spans(&s));
                assert!(session.end > session.start, "empty session: {:?}", spans(&s));
                expect = session.end;
            }
            assert_eq!(expect, 12, "sessions must cover everything: {:?}", spans(&s));
        }
    }

    #[test]
    fn the_two_ends_of_the_sequence_are_not_boundaries() {
        // Photo 0 already starts a session and the last already ends one;
        // a boundary there would describe nothing and must not invent an
        // empty session.
        let mut s = sess(&[(0, 5)]);
        toggle_boundary(&mut s, 0);
        toggle_boundary(&mut s, 5);
        toggle_boundary(&mut s, 99);
        assert_eq!(spans(&s), vec![(0, 5)]);
        let mut empty: Vec<maple_import::Session> = Vec::new();
        toggle_boundary(&mut empty, 3);
        assert!(empty.is_empty());
    }

    // ── The tournament ────────────────────────────────────────────

    use crate::import_tournament::{Tournament, Verdict};

    /// Run a whole pass and report `(marked, passed over)`.
    fn run_pass(
        entries: &mut [Entry],
        groups: &[Vec<usize>],
        script: &[Verdict],
    ) -> (HashSet<usize>, Vec<usize>) {
        let mut t = Tournament::build(groups, Vec::new(), |_| false);
        let mut selected = HashSet::new();
        let mut brushed = HashMap::new();
        for v in script {
            let settled = t.decide(*v);
            settle_marks(entries, &mut selected, &mut brushed, &[], &settled);
        }
        let passed = entries
            .iter()
            .enumerate()
            .filter(|(_, e)| e.passed)
            .map(|(i, _)| i)
            .collect();
        (selected, passed)
    }

    fn six() -> Vec<Entry> {
        (0..6).map(|i| entry(&format!("{i}.jpg"), Some(i as f32))).collect()
    }

    /// The property the whole pass exists to guarantee: nothing it covers
    /// is left undecided, and nothing is both kept and passed over. A
    /// photo left in neither state would be silently dropped from the
    /// import *and* offered again on the next scan.
    #[test]
    fn a_finished_pass_leaves_every_photo_marked_or_passed_and_never_both() {
        let mut entries = six();
        let groups = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let script = [Verdict::Right, Verdict::Both, Verdict::Left, Verdict::Both];
        let (selected, passed) = run_pass(&mut entries, &groups, &script);

        for i in 0..6 {
            assert!(
                selected.contains(&i) != passed.contains(&i),
                "photo {i}: selected={} passed={}",
                selected.contains(&i),
                passed.contains(&i)
            );
        }
    }

    #[test]
    fn holding_the_incumbent_all_the_way_keeps_one_photo_from_the_session() {
        let mut entries = six();
        let groups = vec![vec![0, 1, 2, 3, 4, 5]];
        let (selected, passed) = run_pass(&mut entries, &groups, &[Verdict::Left; 5]);
        assert_eq!(selected, HashSet::from([0]));
        assert_eq!(passed, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn taking_both_every_time_keeps_the_whole_session() {
        let mut entries = six();
        let groups = vec![vec![0, 1, 2, 3, 4, 5]];
        let (selected, passed) = run_pass(&mut entries, &groups, &[Verdict::Both; 5]);
        assert_eq!(selected.len(), 6);
        assert!(passed.is_empty());
    }

    /// A loss is a *skip*, not an absence — which is what puts it in the
    /// Skipped record on the medium so a re-scan does not offer it again.
    #[test]
    fn the_loser_of_a_comparison_is_recorded_as_passed_over() {
        let mut entries = six();
        let groups = vec![vec![0, 1]];
        let (selected, passed) = run_pass(&mut entries, &groups, &[Verdict::Right]);
        assert_eq!(selected, HashSet::from([1]));
        assert_eq!(passed, vec![0]);
        assert!(!entries[1].passed, "the winner must not be marked as skipped");
    }

    #[test]
    fn undo_puts_a_photo_back_to_undecided() {
        let mut entries = six();
        let groups = vec![vec![0, 1, 2]];
        let mut t = Tournament::build(&groups, Vec::new(), |_| false);
        let mut selected = HashSet::new();
        let mut brushed: HashMap<usize, Vec<i64>> = HashMap::new();

        let settled = t.decide(Verdict::Right);
        settle_marks(&mut entries, &mut selected, &mut brushed, &[], &settled);
        assert!(entries[0].passed);

        let undone = t.undo();
        withdraw_marks(&mut entries, &mut selected, &mut brushed, &undone);
        assert!(!entries[0].passed, "an undone loss is not a skip");
        assert!(selected.is_empty());
        assert!(brushed.is_empty());
        assert_eq!(t.pair(), Some((0, 1)), "and the question comes back");
    }

    /// The brush is read when the photo is marked, exactly as it is for a
    /// mark made by hand — so a tag added half way through a pass tags the
    /// second half and not the first.
    #[test]
    fn a_kept_photo_carries_the_brush_it_was_kept_under() {
        let mut entries = six();
        let groups = vec![vec![0, 1], vec![2, 3]];
        let mut t = Tournament::build(&groups, Vec::new(), |_| false);
        let mut selected = HashSet::new();
        let mut brushed = HashMap::new();

        let first = t.decide(Verdict::Left);
        settle_marks(&mut entries, &mut selected, &mut brushed, &[], &first);

        let holiday = vec![Tag { id: 7, name: "Holiday".into(), color: "#aabbcc".into() }];
        let second = t.decide(Verdict::Left);
        settle_marks(&mut entries, &mut selected, &mut brushed, &holiday, &second);

        assert_eq!(brushed.get(&0), Some(&Vec::new()), "kept before the tag existed");
        assert_eq!(brushed.get(&2), Some(&vec![7]));
        // A loser carries no tags at all — there is nothing to tag.
        assert!(!brushed.contains_key(&1));
        assert!(!brushed.contains_key(&3));
    }

    #[test]
    fn the_note_tells_nothing_to_run_apart_from_nothing_left_to_run() {
        let nothing = Tournament::build(&[vec![0]], Vec::new(), |_| false);
        assert!(tournament_note(&nothing).contains("nothing to run"));

        let mut done = Tournament::build(&[vec![0, 1, 2]], Vec::new(), |_| false);
        done.decide(Verdict::Both);
        done.decide(Verdict::Left);
        assert_eq!(tournament_note(&done), "Tournament complete — 2 kept, 1 passed over.");
    }

    #[test]
    fn a_pane_falls_back_to_half_the_window_until_it_reports_its_own_size() {
        // The reported number wins whenever there is one.
        assert_eq!(pane_size((1600, 1000), 700, 880), (700, 880));
        // And a pane that has not reported still gets something drawable
        // rather than nothing at all.
        let (w, h) = pane_size((1600, 1000), 0, 0);
        assert_eq!(w, 800);
        assert!(h > 0 && h < 1000);
        // Including from a window too small to subtract the chrome from.
        assert_eq!(pane_size((0, 0), 0, 0), (1, 1));
    }

    // ── Status text ───────────────────────────────────────────────

    #[test]
    fn scan_status_text_pluralises() {
        let clean = ScanQuality::default();
        assert_eq!(scan_status_text(1, clean), "1 photo found");
        assert_eq!(scan_status_text(0, clean), "0 photos found");
        assert_eq!(scan_status_text(7, clean), "7 photos found");
    }

    #[test]
    fn the_status_line_names_what_went_wrong() {
        assert_eq!(
            scan_status_text(7, ScanQuality { recovered: 0, unreadable: 2 }),
            "7 photos found · 2 without a preview (see the log for paths)"
        );
        assert_eq!(
            scan_status_text(7, ScanQuality { recovered: 3, unreadable: 1 }),
            "7 photos found · 3 previewed from RAW · 1 without a preview (see the log for paths)"
        );
    }

    #[test]
    fn every_notable_photo_is_reported_with_its_reason() {
        let mut entries = vec![
            entry("/card/a.jpg", None),
            entry("/card/stalls.raf", None),
            entry("/card/corrupt.jpg", None),
        ];
        entries[1].readable = Readable::TimedOut;
        entries[2].readable = Readable::NoPreview;

        assert_eq!(
            report_scan_quality(&entries),
            ScanQuality { recovered: 0, unreadable: 2 }
        );
        assert_eq!(Readable::TimedOut.reason(), "timed out");
        assert_eq!(Readable::NoPreview.reason(), "could not be decoded");
        assert!(Readable::Yes.ok() && !Readable::TimedOut.ok());

        // A recovered photo has a picture, so it is not a failure — but it
        // is still reported, because its display file is corrupt and only
        // the user can decide what to do about that.
        assert!(Readable::FromCompanion.ok());
        entries[0].readable = Readable::FromCompanion;
        assert_eq!(
            report_scan_quality(&entries),
            ScanQuality { recovered: 1, unreadable: 2 }
        );
    }

    #[test]
    fn copy_status_text_reports_failures_when_there_are_any() {
        assert_eq!(copy_status_text(1, 0), "Copied 1 photo");
        assert_eq!(copy_status_text(3, 0), "Copied 3 photos");
        assert_eq!(copy_status_text(3, 2), "Copied 3, 2 failed");
    }

    // ── Copy selection ────────────────────────────────────────────

    #[test]
    fn copy_mode_from_index_maps_the_dropdown_rows() {
        assert_eq!(copy_mode_from_index(0), maple_import::CopyMode::DisplayOnly);
        assert_eq!(copy_mode_from_index(1), maple_import::CopyMode::All);
        assert_eq!(copy_mode_from_index(2), maple_import::CopyMode::RawOnly);
        // Anything unexpected copies everything rather than dropping files.
        assert_eq!(copy_mode_from_index(99), maple_import::CopyMode::All);
    }

    #[test]
    fn copy_sources_includes_companions_in_all_mode() {
        let mut e = entry("/src/DSCF0001.JPG", None);
        e.companions = vec![PathBuf::from("/src/DSCF0001.RAF")];
        let entries = vec![e, entry("/src/other.jpg", None)];

        let sources = copy_sources(&entries, &[0], maple_import::CopyMode::All);
        assert_eq!(sources.len(), 2);
        assert!(sources.contains(&PathBuf::from("/src/DSCF0001.JPG")));
        assert!(sources.contains(&PathBuf::from("/src/DSCF0001.RAF")));
    }

    #[test]
    fn copy_sources_skips_out_of_range_indices() {
        let entries = vec![entry("/src/a.jpg", None)];
        assert!(copy_sources(&entries, &[5], maple_import::CopyMode::All).is_empty());
    }

    // ── Model rows ────────────────────────────────────────────────

    #[test]
    fn make_item_flags_a_jpg_with_a_raw_companion_as_having_both() {
        let mut e = entry("/src/DSCF0001.JPG", None);
        e.companions = vec![PathBuf::from("/src/DSCF0001.RAF")];
        let entries = vec![e];

        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(item.has_jpg);
        assert!(item.has_raw);
        assert_eq!(item.filename, "DSCF0001.JPG");
        // The JPG/RAW badges come from the listing, not from having pixels:
        // they are known before the file has been touched.
        assert!(!item.loaded, "no preview has been decoded for it yet");
    }

    #[test]
    fn make_item_flags_a_lone_raw_as_raw_only() {
        let entries = vec![entry("/src/DSCF0002.RAF", None)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(!item.has_jpg);
        assert!(item.has_raw);
    }

    #[test]
    fn make_item_reports_an_unscanned_placeholder_as_not_loaded() {
        // Entries are pre-allocated with an empty path when the scan's count
        // arrives, before their thumbnails do.
        let entries = vec![entry("", None)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(!item.loaded);
    }

    #[test]
    fn make_item_carries_selection_and_session_shape() {
        let entries = vec![entry("/src/a.jpg", None), entry("/src/b.jpg", None)];
        let selected: HashSet<usize> = [1].into_iter().collect();
        let sessions = sess(&[(0, 2)]);

        let first = make_item(&entries, &selected, &sessions, 0);
        assert!(!first.is_selected);
        assert_eq!(first.stack_size, 2);
        assert_eq!(first.session_id, 0);
        assert!(first.session_start && !first.session_end);

        let second = make_item(&entries, &selected, &sessions, 1);
        assert!(second.is_selected);
        assert!(!second.session_start && second.session_end);
    }

    // ── Drawing a session with two clicks ────────────────────────

    /// A click says "the session starts here". Saying it about a photo that
    /// already starts one must leave it starting one — `toggle_boundary`
    /// would merge it backwards, which is the opposite instruction.
    #[test]
    fn clicking_an_existing_start_leaves_it_alone_where_a_key_would_toggle_it() {
        let mut sessions = sess(&[(0, 4), (4, 8)]);

        assert!(!ensure_boundary(&mut sessions, 4), "nothing to do, and it says so");
        assert_eq!(sessions, sess(&[(0, 4), (4, 8)]));

        // The key vocabulary, for contrast: the same call merges.
        let mut keyed = sess(&[(0, 4), (4, 8)]);
        toggle_boundary(&mut keyed, 4);
        assert_eq!(keyed, sess(&[(0, 8)]), "`[` toggles, a click does not");
    }

    /// Two clicks are the two boundaries `[` and `]` set — one photo apart,
    /// which is the identity the whole edit vocabulary rests on.
    #[test]
    fn two_clicks_carve_out_exactly_the_photos_between_them() {
        let mut sessions = sess(&[(0, 10)]);

        // Click on 3: the session starts here.
        assert!(ensure_boundary(&mut sessions, 3));
        // Click on 6: the session stops here — a boundary before 7.
        assert!(ensure_boundary(&mut sessions, 7));

        assert_eq!(sessions, sess(&[(0, 3), (3, 7), (7, 10)]));
        let carved = sessions.iter().find(|s| s.start == 3).unwrap();
        assert_eq!(carved.len(), 4, "photos 3,4,5,6 — both clicks included");
    }

    #[test]
    fn clicking_the_same_photo_twice_makes_a_session_of_one() {
        let mut sessions = sess(&[(0, 6)]);
        ensure_boundary(&mut sessions, 2);
        ensure_boundary(&mut sessions, 3);
        assert_eq!(sessions, sess(&[(0, 2), (2, 3), (3, 6)]));
    }

    #[test]
    fn a_click_outside_the_sequence_changes_nothing() {
        let mut sessions = sess(&[(0, 4)]);
        assert!(!ensure_boundary(&mut sessions, 0), "the sequence already begins there");
        assert!(!ensure_boundary(&mut sessions, 4), "and already ends there");
        assert!(!ensure_boundary(&mut sessions, 99));
        assert_eq!(sessions, sess(&[(0, 4)]));

        let mut empty: Vec<maple_import::Session> = Vec::new();
        assert!(!ensure_boundary(&mut empty, 1), "nothing scanned yet");
    }

    /// Sessions must still tile the sequence after any number of clicks —
    /// the `f` grid needs every photo to belong somewhere so a boundary can
    /// be dragged onto it.
    #[test]
    fn clicking_never_leaves_a_hole_or_an_overlap() {
        let mut sessions = sess(&[(0, 20)]);
        for at in [7usize, 3, 15, 8, 3, 19, 1, 15] {
            ensure_boundary(&mut sessions, at);
        }
        assert_eq!(sessions.first().unwrap().start, 0);
        assert_eq!(sessions.last().unwrap().end, 20);
        for pair in sessions.windows(2) {
            assert_eq!(pair[0].end, pair[1].start, "sessions must tile: {sessions:?}");
        }
    }

    #[test]
    fn a_session_of_one_opens_and_closes_on_the_same_tile_and_is_not_a_group() {
        let entries = vec![entry("/src/a.jpg", None)];
        let item = make_item(&entries, &HashSet::new(), &sess(&[(0, 1)]), 0);
        assert!(item.session_start && item.session_end);
        assert_eq!(item.stack_size, 0, "a session of one must not read as a group");
    }

    #[test]
    fn a_photo_in_no_session_reports_minus_one() {
        // What every tile looks like before the scan finishes, and on a
        // card where detection found nothing at all.
        let entries = vec![entry("/src/a.jpg", None)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert_eq!(item.session_id, -1);
        assert!(!item.session_start && !item.session_end);
    }

    // ── Triage verdicts ──────────────────────────────────────────

    /// An entry in a given triage state, for the badge tests.
    #[test]
    fn the_preview_window_survives_a_filter_that_shrank_the_strip() {
        // The bug: the strip reports rows 700..730 while 816 photos show,
        // then "Hide old images" leaves 60. Re-using that window makes
        // `request_previews` bail on `first > last` and every visible tile
        // stays blank until the button is clicked a second time.
        let (first, last, focus) = preview_window_for(60, 30, 12).unwrap();
        assert!(first <= last, "an unusable window");
        assert!(last < 60, "asked for a row the strip does not have");
        assert!((first..=last).contains(&focus), "focus outside the window");
        assert!((first..=last).contains(&12), "the current photo is not in it");
    }

    #[test]
    fn the_preview_window_clamps_at_both_ends() {
        // Top: nothing to centre behind, so it starts at 0.
        assert_eq!(preview_window_for(100, 20, 0).unwrap().0, 0);
        // Bottom: already at the end, so it simply stops there.
        let (_, last, _) = preview_window_for(100, 20, 99).unwrap();
        assert_eq!(last, 99);
        // A span narrower than the initial one still asks for a usable
        // screenful — a stale zero-width window must not stay zero-width.
        let (first, last, _) = preview_window_for(100, 0, 50).unwrap();
        assert!(last - first >= INITIAL_WINDOW);
    }

    #[test]
    fn an_empty_strip_has_no_preview_window() {
        assert!(preview_window_for(0, 30, 0).is_none());
    }

    fn triaged(path: &str, passed: bool, imported: bool) -> Entry {
        let mut e = entry(path, None);
        e.passed = passed;
        e.is_imported = imported;
        e.decided_before = imported;
        e
    }

    #[test]
    fn a_photo_not_yet_reached_carries_no_verdict() {
        let entries = vec![triaged("/src/a.jpg", false, false)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(!item.is_selected);
        assert!(!item.is_skipped, "a photo nobody has looked at is not a reject");
        assert!(!item.is_imported);
    }

    #[test]
    fn moving_past_a_photo_is_what_paints_the_red_cross() {
        let entries = vec![triaged("/src/a.jpg", true, false)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(item.is_skipped);
    }

    #[test]
    fn marking_a_photo_outranks_having_walked_past_it() {
        // The user stepped past it, came back and marked it: one verdict,
        // and it is the mark.
        let entries = vec![triaged("/src/a.jpg", true, false)];
        let item = make_item(&entries, &HashSet::from([0]), &[], 0);
        assert!(item.is_selected);
        assert!(!item.is_skipped);
    }

    #[test]
    fn a_photo_already_in_the_library_is_never_a_reject() {
        let entries = vec![triaged("/src/a.jpg", true, true)];
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(item.is_imported);
        assert!(!item.is_skipped, "the ✓ scrim already says more than a ✗ would");
        assert!(item.is_old, "an imported photo was decided in an earlier session");
    }

    // ── Hiding what an earlier session decided ───────────────────

    #[test]
    fn hiding_old_photos_leaves_entry_indices_as_the_identity() {
        let entries = vec![
            triaged("/src/a.jpg", false, true),   // old
            triaged("/src/b.jpg", false, false),  // new
            triaged("/src/c.jpg", false, true),   // old
            triaged("/src/d.jpg", false, false),  // new
        ];
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        assert_eq!(visible.rows, vec![1, 3]);
        assert_eq!(visible.row(0), None);
        assert_eq!(visible.row(1), Some(0));
        assert_eq!(visible.row(3), Some(1));

        // The rows shrank to two, but each item still names its own scan
        // index — a click on the second row must reach entry 3, not entry 1.
        let items = build_items(&entries, &HashSet::new(), &[], &visible);
        assert_eq!(items.len(), 2);
        assert_eq!(items[0].index, 1);
        assert_eq!(items[1].index, 3);
    }

    #[test]
    fn with_the_filter_off_every_row_is_its_own_entry() {
        let entries = vec![
            triaged("/src/a.jpg", false, true),
            triaged("/src/b.jpg", false, false),
        ];
        let visible = all_visible(&entries);
        assert_eq!(visible.rows, vec![0, 1]);
        assert_eq!(visible.row(0), Some(0));
        assert_eq!(visible.row(1), Some(1));
    }

    // ── Import tags ──────────────────────────────────────────────

    fn tag(id: i64, name: &str) -> Tag {
        Tag { id, name: name.to_string(), color: tag_color_for(name).to_string() }
    }

    #[test]
    fn a_tag_colour_follows_from_its_name() {
        // Two devices that both create "Holiday" before they ever sync have
        // to agree on what it looks like, so the colour cannot come from a
        // counter or the RNG.
        assert_eq!(tag_color_for("Holiday"), tag_color_for("Holiday"));
        assert!(TAG_PALETTE.contains(&tag_color_for("Holiday")));
        assert!(TAG_PALETTE.contains(&tag_color_for("")));
        // Not a constant dressed up as a function.
        let distinct: std::collections::HashSet<_> =
            ["a", "b", "c", "d", "e", "f", "g", "h"].iter().map(|n| tag_color_for(n)).collect();
        assert!(distinct.len() > 1, "every name got the same colour");
    }

    #[test]
    fn marking_a_photo_stamps_the_brush_as_it_stands() {
        let mut brushed = HashMap::new();
        let brush = vec![tag(7, "Holiday"), tag(9, "Family")];

        record_brush(&mut brushed, 3, true, &brush);
        assert_eq!(brushed.get(&3), Some(&vec![7, 9]));

        // Marking with nothing on the brush is a photo with no tags, not an
        // absent record — `unwrap_or_default` at copy time reads both the
        // same way, but the distinction is what makes unmarking meaningful.
        record_brush(&mut brushed, 4, true, &[]);
        assert_eq!(brushed.get(&4), Some(&vec![]));
    }

    #[test]
    fn changing_the_brush_mid_pass_tags_the_two_halves_differently() {
        // The whole point of reading the brush at mark time: one triage pass
        // can produce two differently-tagged sets without copying twice.
        let mut brushed = HashMap::new();
        let holiday = vec![tag(7, "Holiday")];
        record_brush(&mut brushed, 0, true, &holiday);
        record_brush(&mut brushed, 1, true, &holiday);

        let portraits = vec![tag(11, "Portraits")];
        record_brush(&mut brushed, 2, true, &portraits);

        assert_eq!(brushed.get(&0), Some(&vec![7]));
        assert_eq!(brushed.get(&1), Some(&vec![7]));
        assert_eq!(brushed.get(&2), Some(&vec![11]));
    }

    #[test]
    fn unmarking_a_photo_drops_the_tags_it_was_marked_with() {
        let mut brushed = HashMap::new();
        record_brush(&mut brushed, 5, true, &[tag(7, "Holiday")]);
        record_brush(&mut brushed, 5, false, &[tag(7, "Holiday")]);
        assert_eq!(brushed.get(&5), None);

        // Re-marked under a different brush it takes the new tags, not the
        // ones it happened to carry the first time round.
        record_brush(&mut brushed, 5, true, &[tag(11, "Portraits")]);
        assert_eq!(brushed.get(&5), Some(&vec![11]));
    }

    #[test]
    fn clearing_the_brush_does_not_untag_what_is_already_marked() {
        // `c` means "stop tagging", not "untag" — photos already marked keep
        // what they were marked with.
        let mut brushed = HashMap::new();
        let mut brush = vec![tag(7, "Holiday")];
        record_brush(&mut brushed, 0, true, &brush);

        brush.clear();
        record_brush(&mut brushed, 1, true, &brush);

        assert_eq!(brushed.get(&0), Some(&vec![7]));
        assert_eq!(brushed.get(&1), Some(&vec![]));
    }

    #[test]
    fn the_picker_ticks_exactly_what_is_on_the_brush() {
        let holiday = tag(7, "Holiday");
        assert!(to_ui_tag(&holiday, true).chosen);
        assert!(!to_ui_tag(&holiday, false).chosen);
        assert_eq!(to_ui_tag(&holiday, true).name, "Holiday");
        assert_eq!(to_ui_tag(&holiday, true).id, 7);
    }

    // ── The row the strip scrolls to ─────────────────────────────

    #[test]
    fn the_strip_is_told_the_row_not_the_scan_index() {
        // Hiding the two old photos renumbers the rows: entry 3 is now the
        // *second* tile. Telling the strip "3" would scroll it a screenful
        // past the photo the user is actually looking at.
        let entries = vec![
            triaged("/src/a.jpg", false, true),   // old
            triaged("/src/b.jpg", false, false),  // new
            triaged("/src/c.jpg", false, true),   // old
            triaged("/src/d.jpg", false, false),  // new
        ];
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        assert_eq!(strip_row(&visible, 1), 0);
        assert_eq!(strip_row(&visible, 3), 1);

        // Same photo, filter off: same identity, different row.
        let visible = all_visible(&entries);
        assert_eq!(strip_row(&visible, 3), 3);
    }

    #[test]
    fn a_photo_the_filter_hides_has_no_row_to_park() {
        let entries = vec![
            triaged("/src/a.jpg", false, true),
            triaged("/src/b.jpg", false, false),
        ];
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        // -1, not 0: the strip must leave the scroll alone rather than jump
        // to the top of the card for a photo it is not showing.
        assert_eq!(strip_row(&visible, 0), -1);
        // Out of range reads the same way.
        assert_eq!(strip_row(&visible, 99), -1);
    }

    #[test]
    fn stepping_forward_moves_the_strip_one_row_at_a_time() {
        // The strip parks the current photo one tile down, so consecutive
        // arrow presses have to yield consecutive rows — that is what makes
        // the strip walk instead of jumping a screenful per press.
        let entries: Vec<Entry> = (0..6)
            .map(|i| triaged(&format!("/src/{i}.jpg"), false, i % 2 == 0))
            .collect();
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        let mut cur = 1;
        let mut rows = vec![strip_row(&visible, cur)];
        for _ in 0..2 {
            cur = nav_visible_target(&[], &visible, cur, entries.len(), 1);
            rows.push(strip_row(&visible, cur));
        }
        assert_eq!(rows, vec![0, 1, 2]);
    }

    #[test]
    fn navigation_steps_over_hidden_photos() {
        // b and c were decided earlier; from a, "next" must land on d.
        let entries = vec![
            triaged("/src/a.jpg", false, false),
            triaged("/src/b.jpg", false, true),
            triaged("/src/c.jpg", false, true),
            triaged("/src/d.jpg", false, false),
        ];
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        assert_eq!(nav_visible_target(&[], &visible, 0, 4, 1), 3);
        assert_eq!(nav_visible_target(&[], &visible, 3, 4, -1), 0);
    }

    #[test]
    fn navigation_stays_put_when_nothing_visible_lies_ahead() {
        let entries = vec![
            triaged("/src/a.jpg", false, false),
            triaged("/src/b.jpg", false, true),
            triaged("/src/c.jpg", false, true),
        ];
        let mut visible = Visible::default();
        visible.rebuild(&entries, true);

        // Everything past `a` is hidden — the arrow key must not jump onto
        // a photo the strip is not showing.
        assert_eq!(nav_visible_target(&[], &visible, 0, 3, 1), 0);
    }

    #[test]
    fn navigation_is_unchanged_when_nothing_is_hidden() {
        let entries: Vec<Entry> =
            (0..4).map(|i| triaged(&format!("/src/{i}.jpg"), false, false)).collect();
        let visible = all_visible(&entries);
        for cur in 0..4usize {
            assert_eq!(
                nav_visible_target(&[], &visible, cur, 4, 1),
                nav_target(&[], cur, 4, 1)
            );
            assert_eq!(
                nav_visible_target(&[], &visible, cur, 4, -1),
                nav_target(&[], cur, 4, -1)
            );
        }
    }

    // ── Surviving a photo that will not decode ───────────────────

    /// A real, decodable 8×8 PNG.
    fn write_png(path: &std::path::Path) {
        let img = image::RgbImage::from_pixel(8, 8, image::Rgb([120, 30, 200]));
        image::DynamicImage::ImageRgb8(img).save(path).unwrap();
    }

    /// A named pipe with no writer. Opening it for reading blocks forever,
    /// which is the cheapest faithful stand-in for the card that started
    /// all this — a file the OS never returns from.
    #[cfg(unix)]
    fn make_stalling_file(path: &std::path::Path) {
        let ok = std::process::Command::new("mkfifo")
            .arg(path)
            .status()
            .expect("mkfifo");
        assert!(ok.success());
    }

    #[test]
    fn a_readable_photo_comes_back_with_bytes_and_a_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        write_png(&path);

        let out = read_with_budget(&path, Duration::from_secs(30));
        assert!(out.readable.ok());
        assert!(out.bytes.is_some());
        // One read serves both: the hash of the bytes handed to the decoder
        // is the same identifier the library and `SeenSet` use.
        assert_eq!(
            out.hash.unwrap(),
            maple_import::content_hash(&path).unwrap(),
            "the scan's hash must match the file's content hash"
        );
    }

    #[test]
    fn a_file_that_cannot_be_decoded_still_yields_its_hash() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("truncated.jpg");
        std::fs::write(&path, b"not really a jpeg").unwrap();

        // The *read* succeeds — the bytes are there. Whether they decode is
        // a separate question, asked later and only for photos on screen
        // (see `import_previews`).
        let out = read_with_budget(&path, Duration::from_secs(30));
        assert!(out.hash.is_some());
        assert!(out.bytes.is_some());
        assert!(out.readable.ok());
    }

    #[cfg(unix)]
    #[test]
    fn a_file_that_never_returns_is_abandoned_rather_than_waited_on() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("stalls.jpg");
        make_stalling_file(&path);

        let started = std::time::Instant::now();
        let out = read_with_budget(&path, Duration::from_millis(150));
        let waited = started.elapsed();

        assert_eq!(out.readable, Readable::TimedOut);
        assert!(out.hash.is_none() && out.bytes.is_none());
        assert!(
            waited < Duration::from_secs(5),
            "waited {waited:?} on a file that never comes back"
        );
        // The reader thread is still blocked in `open` and always will be.
        // Outliving it is the whole mechanism; there is nothing to join.
    }

    /// The bug this whole rewrite exists for: one file the card never
    /// returns from must cost its budget once and then be left behind,
    /// rather than stopping the scan.
    #[cfg(unix)]
    #[test]
    fn one_stalled_photo_does_not_stop_the_scan() {
        let dir = tempfile::tempdir().unwrap();
        let mut paths = Vec::new();
        for i in 0..5 {
            let p = dir.path().join(format!("good{i}.png"));
            write_png(&p);
            paths.push(p);
        }
        let stalled = dir.path().join("stalls.jpg");
        make_stalling_file(&stalled);
        // In the middle: before the fix, nothing after it ever arrived.
        paths.insert(2, stalled);

        let prior = PriorDecisions {
            imported: maple_state::SeenSet::new(),
            skipped: maple_state::SeenSet::new(),
        };
        let (tx, rx) = mpsc::channel();
        let budget = Duration::from_millis(150);

        let cache = Mutex::new(maple_import::PreviewCache::detached());
        let started = std::time::Instant::now();
        for (index, path) in paths.iter().enumerate() {
            let group = maple_import::ImageGroup {
                display: maple_import::ImageFile { path: path.clone(), size: 0 },
                companions: vec![],
            };
            let job = read_one(index, &group, budget, &prior, dir.path(), &cache);
            decode_one(job, None, None, None, &tx);
        }
        drop(tx);

        let thumbs: Vec<ScanThumb> = rx
            .into_iter()
            .filter_map(|m| match m {
                ScanMsg::Thumb(t) => Some(t),
                _ => None,
            })
            .collect();

        assert_eq!(thumbs.len(), 6, "every photo must be reported, stalled or not");
        let stalled: Vec<usize> = thumbs
            .iter()
            .filter(|t| t.readable == Readable::TimedOut)
            .map(|t| t.index)
            .collect();
        assert_eq!(stalled, vec![2], "only the stalled file lacks a preview");
        for t in thumbs.iter().filter(|t| t.readable.ok()) {
            assert!(t.content_hash != [0u8; 32], "a readable photo must be hashed");
        }
        // Five good photos plus one 150 ms write-off: anything near a
        // multiple of the budget would mean the stall was contagious.
        assert!(
            started.elapsed() < Duration::from_secs(5),
            "the scan took {:?} — a stalled photo is still blocking",
            started.elapsed()
        );
    }

    /// Run the real scan worker over `dir` and collect what it reported.
    fn scan(dir: &std::path::Path) -> (Vec<ScanThumb>, Vec<(usize, maple_import::Signature)>) {
        let (tx, rx) = mpsc::channel();
        spawn_scan_worker(
            dir.to_path_buf(),
            maple_state::StackSettings { enabled: false, ..Default::default() },
            maple_state::SessionSettings::default(),
            maple_state::ImportSettings::default(),
            Arc::new(PriorDecisions {
                imported: maple_state::SeenSet::new(),
                skipped: maple_state::SeenSet::new(),
            }),
            tx,
        );
        let (mut thumbs, mut signatures) = (Vec::new(), Vec::new());
        loop {
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(ScanMsg::Thumb(t)) => thumbs.push(t),
                Ok(ScanMsg::Signature { index, signature }) => signatures.push((index, signature)),
                Ok(ScanMsg::Done) => break,
                Ok(ScanMsg::Error(e)) => panic!("scan error: {e}"),
                Ok(_) => {}
                Err(e) => panic!("scan stalled: {e}"),
            }
        }
        thumbs.sort_by_key(|t| t.index);
        signatures.sort_by_key(|(i, _)| *i);
        (thumbs, signatures)
    }

    /// Overwrite a file's contents while keeping the size and mtime the
    /// cache keys on — so a scan that still gets the *original* preview
    /// back has provably not opened it.
    fn scribble_over(path: &std::path::Path) {
        let meta = std::fs::metadata(path).unwrap();
        let (len, mtime) = (meta.len(), meta.modified().unwrap());
        std::fs::write(path, vec![0u8; len as usize]).unwrap();
        let f = std::fs::OpenOptions::new().write(true).open(path).unwrap();
        f.set_modified(mtime).unwrap();
        f.sync_all().unwrap();
        let after = std::fs::metadata(path).unwrap();
        assert_eq!(after.len(), len);
        assert_eq!(after.modified().unwrap(), mtime);
    }

    /// The claim the preview cache is built on: **a second scan of an
    /// unchanged card does not open the files at all.**
    ///
    /// Proved by destroying the contents between the two scans while
    /// keeping the size and mtime intact. A scan that read anything would
    /// come back with no preview and a different hash; one that used the
    /// medium's own record comes back with exactly what it found the first
    /// time.
    #[test]
    fn a_rescan_of_an_unchanged_card_never_opens_the_files() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..6 {
            write_png(&dir.path().join(format!("p{i}.png")));
        }

        let (first, _) = scan(dir.path());
        assert_eq!(first.len(), 6);
        assert!(first.iter().all(|t| t.preview.is_some()), "the scan must make previews");
        assert!(
            dir.path().join(maple_import::PREVIEW_CACHE_FILE).exists(),
            "and write them back to the medium"
        );

        for i in 0..6 {
            scribble_over(&dir.path().join(format!("p{i}.png")));
        }

        let (second, _) = scan(dir.path());
        assert_eq!(second.len(), 6);
        for (a, b) in first.iter().zip(&second) {
            assert_eq!(a.content_hash, b.content_hash, "the hash comes from the cache too");
            assert_eq!(a.preview, b.preview, "byte-identical preview, not a re-encode");
            assert!(b.readable.ok());
        }
    }

    /// The other half of that key: a file that actually changed must be
    /// read again. Serving the stale entry would also serve the stale
    /// content hash, which is what decides "already imported".
    #[test]
    fn a_file_that_changed_is_read_again() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("a.png");
        write_png(&path);
        let (first, _) = scan(dir.path());

        // A different image, so both the size and the mtime move.
        let img = image::RgbImage::from_pixel(16, 16, image::Rgb([10, 200, 40]));
        image::DynamicImage::ImageRgb8(img).save(&path).unwrap();

        let (second, _) = scan(dir.path());
        assert_ne!(first[0].content_hash, second[0].content_hash);
        assert_ne!(first[0].preview, second[0].preview);
        assert_eq!(
            second[0].content_hash,
            maple_import::content_hash(&path).unwrap(),
            "the fresh read must produce the file's real hash"
        );
    }

    /// The invariant [`maple_import::preview`] exists for: every check runs
    /// on the frame the *kept* preview decodes to, so recomputing one from
    /// what the scan stored reproduces the scan's own answer exactly.
    ///
    /// If anything ever computes on a pristine decode again, this fails —
    /// which is the point, because that pixel buffer is not kept anywhere
    /// and nothing downstream could reproduce a result from it.
    #[test]
    fn every_check_is_computed_from_the_preview_that_is_kept() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..4 {
            let img = image::RgbImage::from_fn(64, 48, |x, y| {
                image::Rgb([(x * 4) as u8, (y * 5) as u8, (i * 60) as u8])
            });
            image::DynamicImage::ImageRgb8(img)
                .save(dir.path().join(format!("p{i}.png")))
                .unwrap();
        }

        let (thumbs, signatures) = scan(dir.path());
        assert_eq!(signatures.len(), 4, "every photo must reach the session engine");

        let settings = maple_state::SessionSettings::default();
        let mut engine =
            maple_import::session::engine_from_spec(&settings.engine).expect("default engine");

        for (thumb, (index, signature)) in thumbs.iter().zip(&signatures) {
            assert_eq!(thumb.index, *index);
            let webp = thumb.preview.as_ref().expect("a preview was kept");
            let frame = maple_import::preview::decode(webp).expect("it decodes");

            assert_eq!(
                thumb.sharpness.unwrap(),
                maple_import::laplacian_variance(frame.as_raw(), frame.width(), frame.height()),
                "sharpness must be reproducible from the kept preview"
            );
            let recomputed = engine
                .signature(&maple_import::Frame::new(&frame, thumb.taken))
                .expect("signature");
            assert_eq!(
                &recomputed, signature,
                "the signature must be reproducible from the kept preview"
            );
        }
    }

    /// Drive the real `spawn_scan_worker` over a real folder and assert the
    /// message stream the UI depends on: the listing, one row per photo,
    /// then `Done`. A deadlock between reader, decoders and embedder shows
    /// up here as a timeout rather than as an empty grid in the app.
    ///
    /// Pixels are not part of this stream any more — previews are decoded
    /// on demand by `import_previews`.
    #[test]
    fn a_scan_reports_a_count_then_every_photo_then_done() {
        let dir = tempfile::tempdir().unwrap();
        for i in 0..25 {
            write_png(&dir.path().join(format!("p{i:03}.png")));
        }

        let (tx, rx) = mpsc::channel();
        spawn_scan_worker(
            dir.path().to_path_buf(),
            maple_state::StackSettings { enabled: false, ..Default::default() },
            maple_state::SessionSettings::default(),
            maple_state::ImportSettings::default(),
            Arc::new(PriorDecisions {
                imported: maple_state::SeenSet::new(),
                skipped: maple_state::SeenSet::new(),
            }),
            tx,
        );

        let mut count = None;
        let mut seen = std::collections::HashSet::new();
        let mut signed = std::collections::HashSet::new();
        let done;
        loop {
            // Finite: a deadlock must fail the test, not hang it.
            match rx.recv_timeout(Duration::from_secs(30)) {
                Ok(ScanMsg::Found(photos)) => count = Some(photos.len()),
                Ok(ScanMsg::Thumb(t)) => {
                    assert!(t.readable.ok(), "photo {} failed to read", t.index);
                    assert!(seen.insert(t.index), "photo {} arrived twice", t.index);
                }
                Ok(ScanMsg::Embedding { .. }) => {}
                Ok(ScanMsg::Signature { index, .. }) => {
                    signed.insert(index);
                }
                Ok(ScanMsg::Done) => {
                    done = true;
                    break;
                }
                Ok(ScanMsg::Error(e)) => panic!("scan error: {e}"),
                Err(e) => panic!("scan stalled after {} thumbs: {e}", seen.len()),
            }
        }

        assert_eq!(count, Some(25), "the listing must arrive before anything else");
        assert_eq!(seen.len(), 25, "every photo must reach the UI exactly once");
        assert!(done, "the scan must finish");
        // Session detection is on by default, so every photo that decoded
        // must also have been described. A signature arriving for only some
        // of them is the failure that would silently halve the grouping.
        assert_eq!(signed.len(), 25, "every photo must reach the session engine");
    }

    /// Session-detection probe against a real folder, run by hand:
    /// `MAPLE_PROBE_DIR=/path cargo test -p maple-ui session_probe -- --ignored --nocapture`
    ///
    /// Drives the real scan worker and the real `detect_sessions`, so it
    /// answers the question the lab cannot: does the *importer* group these
    /// photos, given that it builds its signatures and its segmentation
    /// engine as two separate instances?
    #[test]
    #[ignore = "needs MAPLE_PROBE_DIR; run by hand"]
    fn session_probe() {
        let Ok(dir) = std::env::var("MAPLE_PROBE_DIR") else { return };
        let dir = std::path::PathBuf::from(dir);
        let settings = maple_state::SessionSettings::default();
        println!("engine = {:?}", settings.engine);

        let (tx, rx) = mpsc::channel();
        spawn_scan_worker(
            dir.clone(),
            maple_state::StackSettings { enabled: false, ..Default::default() },
            settings.clone(),
            maple_state::ImportSettings::default(),
            Arc::new(PriorDecisions {
                imported: maple_state::SeenSet::new(),
                skipped: maple_state::SeenSet::new(),
            }),
            tx,
        );

        let mut entries: Vec<Entry> = Vec::new();
        let mut listed = 0usize;
        loop {
            match rx.recv_timeout(Duration::from_secs(600)) {
                Ok(ScanMsg::Found(photos)) => {
                    listed = photos.len();
                    entries = photos
                        .iter()
                        .map(|p| entry(p.display.to_str().unwrap(), None))
                        .collect();
                }
                Ok(ScanMsg::Thumb(t)) => {
                    if let Some(e) = entries.get_mut(t.index) {
                        e.taken = t.taken;
                        e.sharpness = t.sharpness;
                        e.readable = t.readable;
                    }
                }
                Ok(ScanMsg::Signature { index, signature }) => {
                    if let Some(e) = entries.get_mut(index) {
                        e.signature = Some(signature);
                    }
                }
                Ok(ScanMsg::Done) => break,
                Ok(ScanMsg::Error(e)) => panic!("scan error: {e}"),
                Ok(_) => {}
                Err(e) => panic!("scan stalled: {e}"),
            }
        }

        let signed = entries.iter().filter(|e| e.signature.is_some()).count();
        let timed = entries.iter().filter(|e| e.taken.is_some()).count();
        println!("listed {listed}, signed {signed}, with a capture time {timed}");

        let sessions = detect_sessions(&entries, &settings);
        let groups = groups_from_sessions(&sessions);
        println!("sessions {}, groups {}", sessions.len(), groups.len());
        for s in sessions.iter().take(20) {
            println!("  [{:>4}..{:<4}] {} photos", s.start, s.end - 1, s.end - s.start);
        }
    }

    /// Timing probe against camera-sized files, run by hand:
    /// `cargo test -p maple-ui scan_throughput -- --ignored --nocapture`.
    ///
    /// The synthetic scan test above uses 8×8 pixels and proves only that
    /// the pipeline is wired up. This one asks the question that actually
    /// went wrong on a real card: **how long until the first tile appears**.
    /// Reading twelve files at once made that number worse, not better.
    #[test]
    #[ignore = "writes ~50 full-size JPEGs; run by hand"]
    fn scan_throughput_on_camera_sized_files() {
        const N: usize = 48;
        let dir = tempfile::tempdir().unwrap();
        eprintln!("writing {N} 6000×4000 JPEGs…");
        for i in 0..N {
            let img = image::RgbImage::from_fn(6000, 4000, |x, y| {
                image::Rgb([(x % 251) as u8, (y % 241) as u8, ((x ^ y) % 239) as u8])
            });
            image::DynamicImage::ImageRgb8(img)
                .save(dir.path().join(format!("DSCF{i:04}.jpg")))
                .unwrap();
        }

        let (tx, rx) = mpsc::channel();
        let started = std::time::Instant::now();
        spawn_scan_worker(
            dir.path().to_path_buf(),
            maple_state::StackSettings { enabled: false, ..Default::default() },
            maple_state::SessionSettings::default(),
            maple_state::ImportSettings::default(),
            Arc::new(PriorDecisions {
                imported: maple_state::SeenSet::new(),
                skipped: maple_state::SeenSet::new(),
            }),
            tx,
        );

        let mut first: Option<Duration> = None;
        let mut thumbs = 0usize;
        loop {
            match rx.recv_timeout(Duration::from_secs(120)) {
                Ok(ScanMsg::Thumb(_)) => {
                    thumbs += 1;
                    if first.is_none() {
                        first = Some(started.elapsed());
                        eprintln!("first tile after {:?}", first.unwrap());
                    }
                }
                Ok(ScanMsg::Done) => break,
                Ok(_) => {}
                Err(e) => panic!("stalled after {thumbs} thumbs: {e}"),
            }
        }
        eprintln!(
            "{thumbs} photos in {:?} (first tile {:?}, {:.1}/s)",
            started.elapsed(),
            first.unwrap(),
            thumbs as f64 / started.elapsed().as_secs_f64()
        );
        assert_eq!(thumbs, N);
    }

    #[test]
    fn an_unreadable_photo_still_reaches_its_tile() {
        let mut entries = vec![entry("/src/a.jpg", None)];
        entries[0].readable = Readable::TimedOut;
        let item = make_item(&entries, &HashSet::new(), &[], 0);
        assert!(item.is_unreadable);
        // `loaded` now means "has a decoded preview", so an unreadable
        // photo is not loaded — but the tile still knows its name, which is
        // what stops it looking like a row still waiting its turn.
        assert!(!item.loaded);
        assert_eq!(item.filename, "a.jpg");
    }

    #[test]
    fn build_items_covers_every_entry() {
        let entries = vec![entry("/src/a.jpg", None), entry("/src/b.jpg", None)];
        let items = build_items(&entries, &HashSet::new(), &[], &all_visible(&entries));
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].index, 1);
        // Solo entries carry no stack badge.
        assert!(items.iter().all(|i| i.stack_size == 0));
    }
}
