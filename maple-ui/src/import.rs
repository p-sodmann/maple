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
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use image::{DynamicImage, RgbImage};
use slint::{ComponentHandle, Model, ModelRc, SharedString, Timer, TimerMode, VecModel};

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

thread_local! {
    static IMPORT: RefCell<Option<Import>> = const { RefCell::new(None) };
}

// ── Background worker messages ────────────────────────────────────

enum ScanMsg {
    Count(usize),
    Thumb(ScanThumb),
    Done,
    Error(String),
}

/// One scanned photo, handed from the scan worker to the UI thread.
struct ScanThumb {
    index: usize,
    path: PathBuf,
    companions: Vec<PathBuf>,
    rgb: Vec<u8>,
    width: u32,
    height: u32,
    content_hash: [u8; 32],
    imported: bool,
    /// DINOv2 embedding, if stack detection is enabled and inference
    /// (or an SD-card cache hit) succeeded.
    embedding: Option<Vec<f32>>,
    /// Variance-of-Laplacian sharpness score, computed whenever an
    /// embedding is (used to auto-pick the "best" shot in a burst).
    sharpness: Option<f32>,
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
    /// Decoded thumbnail (None until the scan worker sends it).
    thumb: Option<slint::Image>,
    /// DINOv2 embedding computed during the scan (`None` if stack detection
    /// is disabled or inference failed for this photo).
    embedding: Option<Vec<f32>>,
    /// Variance-of-Laplacian sharpness score, used to auto-pick the "best"
    /// shot within a detected burst group.
    sharpness: Option<f32>,
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
    /// Detected burst groups from the last scan — each a sorted list of
    /// flat `entries` indices. Populated once, when the scan finishes.
    groups: Rc<RefCell<Vec<Vec<usize>>>>,
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
        model,
        preview_shown_idx: Rc::new(Cell::new(None)),
        scanned_count: Rc::new(Cell::new(0)),
        source: Rc::new(RefCell::new(PathBuf::new())),
        dest,
        groups: Rc::new(RefCell::new(Vec::new())),
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

    Ok(Import { window, ctx })
}

// ── Close / pickers ───────────────────────────────────────────────

/// Wire the window chrome: closing, the two folder pickers, and the
/// file-naming template editor.
fn wire_chrome(window: &ImportWindow, ctx: &ImportCtx) {
    // ── Close ─────────────────────────────────────────────────────
    window.on_close_requested({
        let w = ctx.window.clone();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
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

            let imported_set = Arc::new(Mutex::new(
                maple_state::SeenSet::load_imported(&library_dir),
            ));

            spawn_scan_worker(src, stack_settings, imported_set.clone(), tx.clone());

            // Timer to drain scan results.
            let ctx2 = ctx.clone();
            let timer = Timer::default();
            timer.start(
                TimerMode::Repeated,
                Duration::from_millis(30),
                move || {
                    let Some(w) = ctx2.window.upgrade() else { return };

                    for _ in 0..10 {
                        match rx.try_recv() {
                            Ok(ScanMsg::Count(n)) => apply_scan_count(&w, &ctx2, n),
                            Ok(ScanMsg::Thumb(thumb)) => apply_scan_thumb(&w, &ctx2, thumb),
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
                        }
                    }
                },
            );
            *ctx.scan_timer.borrow_mut() = Some(timer);
        }
    });
}

/// Scan `src` on a background thread, streaming one [`ScanThumb`] per photo.
///
/// Burst detection during the scan reuses the [stacks] settings — same
/// toggle/threshold/model as post-import library stacking. If the embedder
/// fails to load (e.g. no network for a first-time model fetch), log once and
/// continue the scan without embeddings/sharpness — this is enrichment, never
/// a hard requirement to finish scanning.
fn spawn_scan_worker(
    src: PathBuf,
    stack_settings: maple_state::StackSettings,
    imported_set: Arc<Mutex<maple_state::SeenSet>>,
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
        let _ = tx.send(ScanMsg::Count(total));

        let algorithm_key = stack_settings.algorithm_key();
        let cache_path = src.join(EMBED_CACHE_FILE);
        let mut embedder = if stack_settings.enabled {
            match maple_db::load_onnx_embedder(&stack_settings) {
                Ok(e) => Some(e),
                Err(err) => {
                    tracing::warn!(
                        "Import scan: failed to load image embedder, skipping burst detection: {err}"
                    );
                    None
                }
            }
        } else {
            None
        };
        let mut embed_cache = stack_settings
            .enabled
            .then(|| maple_import::EmbeddingCache::load_from(&cache_path, &algorithm_key));
        let mut unflushed = 0usize;

        for (idx, group) in scanned_groups.into_iter().enumerate() {
            let display_path = group.display.path.clone();
            let companions: Vec<PathBuf> =
                group.companions.iter().map(|c| c.path.clone()).collect();

            let (hash, imported) = match maple_import::content_hash(&display_path) {
                Ok(h) => {
                    let imp = imported_set.lock().map(|s| s.contains(&h)).unwrap_or(false);
                    (h, imp)
                }
                Err(_) => ([0u8; 32], false),
            };

            let (rgb, width, height) =
                thumbnail::render_to_rgb(&display_path, 256).unwrap_or_default();

            let (sharpness, embedding) = if stack_settings.enabled
                && !rgb.is_empty()
                && width > 0
                && height > 0
            {
                let sharp = maple_import::laplacian_variance(&rgb, width, height);
                let cached = embed_cache.as_ref().and_then(|c| c.get(&hash)).map(|s| s.to_vec());
                let embedding = match cached {
                    Some(e) => Some(e),
                    None => embedder.as_mut().and_then(|embedder| {
                        let img = RgbImage::from_raw(width, height, rgb.clone())?;
                        match embedder.embed(&DynamicImage::ImageRgb8(img)) {
                            Ok(v) => {
                                if let Some(cache) = embed_cache.as_mut() {
                                    cache.insert(hash, v.clone());
                                    unflushed += 1;
                                }
                                Some(v)
                            }
                            Err(err) => {
                                tracing::warn!(
                                    "Import scan: embedding failed for {}: {err}",
                                    display_path.display()
                                );
                                None
                            }
                        }
                    }),
                };
                (Some(sharp), embedding)
            } else {
                (None, None)
            };

            if unflushed >= EMBED_CACHE_FLUSH_EVERY {
                if let Some(cache) = embed_cache.as_ref() {
                    if let Err(err) = cache.save_to(&cache_path) {
                        tracing::warn!("Import scan: failed to write embedding cache: {err}");
                    }
                }
                unflushed = 0;
            }

            let _ = tx.send(ScanMsg::Thumb(ScanThumb {
                index: idx,
                path: display_path,
                companions,
                rgb,
                width,
                height,
                content_hash: hash,
                imported,
                embedding,
                sharpness,
            }));
        }

        if let Some(cache) = embed_cache.as_ref() {
            if let Err(err) = cache.save_to(&cache_path) {
                tracing::warn!("Import scan: failed to write embedding cache: {err}");
            }
        }

        let _ = tx.send(ScanMsg::Done);
    });
}

/// The scan's photo count arrived — size the entry list and the model.
fn apply_scan_count(w: &ImportWindow, ctx: &ImportCtx, n: usize) {
    let mut ents = ctx.entries.borrow_mut();
    ents.reserve(n);
    for _ in 0..n {
        ents.push(Entry {
            path: PathBuf::new(),
            companions: vec![],
            content_hash: [0; 32],
            is_imported: false,
            thumb: None,
            embedding: None,
            sharpness: None,
        });
    }
    drop(ents);
    w.set_total_count(n as i32);
    // Bulk reset — happens once per scan when the
    // count first arrives, not on every click.
    ctx.model.set_vec(build_items(
        &ctx.entries.borrow(),
        &ctx.selected.borrow(),
        &ctx.groups.borrow(),
    ));
}

/// One scanned photo arrived — fill its entry in and refresh its row.
fn apply_scan_thumb(w: &ImportWindow, ctx: &ImportCtx, msg: ScanThumb) {
    let ScanThumb {
        index, path, companions, rgb, width, height, content_hash, imported, embedding, sharpness,
    } = msg;
    let thumb = if !rgb.is_empty() && width > 0 && height > 0 {
        let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
            &rgb, width, height,
        );
        Some(slint::Image::from_rgb8(buf))
    } else {
        None
    };
    let cur = ctx.current.get();
    let is_first = index == 0 && cur == 0;
    {
        let mut ents = ctx.entries.borrow_mut();
        if let Some(e) = ents.get_mut(index) {
            e.path = path.clone();
            e.companions = companions;
            e.content_hash = content_hash;
            e.is_imported = imported;
            e.thumb = thumb.clone();
            e.embedding = embedding;
            e.sharpness = sharpness;
        }
    }
    update_row(
        &ctx.model,
        &ctx.entries.borrow(),
        &ctx.selected.borrow(),
        &ctx.groups.borrow(),
        index,
    );
    ctx.scanned_count.set(ctx.scanned_count.get() + 1);
    w.set_scanned_count(ctx.scanned_count.get() as i32);
    // Show preview for the first result.
    if is_first {
        let filename = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if let Some(img) = thumb {
            w.set_preview_photo(img);
            ctx.preview_shown_idx.set(Some(0));
        }
        w.set_preview_filename(filename.into());
    }
}

/// The scan finished — cluster into burst groups from the embeddings
/// collected during the scan, then auto-select the sharpest member of each
/// group.
fn finish_scan(w: &ImportWindow, ctx: &ImportCtx) {
    let n = ctx.entries.borrow().len();

    let resolved_groups = {
        let ents = ctx.entries.borrow();
        let mut idx_map: Vec<usize> = Vec::new();
        let mut embeddings: Vec<Vec<f32>> = Vec::new();
        for (i, e) in ents.iter().enumerate() {
            if let Some(emb) = &e.embedding {
                idx_map.push(i);
                embeddings.push(emb.clone());
            }
        }
        if embeddings.is_empty() {
            Vec::new()
        } else {
            let threshold = maple_state::Settings::load().stacks.threshold;
            flatten_clusters(maple_db::cluster_embeddings(&embeddings, threshold), &idx_map)
        }
    };

    if !resolved_groups.is_empty() {
        let ents = ctx.entries.borrow();
        let mut sel = ctx.selected.borrow_mut();
        for group in &resolved_groups {
            sel.insert(sharpest_in_group(group, &ents));
        }
        drop(sel);
        w.set_selected_count(ctx.selected.borrow().len() as i32);
    }
    *ctx.groups.borrow_mut() = resolved_groups;

    w.set_scanning(false);
    w.set_status_text(scan_status_text(n).into());
    // Bulk reset — happens once when the scan
    // finishes, not on every click.
    ctx.model.set_vec(build_items(
        &ctx.entries.borrow(),
        &ctx.selected.borrow(),
        &ctx.groups.borrow(),
    ));
}

/// Translate clusters over the embedded-only subset back into flat `entries`
/// indices via `idx_map`, each group sorted ascending.
fn flatten_clusters(clusters: Vec<Vec<usize>>, idx_map: &[usize]) -> Vec<Vec<usize>> {
    clusters
        .into_iter()
        .map(|members| {
            let mut flat: Vec<usize> = members.iter().map(|&m| idx_map[m]).collect();
            flat.sort_unstable();
            flat
        })
        .collect()
}

/// Index of the sharpest entry in `group` — the shot auto-selected out of a
/// burst. Entries with no sharpness score count as 0, and the first member
/// wins a tie.
fn sharpest_in_group(group: &[usize], entries: &[Entry]) -> usize {
    let mut best = group[0];
    let mut best_sharpness = entries[best].sharpness.unwrap_or(0.0);
    for &idx in &group[1..] {
        let s = entries[idx].sharpness.unwrap_or(0.0);
        if s > best_sharpness {
            best = idx;
            best_sharpness = s;
        }
    }
    best
}

fn scan_status_text(n: usize) -> String {
    format!("{n} photo{} found", if n == 1 { "" } else { "s" })
}

// ── Browsing (click + navigation) ─────────────────────────────────

fn wire_browse(window: &ImportWindow, ctx: &ImportCtx) {
    // ── Item clicked (toggle-select + update preview) ──────────────
    window.on_item_clicked({
        let ctx = ctx.clone();
        move |idx| {
            let Some(w) = ctx.window.upgrade() else { return };
            let idx = idx as usize;
            // Only skip the reload if this exact photo is *already* the one
            // on screen — `current` alone isn't enough, since it defaults to
            // 0 before anything has ever been previewed.
            let already_shown = ctx.preview_shown_idx.get() == Some(idx);
            {
                let mut sel = ctx.selected.borrow_mut();
                if sel.contains(&idx) {
                    sel.remove(&idx);
                } else {
                    sel.insert(idx);
                }
            }
            w.set_copy_done(false);
            w.set_selected_count(ctx.selected.borrow().len() as i32);
            // A click only ever changes this one row's selection state —
            // update it in place rather than rebuilding the whole model.
            update_row(&ctx.model, &ctx.entries.borrow(), &ctx.selected.borrow(), &ctx.groups.borrow(), idx);

            // Clicking the already-open photo just toggles its selection —
            // don't re-decode and re-render the big preview for no reason.
            if already_shown {
                return;
            }
            ctx.current.set(idx);
            w.set_current_index(idx as i32);
            ctx.preview_shown_idx.set(Some(idx));

            show_preview(&w, &ctx.entries, idx);
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

            let new_idx = nav_target(&ctx.groups.borrow(), cur, len, delta);

            if new_idx == cur {
                return;
            }
            ctx.current.set(new_idx);
            w.set_current_index(new_idx as i32);
            ctx.preview_shown_idx.set(Some(new_idx));

            show_preview(&w, &ctx.entries, new_idx);
        }
    };
    window.on_nav_prev(make_nav(-1));
    window.on_nav_next(make_nav(1));
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

// ── Copy selected ─────────────────────────────────────────────────

/// Per-run values the copy drain needs alongside the shared [`ImportCtx`].
struct CopyRun {
    /// The selected entries, ascending — the rows to mark imported.
    sel_indices: Vec<usize>,
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
    // Mark as imported in the SeenSet.
    {
        let mut imp = maple_state::SeenSet::load_imported(&run.library_dir);
        let mut ents = ctx.entries.borrow_mut();
        for &i in &run.sel_indices {
            if let Some(e) = ents.get_mut(i) {
                e.is_imported = true;
                imp.insert(&e.content_hash);
            }
        }
        let _ = imp.save_imported(&run.library_dir);
    }
    // Insert display files into library DB, under the path they were copied
    // to. `Entry::path` is the *source* path — an SD-card path that stops
    // existing the moment the card is ejected.
    let to_insert: Vec<ImportEntry> = {
        let ents = ctx.entries.borrow();
        run.sel_indices
            .iter()
            .filter_map(|&i| ents.get(i))
            .filter_map(|e| {
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
                })
            })
            .collect()
    };
    insert_imported_images(&ctx.db, &to_insert, &run.algorithm_key);
    // Backfill EXIF for the records just inserted.
    maple_db::spawn_metadata_filler(ctx.db.clone());
    ctx.selected.borrow_mut().clear();
    w.set_selected_count(0);
    w.set_copying(false);
    w.set_status_text(copy_status_text(copied, failed).into());
    // Only the copied rows' selected/imported flags
    // changed — update those in place.
    {
        let ents = ctx.entries.borrow();
        let sel = ctx.selected.borrow();
        let grp = ctx.groups.borrow();
        for &i in &run.sel_indices {
            update_row(&ctx.model, &ents, &sel, &grp, i);
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
    update_row(&ctx.model, &ctx.entries.borrow(), &ctx.selected.borrow(), &ctx.groups.borrow(), idx);
    w.set_rotating(false);
}

// ── Model rows ────────────────────────────────────────────────────

/// Build the [`ImportItem`] for a single entry.
fn make_item(entries: &[Entry], selected: &HashSet<usize>, groups: &[Vec<usize>], i: usize) -> ImportItem {
    let e = &entries[i];
    let display_is_raw = maple_import::is_raw_format(&e.path);
    let has_jpg =
        !display_is_raw || e.companions.iter().any(|c| !maple_import::is_raw_format(c));
    let has_raw =
        display_is_raw || e.companions.iter().any(|c| maple_import::is_raw_format(c));
    ImportItem {
        index: i as i32,
        filename: e
            .path
            .file_name()
            .map(|n| SharedString::from(n.to_string_lossy().as_ref()))
            .unwrap_or_default(),
        thumb: e.thumb.clone().unwrap_or_default(),
        loaded: !e.path.as_os_str().is_empty(),
        is_selected: selected.contains(&i),
        is_imported: e.is_imported,
        stack_size: find_group(groups, i).map(|g| g.len() as i32).unwrap_or(0),
        has_jpg,
        has_raw,
    }
}

/// Build the full [`ImportItem`] vec from current state, for bulk resets
/// (new scan, count/size known, scan finished). Not for per-click updates —
/// use [`update_row`] for those so a full model swap doesn't tear down and
/// recreate every tile's `TouchArea` (which can drop a click landing
/// mid-rebuild).
fn build_items(entries: &[Entry], selected: &HashSet<usize>, groups: &[Vec<usize>]) -> Vec<ImportItem> {
    (0..entries.len())
        .map(|i| make_item(entries, selected, groups, i))
        .collect()
}

/// Update a single row of the persistent model in place.
fn update_row(
    model: &VecModel<ImportItem>,
    entries: &[Entry],
    selected: &HashSet<usize>,
    groups: &[Vec<usize>],
    i: usize,
) {
    if i < entries.len() {
        model.set_row_data(i, make_item(entries, selected, groups, i));
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
            thumb: None,
            embedding: None,
            sharpness,
        }
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

    #[test]
    fn flatten_clusters_maps_back_to_entry_indices_and_sorts() {
        // Only entries 2, 5 and 9 produced embeddings.
        let idx_map = vec![2, 5, 9];
        let clusters = vec![vec![2, 0], vec![1]];
        assert_eq!(flatten_clusters(clusters, &idx_map), vec![vec![2, 9], vec![5]]);
    }

    #[test]
    fn sharpest_in_group_picks_the_highest_score() {
        let entries = vec![
            entry("a.jpg", Some(10.0)),
            entry("b.jpg", Some(42.0)),
            entry("c.jpg", Some(30.0)),
        ];
        assert_eq!(sharpest_in_group(&[0, 1, 2], &entries), 1);
    }

    #[test]
    fn sharpest_in_group_treats_a_missing_score_as_zero_and_keeps_the_first_on_a_tie() {
        let entries = vec![entry("a.jpg", None), entry("b.jpg", Some(0.0))];
        assert_eq!(sharpest_in_group(&[0, 1], &entries), 0);
        assert_eq!(sharpest_in_group(&[1], &entries), 1);
    }

    // ── Status text ───────────────────────────────────────────────

    #[test]
    fn scan_status_text_pluralises() {
        assert_eq!(scan_status_text(1), "1 photo found");
        assert_eq!(scan_status_text(0), "0 photos found");
        assert_eq!(scan_status_text(7), "7 photos found");
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
        assert!(item.loaded);
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
    fn make_item_carries_selection_and_stack_size() {
        let entries = vec![entry("/src/a.jpg", None), entry("/src/b.jpg", None)];
        let selected: HashSet<usize> = [1].into_iter().collect();
        let groups = vec![vec![0, 1]];

        let first = make_item(&entries, &selected, &groups, 0);
        assert!(!first.is_selected);
        assert_eq!(first.stack_size, 2);

        let second = make_item(&entries, &selected, &groups, 1);
        assert!(second.is_selected);
    }

    #[test]
    fn build_items_covers_every_entry() {
        let entries = vec![entry("/src/a.jpg", None), entry("/src/b.jpg", None)];
        let items = build_items(&entries, &HashSet::new(), &[]);
        assert_eq!(items.len(), 2);
        assert_eq!(items[1].index, 1);
        // Solo entries carry no stack badge.
        assert!(items.iter().all(|i| i.stack_size == 0));
    }
}
