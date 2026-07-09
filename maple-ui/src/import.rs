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
use std::collections::HashSet;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use image::{DynamicImage, RgbImage};
use slint::{ComponentHandle, Model, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::services::import::{insert_imported_images, ImportEntry};
use crate::{ImportItem, ImportWindow};
use crate::thumbnail;

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
    Thumb {
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
    },
    Done,
    Error(String),
}

enum CopyMsg {
    Progress { done: usize, total: usize },
    Done { copied: usize, failed: usize },
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

// Fields keep shared state alive for the lifetime of the singleton even though
// only Rc/Arc clones are accessed by the callbacks themselves.
#[allow(dead_code)]
struct Import {
    window: ImportWindow,
    entries: Rc<RefCell<Vec<Entry>>>,
    selected: Rc<RefCell<HashSet<usize>>>,
    current: Rc<Cell<usize>>,
    source: Rc<RefCell<PathBuf>>,
    dest: Rc<RefCell<PathBuf>>,
    db: Arc<Mutex<maple_db::Database>>,
    /// Detected burst groups from the last scan — each a sorted list of
    /// flat `entries` indices. Populated once, when the scan finishes.
    groups: Rc<RefCell<Vec<Vec<usize>>>>,
    _scan_timer: Rc<RefCell<Option<Timer>>>,
    _copy_timer: Rc<RefCell<Option<Timer>>>,
    _copy_done_timer: Rc<RefCell<Option<Timer>>>,
    _rotate_timer: Rc<RefCell<Option<Timer>>>,
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
                *imp.source.borrow_mut() = source_path;
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

    let entries: Rc<RefCell<Vec<Entry>>> = Rc::new(RefCell::new(Vec::new()));
    let selected: Rc<RefCell<HashSet<usize>>> = Rc::new(RefCell::new(HashSet::new()));
    let current: Rc<Cell<usize>> = Rc::new(Cell::new(0));
    // Persistent model, mutated in place via `set_row_data` for single-row
    // changes (selection toggle, thumb arriving, rotate, …) instead of being
    // replaced wholesale. Swapping in a brand-new `VecModel` on every click
    // forces Slint to tear down and recreate every tile's `TouchArea`; a
    // second click landing mid-rebuild would then hit a fresh TouchArea that
    // never saw the press, silently dropping the click. Reserve full
    // `set_vec` resets for genuinely bulk changes (new scan, all counts known).
    let model: Rc<VecModel<ImportItem>> = Rc::new(VecModel::default());
    window.set_items(ModelRc::from(model.clone()));
    // Which entry's big preview is actually on screen right now — `None`
    // until something has genuinely been rendered into it. `current`
    // defaults to 0 before anything is ever previewed, so comparing against
    // `current` alone can't tell "index 0 is already showing" apart from
    // "nothing has been shown yet, and this happens to be index 0".
    let preview_shown_idx: Rc<Cell<Option<usize>>> = Rc::new(Cell::new(None));
    // Count of thumbnails processed so far during the current scan — drives
    // the progress bar shown while `scanning` is true.
    let scanned_count: Rc<Cell<usize>> = Rc::new(Cell::new(0));
    let source: Rc<RefCell<PathBuf>> = Rc::new(RefCell::new(PathBuf::new()));
    // The embedded sidebar ImportPage only lets the user pick a *source*
    // folder — there is no destination step in that flow anymore, so default
    // to the configured library directory (the same place the scanner,
    // thumbcache, etc. already treat as "the library"). Without this, `dest`
    // stays empty forever and "Copy Selected" silently no-ops.
    let dest: Rc<RefCell<PathBuf>> =
        Rc::new(RefCell::new(maple_state::Settings::load().library_dir));
    window.set_dest_path(SharedString::from(dest.borrow().to_string_lossy().into_owned()));
    let groups: Rc<RefCell<Vec<Vec<usize>>>> = Rc::new(RefCell::new(Vec::new()));
    let scan_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    let copy_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    let copy_done_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    let rotate_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));

    // ── Close ─────────────────────────────────────────────────────
    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    // ── Pick source ───────────────────────────────────────────────
    window.on_pick_source({
        let w = window.as_weak();
        let source = source.clone();
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
        let w = window.as_weak();
        let dest = dest.clone();
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

    // ── Start scan ────────────────────────────────────────────────
    window.on_start_scan({
        let w = window.as_weak();
        let entries = entries.clone();
        let selected = selected.clone();
        let current = current.clone();
        let source = source.clone();
        let groups = groups.clone();
        let preview_shown_idx = preview_shown_idx.clone();
        let scanned_count = scanned_count.clone();
        let scan_timer = scan_timer.clone();
        let model = model.clone();

        move || {
            let Some(w) = w.upgrade() else { return };
            let src = source.borrow().clone();
            if src.as_os_str().is_empty() {
                return;
            }

            // Reset state.
            entries.borrow_mut().clear();
            selected.borrow_mut().clear();
            groups.borrow_mut().clear();
            current.set(0);
            preview_shown_idx.set(None);
            scanned_count.set(0);
            model.set_vec(Vec::new());
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

            let tx_clone = tx.clone();
            let src_clone = src.clone();
            let imported_set_clone = imported_set.clone();
            std::thread::spawn(move || {
                let scanned_groups = match maple_import::scan_grouped(&src_clone) {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = tx_clone.send(ScanMsg::Error(e.to_string()));
                        return;
                    }
                };
                let total = scanned_groups.len();
                let _ = tx_clone.send(ScanMsg::Count(total));

                // Burst detection during the scan reuses the [stacks]
                // settings — same toggle/threshold/model as post-import
                // library stacking. If the embedder fails to load (e.g. no
                // network for a first-time model fetch), log once and
                // continue the scan without embeddings/sharpness — this is
                // enrichment, never a hard requirement to finish scanning.
                let algorithm_key = stack_settings.algorithm_key();
                let cache_path = src_clone.join(EMBED_CACHE_FILE);
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
                            let imp = imported_set_clone
                                .lock()
                                .map(|s| s.probably_contains(&h))
                                .unwrap_or(false);
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

                    let _ = tx_clone.send(ScanMsg::Thumb {
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
                    });
                }

                if let Some(cache) = embed_cache.as_ref() {
                    if let Err(err) = cache.save_to(&cache_path) {
                        tracing::warn!("Import scan: failed to write embedding cache: {err}");
                    }
                }

                let _ = tx_clone.send(ScanMsg::Done);
            });

            // Timer to drain scan results.
            let w_weak = w.as_weak();
            let entries2 = entries.clone();
            let selected2 = selected.clone();
            let current2 = current.clone();
            let groups2 = groups.clone();
            let preview_shown_idx2 = preview_shown_idx.clone();
            let scanned_count2 = scanned_count.clone();
            let model2 = model.clone();
            let timer = Timer::default();
            timer.start(
                TimerMode::Repeated,
                Duration::from_millis(30),
                move || {
                    let Some(w) = w_weak.upgrade() else { return };

                    for _ in 0..10 {
                        match rx.try_recv() {
                            Ok(ScanMsg::Count(n)) => {
                                let mut ents = entries2.borrow_mut();
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
                                model2.set_vec(build_items(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                ));
                            }
                            Ok(ScanMsg::Thumb {
                                index, path, companions, rgb, width, height,
                                content_hash, imported, embedding, sharpness,
                            }) => {
                                let thumb = if !rgb.is_empty() && width > 0 && height > 0 {
                                    let buf = slint::SharedPixelBuffer::<slint::Rgb8Pixel>::clone_from_slice(
                                        &rgb, width, height,
                                    );
                                    Some(slint::Image::from_rgb8(buf))
                                } else {
                                    None
                                };
                                let cur = current2.get();
                                let is_first = index == 0 && cur == 0;
                                {
                                    let mut ents = entries2.borrow_mut();
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
                                    &model2,
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                    index,
                                );
                                scanned_count2.set(scanned_count2.get() + 1);
                                w.set_scanned_count(scanned_count2.get() as i32);
                                // Show preview for the first result.
                                if is_first {
                                    let filename = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_default();
                                    if let Some(img) = thumb {
                                        w.set_preview_photo(img);
                                        preview_shown_idx2.set(Some(0));
                                    }
                                    w.set_preview_filename(filename.into());
                                }
                            }
                            Ok(ScanMsg::Done) => {
                                let n = entries2.borrow().len();

                                // Cluster into burst groups from the embeddings
                                // collected during the scan, then auto-select
                                // the sharpest member of each group.
                                let resolved_groups = {
                                    let ents = entries2.borrow();
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
                                        maple_db::cluster_embeddings(&embeddings, threshold)
                                            .into_iter()
                                            .map(|members| {
                                                let mut flat: Vec<usize> =
                                                    members.iter().map(|&m| idx_map[m]).collect();
                                                flat.sort_unstable();
                                                flat
                                            })
                                            .collect()
                                    }
                                };

                                if !resolved_groups.is_empty() {
                                    let ents = entries2.borrow();
                                    let mut sel = selected2.borrow_mut();
                                    for group in &resolved_groups {
                                        let mut best = group[0];
                                        let mut best_sharpness = ents[best].sharpness.unwrap_or(0.0);
                                        for &idx in &group[1..] {
                                            let s = ents[idx].sharpness.unwrap_or(0.0);
                                            if s > best_sharpness {
                                                best = idx;
                                                best_sharpness = s;
                                            }
                                        }
                                        sel.insert(best);
                                    }
                                    drop(sel);
                                    w.set_selected_count(selected2.borrow().len() as i32);
                                }
                                *groups2.borrow_mut() = resolved_groups;

                                w.set_scanning(false);
                                w.set_status_text(
                                    format!("{n} photo{} found",
                                        if n == 1 { "" } else { "s" }).into(),
                                );
                                // Bulk reset — happens once when the scan
                                // finishes, not on every click.
                                model2.set_vec(build_items(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                ));
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
            *scan_timer.borrow_mut() = Some(timer);
        }
    });

    // ── Item clicked (toggle-select + update preview) ──────────────
    window.on_item_clicked({
        let w = window.as_weak();
        let entries = entries.clone();
        let selected = selected.clone();
        let current = current.clone();
        let groups = groups.clone();
        let preview_shown_idx = preview_shown_idx.clone();
        let model = model.clone();
        move |idx| {
            let Some(w) = w.upgrade() else { return };
            let idx = idx as usize;
            // Only skip the reload if this exact photo is *already* the one
            // on screen — `current` alone isn't enough, since it defaults to
            // 0 before anything has ever been previewed.
            let already_shown = preview_shown_idx.get() == Some(idx);
            {
                let mut sel = selected.borrow_mut();
                if sel.contains(&idx) {
                    sel.remove(&idx);
                } else {
                    sel.insert(idx);
                }
            }
            w.set_copy_done(false);
            w.set_selected_count(selected.borrow().len() as i32);
            // A click only ever changes this one row's selection state —
            // update it in place rather than rebuilding the whole model.
            update_row(&model, &entries.borrow(), &selected.borrow(), &groups.borrow(), idx);

            // Clicking the already-open photo just toggles its selection —
            // don't re-decode and re-render the big preview for no reason.
            if already_shown {
                return;
            }
            current.set(idx);
            w.set_current_index(idx as i32);
            preview_shown_idx.set(Some(idx));

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
    });

    // ── Navigation ────────────────────────────────────────────────
    //
    // If the current photo belongs to a detected burst group, left/right
    // jumps directly to the prev/next member of that group (skipping over
    // unrelated interleaved photos). Moving past the first/last member of
    // the group falls through to ordinary flat navigation — arrow keys
    // never trap the user inside a burst; they page through all of its
    // members, then continue into the rest of the scan. Solo entries behave
    // exactly as a plain flat clamp, same as before this feature existed.
    let make_nav = |delta: i32| {
        let w = window.as_weak();
        let entries = entries.clone();
        let current = current.clone();
        let groups = groups.clone();
        let preview_shown_idx = preview_shown_idx.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let len = entries.borrow().len();
            if len == 0 {
                return;
            }
            let cur = current.get();

            let groups_ref = groups.borrow();
            let new_idx = match find_group(&groups_ref, cur) {
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
            };
            drop(groups_ref);

            if new_idx == cur {
                return;
            }
            current.set(new_idx);
            w.set_current_index(new_idx as i32);
            preview_shown_idx.set(Some(new_idx));

            let ents = entries.borrow();
            if let Some(e) = ents.get(new_idx) {
                let filename = e
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                w.set_preview_filename(filename.into());
                if let Some(thumb) = &e.thumb {
                    w.set_preview_photo(thumb.clone());
                }
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
    };
    window.on_nav_prev(make_nav(-1));
    window.on_nav_next(make_nav(1));

    // ── Copy selected ─────────────────────────────────────────────
    window.on_copy_selected({
        let w = window.as_weak();
        let entries = entries.clone();
        let selected = selected.clone();
        let dest = dest.clone();
        let db = db.clone();
        let groups = groups.clone();
        let copy_timer = copy_timer.clone();
        let copy_done_timer = copy_done_timer.clone();
        let model = model.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let sel = selected.borrow().clone();
            if sel.is_empty() {
                return;
            }
            let dst = dest.borrow().clone();
            if dst.as_os_str().is_empty() {
                w.set_status_text("No destination folder set.".into());
                return;
            }

            w.set_copying(true);
            w.set_copy_done(false);
            w.set_status_text("Copying…".into());

            let settings = maple_state::Settings::load();
            let folder_format = settings.folder_format.clone();
            let library_dir = settings.library_dir.clone();
            let algorithm_key = settings.stacks.algorithm_key();

            let copy_mode = match w.get_copy_mode() {
                0 => maple_import::CopyMode::DisplayOnly,
                2 => maple_import::CopyMode::RawOnly,
                _ => maple_import::CopyMode::All,
            };

            let mut sources: Vec<PathBuf> = Vec::new();
            let mut sel_indices: Vec<usize> = sel.iter().copied().collect();
            sel_indices.sort_unstable();
            let mut entry_data: Vec<(PathBuf, [u8; 32])> = Vec::new();
            {
                let ents = entries.borrow();
                for &i in &sel_indices {
                    if let Some(e) = ents.get(i) {
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
                        entry_data.push((e.path.clone(), e.content_hash));
                    }
                }
            }

            let (tx, rx) = mpsc::channel::<CopyMsg>();
            let dst2 = dst.clone();
            std::thread::spawn(move || {
                let result = maple_import::copy_images(
                    &sources,
                    &dst2,
                    &folder_format,
                    |done, total| {
                        let _ = tx.send(CopyMsg::Progress { done, total });
                    },
                );
                match result {
                    Ok(summary) => {
                        let _ = tx.send(CopyMsg::Done {
                            copied: summary.copied,
                            failed: summary.failed,
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(CopyMsg::Error(e.to_string()));
                    }
                }
            });

            let w_weak = w.as_weak();
            let entries2 = entries.clone();
            let selected2 = selected.clone();
            let groups2 = groups.clone();
            let db2 = db.clone();
            let copy_done_timer = copy_done_timer.clone();
            let model2 = model.clone();
            let timer = Timer::default();
            timer.start(
                TimerMode::Repeated,
                Duration::from_millis(30),
                move || {
                    let Some(w) = w_weak.upgrade() else { return };
                    loop {
                        match rx.try_recv() {
                            Ok(CopyMsg::Progress { done, total }) => {
                                if total > 0 {
                                    w.set_status_text(
                                        format!("Copying… {done} / {total}").into(),
                                    );
                                }
                            }
                            Ok(CopyMsg::Done { copied, failed }) => {
                                // Mark as imported in the SeenSet.
                                {
                                    let mut imp = maple_state::SeenSet::load_imported(&library_dir);
                                    let mut ents = entries2.borrow_mut();
                                    for &i in &sel_indices {
                                        if let Some(e) = ents.get_mut(i) {
                                            e.is_imported = true;
                                            imp.insert(&e.content_hash);
                                        }
                                    }
                                    let _ = imp.save_imported(&library_dir);
                                }
                                // Insert display files into library DB.
                                let to_insert: Vec<ImportEntry> = {
                                    let ents = entries2.borrow();
                                    sel_indices
                                        .iter()
                                        .filter_map(|&i| ents.get(i))
                                        .map(|e| ImportEntry {
                                            path: e.path.clone(),
                                            content_hash: e.content_hash,
                                            embedding: e.embedding.clone(),
                                        })
                                        .collect()
                                };
                                insert_imported_images(&db2, &to_insert, &algorithm_key);
                                // Backfill EXIF for the records just inserted.
                                maple_db::spawn_metadata_filler(db2.clone());
                                selected2.borrow_mut().clear();
                                w.set_selected_count(0);
                                w.set_copying(false);
                                let msg = if failed == 0 {
                                    format!(
                                        "Copied {copied} photo{}",
                                        if copied == 1 { "" } else { "s" }
                                    )
                                } else {
                                    format!("Copied {copied}, {failed} failed")
                                };
                                w.set_status_text(msg.into());
                                // Only the copied rows' selected/imported flags
                                // changed — update those in place.
                                {
                                    let ents = entries2.borrow();
                                    let sel = selected2.borrow();
                                    let grp = groups2.borrow();
                                    for &i in &sel_indices {
                                        update_row(&model2, &ents, &sel, &grp, i);
                                    }
                                }

                                // Flash the button green, then revert to the
                                // normal "Copy Selected" state.
                                w.set_copy_done(true);
                                let w_weak2 = w.as_weak();
                                let done_timer = Timer::default();
                                done_timer.start(
                                    TimerMode::SingleShot,
                                    Duration::from_millis(2500),
                                    move || {
                                        if let Some(w) = w_weak2.upgrade() {
                                            w.set_copy_done(false);
                                        }
                                    },
                                );
                                *copy_done_timer.borrow_mut() = Some(done_timer);
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
            *copy_timer.borrow_mut() = Some(timer);
        }
    });

    // ── Rotate current photo ─────────────────────────────────────────
    //
    // Patches the EXIF Orientation tag on the *source* file in place
    // (mirrors detail.rs's rotate_current), then re-renders the grid thumb
    // and the big preview so the change is visible immediately. The file's
    // bytes (and therefore its content hash) change, so the in-memory
    // Entry's content_hash is updated too — it's what gets recorded in the
    // SeenSet / DB on copy.
    window.on_rotate({
        let w = window.as_weak();
        let entries = entries.clone();
        let current = current.clone();
        let selected = selected.clone();
        let groups = groups.clone();
        let rotate_timer = rotate_timer.clone();
        let model = model.clone();
        move |clockwise| {
            let Some(w) = w.upgrade() else { return };
            if w.get_rotating() {
                return;
            }
            let idx = current.get();
            let path = {
                let ents = entries.borrow();
                match ents.get(idx) {
                    Some(e) => e.path.clone(),
                    None => return,
                }
            };

            w.set_rotating(true);

            enum RotateMsg {
                Done {
                    content_hash: [u8; 32],
                    thumb: (Vec<u8>, u32, u32),
                    preview: (Vec<u8>, u32, u32),
                },
                Error(String),
            }

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

            let w_weak = w.as_weak();
            let entries2 = entries.clone();
            let selected2 = selected.clone();
            let groups2 = groups.clone();
            let rotate_timer_slot = rotate_timer.clone();
            let model2 = model.clone();
            let timer = Timer::default();
            timer.start(TimerMode::Repeated, Duration::from_millis(32), move || {
                let Some(w) = w_weak.upgrade() else { return };
                let outcome = match rx.try_recv() {
                    Ok(m) => m,
                    Err(mpsc::TryRecvError::Empty) => return,
                    Err(mpsc::TryRecvError::Disconnected) => {
                        RotateMsg::Error("Rotation worker vanished".to_owned())
                    }
                };
                if let Some(t) = rotate_timer_slot.borrow().as_ref() {
                    t.stop();
                }
                match outcome {
                    RotateMsg::Done { content_hash, thumb, preview } => {
                        if let Some(e) = entries2.borrow_mut().get_mut(idx) {
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
                        update_row(&model2, &entries2.borrow(), &selected2.borrow(), &groups2.borrow(), idx);
                        w.set_rotating(false);
                    }
                    RotateMsg::Error(msg) => {
                        w.set_status_text(format!("Rotate failed: {msg}").into());
                        w.set_rotating(false);
                    }
                }
            });
            *rotate_timer.borrow_mut() = Some(timer);
        }
    });

    Ok(Import {
        window,
        entries,
        selected,
        current,
        source,
        dest,
        db,
        groups,
        _scan_timer: scan_timer,
        _copy_timer: copy_timer,
        _copy_done_timer: copy_done_timer,
        _rotate_timer: rotate_timer,
    })
}

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
