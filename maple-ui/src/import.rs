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
use slint::{ComponentHandle, ModelRc, SharedString, Timer, TimerMode, VecModel};

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
pub fn open(db: Arc<Mutex<maple_db::Database>>) {
    open_with_source(db, std::path::PathBuf::new());
}

/// Open the import browser window pre-seeded with `source_path`.
///
/// Called when the user clicks "Start Scan" on the embedded ImportPage.
/// If `source_path` is empty the window opens on the picker phase as before.
pub fn open_with_source(db: Arc<Mutex<maple_db::Database>>, source_path: std::path::PathBuf) {
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

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<Import, slint::PlatformError> {
    let window = ImportWindow::new()?;

    let entries: Rc<RefCell<Vec<Entry>>> = Rc::new(RefCell::new(Vec::new()));
    let selected: Rc<RefCell<HashSet<usize>>> = Rc::new(RefCell::new(HashSet::new()));
    let current: Rc<Cell<usize>> = Rc::new(Cell::new(0));
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
        let scan_timer = scan_timer.clone();

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
            w.set_items(ModelRc::from(Rc::new(VecModel::<ImportItem>::default())));
            w.set_selected_count(0);
            w.set_copy_done(false);
            w.set_total_count(0);
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
                                w.set_items(build_model(
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
                                w.set_items(build_model(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                ));
                                // Show preview for the first result.
                                if is_first {
                                    let filename = path
                                        .file_name()
                                        .map(|n| n.to_string_lossy().into_owned())
                                        .unwrap_or_default();
                                    if let Some(img) = thumb {
                                        w.set_preview_photo(img);
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

                                w.set_status_text(
                                    format!("{n} photo{} found",
                                        if n == 1 { "" } else { "s" }).into(),
                                );
                                w.set_items(build_model(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                ));
                                return;
                            }
                            Ok(ScanMsg::Error(e)) => {
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
        move |idx| {
            let Some(w) = w.upgrade() else { return };
            let idx = idx as usize;
            let navigated_to_new_item = current.get() != idx;
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

            // Clicking the already-open photo just toggles its selection —
            // don't re-decode and re-render the big preview for no reason.
            if !navigated_to_new_item {
                w.set_items(build_model(&entries.borrow(), &selected.borrow(), &groups.borrow()));
                return;
            }
            current.set(idx);
            w.set_current_index(idx as i32);

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
            drop(ents);
            w.set_items(build_model(&entries.borrow(), &selected.borrow(), &groups.borrow()));
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
        let selected = selected.clone();
        let groups = groups.clone();
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
            drop(ents);
            w.set_items(build_model(&entries.borrow(), &selected.borrow(), &groups.borrow()));
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
                        for p in group.paths_for_copy(maple_import::CopyMode::default()) {
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
                                w.set_items(build_model(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                    &groups2.borrow(),
                                ));

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
                        w.set_items(build_model(&entries2.borrow(), &selected2.borrow(), &groups2.borrow()));
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

/// Rebuild the full [`ImportItem`] model from current state.
///
/// Called each time the selection or scan state changes. This is not cheap
/// (rebuilds every item) but the import grid is small (100s of items) and
/// we don't update on every frame, so it's acceptable.
fn build_model(entries: &[Entry], selected: &HashSet<usize>, groups: &[Vec<usize>]) -> ModelRc<ImportItem> {
    let items: Vec<ImportItem> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| ImportItem {
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
        })
        .collect();
    ModelRc::from(Rc::new(VecModel::from(items)))
}
