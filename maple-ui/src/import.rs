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

use slint::{ComponentHandle, ModelRc, SharedString, Timer, TimerMode, VecModel};

use crate::services::import::{insert_imported_images, ImportEntry};
use crate::{ImportItem, ImportWindow};
use crate::thumbnail;

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
    _scan_timer: Rc<RefCell<Option<Timer>>>,
    _copy_timer: Rc<RefCell<Option<Timer>>>,
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
    let dest: Rc<RefCell<PathBuf>> = Rc::new(RefCell::new(PathBuf::new()));
    let scan_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    let copy_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));

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
            current.set(0);
            w.set_items(ModelRc::from(Rc::new(VecModel::<ImportItem>::default())));
            w.set_selected_count(0);
            w.set_total_count(0);
            w.set_preview_photo(slint::Image::default());
            w.set_preview_filename(SharedString::default());
            w.set_status_text("Scanning…".into());
            w.set_in_browser(true);

            let (tx, rx) = mpsc::channel::<ScanMsg>();

            let settings = maple_state::Settings::load();
            let library_dir = settings.library_dir.clone();

            let imported_set = Arc::new(Mutex::new(
                maple_state::SeenSet::load_imported(&library_dir),
            ));

            let tx_clone = tx.clone();
            let src_clone = src.clone();
            let imported_set_clone = imported_set.clone();
            std::thread::spawn(move || {
                let groups = match maple_import::scan_grouped(&src_clone) {
                    Ok(g) => g,
                    Err(e) => {
                        let _ = tx_clone.send(ScanMsg::Error(e.to_string()));
                        return;
                    }
                };
                let total = groups.len();
                let _ = tx_clone.send(ScanMsg::Count(total));

                for (idx, group) in groups.into_iter().enumerate() {
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

                    let _ = tx_clone.send(ScanMsg::Thumb {
                        index: idx,
                        path: display_path,
                        companions,
                        rgb,
                        width,
                        height,
                        content_hash: hash,
                        imported,
                    });
                }
                let _ = tx_clone.send(ScanMsg::Done);
            });

            // Timer to drain scan results.
            let w_weak = w.as_weak();
            let entries2 = entries.clone();
            let selected2 = selected.clone();
            let current2 = current.clone();
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
                                    });
                                }
                                drop(ents);
                                w.set_total_count(n as i32);
                                w.set_items(build_model(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
                                ));
                            }
                            Ok(ScanMsg::Thumb {
                                index, path, companions, rgb, width, height,
                                content_hash, imported,
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
                                    }
                                }
                                w.set_items(build_model(
                                    &entries2.borrow(),
                                    &selected2.borrow(),
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
                                w.set_status_text(
                                    format!("{n} photo{} found",
                                        if n == 1 { "" } else { "s" }).into(),
                                );
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
        move |idx| {
            let Some(w) = w.upgrade() else { return };
            let idx = idx as usize;
            {
                let mut sel = selected.borrow_mut();
                if sel.contains(&idx) {
                    sel.remove(&idx);
                } else {
                    sel.insert(idx);
                }
            }
            current.set(idx);
            w.set_current_index(idx as i32);
            w.set_selected_count(selected.borrow().len() as i32);

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
            w.set_items(build_model(&entries.borrow(), &selected.borrow()));
        }
    });

    // ── Navigation ────────────────────────────────────────────────
    let make_nav = |delta: i32| {
        let w = window.as_weak();
        let entries = entries.clone();
        let current = current.clone();
        let selected = selected.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let len = entries.borrow().len();
            if len == 0 {
                return;
            }
            let cur = current.get() as i64;
            let new_idx = (cur + delta as i64).clamp(0, len as i64 - 1) as usize;
            if new_idx as i64 == cur {
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
            w.set_items(build_model(&entries.borrow(), &selected.borrow()));
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
        let copy_timer = copy_timer.clone();
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
            w.set_status_text("Copying…".into());

            let settings = maple_state::Settings::load();
            let folder_format = settings.folder_format.clone();
            let library_dir = settings.library_dir.clone();

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
            let db2 = db.clone();
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
                                        })
                                        .collect()
                                };
                                insert_imported_images(&db2, &to_insert);
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
                                ));
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

    Ok(Import {
        window,
        entries,
        selected,
        current,
        source,
        dest,
        db,
        _scan_timer: scan_timer,
        _copy_timer: copy_timer,
    })
}

/// Rebuild the full [`ImportItem`] model from current state.
///
/// Called each time the selection or scan state changes. This is not cheap
/// (rebuilds every item) but the import grid is small (100s of items) and
/// we don't update on every frame, so it's acceptable.
fn build_model(entries: &[Entry], selected: &HashSet<usize>) -> ModelRc<ImportItem> {
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
        })
        .collect();
    ModelRc::from(Rc::new(VecModel::from(items)))
}
