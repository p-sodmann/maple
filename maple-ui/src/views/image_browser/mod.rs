//! Image browser view — large preview + filmstrip + keyboard selection.
//!
//! Reached from `source_picker` after the user picks source + destination.
//!
//! Layout:
//!   ┌──────────────────────────┬───────────┐
//!   │                          │  filmstrip │
//!   │     large preview        │  (scroll)  │
//!   │                          │            │
//!   │  [filename]              │            │
//!   │  [selected indicator]    │            │
//!   ├──────────────────────────┴───────────┤
//!   │  progress bar                         │
//!   └──────────────────────────────────────┘
//!
//! Keys:  ↑/← previous  ↓/→ next  X toggle-select
//!
//! Background workers:
//!   • **Scan** (`start_scan`) — walks source directory, generates thumbnails,
//!     sends `ScanMsg` to the main loop.
//!   • **Full-res loader** — loads full-resolution images on demand (via
//!     `FullResMsg` channel) for the preview pane.
//!   • **Copy** — copies selected images to the destination directory in a
//!     background thread, reporting progress via `CopyMsg`.
//!
//! State is centralised in `BrowserState` (`Rc<RefCell<BrowserState>>`).

mod filmstrip;
mod preview;
mod scan;

use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use filmstrip::{update_strip_opacity, update_strip_visibility};
use preview::{apply_zoom, compute_buffer_window, update_preview};
use scan::{scan_summary_text, start_scan};

// ── Thumbnail size constants ────────────────────────────────────

const THUMB_SIZE: u32 = 256;
const STRIP_THUMB_PX: i32 = 100;

// ── Background worker messages ──────────────────────────────────

enum ScanMsg {
    Count(usize),
    Thumb {
        index: usize,
        path: PathBuf,
        companions: Vec<PathBuf>,
        png_bytes: Vec<u8>,
        content_hash: [u8; 32],
        imported: bool,
        rejected: bool,
    },
    Done,
    Error(String),
}

/// Message from the background full-res loader to the UI thread.
///
/// Carries raw file bytes (JPEG/PNG) — decoding happens on the UI thread
/// via `gdk::Texture::from_bytes()` which uses the system's libjpeg-turbo,
/// making it 3-5× faster than the pure-Rust `image` crate decoder.
struct FullResMsg {
    index: usize,
    file_bytes: Vec<u8>,
}

/// Message from the background copy worker to the UI thread.
enum CopyMsg {
    Progress { done: usize, total: usize },
    /// Carries the full summary and per-group info for DB insertion.
    Finished {
        summary: maple_import::CopySummary,
        /// One entry per source file: `(display_hash, is_display_file)`.
        /// The copy results and this vec share the same order/length.
        copy_file_info: Vec<CopyFileInfo>,
    },
    Error(String),
}

/// Metadata about one file in the flat copy list.
#[derive(Clone)]
struct CopyFileInfo {
    /// BLAKE3 hash of the *display* file for this group.
    display_hash: [u8; 32],
    /// Whether this file is the display file (JPG) or a companion (RAF).
    is_display: bool,
}

// ── Per-image data stored on the UI thread ──────────────────────

struct ImageEntry {
    path: PathBuf,
    /// Companion files (e.g. RAF when display is JPG).
    companions: Vec<PathBuf>,
    texture: Option<gdk::Texture>,
    content_hash: [u8; 32],
    imported: bool,
    rejected: bool,
}

// ── Shared browser state ────────────────────────────────────────

struct BrowserState {
    images: Vec<ImageEntry>,
    current: usize,
    selected: HashSet<usize>,
    total: usize,
    generated: usize,
    imported_count: usize,
    rejected_count: usize,
    zoom: f64,
    /// Whether to skip imported/rejected images during navigation.
    filter_seen: bool,
    /// Destination directory where selected files will be copied.
    destination: PathBuf,
    /// Sender for dispatching full-res load results back to the UI thread.
    /// Wrapped in Arc<Mutex> so background threads can use it.
    fullres_tx: Arc<Mutex<mpsc::Sender<FullResMsg>>>,
    /// Cache of decoded full-resolution textures keyed by image index.
    fullres_cache: HashMap<usize, gdk::Texture>,
    /// Set of indices currently being loaded to avoid duplicate spawns.
    fullres_loading: HashSet<usize>,
    /// Number of full-res images to keep buffered around the current image.
    buffer_size: usize,
    /// Previously imported image hashes (bloom filter + 32-byte hashes).
    imported_set: Arc<Mutex<maple_state::SeenSet>>,
    /// Explicitly rejected image hashes.
    rejected_set: Arc<Mutex<maple_state::SeenSet>>,
    /// Directory where library data files are read/written.
    library_dir: PathBuf,
    /// What to include when copying (all files / raw only / display only).
    copy_mode: maple_import::CopyMode,
}

impl BrowserState {
    fn new(
        fullres_tx: mpsc::Sender<FullResMsg>,
        buffer_size: usize,
        destination: PathBuf,
        library_dir: PathBuf,
    ) -> Self {
        Self {
            images: Vec::new(),
            current: 0,
            selected: HashSet::new(),
            total: 0,
            generated: 0,
            imported_count: 0,
            rejected_count: 0,
            zoom: 1.0,
            filter_seen: false,
            destination,
            fullres_tx: Arc::new(Mutex::new(fullres_tx)),
            fullres_cache: HashMap::new(),
            fullres_loading: HashSet::new(),
            buffer_size,
            imported_set: Arc::new(Mutex::new(maple_state::SeenSet::load_imported(&library_dir))),
            rejected_set: Arc::new(Mutex::new(maple_state::SeenSet::load_rejected(&library_dir))),
            library_dir,
            copy_mode: maple_import::CopyMode::default(),
        }
    }

    /// Whether `idx` is visible given the current filter.
    fn is_visible(&self, idx: usize) -> bool {
        if !self.filter_seen {
            return true;
        }
        !self.images[idx].imported && !self.images[idx].rejected
    }

    /// Navigate to the previous visible image; returns `true` if moved.
    fn go_prev(&mut self) -> bool {
        if self.current == 0 {
            return false;
        }
        let mut i = self.current - 1;
        loop {
            if self.is_visible(i) {
                self.current = i;
                return true;
            }
            if i == 0 {
                break;
            }
            i -= 1;
        }
        false
    }

    /// Navigate to the next visible image; returns `true` if moved.
    fn go_next(&mut self) -> bool {
        let len = self.images.len();
        for i in (self.current + 1)..len {
            if self.is_visible(i) {
                self.current = i;
                return true;
            }
        }
        false
    }

    /// Mark the current image as rejected if it has not been selected or
    /// imported. Returns the content hash if the image was newly rejected.
    fn auto_reject_current(&mut self) -> Option<[u8; 32]> {
        let idx = self.current;
        if self.images.is_empty() {
            return None;
        }
        let img = &self.images[idx];
        if !img.rejected && !img.imported && !self.selected.contains(&idx) {
            self.images[idx].rejected = true;
            self.rejected_count += 1;
            Some(self.images[idx].content_hash)
        } else {
            None
        }
    }
}

// ── Public API ──────────────────────────────────────────────────

/// Build the image browser page and start scanning.
pub fn build_browser_page(
    source: &Path,
    destination: &Path,
    toast_overlay: &adw::ToastOverlay,
    db: std::sync::Arc<std::sync::Mutex<maple_db::Database>>,
) -> adw::NavigationPage {
    let settings = maple_state::Settings::load();

    let (fullres_tx, fullres_rx) = mpsc::channel::<FullResMsg>();
    let state = Rc::new(RefCell::new(BrowserState::new(
        fullres_tx,
        settings.preview_buffer_size,
        destination.to_path_buf(),
        settings.library_dir,
    )));

    // ── Large preview (left) ────────────────────────────────────
    let preview_picture = gtk4::Picture::builder()
        .content_fit(gtk4::ContentFit::Contain)
        .can_shrink(true)
        .hexpand(true)
        .vexpand(true)
        .build();

    let preview_scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Never)
        .hexpand(true)
        .vexpand(true)
        .build();
    preview_scroll.set_kinetic_scrolling(false);
    preview_scroll.set_child(Some(&preview_picture));

    let filename_label = gtk4::Label::new(Some(""));
    filename_label.add_css_class("title-4");
    filename_label.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
    filename_label.set_max_width_chars(60);

    let selected_label = gtk4::Label::new(Some(""));
    selected_label.add_css_class("caption");
    selected_label.add_css_class("dim-label");

    let counter_label = gtk4::Label::new(Some(""));
    counter_label.add_css_class("caption");

    let info_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .halign(gtk4::Align::Center)
        .margin_bottom(8)
        .build();
    info_box.append(&filename_label);
    info_box.append(&selected_label);
    info_box.append(&counter_label);

    let preview_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(8)
        .hexpand(true)
        .vexpand(true)
        .build();
    preview_box.append(&preview_scroll);
    preview_box.append(&info_box);

    // ── Filmstrip (right) ───────────────────────────────────────
    let strip_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .margin_top(4)
        .margin_bottom(4)
        .margin_start(4)
        .margin_end(4)
        .build();

    let strip_scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .width_request(STRIP_THUMB_PX + 24)
        .vexpand(true)
        .build();
    strip_scroll.set_child(Some(&strip_box));

    // ── Separator ───────────────────────────────────────────────
    let separator = gtk4::Separator::new(gtk4::Orientation::Vertical);

    // ── Main horizontal split ───────────────────────────────────
    let hbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(0)
        .hexpand(true)
        .vexpand(true)
        .build();
    hbox.append(&preview_box);
    hbox.append(&separator);
    hbox.append(&strip_scroll);

    // ── Progress bar ────────────────────────────────────────────
    let progress_bar = gtk4::ProgressBar::builder()
        .show_text(true)
        .text("Scanning…")
        .margin_start(12)
        .margin_end(12)
        .margin_top(4)
        .margin_bottom(4)
        .build();

    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .build();
    content.append(&hbox);
    content.append(&progress_bar);

    // ── Header bar with selection counter ────────────────────────
    let header = adw::HeaderBar::new();

    let sel_count_label = gtk4::Label::new(Some("0 selected"));
    sel_count_label.add_css_class("caption");
    header.pack_end(&sel_count_label);

    let copy_btn = gtk4::Button::builder()
        .label("Copy Selected")
        .sensitive(false)
        .build();
    copy_btn.add_css_class("suggested-action");
    header.pack_start(&copy_btn);

    // ── Copy mode dropdown (what files to include when copying) ──
    let copy_mode_dropdown = gtk4::DropDown::from_strings(&[
        "Copy all (JPG+RAW)",
        "Copy RAW only",
        "Copy JPG only",
    ]);
    copy_mode_dropdown.set_selected(0);
    copy_mode_dropdown.set_tooltip_text(Some("Choose which files to copy for paired JPG+RAW images"));
    header.pack_start(&copy_mode_dropdown);

    let filter_btn = gtk4::ToggleButton::builder()
        .label("Hide seen")
        .tooltip_text("Hide previously imported or rejected images")
        .build();
    header.pack_start(&filter_btn);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let page = adw::NavigationPage::builder()
        .title("Browse Images")
        .child(&toolbar_view)
        .build();

    // ── Keyboard controller ─────────────────────────────────────
    {
        let state = state.clone();
        let preview_picture = preview_picture.clone();
        let preview_scroll = preview_scroll.clone();
        let filename_label = filename_label.clone();
        let selected_label = selected_label.clone();
        let counter_label = counter_label.clone();
        let sel_count_label = sel_count_label.clone();
        let copy_btn = copy_btn.clone();
        let strip_box = strip_box.clone();
        let strip_scroll = strip_scroll.clone();

        let key_ctrl = gtk4::EventControllerKey::new();
        key_ctrl.connect_key_pressed(move |_, keyval, _, _| {
            let mut st = state.borrow_mut();
            let len = st.images.len();
            if len == 0 {
                return glib::Propagation::Proceed;
            }

            match keyval {
                gdk::Key::Up | gdk::Key::Left => {
                    st.zoom = 1.0;
                    let old_idx = st.current;
                    let newly_rejected = st.auto_reject_current();
                    st.go_prev();
                    drop(st);
                    if let Some(hash) = newly_rejected {
                        update_strip_opacity(&strip_box, old_idx, true);
                        let rejected_set = state.borrow().rejected_set.clone();
                        let library_dir = state.borrow().library_dir.clone();
                        std::thread::spawn(move || {
                            let mut set = rejected_set.lock().unwrap();
                            set.insert(&hash);
                            if let Err(e) = set.save_rejected(&library_dir) {
                                tracing::warn!("Failed to save rejected set: {e}");
                            }
                        });
                    }
                    update_preview(
                        &state,
                        &preview_picture,
                        &preview_scroll,
                        &filename_label,
                        &selected_label,
                        &counter_label,
                        &strip_box,
                        &strip_scroll,
                    );
                    glib::Propagation::Stop
                }
                gdk::Key::Down | gdk::Key::Right => {
                    st.zoom = 1.0;
                    let old_idx = st.current;
                    let newly_rejected = st.auto_reject_current();
                    st.go_next();
                    drop(st);
                    if let Some(hash) = newly_rejected {
                        update_strip_opacity(&strip_box, old_idx, true);
                        let rejected_set = state.borrow().rejected_set.clone();
                        let library_dir = state.borrow().library_dir.clone();
                        std::thread::spawn(move || {
                            let mut set = rejected_set.lock().unwrap();
                            set.insert(&hash);
                            if let Err(e) = set.save_rejected(&library_dir) {
                                tracing::warn!("Failed to save rejected set: {e}");
                            }
                        });
                    }
                    update_preview(
                        &state,
                        &preview_picture,
                        &preview_scroll,
                        &filename_label,
                        &selected_label,
                        &counter_label,
                        &strip_box,
                        &strip_scroll,
                    );
                    glib::Propagation::Stop
                }
                gdk::Key::x => {
                    let idx = st.current;
                    let was_rejected = st.images[idx].rejected;
                    if st.selected.contains(&idx) {
                        st.selected.remove(&idx);
                    } else {
                        // Un-reject if the image was previously auto-rejected.
                        if was_rejected {
                            st.images[idx].rejected = false;
                            if st.rejected_count > 0 {
                                st.rejected_count -= 1;
                            }
                        }
                        st.selected.insert(idx);
                    }
                    let sel_count = st.selected.len();
                    drop(st);
                    if was_rejected {
                        update_strip_opacity(&strip_box, idx, false);
                    }
                    sel_count_label.set_label(&format!("{sel_count} selected"));
                    copy_btn.set_sensitive(sel_count > 0);
                    update_preview(
                        &state,
                        &preview_picture,
                        &preview_scroll,
                        &filename_label,
                        &selected_label,
                        &counter_label,
                        &strip_box,
                        &strip_scroll,
                    );
                    glib::Propagation::Stop
                }
                _ => glib::Propagation::Proceed,
            }
        });
        page.child().unwrap().add_controller(key_ctrl);
    }

    // ── Filter toggle ─────────────────────────────────────────────
    {
        let state = state.clone();
        let strip_box = strip_box.clone();
        let preview_picture = preview_picture.clone();
        let preview_scroll = preview_scroll.clone();
        let filename_label = filename_label.clone();
        let selected_label = selected_label.clone();
        let counter_label = counter_label.clone();
        let strip_scroll = strip_scroll.clone();

        filter_btn.connect_toggled(move |btn| {
            let active = btn.is_active();
            {
                let mut st = state.borrow_mut();
                st.filter_seen = active;
                // Ensure current is on a visible image.
                if active && !st.images.is_empty() && !st.is_visible(st.current) {
                    st.go_next();
                }
            }
            update_strip_visibility(&strip_box, &state);
            update_preview(
                &state,
                &preview_picture,
                &preview_scroll,
                &filename_label,
                &selected_label,
                &counter_label,
                &strip_box,
                &strip_scroll,
            );
        });
    }

    // ── Copy mode dropdown handler ────────────────────────────
    {
        let state = state.clone();
        copy_mode_dropdown.connect_selected_notify(move |dd| {
            let mode = match dd.selected() {
                1 => maple_import::CopyMode::RawOnly,
                2 => maple_import::CopyMode::DisplayOnly,
                _ => maple_import::CopyMode::All,
            };
            state.borrow_mut().copy_mode = mode;
        });
    }

    // ── Scroll-to-zoom on preview ───────────────────────────────
    {
        let state = state.clone();
        let preview_scroll_inner = preview_scroll.clone();
        let preview_picture = preview_picture.clone();
        let zoom_ctrl = gtk4::EventControllerScroll::new(
            gtk4::EventControllerScrollFlags::VERTICAL,
        );
        zoom_ctrl.set_propagation_phase(gtk4::PropagationPhase::Capture);
        zoom_ctrl.connect_scroll(move |_, _dx, dy| {
            let factor = if dy < 0.0 { 1.15 } else { 1.0 / 1.15 };
            let mut st = state.borrow_mut();
            st.zoom = (st.zoom * factor).clamp(1.0, 30.0);
            let zoom = st.zoom;
            drop(st);
            apply_zoom(&preview_scroll_inner, &preview_picture, zoom);
            glib::Propagation::Stop
        });
        preview_scroll.add_controller(zoom_ctrl);
    }

    // ── Click-drag to pan the zoomed preview ────────────────────
    {
        let drag_start = Rc::new(Cell::new((0.0_f64, 0.0_f64)));
        let preview_scroll_inner = preview_scroll.clone();
        let drag = gtk4::GestureDrag::new();
        {
            let ps = preview_scroll_inner.clone();
            let ds = drag_start.clone();
            drag.connect_drag_begin(move |_, _, _| {
                let h = ps.hadjustment().value();
                let v = ps.vadjustment().value();
                ds.set((h, v));
            });
        }
        {
            let ps = preview_scroll_inner;
            let ds = drag_start;
            drag.connect_drag_update(move |_, dx, dy| {
                let (start_h, start_v) = ds.get();
                ps.hadjustment().set_value(start_h - dx);
                ps.vadjustment().set_value(start_v - dy);
            });
        }
        preview_scroll.add_controller(drag);
    }

    // ── Copy button handler ─────────────────────────────────────
    {
        let state = state.clone();
        let toast_overlay = toast_overlay.clone();
        let progress_bar = progress_bar.clone();
        let copy_btn_inner = copy_btn.clone();
        let sel_count_label = sel_count_label.clone();
        let strip_box = strip_box.clone();

        let db = db.clone();
        copy_btn.connect_clicked(move |btn| {
            let st = state.borrow();
            if st.selected.is_empty() {
                return;
            }

            // Build the list of files to copy, respecting copy_mode.
            // copy_file_info tracks per-file metadata so the DB insertion
            // knows which dest paths are display files vs raw companions.
            let copy_mode = st.copy_mode;
            let mut sources: Vec<PathBuf> = Vec::new();
            let mut copy_file_info: Vec<CopyFileInfo> = Vec::new();
            let mut copied_hashes: Vec<[u8; 32]> = Vec::new();
            for &i in &st.selected {
                let entry = &st.images[i];
                if entry.path.as_os_str().is_empty() {
                    continue;
                }
                let group = maple_import::ImageGroup {
                    display: maple_import::ImageFile {
                        path: entry.path.clone(),
                        size: 0,
                    },
                    companions: entry
                        .companions
                        .iter()
                        .map(|p| maple_import::ImageFile {
                            path: p.clone(),
                            size: 0,
                        })
                        .collect(),
                };
                let paths = group.paths_for_copy(copy_mode);
                let display_path = &entry.path;
                for p in &paths {
                    copy_file_info.push(CopyFileInfo {
                        display_hash: entry.content_hash,
                        is_display: p == display_path,
                    });
                }
                sources.extend(paths);
                copied_hashes.push(entry.content_hash);
            }

            let destination = st.destination.clone();
            let copied_indices: Vec<usize> = st.selected.iter().copied().collect();
            drop(st);

            // Disable button during copy.
            btn.set_sensitive(false);
            btn.set_label("Copying…");
            progress_bar.set_fraction(0.0);
            progress_bar.set_text(Some("Copying…"));

            let (copy_tx, copy_rx) = mpsc::channel::<CopyMsg>();

            let folder_format = maple_state::Settings::load().folder_format;

            // Background thread for the copy.
            std::thread::spawn(move || {
                let result = maple_import::copy_images(&sources, &destination, &folder_format, |done, total| {
                    let _ = copy_tx.send(CopyMsg::Progress { done, total });
                });
                match result {
                    Ok(summary) => {
                        let _ = copy_tx.send(CopyMsg::Finished {
                            summary,
                            copy_file_info,
                        });
                    }
                    Err(e) => {
                        let _ = copy_tx.send(CopyMsg::Error(e.to_string()));
                    }
                }
            });

            // Poll for copy progress on the UI thread.
            let toast_overlay = toast_overlay.clone();
            let progress_bar = progress_bar.clone();
            let copy_btn_inner = copy_btn_inner.clone();
            let sel_count_label = sel_count_label.clone();
            let state = state.clone();
            let strip_box = strip_box.clone();
            let db = db.clone();
            glib::timeout_add_local(Duration::from_millis(32), move || {
                while let Ok(msg) = copy_rx.try_recv() {
                    match msg {
                        CopyMsg::Progress { done, total } => {
                            if total > 0 {
                                progress_bar.set_fraction(done as f64 / total as f64);
                                progress_bar.set_text(Some(&format!(
                                    "Copying… {done} / {total}"
                                )));
                            }
                        }
                        CopyMsg::Finished { summary, copy_file_info } => {
                            let (copied, failed) = (summary.copied, summary.failed);
                            let msg = if failed == 0 {
                                format!("Copied {copied} image{}", if copied == 1 { "" } else { "s" })
                            } else {
                                format!("Copied {copied}, {failed} failed")
                            };
                            toast_overlay.add_toast(adw::Toast::new(&msg));

                            // Insert successfully copied files into the library DB.
                            // Only display files get a DB row; raw companions are
                            // stored as `raw_path` on the display row.
                            if let Ok(db_guard) = db.lock() {
                                // First pass: collect dest paths for raw companions
                                // keyed by display_hash so we can attach them.
                                let mut raw_dest_paths: HashMap<[u8; 32], PathBuf> = HashMap::new();
                                for (info, result) in
                                    copy_file_info.iter().zip(summary.results.iter())
                                {
                                    if let maple_import::CopyResult::Ok(dest_path) = result {
                                        if !info.is_display {
                                            raw_dest_paths
                                                .insert(info.display_hash, dest_path.clone());
                                        }
                                    }
                                }

                                // Second pass: insert display files with raw_path.
                                for (info, result) in
                                    copy_file_info.iter().zip(summary.results.iter())
                                {
                                    if !info.is_display {
                                        continue;
                                    }
                                    if let maple_import::CopyResult::Ok(dest_path) = result {
                                        let file_size = dest_path
                                            .metadata()
                                            .map(|m| m.len())
                                            .unwrap_or(0);
                                        let raw_path =
                                            raw_dest_paths.get(&info.display_hash);
                                        if let Err(e) = db_guard.insert_image_with_raw(
                                            dest_path,
                                            &info.display_hash,
                                            file_size,
                                            raw_path.map(|p| p.as_path()),
                                        ) {
                                            tracing::warn!(
                                                "Failed to insert {} into library DB: {e}",
                                                dest_path.display()
                                            );
                                        }
                                    }
                                }
                            }

                            // Mark copied images as imported and persist.
                            {
                                let mut st = state.borrow_mut();
                                let mut imported_set = st.imported_set.lock().unwrap();
                                for hash in &copied_hashes {
                                    imported_set.insert(hash);
                                }
                                if let Err(e) = imported_set.save_imported(&st.library_dir) {
                                    tracing::warn!("Failed to save imported set: {e}");
                                }
                                drop(imported_set);
                                for &i in &copied_indices {
                                    if i < st.images.len() && !st.images[i].imported {
                                        st.images[i].imported = true;
                                        st.imported_count += 1;
                                    }
                                }
                                st.selected.clear();
                            }

                            // Update strip opacity for newly-imported images.
                            {
                                let st = state.borrow();
                                for &i in &copied_indices {
                                    if i < st.images.len() {
                                        update_strip_opacity(&strip_box, i, true);
                                    }
                                }
                            }

                            sel_count_label.set_label("0 selected");
                            copy_btn_inner.set_label("Copy Selected");
                            copy_btn_inner.set_sensitive(false);

                            let st = state.borrow();
                            progress_bar.set_fraction(1.0);
                            progress_bar.set_text(Some(&scan_summary_text(&st)));

                            return glib::ControlFlow::Break;
                        }
                        CopyMsg::Error(e) => {
                            toast_overlay.add_toast(adw::Toast::new(&format!(
                                "Copy error: {e}"
                            )));
                            copy_btn_inner.set_label("Copy Selected");
                            copy_btn_inner.set_sensitive(true);

                            let st = state.borrow();
                            progress_bar.set_fraction(1.0);
                            progress_bar.set_text(Some(&scan_summary_text(&st)));

                            return glib::ControlFlow::Break;
                        }
                    }
                }
                glib::ControlFlow::Continue
            });
        });
    }

    // ── Full-res poller: receives raw file bytes from background threads ──
    //
    // Decodes one image per tick using gdk::Texture::from_bytes() which
    // delegates to the system's gdk-pixbuf (libjpeg-turbo for JPEGs).
    // Processing one-at-a-time limits UI-thread blocking to ~50-100 ms
    // per decode while still being much faster overall.
    {
        let state = state.clone();
        let preview_picture = preview_picture.clone();
        let preview_scroll = preview_scroll.clone();
        glib::timeout_add_local(Duration::from_millis(16), move || {
            // Process at most one image per tick to keep UI responsive.
            if let Ok(msg) = fullres_rx.try_recv() {
                let stream = gtk4::gio::MemoryInputStream::from_bytes(&glib::Bytes::from(&msg.file_bytes));
                let pixbuf = match gtk4::gdk_pixbuf::Pixbuf::from_stream(
                    &stream,
                    gtk4::gio::Cancellable::NONE,
                ) {
                    Ok(pb) => pb,
                    Err(e) => {
                        tracing::warn!("Full-res decode failed for index {}: {e}", msg.index);
                        let mut st = state.borrow_mut();
                        st.fullres_loading.remove(&msg.index);
                        return glib::ControlFlow::Continue;
                    }
                };
                // Apply EXIF orientation (rotation/flip).
                let pixbuf = pixbuf
                    .apply_embedded_orientation()
                    .unwrap_or(pixbuf);
                let texture = gdk::Texture::for_pixbuf(&pixbuf);

                let mut st = state.borrow_mut();
                st.fullres_loading.remove(&msg.index);

                // Evict entries outside the current buffer window.
                let (buf_start, buf_end) =
                    compute_buffer_window(st.current, st.images.len(), st.buffer_size);
                st.fullres_cache.retain(|idx, _| *idx >= buf_start && *idx < buf_end);

                // Store in cache if still in window.
                if msg.index >= buf_start && msg.index < buf_end {
                    st.fullres_cache.insert(msg.index, texture.clone());
                }

                // If this is the current image, show it.
                if msg.index == st.current {
                    let zoom = st.zoom;
                    drop(st);
                    preview_picture.set_paintable(Some(&texture));
                    apply_zoom(&preview_scroll, &preview_picture, zoom);
                }
            }
            glib::ControlFlow::Continue
        });
    }

    // ── Kick off background scan ────────────────────────────────
    start_scan(
        source,
        &state,
        &preview_picture,
        &preview_scroll,
        &filename_label,
        &selected_label,
        &counter_label,
        &strip_box,
        &strip_scroll,
        &progress_bar,
        toast_overlay,
    );

    page
}
