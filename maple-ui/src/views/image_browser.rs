//! Image browser view — large preview + filmstrip + keyboard selection.
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

use maple_import::loadable_image_bytes;

use crate::thumbnail::generate_thumbnail;

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

// ── Summary helpers ──────────────────────────────────────────────

/// Format the progress-bar summary line shown after scanning finishes.
fn scan_summary_text(st: &BrowserState) -> String {
    let seen_total = st.imported_count + st.rejected_count;
    if seen_total > 0 {
        format!(
            "{} images ({} imported, {} skipped)",
            st.generated, st.imported_count, st.rejected_count
        )
    } else {
        format!("{} images", st.generated)
    }
}

// ── Buffer window computation ───────────────────────────────────

/// Compute the `[start, end)` range of indices to buffer around `current`.
///
/// The window has `buffer_size` elements centred on `current`, with
/// `(buffer_size - 1) / 2` images before and after.  When `current` is
/// close to the start or end the window shifts so we always buffer up to
/// `buffer_size` images (or `total` if fewer exist).
fn compute_buffer_window(current: usize, total: usize, buffer_size: usize) -> (usize, usize) {
    if total == 0 || buffer_size == 0 {
        return (0, 0);
    }
    let n = buffer_size.min(total);
    let half = (n - 1) / 2;
    let start = if current <= half {
        0
    } else if current + n - half > total {
        total.saturating_sub(n)
    } else {
        current - half
    };
    let end = (start + n).min(total);
    (start, end)
}

// ── Update the preview panel to reflect current state ───────────

fn update_preview(
    state: &Rc<RefCell<BrowserState>>,
    preview: &gtk4::Picture,
    preview_scroll: &gtk4::ScrolledWindow,
    filename_label: &gtk4::Label,
    selected_label: &gtk4::Label,
    counter_label: &gtk4::Label,
    strip_box: &gtk4::Box,
    strip_scroll: &gtk4::ScrolledWindow,
) {
    let mut st = state.borrow_mut();
    let len = st.images.len();
    if len == 0 {
        return;
    }
    let idx = st.current;
    let zoom = st.zoom;
    let is_selected = st.selected.contains(&idx);
    let imported = st.images[idx].imported;
    let rejected = st.images[idx].rejected;

    let filename = st.images[idx]
        .path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?")
        .to_owned();

    // ── Show the best available texture for the current image ────
    if let Some(tex) = st.fullres_cache.get(&idx) {
        // Full-res already cached — use it immediately.
        preview.set_paintable(Some(tex));
    } else if let Some(ref tex) = st.images[idx].texture {
        // Fall back to the thumbnail while full-res loads.
        preview.set_paintable(Some(tex));
    }

    // ── Compute buffer window & preload ─────────────────────────
    let buffer_size = st.buffer_size;
    let (buf_start, buf_end) = compute_buffer_window(idx, len, buffer_size);

    // Evict out-of-window entries.
    st.fullres_cache.retain(|k, _| *k >= buf_start && *k < buf_end);

    // Collect indices that need loading (have a known path, not cached, not in-flight).
    let to_load: Vec<(usize, PathBuf)> = (buf_start..buf_end)
        .filter(|i| {
            !st.fullres_cache.contains_key(i)
                && !st.fullres_loading.contains(i)
                && !st.images[*i].path.as_os_str().is_empty()
        })
        .map(|i| (i, st.images[i].path.clone()))
        .collect();

    let fullres_tx = st.fullres_tx.clone();
    for &(i, _) in &to_load {
        st.fullres_loading.insert(i);
    }

    drop(st); // release borrow before spawning threads

    // Spawn a background thread per image that needs loading.
    // Each thread just reads the raw file bytes; decoding happens on
    // the UI thread via gdk::Texture::from_bytes() (libjpeg-turbo).
    for (i, path) in to_load {
        let tx = fullres_tx.clone();
        std::thread::spawn(move || {
            match loadable_image_bytes(&path) {
                Ok(file_bytes) => {
                    if let Ok(tx) = tx.lock() {
                        let _ = tx.send(FullResMsg {
                            index: i,
                            file_bytes,
                        });
                    }
                }
                Err(e) => {
                    tracing::warn!("Full-res read failed for {}: {e}", path.display());
                }
            }
        });
    }

    // Apply current zoom level
    apply_zoom(preview_scroll, preview, zoom);

    // Filename
    filename_label.set_label(&filename);

    // Selected indicator
    if is_selected {
        selected_label.set_label("✓ Selected for import");
        selected_label.remove_css_class("dim-label");
        selected_label.add_css_class("success");
    } else {
        selected_label.set_label("Press X to select");
        selected_label.remove_css_class("success");
        selected_label.add_css_class("dim-label");
    }

    // Counter + seen indicator
    if imported {
        counter_label.set_label(&format!("{} / {}  ·  Previously imported", idx + 1, len));
        counter_label.add_css_class("warning");
        counter_label.remove_css_class("error");
    } else if rejected {
        counter_label.set_label(&format!("{} / {}  ·  Skipped", idx + 1, len));
        counter_label.add_css_class("error");
        counter_label.remove_css_class("warning");
    } else {
        counter_label.set_label(&format!("{} / {}", idx + 1, len));
        counter_label.remove_css_class("warning");
        counter_label.remove_css_class("error");
    }

    // Highlight current thumbnail in strip
    update_strip_highlight(strip_box, idx);

    // Scroll to make current thumbnail visible
    scroll_strip_to(strip_scroll, strip_box, idx);
}

// ── Zoom helpers ────────────────────────────────────────────────

/// Apply the current zoom level to the preview picture and its scroll window.
fn apply_zoom(
    preview_scroll: &gtk4::ScrolledWindow,
    preview: &gtk4::Picture,
    zoom: f64,
) {
    if zoom <= 1.0 {
        // Fit-to-viewport mode
        preview.set_content_fit(gtk4::ContentFit::Contain);
        preview.set_can_shrink(true);
        preview.set_size_request(-1, -1);
        preview_scroll.set_hscrollbar_policy(gtk4::PolicyType::Never);
        preview_scroll.set_vscrollbar_policy(gtk4::PolicyType::Never);
    } else {
        // Zoomed mode – size the picture larger than the viewport
        let Some(paintable) = preview.paintable() else {
            return;
        };
        let tex_w = paintable.intrinsic_width() as f64;
        let tex_h = paintable.intrinsic_height() as f64;
        if tex_w <= 0.0 || tex_h <= 0.0 {
            return;
        }
        let viewport_w = preview_scroll.width() as f64;
        let viewport_h = preview_scroll.height() as f64;
        if viewport_w <= 0.0 || viewport_h <= 0.0 {
            return;
        }

        // "Fit" scale then multiply by zoom
        let scale = (viewport_w / tex_w).min(viewport_h / tex_h);
        let zoomed_w = (tex_w * scale * zoom) as i32;
        let zoomed_h = (tex_h * scale * zoom) as i32;

        preview.set_content_fit(gtk4::ContentFit::Contain);
        preview.set_can_shrink(false);
        preview.set_size_request(zoomed_w, zoomed_h);
        preview_scroll.set_hscrollbar_policy(gtk4::PolicyType::Automatic);
        preview_scroll.set_vscrollbar_policy(gtk4::PolicyType::Automatic);
    }
}

fn update_strip_highlight(strip_box: &gtk4::Box, current: usize) {
    let mut i = 0;
    let mut child = strip_box.first_child();
    while let Some(widget) = child {
        if i == current {
            widget.add_css_class("maple-strip-active");
        } else {
            widget.remove_css_class("maple-strip-active");
        }
        child = widget.next_sibling();
        i += 1;
    }
}

fn scroll_strip_to(
    strip_scroll: &gtk4::ScrolledWindow,
    strip_box: &gtk4::Box,
    index: usize,
) {
    // Find the child widget at `index` and scroll it into view.
    let mut i = 0;
    let mut child = strip_box.first_child();
    while let Some(widget) = child {
        if i == index {
            // Compute the widget's Y position relative to the strip_box
            let point = gtk4::graphene::Point::new(0.0, 0.0);
            if let Some(pos) = widget.compute_point(strip_box, &point) {
                let y = pos.y() as f64;
                let vadj = strip_scroll.vadjustment();
                let widget_height = widget.height() as f64;
                let page_size = vadj.page_size();
                let current_val = vadj.value();

                // Scroll only if the widget is not fully visible
                if y < current_val {
                    vadj.set_value(y);
                } else if y + widget_height > current_val + page_size {
                    vadj.set_value(y + widget_height - page_size);
                }
            }
            return;
        }
        child = widget.next_sibling();
        i += 1;
    }
}

// ── Background scan ─────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn start_scan(
    source: &Path,
    state: &Rc<RefCell<BrowserState>>,
    preview: &gtk4::Picture,
    preview_scroll: &gtk4::ScrolledWindow,
    filename_label: &gtk4::Label,
    selected_label: &gtk4::Label,
    counter_label: &gtk4::Label,
    strip_box: &gtk4::Box,
    strip_scroll: &gtk4::ScrolledWindow,
    progress_bar: &gtk4::ProgressBar,
    toast_overlay: &adw::ToastOverlay,
) {
    let (sender, receiver) = mpsc::channel::<ScanMsg>();
    let source = source.to_path_buf();
    let imported_set = state.borrow().imported_set.clone();
    let rejected_set = state.borrow().rejected_set.clone();

    // Worker thread: scan, generate thumbnails, and check seen status.
    // Uses scan_grouped so that JPG+RAF pairs are shown as a single entry.
    std::thread::spawn(move || {
        match maple_import::scan_grouped(&source) {
            Ok(groups) => {
                let total = groups.len();
                let _ = sender.send(ScanMsg::Count(total));

                let parallelism = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);

                std::thread::scope(|scope| {
                    let sender = &sender;
                    let imported_set = &imported_set;
                    let rejected_set = &rejected_set;
                    let groups = &groups;

                    // Round-robin: thread T processes groups T, T+P, T+2P, …
                    // so all threads start near the beginning of the list and
                    // thumbnails arrive roughly in order.
                    for thread_id in 0..parallelism {
                        scope.spawn(move || {
                            let mut idx = thread_id;
                            while idx < groups.len() {
                                let group = &groups[idx];
                                let display_path = &group.display.path;
                                match generate_thumbnail(display_path, THUMB_SIZE) {
                                    Ok(bytes) => {
                                        let (content_hash, imported, rejected) =
                                            match maple_import::content_hash(display_path) {
                                                Ok(hash) => {
                                                    let imported = imported_set
                                                        .lock()
                                                        .unwrap()
                                                        .probably_contains(&hash);
                                                    let rejected = rejected_set
                                                        .lock()
                                                        .unwrap()
                                                        .probably_contains(&hash);
                                                    (hash, imported, rejected)
                                                }
                                                Err(_) => ([0u8; 32], false, false),
                                            };
                                        let _ = sender.send(ScanMsg::Thumb {
                                            index: idx,
                                            path: display_path.clone(),
                                            companions: group
                                                .companions
                                                .iter()
                                                .map(|c| c.path.clone())
                                                .collect(),
                                            png_bytes: bytes,
                                            content_hash,
                                            imported,
                                            rejected,
                                        });
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Thumbnail failed for {}: {e}",
                                            display_path.display()
                                        );
                                    }
                                }
                                idx += parallelism;
                            }
                        });
                    }
                });

                let _ = sender.send(ScanMsg::Done);
            }
            Err(e) => {
                let _ = sender.send(ScanMsg::Error(e.to_string()));
            }
        }
    });

    // UI-thread receiver
    let state = state.clone();
    let preview = preview.clone();
    let preview_scroll = preview_scroll.clone();
    let filename_label = filename_label.clone();
    let selected_label = selected_label.clone();
    let counter_label = counter_label.clone();
    let strip_box = strip_box.clone();
    let strip_scroll = strip_scroll.clone();
    let progress_bar = progress_bar.clone();
    let toast_overlay = toast_overlay.clone();

    glib::timeout_add_local(Duration::from_millis(32), move || {
        while let Ok(msg) = receiver.try_recv() {
            match msg {
                ScanMsg::Count(n) => {
                    let mut st = state.borrow_mut();
                    st.total = n;
                    if n == 0 {
                        progress_bar.set_fraction(1.0);
                        progress_bar.set_text(Some("No images found"));
                        return glib::ControlFlow::Break;
                    }
                    // Pre-populate image entries (no texture yet)
                    // and add placeholder thumbnails to the strip.
                    for _ in 0..n {
                        st.images.push(ImageEntry {
                            path: PathBuf::new(),
                            companions: Vec::new(),
                            texture: None,
                            content_hash: [0u8; 32],
                            imported: false,
                            rejected: false,
                        });
                        strip_box.append(&build_strip_placeholder());
                    }
                    progress_bar.set_text(Some(&format!(
                        "Generating thumbnails… 0 / {n}"
                    )));
                }

                ScanMsg::Thumb {
                    index,
                    path,
                    companions,
                    png_bytes,
                    content_hash,
                    imported,
                    rejected,
                } => {
                    let bytes = glib::Bytes::from(&png_bytes);
                    let texture = match gdk::Texture::from_bytes(&bytes) {
                        Ok(t) => t,
                        Err(e) => {
                            tracing::warn!("Texture failed: {e}");
                            continue;
                        }
                    };

                    // Store in state
                    {
                        let mut st = state.borrow_mut();
                        if index < st.images.len() {
                            st.images[index] = ImageEntry {
                                path: path.clone(),
                                companions,
                                texture: Some(texture.clone()),
                                content_hash,
                                imported,
                                rejected,
                            };
                        }
                        st.generated += 1;
                        if imported { st.imported_count += 1; }
                        if rejected { st.rejected_count += 1; }
                        let frac = if st.total > 0 {
                            st.generated as f64 / st.total as f64
                        } else {
                            0.0
                        };
                        progress_bar.set_fraction(frac);
                        let seen_total = st.imported_count + st.rejected_count;
                        if seen_total > 0 {
                            progress_bar.set_text(Some(&format!(
                                "Generating thumbnails… {} / {} ({} seen)",
                                st.generated, st.total, seen_total
                            )));
                        } else {
                            progress_bar.set_text(Some(&format!(
                                "Generating thumbnails… {} / {}",
                                st.generated, st.total
                            )));
                        }
                    }

                    // Replace strip placeholder (dim if imported or rejected)
                    replace_strip_thumb(&strip_box, index, &texture, &path, imported, rejected);

                    // If this is the current image, update the preview
                    let cur = state.borrow().current;
                    if index == cur {
                        update_preview(
                            &state,
                            &preview,
                            &preview_scroll,
                            &filename_label,
                            &selected_label,
                            &counter_label,
                            &strip_box,
                            &strip_scroll,
                        );
                    }
                }

                ScanMsg::Done => {
                    let st = state.borrow();
                    progress_bar.set_fraction(1.0);
                    progress_bar.set_text(Some(&scan_summary_text(&st)));
                    drop(st);

                    // Show first image
                    update_preview(
                        &state,
                        &preview,
                        &preview_scroll,
                        &filename_label,
                        &selected_label,
                        &counter_label,
                        &strip_box,
                        &strip_scroll,
                    );
                    return glib::ControlFlow::Break;
                }

                ScanMsg::Error(e) => {
                    toast_overlay.add_toast(adw::Toast::new(&format!("Scan error: {e}")));
                    return glib::ControlFlow::Break;
                }
            }
        }

        glib::ControlFlow::Continue
    });
}



// ── Strip widgets ───────────────────────────────────────────────

fn build_strip_placeholder() -> gtk4::Box {
    let spinner = gtk4::Spinner::builder()
        .spinning(true)
        .width_request(24)
        .height_request(24)
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .hexpand(true)
        .vexpand(true)
        .build();
    spinner.add_css_class("maple-slow-spinner");

    let card = gtk4::Box::builder()
        .width_request(STRIP_THUMB_PX)
        .height_request(STRIP_THUMB_PX)
        .halign(gtk4::Align::Center)
        .css_classes(["maple-strip-thumb"])
        .build();
    card.append(&spinner);

    card
}

fn build_strip_thumb(
    texture: &gdk::Texture,
    path: &Path,
    imported: bool,
    rejected: bool,
) -> gtk4::Box {
    let picture = gtk4::Picture::for_paintable(texture);
    picture.set_size_request(STRIP_THUMB_PX, STRIP_THUMB_PX);
    picture.set_content_fit(gtk4::ContentFit::Contain);

    let name = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");

    let card = gtk4::Box::builder()
        .halign(gtk4::Align::Center)
        .css_classes(["maple-strip-thumb"])
        .build();

    if imported {
        card.set_opacity(0.45);
        card.set_tooltip_text(Some(&format!("{name} (previously imported)")));
    } else if rejected {
        card.set_opacity(0.35);
        card.set_tooltip_text(Some(&format!("{name} (skipped)")));
    } else {
        card.set_tooltip_text(Some(name));
    }

    card.append(&picture);

    card
}

fn replace_strip_thumb(
    strip_box: &gtk4::Box,
    index: usize,
    texture: &gdk::Texture,
    path: &Path,
    imported: bool,
    rejected: bool,
) {
    let mut i = 0;
    let mut child = strip_box.first_child();
    while let Some(widget) = child {
        if i == index {
            let new_thumb = build_strip_thumb(texture, path, imported, rejected);
            // Copy active highlight if present
            if widget.has_css_class("maple-strip-active") {
                new_thumb.add_css_class("maple-strip-active");
            }
            strip_box.insert_child_after(&new_thumb, Some(&widget));
            strip_box.remove(&widget);
            return;
        }
        child = widget.next_sibling();
        i += 1;
    }
}

/// Update the opacity of a strip thumbnail after its status changes.
fn update_strip_opacity(strip_box: &gtk4::Box, index: usize, seen: bool) {
    let mut i = 0;
    let mut child = strip_box.first_child();
    while let Some(widget) = child {
        if i == index {
            widget.set_opacity(if seen { 0.45 } else { 1.0 });
            return;
        }
        child = widget.next_sibling();
        i += 1;
    }
}

/// Show/hide strip thumbnails based on the current filter state.
fn update_strip_visibility(strip_box: &gtk4::Box, state: &Rc<RefCell<BrowserState>>) {
    let st = state.borrow();
    let mut i = 0;
    let mut child = strip_box.first_child();
    while let Some(widget) = child {
        if i < st.images.len() {
            widget.set_visible(st.is_visible(i));
        }
        child = widget.next_sibling();
        i += 1;
    }
}
