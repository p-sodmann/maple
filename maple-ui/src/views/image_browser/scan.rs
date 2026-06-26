//! Background scan — walks source directory (or processes a dropped file list),
//! generates thumbnails, and reports progress via `ScanMsg`.

use std::cell::RefCell;
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;

use super::filmstrip::{build_strip_placeholder, replace_strip_thumb};
use super::preview::update_preview;
use super::{BrowserState, ImageEntry, ScanMsg, THUMB_SIZE};

/// Format the progress-bar summary line shown after scanning finishes.
pub(super) fn scan_summary_text(st: &BrowserState) -> String {
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

// ── Directory scan ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
pub(super) fn start_scan(
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

                    for thread_id in 0..parallelism {
                        scope.spawn(move || {
                            let mut idx = thread_id;
                            while idx < groups.len() {
                                let group = &groups[idx];
                                let display_path = &group.display.path;
                                match crate::thumbnail::render_to_rgb(display_path, THUMB_SIZE) {
                                    Ok((rgb, width, height)) => {
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
                                            rgb,
                                            width,
                                            height,
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

    start_scan_poller(
        receiver,
        state.clone(),
        preview.clone(),
        preview_scroll.clone(),
        filename_label.clone(),
        selected_label.clone(),
        counter_label.clone(),
        strip_box.clone(),
        strip_scroll.clone(),
        progress_bar.clone(),
        toast_overlay.clone(),
    );
}

// ── File-list scan (drag-and-drop) ───────────────────────────────

/// Like `start_scan` but operates on an explicit list of files instead of
/// walking a source directory.  Files are grouped so that JPG+RAW pairs with
/// matching stems appear as a single entry.
#[allow(clippy::too_many_arguments)]
pub(super) fn start_scan_from_files(
    files: Vec<PathBuf>,
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
    let imported_set = state.borrow().imported_set.clone();
    let rejected_set = state.borrow().rejected_set.clone();

    std::thread::spawn(move || {
        let groups = group_dropped_files(files);
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

            for thread_id in 0..parallelism {
                scope.spawn(move || {
                    let mut idx = thread_id;
                    while idx < groups.len() {
                        let (display_path, companions) = &groups[idx];
                        match crate::thumbnail::render_to_rgb(display_path, THUMB_SIZE) {
                            Ok((rgb, width, height)) => {
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
                                    companions: companions.clone(),
                                    rgb,
                                    width,
                                    height,
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
    });

    start_scan_poller(
        receiver,
        state.clone(),
        preview.clone(),
        preview_scroll.clone(),
        filename_label.clone(),
        selected_label.clone(),
        counter_label.clone(),
        strip_box.clone(),
        strip_scroll.clone(),
        progress_bar.clone(),
        toast_overlay.clone(),
    );
}

/// Group a flat file list into (display, companions) pairs.
///
/// Non-raw files are displays; raw files with a matching stem become
/// companions of the display.  Orphan raw files (no matching display) get
/// their own entry.
fn group_dropped_files(files: Vec<PathBuf>) -> Vec<(PathBuf, Vec<PathBuf>)> {
    let mut displays: Vec<PathBuf> = Vec::new();
    let mut raws: Vec<PathBuf> = Vec::new();

    for path in files {
        if maple_import::is_raw_format(&path) {
            raws.push(path);
        } else {
            displays.push(path);
        }
    }

    let mut used_raws: HashSet<usize> = HashSet::new();
    let mut groups: Vec<(PathBuf, Vec<PathBuf>)> = Vec::new();

    for display in &displays {
        let stem = display.file_stem().unwrap_or_default();
        let companions: Vec<PathBuf> = raws
            .iter()
            .enumerate()
            .filter_map(|(i, raw)| {
                if !used_raws.contains(&i) && raw.file_stem().unwrap_or_default() == stem {
                    used_raws.insert(i);
                    Some(raw.clone())
                } else {
                    None
                }
            })
            .collect();
        groups.push((display.clone(), companions));
    }

    for (i, raw) in raws.iter().enumerate() {
        if !used_raws.contains(&i) {
            groups.push((raw.clone(), vec![]));
        }
    }

    groups
}

// ── Shared UI poller ─────────────────────────────────────────────

/// Register the glib timeout poller that drains `receiver` and updates all
/// scan-related widgets.  Shared by both `start_scan` and `start_scan_from_files`.
#[allow(clippy::too_many_arguments)]
fn start_scan_poller(
    receiver: mpsc::Receiver<ScanMsg>,
    state: Rc<RefCell<BrowserState>>,
    preview: gtk4::Picture,
    preview_scroll: gtk4::ScrolledWindow,
    filename_label: gtk4::Label,
    selected_label: gtk4::Label,
    counter_label: gtk4::Label,
    strip_box: gtk4::Box,
    strip_scroll: gtk4::ScrolledWindow,
    progress_bar: gtk4::ProgressBar,
    toast_overlay: adw::ToastOverlay,
) {
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
                    rgb,
                    width,
                    height,
                    content_hash,
                    imported,
                    rejected,
                } => {
                    let bytes = glib::Bytes::from(&rgb);
                    let pixbuf = gtk4::gdk_pixbuf::Pixbuf::from_bytes(
                        &bytes,
                        gtk4::gdk_pixbuf::Colorspace::Rgb,
                        false,
                        8,
                        width as i32,
                        height as i32,
                        (width * 3) as i32,
                    );
                    let texture = gdk::Texture::for_pixbuf(&pixbuf);

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
                        if imported {
                            st.imported_count += 1;
                        }
                        if rejected {
                            st.rejected_count += 1;
                        }
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

                    replace_strip_thumb(&strip_box, index, &texture, &path, imported, rejected);

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
