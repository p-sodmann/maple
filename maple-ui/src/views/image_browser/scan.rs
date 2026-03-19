//! Background scan — walks source directory, generates thumbnails, reports progress.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;

use crate::thumbnail::generate_thumbnail;

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
