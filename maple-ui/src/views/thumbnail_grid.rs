//! Thumbnail grid view — displays scanned images as a scrollable grid.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;

use crate::thumbnail::generate_thumbnail;

// ── Messages from the worker thread ─────────────────────────────

enum ScanMsg {
    /// Total number of images found by the scanner.
    Count(usize),
    /// A single thumbnail was generated (with its original index for ordering).
    Thumb {
        index: usize,
        path: PathBuf,
        png_bytes: Vec<u8>,
    },
    /// All thumbnails have been generated.
    Done,
    /// An error occurred during scanning.
    Error(String),
}

// ── Public API ──────────────────────────────────────────────────

/// Build the thumbnail grid page and immediately start scanning `source`.
pub fn build_grid_page(
    source: &Path,
    _destination: &Path,
    toast_overlay: &adw::ToastOverlay,
) -> adw::NavigationPage {
    // ── FlowBox ─────────────────────────────────────────────────
    let flow_box = gtk4::FlowBox::builder()
        .valign(gtk4::Align::Start)
        .max_children_per_line(30)
        .min_children_per_line(2)
        .selection_mode(gtk4::SelectionMode::None)
        .homogeneous(true)
        .row_spacing(8)
        .column_spacing(8)
        .margin_start(12)
        .margin_end(12)
        .margin_top(12)
        .margin_bottom(12)
        .build();

    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .build();
    scrolled.set_child(Some(&flow_box));

    // ── Progress bar ────────────────────────────────────────────
    let progress_bar = gtk4::ProgressBar::builder()
        .show_text(true)
        .text("Scanning…")
        .margin_start(12)
        .margin_end(12)
        .margin_top(8)
        .margin_bottom(4)
        .build();

    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .build();
    content.append(&progress_bar);
    content.append(&scrolled);

    // ── Header ──────────────────────────────────────────────────
    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let page = adw::NavigationPage::builder()
        .title("Scan Results")
        .child(&toolbar_view)
        .build();

    // ── Kick off background scan ────────────────────────────────
    start_scan(source, &flow_box, &progress_bar, toast_overlay);

    page
}

// ── Background scan + thumbnail generation ──────────────────────

fn start_scan(
    source: &Path,
    flow_box: &gtk4::FlowBox,
    progress_bar: &gtk4::ProgressBar,
    toast_overlay: &adw::ToastOverlay,
) {
    let (sender, receiver) = mpsc::channel::<ScanMsg>();
    let source = source.to_path_buf();

    // Worker thread: scan then generate thumbnails in parallel
    std::thread::spawn(move || {
        match maple_import::scan_images(&source) {
            Ok(images) => {
                let total = images.len();
                let _ = sender.send(ScanMsg::Count(total));

                let parallelism = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4);

                // Pair each image with its original index for ordered display
                let indexed: Vec<_> = images.into_iter().enumerate()
                    .map(|(i, img)| (img, i))
                    .collect();

                std::thread::scope(|scope| {
                    // Fan out across worker threads
                    let chunk_size = (indexed.len() / parallelism).max(1);
                    let sender = &sender;

                    for chunk in indexed.chunks(chunk_size) {
                        scope.spawn(move || {
                            for (img, idx) in chunk {
                                match generate_thumbnail(&img.path, 256) {
                                    Ok(bytes) => {
                                        let _ = sender.send(ScanMsg::Thumb {
                                            index: *idx,
                                            path: img.path.clone(),
                                            png_bytes: bytes,
                                        });
                                    }
                                    Err(e) => {
                                        tracing::warn!(
                                            "Thumbnail failed for {}: {e}",
                                            img.path.display()
                                        );
                                    }
                                }
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

    // UI-thread receiver — replaces placeholders in-place to preserve order.
    let flow_box = flow_box.clone();
    let progress_bar = progress_bar.clone();
    let toast_overlay = toast_overlay.clone();
    let mut generated = 0usize;
    let mut total = 0usize;

    glib::timeout_add_local(Duration::from_millis(32), move || {
        // Drain all pending messages from the worker thread.
        while let Ok(msg) = receiver.try_recv() {
            match msg {
                ScanMsg::Count(n) => {
                    total = n;
                    if total == 0 {
                        progress_bar.set_fraction(1.0);
                        progress_bar.set_text(Some("No images found"));
                        return glib::ControlFlow::Break;
                    }
                    // Pre-populate the grid with placeholder cards.
                    for _ in 0..total {
                        flow_box.append(&build_placeholder_card());
                    }
                    progress_bar.set_text(Some(&format!(
                        "Generating thumbnails… 0 / {total}"
                    )));
                }

                ScanMsg::Thumb { index, path, png_bytes } => {
                    generated += 1;
                    let frac = if total > 0 {
                        generated as f64 / total as f64
                    } else {
                        0.0
                    };
                    progress_bar.set_fraction(frac);
                    progress_bar.set_text(Some(&format!(
                        "Generating thumbnails… {generated} / {total}"
                    )));

                    // Replace the placeholder at `index` with the real thumbnail.
                    let bytes = glib::Bytes::from(&png_bytes);
                    match gdk::Texture::from_bytes(&bytes) {
                        Ok(texture) => {
                            if let Some(child) = flow_box.child_at_index(index as i32) {
                                let card = build_image_card(&texture, &path);
                                child.set_child(Some(&card));
                            }
                        }
                        Err(e) => {
                            tracing::warn!("Texture creation failed: {e}");
                        }
                    }
                }

                ScanMsg::Done => {
                    progress_bar.set_fraction(1.0);
                    progress_bar.set_text(Some(&format!("{generated} images")));
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

// ── Placeholder card ────────────────────────────────────────────

/// Build a placeholder card shown while the thumbnail is being generated.
fn build_placeholder_card() -> gtk4::Box {
    let spinner = gtk4::Spinner::builder()
        .spinning(true)
        .width_request(32)
        .height_request(32)
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .hexpand(true)
        .vexpand(true)
        .build();
    // Slow down the spin animation via CSS
    spinner.add_css_class("maple-slow-spinner");

    let frame = gtk4::Box::builder()
        .width_request(180)
        .height_request(180)
        .halign(gtk4::Align::Fill)
        .valign(gtk4::Align::Fill)
        .hexpand(true)
        .vexpand(true)
        .build();
    frame.append(&spinner);

    let label = gtk4::Label::new(Some("Loading…"));
    label.add_css_class("caption");
    label.add_css_class("dim-label");
    label.set_halign(gtk4::Align::Center);

    let card = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .build();
    card.append(&frame);
    card.append(&label);

    card
}

// ── Image card widget ───────────────────────────────────────────

fn build_image_card(texture: &gdk::Texture, path: &Path) -> gtk4::Box {
    let picture = gtk4::Picture::for_paintable(texture);
    picture.set_size_request(180, 180);
    picture.set_content_fit(gtk4::ContentFit::Contain);

    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("?");

    let label = gtk4::Label::new(Some(filename));
    label.set_ellipsize(gtk4::pango::EllipsizeMode::Middle);
    label.set_max_width_chars(18);
    label.add_css_class("caption");

    let card = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .build();
    card.append(&picture);
    card.append(&label);

    card
}
