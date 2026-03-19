//! Preview panel helpers — update preview, zoom, buffer window computation.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use gtk4::prelude::*;

use maple_import::loadable_image_bytes;

use super::filmstrip::{scroll_strip_to, update_strip_highlight};
use super::{BrowserState, FullResMsg};

/// Compute the `[start, end)` range of indices to buffer around `current`.
///
/// The window has `buffer_size` elements centred on `current`, with
/// `(buffer_size - 1) / 2` images before and after.  When `current` is
/// close to the start or end the window shifts so we always buffer up to
/// `buffer_size` images (or `total` if fewer exist).
pub(super) fn compute_buffer_window(current: usize, total: usize, buffer_size: usize) -> (usize, usize) {
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

pub(super) fn update_preview(
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
        selected_label.set_label("\u{2713} Selected for import");
        selected_label.remove_css_class("dim-label");
        selected_label.add_css_class("success");
    } else {
        selected_label.set_label("Press X to select");
        selected_label.remove_css_class("success");
        selected_label.add_css_class("dim-label");
    }

    // Counter + seen indicator
    if imported {
        counter_label.set_label(&format!("{} / {}  \u{b7}  Previously imported", idx + 1, len));
        counter_label.add_css_class("warning");
        counter_label.remove_css_class("error");
    } else if rejected {
        counter_label.set_label(&format!("{} / {}  \u{b7}  Skipped", idx + 1, len));
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

/// Apply the current zoom level to the preview picture and its scroll window.
pub(super) fn apply_zoom(
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
