//! Filmstrip strip widget helpers — placeholders, thumbnails, highlighting, scrolling.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use gtk4::gdk;
use gtk4::prelude::*;

use super::{BrowserState, STRIP_THUMB_PX};

pub(super) fn build_strip_placeholder() -> gtk4::Box {
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

pub(super) fn build_strip_thumb(
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

pub(super) fn replace_strip_thumb(
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
pub(super) fn update_strip_opacity(strip_box: &gtk4::Box, index: usize, seen: bool) {
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

pub(super) fn update_strip_highlight(strip_box: &gtk4::Box, current: usize) {
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

pub(super) fn scroll_strip_to(
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

/// Show/hide strip thumbnails based on the current filter state.
pub(super) fn update_strip_visibility(strip_box: &gtk4::Box, state: &Rc<RefCell<BrowserState>>) {
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
