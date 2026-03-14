//! Zoom and pan gesture wiring for the detail picture widget.

use std::cell::Cell;
use std::rc::Rc;

use gtk4::glib;
use gtk4::prelude::*;

const ZOOM_STEP: f64 = 1.15;
const ZOOM_MAX: f64  = 20.0;

/// Wire scroll-wheel zoom and left-button-drag pan onto `scrolled`.
pub(super) fn wire_zoom_and_pan(
    scrolled: &gtk4::ScrolledWindow,
    picture: &gtk4::Picture,
    zoom: &Rc<Cell<f64>>,
    img_dims: &Rc<Cell<(i32, i32)>>,
) {
    // Track pointer position (widget-local coords) for pointer-anchored zoom.
    let pointer: Rc<Cell<(f64, f64)>> = Rc::new(Cell::new((0.0, 0.0)));
    let motion = gtk4::EventControllerMotion::new();
    motion.connect_motion({
        let pointer = pointer.clone();
        move |_, x, y| pointer.set((x, y))
    });
    scrolled.add_controller(motion);

    // Scroll wheel → zoom
    let scroll_ctrl =
        gtk4::EventControllerScroll::new(gtk4::EventControllerScrollFlags::VERTICAL);
    scroll_ctrl.connect_scroll({
        let picture = picture.clone();
        let scrolled = scrolled.clone();
        let zoom = zoom.clone();
        let img_dims = img_dims.clone();
        let pointer = pointer.clone();
        move |_, _dx, dy| {
            let old = zoom.get();
            let next = if dy > 0.0 {
                old / ZOOM_STEP  // scroll down → zoom out
            } else {
                old * ZOOM_STEP  // scroll up   → zoom in
            }
            .clamp(1.0, ZOOM_MAX);
            zoom.set(next);
            apply_zoom(&picture, &scrolled, old, next, img_dims.get(), pointer.get());
            glib::Propagation::Stop
        }
    });
    scrolled.add_controller(scroll_ctrl);

    // Left-button drag → pan
    let drag_start_h: Rc<Cell<f64>> = Rc::new(Cell::new(0.0));
    let drag_start_v: Rc<Cell<f64>> = Rc::new(Cell::new(0.0));
    let drag = gtk4::GestureDrag::new();

    drag.connect_drag_begin({
        let scrolled = scrolled.clone();
        let drag_start_h = drag_start_h.clone();
        let drag_start_v = drag_start_v.clone();
        move |_, _, _| {
            drag_start_h.set(scrolled.hadjustment().value());
            drag_start_v.set(scrolled.vadjustment().value());
        }
    });

    drag.connect_drag_update({
        let scrolled = scrolled.clone();
        let drag_start_h = drag_start_h.clone();
        let drag_start_v = drag_start_v.clone();
        move |_, offset_x, offset_y| {
            scrolled.hadjustment().set_value(drag_start_h.get() - offset_x);
            scrolled.vadjustment().set_value(drag_start_v.get() - offset_y);
        }
    });

    scrolled.add_controller(drag);
}

pub(super) fn reset_zoom(
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    zoom: &Cell<f64>,
) {
    zoom.set(1.0);
    picture.set_content_fit(gtk4::ContentFit::Contain);
    picture.set_size_request(-1, -1);
    scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Never);
}

/// Apply `new_zoom` (1.0 = fit-to-window, >1.0 = zoomed in).
/// `(px, py)` is the pointer position in scrolled-widget coordinates; the
/// image pixel under that point is kept stationary through the zoom.
fn apply_zoom(
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    old_zoom: f64,
    new_zoom: f64,
    (img_w, img_h): (i32, i32),
    (px, py): (f64, f64),
) {
    if img_w == 0 || img_h == 0 {
        return;
    }

    let vw = scrolled.width() as f64;
    let vh = scrolled.height() as f64;
    // Screen pixels per image pixel at zoom == 1.0 (the "fit" scale).
    let fit = if vw > 0.0 && vh > 0.0 {
        f64::min(vw / img_w as f64, vh / img_h as f64)
    } else {
        1.0
    };

    // Image-space coordinates of the pixel currently under the pointer.
    let (cx, cy) = if old_zoom <= 1.0 {
        // Fit mode: ContentFit::Contain centers the image in the viewport.
        let img_left = (vw - img_w as f64 * fit) / 2.0;
        let img_top  = (vh - img_h as f64 * fit) / 2.0;
        (
            ((px - img_left) / fit).clamp(0.0, img_w as f64),
            ((py - img_top)  / fit).clamp(0.0, img_h as f64),
        )
    } else {
        let ppx = fit * old_zoom;
        (
            (scrolled.hadjustment().value() + px) / ppx,
            (scrolled.vadjustment().value() + py) / ppx,
        )
    };

    if new_zoom <= 1.0 {
        picture.set_content_fit(gtk4::ContentFit::Contain);
        picture.set_size_request(-1, -1);
        scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Never);
    } else {
        picture.set_content_fit(gtk4::ContentFit::Fill);
        let ppx = fit * new_zoom;
        let w = (img_w as f64 * ppx).round() as i32;
        let h = (img_h as f64 * ppx).round() as i32;
        picture.set_size_request(w, h);
        scrolled.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);

        // Scroll so the image pixel under the pointer stays under the pointer.
        // Pre-configure adjustment bounds before set_value to avoid clamping.
        let th = (cx * ppx - px).max(0.0);
        let tv = (cy * ppx - py).max(0.0);
        let hadj = scrolled.hadjustment();
        hadj.set_upper(w as f64);
        hadj.set_page_size(vw);
        hadj.set_value(th);
        let vadj = scrolled.vadjustment();
        vadj.set_upper(h as f64);
        vadj.set_page_size(vh);
        vadj.set_value(tv);
    }
}
