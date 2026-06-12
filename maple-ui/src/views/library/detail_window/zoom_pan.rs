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

    // Viewport resizes (window resize, fullscreen toggle) invalidate the
    // picture's pixel size request, which was computed against the old fit
    // scale. Re-apply the current zoom factor for the new viewport, keeping
    // the image point at the viewport centre fixed. The adjustments emit
    // "changed" whenever their page size (= viewport) changes.
    let last_viewport: Rc<Cell<(f64, f64)>> = Rc::new(Cell::new((0.0, 0.0)));
    for adj in [scrolled.hadjustment(), scrolled.vadjustment()] {
        adj.connect_changed({
            let picture = picture.clone();
            let scrolled = scrolled.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let last_viewport = last_viewport.clone();
            move |_| {
                let vw = scrolled.width() as f64;
                let vh = scrolled.height() as f64;
                let (ovw, ovh) = last_viewport.replace((vw, vh));
                if (ovw, ovh) == (vw, vh) || vw <= 0.0 || vh <= 0.0 {
                    return;
                }
                let z = zoom.get();
                let (img_w, img_h) = img_dims.get();
                if z <= 1.0 || img_w == 0 || img_h == 0 || ovw <= 0.0 || ovh <= 0.0 {
                    return;
                }
                // Image point that was at the old viewport centre…
                let old_fit = f64::min(ovw / img_w as f64, ovh / img_h as f64);
                let anchor = anchor_image_point(
                    &scrolled,
                    (img_w, img_h),
                    (ovw, ovh),
                    old_fit * z,
                    (ovw / 2.0, ovh / 2.0),
                );
                // …stays at the new viewport centre.
                let fit = f64::min(vw / img_w as f64, vh / img_h as f64);
                apply_zoomed_geometry(
                    &picture,
                    &scrolled,
                    (img_w, img_h),
                    (vw, vh),
                    fit * z,
                    anchor,
                    (vw / 2.0, vh / 2.0),
                );
            }
        });
    }
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

    let anchor = anchor_image_point(
        scrolled,
        (img_w, img_h),
        (vw, vh),
        fit * old_zoom,
        (px, py),
    );

    if new_zoom <= 1.0 {
        picture.set_content_fit(gtk4::ContentFit::Contain);
        picture.set_size_request(-1, -1);
        scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Never);
    } else {
        apply_zoomed_geometry(
            picture,
            scrolled,
            (img_w, img_h),
            (vw, vh),
            fit * new_zoom,
            anchor,
            (px, py),
        );
    }
}

/// Image-space coordinates of the pixel at `(px, py)` (widget coords),
/// given `ppx` screen pixels per image pixel.
///
/// The picture is allocated at least the viewport size, and
/// ContentFit::Contain centres the image inside that allocation, so an
/// axis that doesn't overflow the viewport has a centring margin instead
/// of a scroll offset (the adjustment value is 0 there).
fn anchor_image_point(
    scrolled: &gtk4::ScrolledWindow,
    (img_w, img_h): (i32, i32),
    (vw, vh): (f64, f64),
    ppx: f64,
    (px, py): (f64, f64),
) -> (f64, f64) {
    let img_left = ((vw - img_w as f64 * ppx) / 2.0).max(0.0);
    let img_top  = ((vh - img_h as f64 * ppx) / 2.0).max(0.0);
    (
        ((scrolled.hadjustment().value() + px - img_left) / ppx).clamp(0.0, img_w as f64),
        ((scrolled.vadjustment().value() + py - img_top) / ppx).clamp(0.0, img_h as f64),
    )
}

/// Size the picture for `ppx` screen pixels per image pixel and scroll so
/// that image point `(cx, cy)` lands at widget point `(px, py)`.
fn apply_zoomed_geometry(
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    (img_w, img_h): (i32, i32),
    (vw, vh): (f64, f64),
    ppx: f64,
    (cx, cy): (f64, f64),
    (px, py): (f64, f64),
) {
    // Contain (not Fill): when the requested size is still smaller than
    // the viewport on one axis, the allocation is grown to the viewport
    // and Fill would stretch the image to that aspect ratio.
    picture.set_content_fit(gtk4::ContentFit::Contain);
    let w = (img_w as f64 * ppx).round() as i32;
    let h = (img_h as f64 * ppx).round() as i32;
    picture.set_size_request(w, h);
    scrolled.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);

    // Pre-configure adjustment bounds before set_value to avoid clamping.
    let img_left = ((vw - w as f64) / 2.0).max(0.0);
    let img_top  = ((vh - h as f64) / 2.0).max(0.0);
    let th = (cx * ppx + img_left - px).max(0.0);
    let tv = (cy * ppx + img_top - py).max(0.0);
    let hadj = scrolled.hadjustment();
    hadj.set_upper((w as f64).max(vw));
    hadj.set_page_size(vw);
    hadj.set_value(th);
    let vadj = scrolled.vadjustment();
    vadj.set_upper((h as f64).max(vh));
    vadj.set_page_size(vh);
    vadj.set_value(tv);
}
