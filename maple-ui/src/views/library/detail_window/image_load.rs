//! Detail-window image loading — thin wrapper around the shared loader.
//!
//! After the image is decoded and displayed the window is resized to fit and
//! zoom/pan state is reset.  All pixel decoding and EXIF orientation
//! correction is handled by `library::image_loader::load_image_async`.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;

use gtk4::prelude::*;
use libadwaita as adw;

use super::super::image_loader::load_image_async;
use super::zoom_pan::reset_zoom;

const MAX_WIN_W: i32 = 1400;
const MAX_WIN_H: i32 = 860;
/// Approximate pixel height of header bar + info strip combined.
const CHROME_H: i32 = 82;

/// Load `path` asynchronously, apply EXIF orientation, display it in
/// `picture`, resize the window to fit, and reset zoom/pan.
pub(super) fn load_image(
    path: PathBuf,
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    zoom: &Rc<Cell<f64>>,
    img_dims: &Rc<Cell<(i32, i32)>>,
    window: &adw::Window,
) {
    let scrolled = scrolled.clone();
    let zoom = zoom.clone();
    let picture_ref = picture.clone();
    let window = window.clone();

    load_image_async(path, picture, img_dims, move |img_w, img_h| {
        let (dw, dh) = display_size(img_w, img_h);
        window.set_default_size(dw, dh);
        reset_zoom(&picture_ref, &scrolled, &zoom);
    });
}

/// Compute a window size that fits `img_w × img_h` within the screen budget.
fn display_size(img_w: i32, img_h: i32) -> (i32, i32) {
    if img_w <= 0 || img_h <= 0 {
        return (960, 720);
    }
    let usable_h = MAX_WIN_H - CHROME_H;
    let scale = f64::min(
        MAX_WIN_W as f64 / img_w as f64,
        usable_h as f64 / img_h as f64,
    );
    let w = (img_w as f64 * scale).round() as i32;
    let h = (img_h as f64 * scale).round() as i32 + CHROME_H;
    (w.max(400), h.max(300))
}
