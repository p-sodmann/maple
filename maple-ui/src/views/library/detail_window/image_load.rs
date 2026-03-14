//! Async full-resolution image loader with EXIF orientation correction.
//!
//! Uses gdk-pixbuf (libjpeg-turbo / libpng / …) for decoding and lets it
//! apply the embedded EXIF orientation automatically — the same pipeline as
//! the thumbnail grid.  The raw RGBA pixel buffer is extracted in the
//! background thread; the main thread reconstructs a `gdk::Texture` from it
//! without any additional I/O or pixel work.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::gdk_pixbuf;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;

use super::zoom_pan::reset_zoom;

const MAX_WIN_W: i32 = 1400;
const MAX_WIN_H: i32 = 860;
/// Approximate pixel height of header bar + info strip combined.
const CHROME_H: i32 = 82;

/// Raw pixel data sent from the background thread to the main thread.
struct PixelBuffer {
    width: i32,
    height: i32,
    rowstride: i32,
    has_alpha: bool,
    data: Vec<u8>,
}

/// Load `path` asynchronously, apply EXIF orientation, and display the result
/// in `picture`.  The window is resized to fit the image.
pub(super) fn load_image(
    path: PathBuf,
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    zoom: &Rc<Cell<f64>>,
    img_dims: &Rc<Cell<(i32, i32)>>,
    window: &adw::Window,
) {
    let (tx, rx) = mpsc::channel::<PixelBuffer>();

    std::thread::spawn(move || {
        // Load via gdk-pixbuf — same engine as the thumbnail grid.
        // apply_embedded_orientation() reads the EXIF tag and rotates/flips
        // the pixel buffer automatically; returns None when the image is
        // already correctly oriented (orientation == 1).
        let Ok(pixbuf) = gdk_pixbuf::Pixbuf::from_file(&path) else { return };
        let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);

        let width = pixbuf.width();
        let height = pixbuf.height();
        let rowstride = pixbuf.rowstride();
        let has_alpha = pixbuf.has_alpha();

        // Extract a plain Vec<u8> so we don't send GObject types across threads.
        let Some(glib_bytes) = pixbuf.pixel_bytes() else { return };
        let data = glib_bytes.as_ref().to_vec();

        let _ = tx.send(PixelBuffer { width, height, rowstride, has_alpha, data });
    });

    let picture = picture.clone();
    let scrolled = scrolled.clone();
    let zoom = zoom.clone();
    let img_dims = img_dims.clone();
    let window = window.clone();

    glib::timeout_add_local(Duration::from_millis(32), move || {
        match rx.try_recv() {
            Ok(buf) => {
                let gb = glib::Bytes::from(&buf.data);
                let pixbuf = gdk_pixbuf::Pixbuf::from_bytes(
                    &gb,
                    gdk_pixbuf::Colorspace::Rgb,
                    buf.has_alpha,
                    8,
                    buf.width,
                    buf.height,
                    buf.rowstride,
                );
                let texture = gdk::Texture::for_pixbuf(&pixbuf);
                let iw = texture.width();
                let ih = texture.height();
                img_dims.set((iw, ih));
                let (dw, dh) = display_size(iw, ih);
                window.set_default_size(dw, dh);
                reset_zoom(&picture, &scrolled, &zoom);
                picture.set_paintable(Some(&texture));
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
        }
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
