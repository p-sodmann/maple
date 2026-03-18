//! Shared async image loader with EXIF orientation correction.
//!
//! `load_image_async` is the single place where images are decoded.  It uses
//! gdk-pixbuf (the same engine as the thumbnail grid) so that
//! `apply_embedded_orientation()` is called on every image, correcting any
//! EXIF rotation/flip tag.  The decoded RGBA buffer is shipped back to the
//! main thread over an `mpsc` channel so no GObject types cross thread
//! boundaries.
//!
//! Callers supply an `on_loaded` closure that receives `(img_w, img_h)` —
//! the *post-rotation* pixel dimensions of the decoded image — giving each
//! caller a chance to resize windows, reset zoom, queue redraws, etc.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::gdk_pixbuf;
use gtk4::glib;
use maple_import::{is_raw_format, loadable_image_bytes};

/// Raw pixel data that can safely be moved across threads.
struct PixelBuffer {
    width: i32,
    height: i32,
    rowstride: i32,
    has_alpha: bool,
    data: Vec<u8>,
}

/// Load `path` asynchronously, apply EXIF orientation, display the result in
/// `picture`, and set `img_dims` to the post-rotation pixel size.
///
/// `on_loaded` is called on the main thread once the texture is displayed.
/// It receives the decoded `(width, height)` in pixels.
pub fn load_image_async(
    path: PathBuf,
    picture: &gtk4::Picture,
    img_dims: &Rc<Cell<(i32, i32)>>,
    on_loaded: impl Fn(i32, i32) + 'static,
) {
    let (tx, rx) = mpsc::channel::<PixelBuffer>();

    std::thread::spawn(move || {
        let pixbuf = if is_raw_format(&path) {
            let Ok(bytes) = loadable_image_bytes(&path) else { return };
            let stream = gtk4::gio::MemoryInputStream::from_bytes(&glib::Bytes::from(&bytes));
            let Ok(pb) = gdk_pixbuf::Pixbuf::from_stream(&stream, gtk4::gio::Cancellable::NONE)
            else {
                return;
            };
            pb
        } else {
            let Ok(pb) = gdk_pixbuf::Pixbuf::from_file(&path) else { return };
            pb
        };
        // apply_embedded_orientation() rotates/flips according to the EXIF tag.
        // It returns None when orientation is already 1 (top-left), so fall
        // back to the original in that case.
        let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);

        let width = pixbuf.width();
        let height = pixbuf.height();
        let rowstride = pixbuf.rowstride();
        let has_alpha = pixbuf.has_alpha();
        let Some(bytes) = pixbuf.pixel_bytes() else { return };
        let data = bytes.as_ref().to_vec();

        let _ = tx.send(PixelBuffer { width, height, rowstride, has_alpha, data });
    });

    let picture = picture.clone();
    let img_dims = img_dims.clone();

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
                img_dims.set((buf.width, buf.height));
                picture.set_paintable(Some(&texture));
                on_loaded(buf.width, buf.height);
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
        }
    });
}
