//! Shared async image loader with EXIF orientation correction.
//!
//! `load_image_async` is the single place where images are decoded.  It uses
//! gdk-pixbuf (the same engine as the thumbnail grid) so that
//! `apply_embedded_orientation()` is called on every image, correcting any
//! EXIF rotation/flip tag.  The decoded RGBA buffer is shipped back to the
//! main thread over an `mpsc` channel so no GObject types cross thread
//! boundaries.
//!
//! Callers supply:
//!   • `on_loaded` — called with `(img_w, img_h)` on success.
//!   • `on_error`  — called when decoding fails *or* the 30-second timeout
//!                   fires (hung thread / unavailable mount).
//!
//! The channel always receives exactly one message (or Disconnected if the
//! thread panics), so the glib poller always terminates.

use std::cell::Cell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use gtk4::gdk;
use gtk4::gdk_pixbuf;
use gtk4::glib;
use maple_import::{is_raw_format, loadable_image_bytes};

/// How long to wait for the decode thread before giving up.
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);

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
/// `on_loaded` is called on the main thread on success with `(width, height)`.
/// `on_error` is called when decoding fails or the 30-second timeout fires.
pub fn load_image_async(
    path: PathBuf,
    picture: &gtk4::Picture,
    img_dims: &Rc<Cell<(i32, i32)>>,
    on_loaded: impl Fn(i32, i32) + 'static,
    on_error: impl Fn() + 'static,
) {
    let (tx, rx) = mpsc::channel::<Option<PixelBuffer>>();

    std::thread::spawn(move || {
        let result = (|| {
            let pixbuf = if is_raw_format(&path) {
                let bytes = loadable_image_bytes(&path).ok()?;
                let stream =
                    gtk4::gio::MemoryInputStream::from_bytes(&glib::Bytes::from(&bytes));
                gdk_pixbuf::Pixbuf::from_stream(&stream, gtk4::gio::Cancellable::NONE).ok()?
            } else {
                gdk_pixbuf::Pixbuf::from_file(&path).ok()?
            };
            let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);
            let width = pixbuf.width();
            let height = pixbuf.height();
            let rowstride = pixbuf.rowstride();
            let has_alpha = pixbuf.has_alpha();
            let bytes = pixbuf.pixel_bytes()?;
            let data = bytes.as_ref().to_vec();
            Some(PixelBuffer { width, height, rowstride, has_alpha, data })
        })();
        let _ = tx.send(result);
    });

    let picture = picture.clone();
    let img_dims = img_dims.clone();
    let deadline = Instant::now() + LOAD_TIMEOUT;

    glib::timeout_add_local(Duration::from_millis(32), move || {
        if Instant::now() > deadline {
            on_error();
            return glib::ControlFlow::Break;
        }
        match rx.try_recv() {
            Ok(Some(buf)) => {
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
            Ok(None) | Err(mpsc::TryRecvError::Disconnected) => {
                on_error();
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        }
    });
}
