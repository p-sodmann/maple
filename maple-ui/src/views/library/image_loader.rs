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
use maple_import::{is_raw_format, loadable_image_bytes, raw_preview_supported};

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
/// `on_error` is called with a user-facing message when decoding fails or
/// the 30-second timeout fires.
pub fn load_image_async(
    path: PathBuf,
    picture: &gtk4::Picture,
    img_dims: &Rc<Cell<(i32, i32)>>,
    on_loaded: impl Fn(i32, i32) + 'static,
    on_error: impl Fn(&str) + 'static,
) {
    let (tx, rx) = mpsc::channel::<Result<PixelBuffer, String>>();

    std::thread::spawn(move || {
        let result: Result<PixelBuffer, String> = (|| {
            // Check before attempting extraction — gives a clearer error than
            // the generic bail! inside the handler.
            if is_raw_format(&path) && !raw_preview_supported(&path) {
                let msg = format!(
                    "Preview not available for {} — raw format not yet supported",
                    path.extension()
                        .and_then(|e| e.to_str())
                        .unwrap_or("unknown")
                        .to_uppercase()
                );
                tracing::warn!("image_loader: {msg} ({})", path.display());
                return Err(msg);
            }

            let pixbuf = if is_raw_format(&path) {
                let bytes = loadable_image_bytes(&path).map_err(|e| {
                    let msg = format!("Failed to read raw preview: {e}");
                    tracing::warn!("image_loader: {msg} ({})", path.display());
                    msg
                })?;
                let stream =
                    gtk4::gio::MemoryInputStream::from_bytes(&glib::Bytes::from(&bytes));
                gdk_pixbuf::Pixbuf::from_stream(&stream, gtk4::gio::Cancellable::NONE)
                    .map_err(|e| {
                        let msg = format!("Failed to decode raw preview: {e}");
                        tracing::warn!("image_loader: {msg} ({})", path.display());
                        msg
                    })?
            } else {
                gdk_pixbuf::Pixbuf::from_file(&path).map_err(|e| {
                    let msg = format!("Failed to decode image: {e}");
                    tracing::warn!("image_loader: {msg} ({})", path.display());
                    msg
                })?
            };

            let pixbuf = pixbuf.apply_embedded_orientation().unwrap_or(pixbuf);
            let width = pixbuf.width();
            let height = pixbuf.height();
            let rowstride = pixbuf.rowstride();
            let has_alpha = pixbuf.has_alpha();
            let bytes = pixbuf.pixel_bytes().ok_or_else(|| {
                let msg = "Failed to access pixel data".to_owned();
                tracing::warn!("image_loader: {msg} ({})", path.display());
                msg
            })?;
            let data = bytes.as_ref().to_vec();
            Ok(PixelBuffer { width, height, rowstride, has_alpha, data })
        })();
        let _ = tx.send(result);
    });

    let picture = picture.clone();
    let img_dims = img_dims.clone();
    let deadline = Instant::now() + LOAD_TIMEOUT;

    glib::timeout_add_local(Duration::from_millis(32), move || {
        if Instant::now() > deadline {
            tracing::warn!("image_loader: 30-second decode timeout");
            on_error("Image took too long to load");
            return glib::ControlFlow::Break;
        }
        match rx.try_recv() {
            Ok(Ok(buf)) => {
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
            Ok(Err(msg)) => {
                on_error(&msg);
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                tracing::warn!("image_loader: decode thread panicked");
                on_error("Failed to load image");
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
        }
    });
}
