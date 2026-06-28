//! Full-resolution async image loader for the detail window (pure-Rust).
//!
//! Threading model: a watchdog thread spawns an inner decode thread and waits
//! at most 30 seconds for a result, then delivers exactly one outcome to the
//! Slint event loop via [`slint::Weak::upgrade_in_event_loop`].  The window's
//! `loading` flag is always cleared exactly once, on success *or* timeout.

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Duration;

use slint::{Image, Rgb8Pixel, SharedPixelBuffer, SharedString};

use maple_import::{decode_image, is_raw_format, raw_preview_supported};

use crate::DetailWindow;

/// How long to wait for the decode thread before giving up.
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// Decoded pixels, safe to move across threads (no GObject / Slint types).
struct RgbBuf {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// Decode `path` at full resolution on a background thread and display it in
/// `window`, applying EXIF orientation.
///
/// Exactly one outcome reaches the UI thread:
///   • success → sets `photo` + `img-w`/`img-h`, clears `loading`, resets zoom;
///   • failure / 30-second timeout → sets `error-text`, clears `loading`.
pub fn load_full_image(path: PathBuf, window: slint::Weak<DetailWindow>) {
    std::thread::spawn(move || {
        let (tx, rx) = mpsc::channel::<Result<RgbBuf, String>>();

        // Inner decode thread so a hung decode (e.g. unavailable mount) can
        // be abandoned by the timeout below instead of blocking forever.
        {
            let path = path.clone();
            std::thread::spawn(move || {
                let _ = tx.send(decode_to_rgb(&path));
            });
        }

        let outcome = rx
            .recv_timeout(LOAD_TIMEOUT)
            .unwrap_or_else(|_| Err("Image took too long to load".to_owned()));

        let _ = window.upgrade_in_event_loop(move |w| match outcome {
            Ok(buf) => {
                let mut pb = SharedPixelBuffer::<Rgb8Pixel>::new(buf.width, buf.height);
                pb.make_mut_bytes().copy_from_slice(&buf.data);
                w.set_photo(Image::from_rgb8(pb));
                w.set_img_w(buf.width as i32);
                w.set_img_h(buf.height as i32);
                w.set_error_text(SharedString::new());
                w.set_loading(false);
                w.invoke_reset_view();
            }
            Err(msg) => {
                w.set_error_text(msg.into());
                w.set_loading(false);
            }
        });
    });
}

/// Decode + orient `path` into a tight RGB buffer (worker-thread side).
///
/// Using RGB (not RGBA) avoids an unnecessary pixel-by-pixel expansion:
/// JPEG decode returns `DynamicImage::ImageRgb8`, so `into_rgb8()` is a
/// zero-copy extract whereas `into_rgba8()` would allocate a fresh 25%-larger
/// buffer and write α=255 into every pixel.
fn decode_to_rgb(path: &Path) -> Result<RgbBuf, String> {
    if is_raw_format(path) && !raw_preview_supported(path) {
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_uppercase();
        let msg = format!("Preview not available for {ext} — raw format not yet supported");
        tracing::warn!("image_loader: {msg} ({})", path.display());
        return Err(msg);
    }

    let img = decode_image(path).map_err(|e| {
        let msg = format!("Failed to load image: {e}");
        tracing::warn!("image_loader: {msg} ({})", path.display());
        msg
    })?;

    let rgb = img.into_rgb8();
    let (width, height) = rgb.dimensions();
    Ok(RgbBuf { width, height, data: rgb.into_raw() })
}
