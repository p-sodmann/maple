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

use maple_import::{decode_image, decode_image_bytes, is_raw_format, raw_preview_supported};

use crate::remote::RemoteBlobs;
use crate::DetailWindow;

/// How long to wait for the decode thread before giving up.
const LOAD_TIMEOUT: Duration = Duration::from_secs(30);

/// Decoded pixels, safe to move across threads (no GObject / Slint types).
struct RgbBuf {
    width: u32,
    height: u32,
    data: Vec<u8>,
}

/// Where a full-resolution image comes from.
///
/// The remote arm is the relay contract in one place: the bytes are fetched
/// into memory, decoded, and dropped. Nothing is written to disk — a relay
/// servant that cached originals would stop being a relay, and the user chose
/// that mode precisely to keep their disk empty.
pub enum Source {
    Disk(PathBuf),
    Master {
        blobs: RemoteBlobs,
        hash: [u8; 32],
        /// Kept for the log line and the error text: a remote row's `path` is
        /// the master's, and is never opened here.
        origin_path: PathBuf,
    },
}

/// Decode `path` at full resolution on a background thread and display it in
/// `window`, applying EXIF orientation.
///
/// Exactly one outcome reaches the UI thread:
///   • success → sets `photo` + `img-w`/`img-h`, clears `loading`, resets zoom;
///   • failure / 30-second timeout → sets `error-text`, clears `loading`.
pub fn load_full_image(source: Source, window: slint::Weak<DetailWindow>) {
    std::thread::spawn(move || {
        let (tx, rx) = mpsc::channel::<Result<RgbBuf, String>>();

        // Inner decode thread so a hung decode (e.g. unavailable mount, or a
        // master that accepted the connection and then went quiet) can be
        // abandoned by the timeout below instead of blocking forever.
        std::thread::spawn(move || {
            let _ = tx.send(match &source {
                Source::Disk(path) => decode_to_rgb(path),
                Source::Master { blobs, hash, origin_path } => {
                    fetch_and_decode(blobs, hash, origin_path)
                }
            });
        });

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

/// Fetch an original from the master and decode it, in memory.
///
/// The 30-second watchdog above covers a slow or absent master, so there is
/// no separate timeout here — one deadline for "the pixels did not arrive"
/// is easier to reason about than two that can disagree.
fn fetch_and_decode(
    blobs: &RemoteBlobs,
    hash: &[u8; 32],
    origin_path: &Path,
) -> Result<RgbBuf, String> {
    // `/blob/orig` serves the file *verbatim* — it has to, because P7 will
    // verify the BLAKE3 of what it downloads. For a raw original that means
    // sensor data, and the preview extractor that makes RAF viewable locally
    // (`maple_import::loadable_image_bytes`) reads from a path, not a buffer.
    // Say so rather than letting the decoder fail with something generic.
    if is_raw_format(origin_path) {
        let ext = origin_path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("raw")
            .to_uppercase();
        let msg = format!("{ext} originals cannot be viewed over sync yet");
        tracing::warn!("image_loader: {msg} ({})", origin_path.display());
        return Err(msg);
    }

    let bytes = blobs.original(hash, false).map_err(|e| {
        let msg = format!("Could not load this photo from the master: {e}");
        tracing::warn!("image_loader: {msg} ({})", origin_path.display());
        msg
    })?;

    let img = decode_image_bytes(&bytes).map_err(|e| {
        let msg = format!("Failed to decode image: {e}");
        tracing::warn!("image_loader: {msg} ({})", origin_path.display());
        msg
    })?;

    let rgb = img.into_rgb8();
    let (width, height) = rgb.dimensions();
    Ok(RgbBuf { width, height, data: rgb.into_raw() })
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
