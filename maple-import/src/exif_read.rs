//! Reading EXIF off a file, raw containers included.
//!
//! One reader pass per file, shared by the two callers that need it for
//! different reasons: [`crate::copy`] wants the date and camera for the
//! `{camera}` path token, and [`crate::session`] wants the capture *instant*
//! — with sub-second precision, because that is what separates two frames
//! of a drive-mode burst from two deliberate shots a second apart.

use std::path::Path;

use crate::{is_raw_format, loadable_image_bytes_named, ExifDateTime};

/// What one EXIF pass yields.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ExifContext {
    pub datetime: Option<ExifDateTime>,
    /// `SubSecTimeOriginal`, as written — a decimal fraction of a second
    /// with no leading `"0."`.
    pub subsec: Option<String>,
    /// `Make` and `Model` joined, whichever of the two exists.
    pub camera: Option<String>,
}

impl ExifContext {
    /// Capture time as fractional seconds since the epoch.
    ///
    /// No timezone adjustment — [`ExifDateTime`] treats the stamp as UTC,
    /// and every photo on one card is offset identically, so a *gap*
    /// between two of them is right regardless.
    pub fn capture_secs(&self) -> Option<f64> {
        let base = self.datetime?.to_unix_timestamp() as f64;
        let frac = self
            .subsec
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()))
            .and_then(|s| format!("0.{s}").parse::<f64>().ok())
            .unwrap_or(0.0);
        Some(base + frac)
    }
}

/// Read EXIF from `path`. Returns an empty context on any read or parse
/// failure rather than erroring — every caller treats missing EXIF as a
/// normal case.
///
/// For raw files this reads the embedded preview: raw containers are not
/// parsed directly (see [`crate::loadable_image_bytes`]), and the preview
/// carries the same EXIF block.
pub fn read(path: &Path) -> ExifContext {
    read_named(path, path)
}

/// Read EXIF from `path`, deciding whether it is a raw container from a
/// *different* name than the one it currently sits under.
///
/// The seam sync needs. A downloaded blob is staged as `<hex-hash>.orig` —
/// its extension says nothing about what is inside it — while the sender's
/// own filename, which does, travelled beside it. Passing that name here is
/// the difference between a RAF read as a raw container and a RAF handed to
/// a JPEG parser, which returns nothing and sends the photo to the mtime
/// fallback: filed under the day it arrived rather than the day it was taken.
///
/// Deliberately *not* container sniffing. The name is a fact the caller
/// already holds; guessing from the bytes would be a second, weaker source of
/// truth for something `is_raw_format` already decides everywhere else.
pub fn read_named(path: &Path, name: &Path) -> ExifContext {
    let exif = if is_raw_format(name) {
        let Ok(bytes) = loadable_image_bytes_named(path, name) else {
            return ExifContext::default();
        };
        exif::Reader::new().read_from_container(&mut std::io::Cursor::new(bytes))
    } else {
        let Ok(file) = std::fs::File::open(path) else {
            return ExifContext::default();
        };
        exif::Reader::new().read_from_container(&mut std::io::BufReader::new(file))
    };
    let Ok(exif) = exif else {
        return ExifContext::default();
    };
    from_exif(&exif)
}

/// Read EXIF out of bytes already in memory.
///
/// The import scan holds every file's bytes to hash them, and for a raw it
/// holds the extracted preview — which carries the same EXIF block. Parsing
/// those instead of reopening the file is what makes capture time free
/// during a scan: the card is the slowest link in the pipeline, and reading
/// each photo a second time just for its timestamp would be paying the
/// whole cost of session detection over again.
pub fn read_bytes(bytes: &[u8]) -> ExifContext {
    match exif::Reader::new().read_from_container(&mut std::io::Cursor::new(bytes)) {
        Ok(exif) => from_exif(&exif),
        Err(_) => ExifContext::default(),
    }
}

fn from_exif(exif: &exif::Exif) -> ExifContext {
    let ascii = |tag: exif::Tag| exif.fields().find(|f| f.tag == tag).and_then(ascii_value);

    let camera = match (ascii(exif::Tag::Make), ascii(exif::Tag::Model)) {
        (Some(make), Some(model)) => Some(format!("{} {}", make.trim(), model.trim())),
        (Some(v), None) | (None, Some(v)) => Some(v.trim().to_owned()),
        (None, None) => None,
    };

    ExifContext {
        datetime: ascii(exif::Tag::DateTimeOriginal).and_then(|s| ExifDateTime::parse(&s)),
        subsec: ascii(exif::Tag::SubSecTimeOriginal),
        camera,
    }
}

fn ascii_value(field: &exif::Field) -> Option<String> {
    if let exif::Value::Ascii(ref v) = field.value {
        let bytes = v.first()?;
        std::str::from_utf8(bytes).ok().map(|s| s.to_owned())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(second: u32) -> ExifDateTime {
        ExifDateTime { year: 2026, month: 8, day: 26, hour: 12, minute: 0, second }
    }

    #[test]
    fn subsec_refines_the_capture_instant() {
        let ctx = ExifContext { datetime: Some(at(30)), subsec: Some("25".into()), camera: None };
        let secs = ctx.capture_secs().unwrap();
        assert!((secs - at(30).to_unix_timestamp() as f64 - 0.25).abs() < 1e-9, "got {secs}");
    }

    #[test]
    fn two_frames_in_the_same_second_are_still_ordered() {
        // Without SubSecTimeOriginal a 10 fps burst is ten photos at the
        // same instant, and every gap is zero.
        let a = ExifContext { datetime: Some(at(30)), subsec: Some("1".into()), camera: None };
        let b = ExifContext { datetime: Some(at(30)), subsec: Some("7".into()), camera: None };
        assert!(b.capture_secs().unwrap() > a.capture_secs().unwrap());
    }

    #[test]
    fn a_junk_subsec_falls_back_to_whole_seconds() {
        let ctx = ExifContext { datetime: Some(at(30)), subsec: Some("  ".into()), camera: None };
        assert_eq!(ctx.capture_secs(), Some(at(30).to_unix_timestamp() as f64));
        let ctx = ExifContext { datetime: Some(at(30)), subsec: Some("1/2".into()), camera: None };
        assert_eq!(ctx.capture_secs(), Some(at(30).to_unix_timestamp() as f64));
    }

    #[test]
    fn no_datetime_means_no_capture_time() {
        assert_eq!(ExifContext::default().capture_secs(), None);
    }

    #[test]
    fn a_missing_file_reads_as_an_empty_context() {
        assert_eq!(read(Path::new("/nonexistent/photo.jpg")), ExifContext::default());
    }
}
