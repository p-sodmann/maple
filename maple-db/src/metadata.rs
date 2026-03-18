//! EXIF metadata extraction and background population worker.
//!
//! `extract_metadata` reads a single image file and returns whatever EXIF
//! fields are present.  It never panics — on any I/O or parse failure it
//! returns a struct with only `filename` populated.
//!
//! `spawn_metadata_filler` runs once at library-open time to populate
//! metadata for any records that were inserted before EXIF extraction
//! existed (i.e. `filename IS NULL` in the DB).

use std::fs::File;
use std::io::{BufReader, Cursor};
use std::path::Path;
use std::sync::{Arc, Mutex};

use exif::{Tag, Value};
use maple_import::{is_raw_format, loadable_image_bytes};

use crate::Database;

// ── Data model ───────────────────────────────────────────────────

/// EXIF and file-level metadata for one image.
///
/// All fields are `Option` because any of them may be absent — not all
/// cameras write all tags, and non-JPEG formats may carry no EXIF at all.
#[derive(Debug, Clone, Default)]
pub struct ImageMetadata {
    pub filename: Option<String>,
    pub taken_at: Option<i64>,
    pub make: Option<String>,
    pub model: Option<String>,
    pub lens: Option<String>,
    pub focal_length: Option<f64>,
    pub aperture: Option<f64>,
    pub iso: Option<i64>,
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub orientation: Option<i64>,
}

// ── Extraction ───────────────────────────────────────────────────

/// Extract metadata from `path`.  Returns a best-effort struct — fields
/// that cannot be read are left as `None`.
pub fn extract_metadata(path: &Path) -> ImageMetadata {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .map(ToOwned::to_owned);

    // For raw files, read EXIF from the embedded JPEG preview.
    let exif = if is_raw_format(path) {
        let Ok(bytes) = loadable_image_bytes(path) else {
            return ImageMetadata { filename, ..Default::default() };
        };
        let mut cursor = Cursor::new(bytes);
        match exif::Reader::new().read_from_container(&mut cursor) {
            Ok(e) => e,
            Err(_) => return ImageMetadata { filename, ..Default::default() },
        }
    } else {
        let Ok(file) = File::open(path) else {
            return ImageMetadata { filename, ..Default::default() };
        };
        let mut reader = BufReader::new(file);
        match exif::Reader::new().read_from_container(&mut reader) {
            Ok(e) => e,
            Err(_) => return ImageMetadata { filename, ..Default::default() },
        }
    };

    // ── Helper closures that search all IFDs ─────────────────────

    let get_ascii = |tag: Tag| -> Option<String> {
        exif.fields().find(|f| f.tag == tag).and_then(|f| {
            if let Value::Ascii(ref v) = f.value {
                v.first()
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .map(|s| s.trim_end_matches('\0').trim().to_owned())
                    .filter(|s| !s.is_empty())
            } else {
                None
            }
        })
    };

    let get_rational = |tag: Tag| -> Option<f64> {
        exif.fields().find(|f| f.tag == tag).and_then(|f| {
            if let Value::Rational(ref v) = f.value {
                v.first()
                    .filter(|r| r.denom != 0)
                    .map(|r| r.num as f64 / r.denom as f64)
            } else {
                None
            }
        })
    };

    let get_uint = |tag: Tag| -> Option<i64> {
        exif.fields()
            .find(|f| f.tag == tag)
            .and_then(|f| f.value.get_uint(0))
            .map(|v| v as i64)
    };

    // DateTimeOriginal is ASCII: "YYYY:MM:DD HH:MM:SS"
    let taken_at = exif
        .fields()
        .find(|f| f.tag == Tag::DateTimeOriginal)
        .and_then(|f| {
            if let Value::Ascii(ref v) = f.value {
                v.first()
                    .and_then(|b| std::str::from_utf8(b).ok())
                    .and_then(parse_exif_datetime)
            } else {
                None
            }
        });

    ImageMetadata {
        filename,
        taken_at,
        make: get_ascii(Tag::Make),
        model: get_ascii(Tag::Model),
        lens: get_ascii(Tag::LensModel),
        focal_length: get_rational(Tag::FocalLength),
        aperture: get_rational(Tag::FNumber),
        iso: get_uint(Tag::PhotographicSensitivity),
        width: get_uint(Tag::PixelXDimension),
        height: get_uint(Tag::PixelYDimension),
        orientation: get_uint(Tag::Orientation),
    }
}

/// Parse an EXIF datetime string `"YYYY:MM:DD HH:MM:SS"` into a Unix
/// timestamp (seconds since 1970-01-01 UTC, no timezone adjustment).
///
/// Returns `None` on any parse failure rather than panicking.
fn parse_exif_datetime(s: &str) -> Option<i64> {
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s[0..4].parse().ok()?;
    let month: i64 = s[5..7].parse().ok()?;
    let day: i64 = s[8..10].parse().ok()?;
    let hour: i64 = s[11..13].parse().ok()?;
    let minute: i64 = s[14..16].parse().ok()?;
    let second: i64 = s[17..19].parse().ok()?;

    // Days from 1970-01-01 (proleptic Gregorian calendar).
    // Algorithm: https://howardhinnant.github.io/date_algorithms.html#days_from_civil
    let y = if month <= 2 { year - 1 } else { year };
    let m = if month <= 2 { month + 9 } else { month - 3 };
    let era = y.div_euclid(400);
    let yoe = y - era * 400;
    let doy = (153 * m + 2) / 5 + day - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;

    Some(days * 86400 + hour * 3600 + minute * 60 + second)
}

// ── Background worker ────────────────────────────────────────────

/// Spawn a one-shot background thread that fills EXIF metadata for all
/// library records where `filename IS NULL` (not yet processed).
///
/// Safe to call multiple times — only unprocessed records are touched.
pub fn spawn_metadata_filler(db: Arc<Mutex<Database>>) {
    std::thread::Builder::new()
        .name("maple-metadata-filler".into())
        .spawn(move || {
            let to_fill = match db.lock() {
                Ok(d) => d.records_needing_metadata().unwrap_or_default(),
                Err(_) => return,
            };

            if to_fill.is_empty() {
                return;
            }

            tracing::info!("Metadata filler: {} records to process", to_fill.len());

            for (id, path) in to_fill {
                let meta = extract_metadata(&path);
                if let Ok(d) = db.lock() {
                    if let Err(e) = d.update_metadata(id, &meta) {
                        tracing::warn!("Metadata filler: failed for {}: {e}", path.display());
                    }
                }
            }

            tracing::info!("Metadata filler: done");
        })
        .ok();
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_valid_exif_datetime() {
        // 2024-06-15 10:30:00 UTC
        let ts = parse_exif_datetime("2024:06:15 10:30:00").unwrap();
        // Rough sanity: should be > 2020-01-01 and < 2030-01-01
        assert!(ts > 1_577_836_800);
        assert!(ts < 1_893_456_000);
    }

    #[test]
    fn parse_invalid_exif_datetime_returns_none() {
        assert!(parse_exif_datetime("not-a-date").is_none());
        assert!(parse_exif_datetime("").is_none());
        assert!(parse_exif_datetime("2024:06:15").is_none()); // too short
    }

    #[test]
    fn extract_metadata_missing_file_returns_filename_only() {
        let meta = extract_metadata(Path::new("/nonexistent/photo.jpg"));
        assert_eq!(meta.filename.as_deref(), Some("photo.jpg"));
        assert!(meta.make.is_none());
    }
}
