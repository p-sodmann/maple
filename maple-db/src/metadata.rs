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
fn parse_exif_datetime(s: &str) -> Option<i64> {
    maple_import::ExifDateTime::parse(s).map(|dt| dt.to_unix_timestamp())
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
            let to_fill = crate::lock_db(&db).records_needing_metadata().unwrap_or_default();

            if to_fill.is_empty() {
                return;
            }

            tracing::info!("Metadata filler: {} records to process", to_fill.len());

            for (id, path) in to_fill {
                let meta = extract_metadata(&path);
                if let Err(e) = crate::lock_db(&db).update_metadata(id, &meta) {
                    tracing::warn!("Metadata filler: failed for {}: {e}", path.display());
                }
            }

            tracing::info!("Metadata filler: done");
        })
        .ok();
}

// ── Rotation ─────────────────────────────────────────────────────

/// Rotate the EXIF Orientation tag in a JPEG file 90° CW or CCW.
///
/// Patches the Orientation tag byte in-place — the JPEG pixel data is
/// not re-encoded.  Returns `(new_orientation, new_blake3_hash)` so the
/// caller can update the DB record in one shot.
///
/// Fails if the file is a raw format, is not a JPEG, or has no
/// existing EXIF Orientation tag.
pub fn rotate_image_file(path: &Path, clockwise: bool) -> anyhow::Result<(u16, [u8; 32])> {
    if maple_import::is_raw_format(path) {
        anyhow::bail!("RAW files cannot be rotated via EXIF");
    }

    let data = std::fs::read(path)?;
    let current = read_exif_orientation_from_bytes(&data);
    let new_orientation = if clockwise {
        rotate_cw(current)
    } else {
        rotate_ccw(current)
    };

    let patched = patch_jpeg_exif_orientation(&data, new_orientation)
        .ok_or_else(|| anyhow::anyhow!("No EXIF Orientation tag found in this image"))?;

    std::fs::write(path, &patched)?;
    let new_hash: [u8; 32] = *blake3::hash(&patched).as_bytes();
    Ok((new_orientation, new_hash))
}

fn rotate_cw(o: u32) -> u16 {
    match o {
        1 => 6, 2 => 7, 3 => 8, 4 => 5,
        5 => 2, 6 => 3, 7 => 4, 8 => 1,
        _ => 6,
    }
}

fn rotate_ccw(o: u32) -> u16 {
    match o {
        1 => 8, 2 => 5, 3 => 6, 4 => 7,
        5 => 4, 6 => 1, 7 => 2, 8 => 3,
        _ => 8,
    }
}

/// Read the EXIF Orientation tag directly from JPEG bytes (no file I/O).
fn read_exif_orientation_from_bytes(data: &[u8]) -> u32 {
    jpeg_exif_tiff(data)
        .and_then(|(tiff, le)| {
            find_orientation_entry(tiff, le).map(|(val_off, _)| {
                let b = &tiff[val_off..val_off + 2];
                (if le { u16::from_le_bytes([b[0], b[1]]) } else { u16::from_be_bytes([b[0], b[1]]) }) as u32
            })
        })
        .unwrap_or(1)
}

/// Patch the EXIF Orientation tag in JPEG bytes.  Returns `None` if the
/// tag cannot be located.
fn patch_jpeg_exif_orientation(data: &[u8], new_orientation: u16) -> Option<Vec<u8>> {
    let (tiff, le) = jpeg_exif_tiff(data)?;
    let tiff_offset = jpeg_exif_tiff_offset(data)?;
    let (val_off, _) = find_orientation_entry(tiff, le)?;

    let abs = tiff_offset + val_off;
    if abs + 2 > data.len() {
        return None;
    }
    let mut out = data.to_vec();
    let encoded = if le { new_orientation.to_le_bytes() } else { new_orientation.to_be_bytes() };
    out[abs] = encoded[0];
    out[abs + 1] = encoded[1];
    Some(out)
}

/// Return a slice pointing to the TIFF header inside the JPEG APP1/Exif
/// segment together with a `little_endian` flag, or `None`.
fn jpeg_exif_tiff(data: &[u8]) -> Option<(&[u8], bool)> {
    let off = jpeg_exif_tiff_offset(data)?;
    let tiff = &data[off..];
    let le = if tiff.starts_with(b"II") {
        true
    } else if tiff.starts_with(b"MM") {
        false
    } else {
        return None;
    };
    Some((tiff, le))
}

/// Absolute byte offset of the TIFF header within `data`.
fn jpeg_exif_tiff_offset(data: &[u8]) -> Option<usize> {
    if data.len() < 4 || data[0] != 0xFF || data[1] != 0xD8 {
        return None;
    }
    let mut i = 2usize;
    while i + 4 <= data.len() {
        if data[i] != 0xFF {
            return None;
        }
        let marker = data[i + 1];
        if marker == 0xD9 || marker == 0xDA {
            break;
        }
        let seg_len = u16::from_be_bytes([data[i + 2], data[i + 3]]) as usize;
        if seg_len < 2 || i + 2 + seg_len > data.len() {
            return None;
        }
        if marker == 0xE1 {
            let payload = i + 4;
            if data[payload..].starts_with(b"Exif\0\0") {
                return Some(payload + 6);
            }
        }
        i += 2 + seg_len;
    }
    None
}

/// Scan IFD0 for the Orientation tag (0x0112).
///
/// Returns `(value_field_offset_within_tiff, little_endian)` where
/// `value_field_offset` is byte 8 of the matching 12-byte IFD entry.
fn find_orientation_entry(tiff: &[u8], le: bool) -> Option<(usize, bool)> {
    let ru16 = |off: usize| -> u16 {
        if le { u16::from_le_bytes([tiff[off], tiff[off + 1]]) }
        else   { u16::from_be_bytes([tiff[off], tiff[off + 1]]) }
    };
    let ru32 = |off: usize| -> u32 {
        if le { u32::from_le_bytes([tiff[off], tiff[off+1], tiff[off+2], tiff[off+3]]) }
        else  { u32::from_be_bytes([tiff[off], tiff[off+1], tiff[off+2], tiff[off+3]]) }
    };

    if tiff.len() < 8 { return None; }
    if ru16(2) != 42 { return None; } // TIFF magic

    let ifd0 = ru32(4) as usize;
    if ifd0 + 2 > tiff.len() { return None; }

    let count = ru16(ifd0) as usize;
    let entries = ifd0 + 2;

    for n in 0..count {
        let e = entries + n * 12;
        if e + 12 > tiff.len() { break; }
        if ru16(e) == 0x0112 {
            return Some((e + 8, le));
        }
    }
    None
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
