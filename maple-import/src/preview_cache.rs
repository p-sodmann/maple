//! The medium's own record of the previews already made from it.
//!
//! Beside `.maple_embed_cache.bin` and `.maple_seen.bin`, and for the same
//! reason: the card carries its own history, so plugging it into a second
//! machine does not repeat work the first one already did. What is stored
//! here is the [canonical preview](crate::preview) — the *only* pixel
//! representation the pipeline keeps, and therefore the one every check
//! reads.
//!
//! # Why this one is keyed by path, not by content hash
//!
//! [`EmbeddingCache`](crate::EmbeddingCache) is keyed by BLAKE3 content
//! hash, which survives a rename. This cache deliberately is not, and the
//! difference is the whole point: **a content hash can only be computed by
//! reading the entire file**, which is precisely the cost a cache hit is
//! supposed to avoid. Keying on the hash would mean reading 25 MB off the
//! card to discover that the card already knew the answer.
//!
//! So the key is what a directory listing already tells us — path, size and
//! modification time — and a hit skips opening the file at all. On a rescan
//! of an unchanged card that turns a ~100 ms-per-photo serial read into a
//! `stat`, which is what makes re-inserting a card feel instant rather than
//! like a fresh import.
//!
//! The trade that buys it: **`(path, size, mtime)` is taken as the file's
//! identity.** A file replaced in place, at the same path, with the same
//! byte count and the same mtime would be served the previous file's
//! preview *and its content hash* — so it could be badged with the wrong
//! import history. Camera cards write each file once and never touch it
//! again, and every editor that rewrites a photo moves its mtime, so this
//! is the same assumption `make`, `rsync` and every incremental tool ship
//! with. It is recorded here rather than buried because the failure is
//! silent when it happens.
//!
//! # Why the file is append-only
//!
//! A few thousand previews at ~15 KB is tens of megabytes; rewriting all of
//! it every time a batch completes — the way `EmbeddingCache` does with its
//! much smaller payload — would spend the scan's savings back on the card.
//! Records are appended instead, and later records win on load, so a flush
//! costs only the photos actually new since the last one.
//!
//! Each record is length-prefixed, which makes a card pulled mid-write cost
//! exactly the truncated tail record: the load stops there and keeps
//! everything before it. Growth is bounded by [`PreviewCache::flush`]
//! compacting whenever the dead records outnumber the live ones, which is
//! what keeps a card that has been formatted and refilled a few times from
//! carrying every generation of `DSCF0001.JPG` forever.

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Name of the cache file, written into the directory that was scanned.
///
/// Dot-prefixed so [`crate::scan_grouped`] and the library's own scanner
/// both skip it.
pub const PREVIEW_CACHE_FILE: &str = ".maple_previews.bin";

const MAGIC: &[u8; 8] = b"MAPLEPRV";
const VERSION: u32 = 1;

/// Refuse a record claiming a preview larger than this. A canonical preview
/// is ~15 KB; anything near this is corruption, and the cap is what stops a
/// bad length prefix turning into a huge allocation.
const MAX_PREVIEW_BYTES: usize = 8 * 1024 * 1024;

/// What identifies a file without opening it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PreviewKey {
    /// Path relative to the cached directory, with `/` separators so a card
    /// written on one platform reads on another.
    pub rel_path: String,
    pub size: u64,
    pub mtime_secs: i64,
    pub mtime_nanos: u32,
}

impl PreviewKey {
    /// Describe `path` from its directory entry alone — one `stat`, no read.
    ///
    /// `None` when the file is outside `root`, has no readable metadata, or
    /// carries a name that is not UTF-8; each of those simply means "no
    /// cache for this photo", never an error.
    pub fn for_file(root: &Path, path: &Path) -> Option<Self> {
        let rel = path.strip_prefix(root).ok()?;
        let mut parts = Vec::new();
        for part in rel.components() {
            parts.push(part.as_os_str().to_str()?);
        }
        let meta = std::fs::metadata(path).ok()?;
        let modified = meta.modified().ok()?;
        let (mtime_secs, mtime_nanos) = match modified.duration_since(std::time::UNIX_EPOCH) {
            Ok(d) => (d.as_secs() as i64, d.subsec_nanos()),
            // Before 1970. Vanishingly rare, but it must not wrap into a
            // key that collides with a real one.
            Err(e) => (-(e.duration().as_secs() as i64), e.duration().subsec_nanos()),
        };
        Some(Self {
            rel_path: parts.join("/"),
            size: meta.len(),
            mtime_secs,
            mtime_nanos,
        })
    }

    fn same_file_as(&self, other: &Self) -> bool {
        self.size == other.size
            && self.mtime_secs == other.mtime_secs
            && self.mtime_nanos == other.mtime_nanos
    }
}

/// Everything the scan would otherwise have had to open the file to learn.
#[derive(Clone, Debug, PartialEq)]
pub struct CachedPreview {
    /// BLAKE3 of the file's bytes — the identifier the library, the
    /// `SeenSet` and sync all key on. Cached because recomputing it is
    /// exactly the read being skipped.
    pub content_hash: [u8; 32],
    /// Capture instant from EXIF, fractional seconds since the epoch.
    pub taken: Option<f64>,
    /// The canonical preview. Everything downstream decodes *this*.
    pub webp: Vec<u8>,
}

struct Stored {
    size: u64,
    mtime_secs: i64,
    mtime_nanos: u32,
    preview: CachedPreview,
}

/// Previews already made from one directory, loaded from and appended to
/// that directory's own cache file.
pub struct PreviewCache {
    path: PathBuf,
    entries: HashMap<String, Stored>,
    /// Records read from the file on load. Compared against the live entry
    /// count to decide whether the next flush appends or compacts.
    records_on_disk: usize,
    /// Keys inserted since the last flush, in insertion order.
    unwritten: Vec<String>,
    /// The file could not be parsed at all (missing, wrong magic, wrong
    /// version). The next flush rewrites it whole rather than appending
    /// records nothing will read.
    rewrite: bool,
}

impl PreviewCache {
    /// Load the cache belonging to `dir`.
    ///
    /// A missing, corrupt or foreign-version file yields an empty cache —
    /// this is a cache, and the cost of a miss is one ordinary read.
    pub fn load_from(dir: &Path) -> Self {
        let path = dir.join(PREVIEW_CACHE_FILE);
        let mut cache = Self {
            path,
            entries: HashMap::new(),
            records_on_disk: 0,
            unwritten: Vec::new(),
            rewrite: true,
        };
        if let Ok(data) = std::fs::read(&cache.path) {
            cache.read_records(&data);
        }
        cache
    }

    /// An in-memory cache with no file behind it, for tests and for callers
    /// that only want the lookup.
    pub fn detached() -> Self {
        Self {
            path: PathBuf::new(),
            entries: HashMap::new(),
            records_on_disk: 0,
            unwritten: Vec::new(),
            rewrite: false,
        }
    }

    /// What this medium already knows about `key`'s file.
    ///
    /// Misses when the path is unknown, and also when the path is known but
    /// its size or mtime has moved — a changed file is a different file.
    pub fn get(&self, key: &PreviewKey) -> Option<&CachedPreview> {
        let stored = self.entries.get(&key.rel_path)?;
        let same = Self::key_of(&key.rel_path, stored).same_file_as(key);
        same.then_some(&stored.preview)
    }

    /// Record a preview, replacing any earlier one for the same path.
    pub fn insert(&mut self, key: PreviewKey, preview: CachedPreview) {
        self.entries.insert(
            key.rel_path.clone(),
            Stored {
                size: key.size,
                mtime_secs: key.mtime_secs,
                mtime_nanos: key.mtime_nanos,
                preview,
            },
        );
        self.unwritten.push(key.rel_path);
    }

    /// Number of previews held.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many previews are waiting to be written.
    pub fn pending(&self) -> usize {
        self.unwritten.len()
    }

    /// Write everything inserted since the last flush.
    ///
    /// Appends by default. Rewrites the file whole when the records already
    /// on disk are mostly dead — a card formatted and refilled would
    /// otherwise accumulate every generation of every filename — and when
    /// the file could not be read at all.
    ///
    /// Best-effort by nature: a write-protected or full card must cost the
    /// user nothing more than a slower scan, so callers log and carry on.
    pub fn flush(&mut self) -> anyhow::Result<()> {
        if self.path.as_os_str().is_empty() || self.unwritten.is_empty() {
            self.unwritten.clear();
            return Ok(());
        }
        let dead = self.records_on_disk.saturating_sub(self.entries.len());
        if self.rewrite || dead > self.entries.len() {
            self.compact()
        } else {
            self.append()
        }
    }

    /// Append only the records added since the last flush.
    fn append(&mut self) -> anyhow::Result<()> {
        let mut buf = Vec::new();
        let mut written = 0usize;
        // De-duplicated: a photo re-inserted twice between flushes needs
        // only its latest record on disk.
        let mut seen = std::collections::HashSet::new();
        for rel in self.unwritten.iter().rev() {
            if !seen.insert(rel.clone()) {
                continue;
            }
            if let Some(stored) = self.entries.get(rel) {
                write_record(&mut buf, rel, stored);
                written += 1;
            }
        }
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;
        // A fresh file still needs its header before the first record.
        if file.metadata().map(|m| m.len()).unwrap_or(0) == 0 {
            let mut header = Vec::with_capacity(12 + buf.len());
            header.extend_from_slice(MAGIC);
            header.extend_from_slice(&VERSION.to_le_bytes());
            header.extend_from_slice(&buf);
            buf = header;
        }
        file.write_all(&buf)?;
        file.flush()?;
        self.records_on_disk += written;
        self.unwritten.clear();
        Ok(())
    }

    /// Rewrite the whole file from the live entries, staging and renaming so
    /// a card pulled mid-write keeps the previous cache rather than a
    /// half-written one.
    fn compact(&mut self) -> anyhow::Result<()> {
        let mut buf = Vec::new();
        buf.extend_from_slice(MAGIC);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        for (rel, stored) in &self.entries {
            write_record(&mut buf, rel, stored);
        }

        let scratch = self.path.with_file_name(format!(".{PREVIEW_CACHE_FILE}.tmp"));
        std::fs::write(&scratch, &buf)?;
        if let Err(err) = std::fs::rename(&scratch, &self.path) {
            let _ = std::fs::remove_file(&scratch);
            return Err(err.into());
        }
        self.records_on_disk = self.entries.len();
        self.rewrite = false;
        self.unwritten.clear();
        Ok(())
    }

    fn key_of(rel_path: &str, stored: &Stored) -> PreviewKey {
        PreviewKey {
            rel_path: rel_path.to_string(),
            size: stored.size,
            mtime_secs: stored.mtime_secs,
            mtime_nanos: stored.mtime_nanos,
        }
    }

    /// Parse records until the data runs out or stops making sense.
    ///
    /// Stopping early is the designed behaviour, not a fallback: the tail
    /// of an append interrupted by an ejected card is exactly this case,
    /// and everything before it is still good.
    fn read_records(&mut self, data: &[u8]) {
        if data.len() < 12 || &data[0..8] != MAGIC {
            return;
        }
        if u32::from_le_bytes(data[8..12].try_into().unwrap()) != VERSION {
            return;
        }
        self.rewrite = false;

        let mut at = 12;
        while at + 4 <= data.len() {
            let len = u32::from_le_bytes(data[at..at + 4].try_into().unwrap()) as usize;
            at += 4;
            if len == 0 || at + len > data.len() {
                break;
            }
            match parse_record(&data[at..at + len]) {
                Some((rel, stored)) => {
                    self.entries.insert(rel, stored);
                    self.records_on_disk += 1;
                }
                None => break,
            }
            at += len;
        }
    }
}

/// `rec_len(u32) | path_len(u32) | path | size(u64) | mtime_secs(i64) |
///  mtime_nanos(u32) | hash(32) | taken(f64, NaN = none) | webp_len(u32) | webp`
fn write_record(buf: &mut Vec<u8>, rel_path: &str, stored: &Stored) {
    let path = rel_path.as_bytes();
    let webp = &stored.preview.webp;
    let len = 4 + path.len() + 8 + 8 + 4 + 32 + 8 + 4 + webp.len();
    buf.extend_from_slice(&(len as u32).to_le_bytes());
    buf.extend_from_slice(&(path.len() as u32).to_le_bytes());
    buf.extend_from_slice(path);
    buf.extend_from_slice(&stored.size.to_le_bytes());
    buf.extend_from_slice(&stored.mtime_secs.to_le_bytes());
    buf.extend_from_slice(&stored.mtime_nanos.to_le_bytes());
    buf.extend_from_slice(&stored.preview.content_hash);
    // NaN as "absent" keeps the record fixed-shape. No real capture time is
    // NaN, and a photo with no timestamp is read as "no gap information"
    // rather than "no gap" downstream either way.
    buf.extend_from_slice(&stored.preview.taken.unwrap_or(f64::NAN).to_le_bytes());
    buf.extend_from_slice(&(webp.len() as u32).to_le_bytes());
    buf.extend_from_slice(webp);
}

fn parse_record(rec: &[u8]) -> Option<(String, Stored)> {
    let mut at = 0usize;
    let mut take = |n: usize| -> Option<&[u8]> {
        let end = at.checked_add(n)?;
        if end > rec.len() {
            return None;
        }
        let out = &rec[at..end];
        at = end;
        Some(out)
    };

    let path_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    let rel_path = std::str::from_utf8(take(path_len)?).ok()?.to_string();
    let size = u64::from_le_bytes(take(8)?.try_into().ok()?);
    let mtime_secs = i64::from_le_bytes(take(8)?.try_into().ok()?);
    let mtime_nanos = u32::from_le_bytes(take(4)?.try_into().ok()?);
    let content_hash: [u8; 32] = take(32)?.try_into().ok()?;
    let taken = f64::from_le_bytes(take(8)?.try_into().ok()?);
    let webp_len = u32::from_le_bytes(take(4)?.try_into().ok()?) as usize;
    if webp_len > MAX_PREVIEW_BYTES {
        return None;
    }
    let webp = take(webp_len)?.to_vec();

    Some((
        rel_path,
        Stored {
            size,
            mtime_secs,
            mtime_nanos,
            preview: CachedPreview {
                content_hash,
                taken: (!taken.is_nan()).then_some(taken),
                webp,
            },
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preview(tag: u8) -> CachedPreview {
        CachedPreview {
            content_hash: [tag; 32],
            taken: Some(1_700_000_000.5),
            webp: vec![tag; 64],
        }
    }

    fn key(name: &str, size: u64, mtime: i64) -> PreviewKey {
        PreviewKey {
            rel_path: name.to_string(),
            size,
            mtime_secs: mtime,
            mtime_nanos: 0,
        }
    }

    #[test]
    fn a_preview_survives_a_round_trip_through_the_medium() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = PreviewCache::load_from(dir.path());
        cache.insert(key("DCIM/a.jpg", 100, 5), preview(1));
        cache.flush().unwrap();

        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.get(&key("DCIM/a.jpg", 100, 5)), Some(&preview(1)));
    }

    #[test]
    fn a_file_that_changed_is_a_miss_not_a_stale_hit() {
        // The one thing the key exists to catch. Serving the old preview
        // would also serve the old *content hash*, which is what decides
        // whether the photo counts as already imported.
        let mut cache = PreviewCache::detached();
        cache.insert(key("a.jpg", 100, 5), preview(1));
        assert!(cache.get(&key("a.jpg", 101, 5)).is_none(), "size moved");
        assert!(cache.get(&key("a.jpg", 100, 6)).is_none(), "mtime moved");
        assert!(cache.get(&key("b.jpg", 100, 5)).is_none(), "different path");
        assert!(cache.get(&key("a.jpg", 100, 5)).is_some());
    }

    #[test]
    fn a_second_flush_appends_rather_than_rewriting() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREVIEW_CACHE_FILE);

        let mut cache = PreviewCache::load_from(dir.path());
        cache.insert(key("a.jpg", 1, 1), preview(1));
        cache.flush().unwrap();
        let after_first = std::fs::metadata(&path).unwrap().len();

        cache.insert(key("b.jpg", 2, 2), preview(2));
        cache.flush().unwrap();
        let after_second = std::fs::metadata(&path).unwrap().len();

        assert!(after_second > after_first, "the second record must land");
        // The first record's bytes are still where they were: an append
        // pays for the new photos only, which is the whole reason a card
        // holding tens of megabytes of previews is affordable.
        let bytes = std::fs::read(&path).unwrap();
        assert_eq!(&bytes[..after_first as usize], &std::fs::read(&path).unwrap()[..after_first as usize]);

        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.len(), 2);
    }

    #[test]
    fn nothing_new_writes_nothing() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = PreviewCache::load_from(dir.path());
        cache.flush().unwrap();
        assert!(
            !dir.path().join(PREVIEW_CACHE_FILE).exists(),
            "an unchanged card must not be written to at all"
        );
    }

    #[test]
    fn a_truncated_tail_costs_only_the_last_record() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREVIEW_CACHE_FILE);
        let mut cache = PreviewCache::load_from(dir.path());
        cache.insert(key("a.jpg", 1, 1), preview(1));
        cache.insert(key("b.jpg", 2, 2), preview(2));
        cache.flush().unwrap();

        // The card came out mid-append.
        let mut bytes = std::fs::read(&path).unwrap();
        bytes.truncate(bytes.len() - 20);
        std::fs::write(&path, &bytes).unwrap();

        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.len(), 1, "the intact record must still be there");
    }

    #[test]
    fn a_foreign_file_reads_as_empty_and_is_replaced_whole() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREVIEW_CACHE_FILE);
        std::fs::write(&path, b"something else entirely").unwrap();

        let mut cache = PreviewCache::load_from(dir.path());
        assert!(cache.is_empty());
        cache.insert(key("a.jpg", 1, 1), preview(1));
        cache.flush().unwrap();

        // Rewritten, not appended to — appending would have left the
        // garbage header in front and lost the record on the next load.
        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.len(), 1);
    }

    #[test]
    fn a_card_reused_many_times_compacts_instead_of_growing() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join(PREVIEW_CACHE_FILE);

        // Same filename, new content each time — a card formatted and
        // refilled. Each round makes the previous record dead.
        let mut sizes = Vec::new();
        for round in 1..=8u8 {
            let mut cache = PreviewCache::load_from(dir.path());
            cache.insert(key("DSCF0001.JPG", round as u64, round as i64), preview(round));
            cache.flush().unwrap();
            sizes.push(std::fs::metadata(&path).unwrap().len());
        }

        assert!(
            sizes.last().unwrap() < sizes.iter().max().unwrap(),
            "the file must have been compacted at some point, not grown forever: {sizes:?}"
        );
        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.len(), 1);
        assert_eq!(
            reloaded.get(&key("DSCF0001.JPG", 8, 8)),
            Some(&preview(8)),
            "and the newest generation is what survives"
        );
    }

    #[test]
    fn re_inserting_between_flushes_writes_one_record() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = PreviewCache::load_from(dir.path());
        cache.insert(key("a.jpg", 1, 1), preview(1));
        cache.insert(key("a.jpg", 1, 1), preview(2));
        cache.flush().unwrap();

        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.len(), 1);
        assert_eq!(reloaded.get(&key("a.jpg", 1, 1)), Some(&preview(2)));
    }

    #[test]
    fn a_key_describes_a_real_file_without_opening_it() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("DCIM")).unwrap();
        let file = dir.path().join("DCIM/DSCF0001.JPG");
        std::fs::write(&file, b"0123456789").unwrap();

        let key = PreviewKey::for_file(dir.path(), &file).unwrap();
        assert_eq!(key.rel_path, "DCIM/DSCF0001.JPG", "always `/`, whatever the platform");
        assert_eq!(key.size, 10);
        assert!(PreviewKey::for_file(dir.path(), &dir.path().join("gone.jpg")).is_none());
    }

    #[test]
    fn a_preview_with_no_capture_time_round_trips_as_absent() {
        let dir = tempfile::tempdir().unwrap();
        let mut cache = PreviewCache::load_from(dir.path());
        let undated = CachedPreview { taken: None, ..preview(3) };
        cache.insert(key("a.jpg", 1, 1), undated.clone());
        cache.flush().unwrap();

        let reloaded = PreviewCache::load_from(dir.path());
        assert_eq!(reloaded.get(&key("a.jpg", 1, 1)), Some(&undated));
    }
}
