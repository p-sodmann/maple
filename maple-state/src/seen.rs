//! Persistent sets for tracking previously imported and rejected images.
//!
//! Two [`Record`]s are kept, [`Record::Imported`] and [`Record::Skipped`] —
//! what was copied into a library, and what the user deliberately passed over
//! in the import browser. Together they are "what an earlier session already
//! decided about this photo", which is what the browser's *Hide old images*
//! filter asks.
//!
//! Each record lives in two places:
//!
//! * `<source>/.maple_seen.bin` and `<source>/.maple_skipped.bin` — the
//!   **authoritative** copies, written to the medium itself so the card
//!   carries its own history to whichever machine it is plugged into next.
//!   Same idiom as `maple_import::EmbeddingCache`'s `.maple_embed_cache.bin`,
//!   which sits beside them.
//! * `seen_imported.bin` and `seen_skipped.bin` — non-authoritative replicas
//!   in the library directory, read only when a source carries no record of
//!   its own: read-only cards, network shares, and plain folders that later
//!   move would otherwise have no memory at all.
//!
//! Each stores the full 32-byte BLAKE3 content hashes of its members, both in
//! memory and on disk.
//!
//! The set is **grow-only**, which is what lets [`SeenSet::merge_save_to_source`]
//! be a read-merge-write with no locking and no conflict resolution: a union
//! is the same operation whether it is combining two concurrent importers or
//! folding one card's history into another's. Nothing ever removes a hash, so
//! there is no delete for a merge to lose.
//!
//! Membership queries are **exact** in both directions: [`SeenSet::contains`]
//! answers from the hash set itself, so the only way two distinct images can
//! collide is a genuine BLAKE3 collision (2⁻²⁵⁶ per pair).
//!
//! A bloom filter sits in front of the hash set purely as a fast-reject stage:
//! all-bits-set is a *maybe*, which is then confirmed against the hash set; a
//! single clear bit is a definitive "not present" and short-circuits before any
//! 32-byte comparison happens. It is a cache-friendly early-out, not a
//! correctness mechanism — the hash set alone is already O(1).

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// Number of hash functions for the bloom filter.
const K: u32 = 7;

/// Minimum bloom filter size in bits.
const MIN_BITS: usize = 8192;

/// File format version.
const VERSION: u32 = 1;

/// Placeholder the import scan stores when hashing a file fails
/// (`maple-ui/src/import.rs`). It is not a content hash, and letting it into
/// the set would badge *every* unreadable photo on the next scan as already
/// imported. [`SeenSet::insert`] refuses it on the way in, and the loader
/// drops it on the way out of a file that already caught it.
const UNHASHED: [u8; 32] = [0u8; 32];

/// Makes scratch filenames unique within one process; the pid separates
/// processes. Two importers saving at the same moment must not end up
/// writing the same temp file out from under each other.
static TEMP_SEQ: AtomicU64 = AtomicU64::new(0);

/// Which of the two per-medium records a load or save is talking about.
///
/// The pair is deliberately not one set with a flag per hash: both are
/// grow-only, and keeping them apart is what lets a union be the only
/// combining operation either one ever needs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Record {
    /// Photos copied into a library.
    Imported,
    /// Photos the user moved past without marking — the import browser's
    /// red ✗. Kept so a card triaged in one sitting does not present the
    /// same rejects again in the next.
    Skipped,
}

impl Record {
    /// Filename at the root of an import medium.
    ///
    /// Dotfile-prefixed so the import scanner's existing hidden-file
    /// filtering already skips it, exactly like `.maple_embed_cache.bin`
    /// next to it.
    pub fn on_medium(self) -> &'static str {
        match self {
            Self::Imported => ".maple_seen.bin",
            Self::Skipped => ".maple_skipped.bin",
        }
    }

    /// Filename of the library-side replica.
    pub fn replica(self) -> &'static str {
        match self {
            Self::Imported => "seen_imported.bin",
            Self::Skipped => "seen_skipped.bin",
        }
    }
}

/// Persistent set keyed by full 32-byte BLAKE3 content hashes.
pub struct SeenSet {
    /// Bloom filter bit array — a fast-reject stage in front of `hashes`.
    bits: Vec<u64>,
    /// Number of usable bits (`bits.len() * 64`).
    num_bits: usize,
    /// The authoritative membership set. Also what gets persisted; the
    /// hashes carry no meaningful order, so one container serves both roles
    /// and there is no second copy that could drift out of sync.
    hashes: HashSet<[u8; 32]>,
}

impl SeenSet {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    fn with_capacity(expected: usize) -> Self {
        let num_bits = optimal_bits(expected).max(MIN_BITS);
        let words = num_bits.div_ceil(64);
        Self {
            bits: vec![0u64; words],
            num_bits: words * 64,
            hashes: HashSet::with_capacity(expected),
        }
    }

    // ── Named constructors ───────────────────────────────────────

    /// Load a record's library-side replica out of `dir`.
    pub fn load_replica(dir: &Path, record: Record) -> Self {
        Self::load_from(&dir.join(record.replica()))
    }

    /// Load `record` for an import medium rooted at `root`.
    ///
    /// The medium's own copy is authoritative. The library replica is
    /// consulted **only** when the medium carries no readable record of its
    /// own — a card that was never written to because it is read-only, a
    /// network share, a folder that has since moved. A medium record that
    /// exists but is corrupt falls back the same way: an empty set there
    /// would silently forget every decision ever made about it.
    pub fn load_for_source(root: &Path, library_dir: &Path, record: Record) -> Self {
        Self::read_file(&root.join(record.on_medium()))
            .unwrap_or_else(|| Self::load_replica(library_dir, record))
    }

    // ── Load / Save ─────────────────────────────────────────────

    /// Load from a specific file (`Self::new()` on any error).
    pub fn load_from(path: &Path) -> Self {
        Self::read_file(path).unwrap_or_default()
    }

    /// Load from a specific file, distinguishing "nothing readable there"
    /// from "an empty set". Only [`Self::load_for_source`] needs the
    /// difference, and for it the difference is the whole point.
    fn read_file(path: &Path) -> Option<Self> {
        Self::parse(&std::fs::read(path).ok()?)
    }

    /// Save a record's library-side replica into `dir`.
    pub fn save_replica(&self, dir: &Path, record: Record) -> anyhow::Result<()> {
        self.save_to(&dir.join(record.replica()))
    }

    /// Save to a specific file, replacing it atomically.
    ///
    /// The bytes land in a scratch file beside the target and are renamed
    /// over it, so a reader never sees a half-written set and a crash or an
    /// ejected card mid-write leaves the previous record intact rather than
    /// a truncated one.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        let tmp = scratch_path(path);
        std::fs::write(&tmp, self.to_bytes())?;
        if let Err(err) = std::fs::rename(&tmp, path) {
            let _ = std::fs::remove_file(&tmp);
            return Err(err.into());
        }
        Ok(())
    }

    /// Fold this set into the record on the import medium at `root`, and
    /// into the library replica.
    ///
    /// Read-merge-write, both times: whatever is on disk is loaded, unioned
    /// with `self`, and written back. Two importers running at once
    /// therefore combine instead of clobbering each other — the old
    /// load-modify-write in `maple-ui/src/import.rs` dropped one of the two
    /// runs entirely — and the same code path merges a card's history into
    /// a library that has never seen it.
    ///
    /// Returns `Ok(true)` when the medium itself took the write, `Ok(false)`
    /// when only the replica did (a read-only card — expected, not a
    /// failure), and `Err` when neither did and the record was lost.
    pub fn merge_save_to_source(
        &self,
        root: &Path,
        library_dir: &Path,
        record: Record,
    ) -> anyhow::Result<bool> {
        // An empty root joins to a bare relative name, i.e. the process's
        // working directory — nobody's import medium. Treat it as unwritable
        // rather than littering wherever the app happens to have been
        // launched from.
        let on_medium = if root.as_os_str().is_empty() {
            Err(anyhow::anyhow!("no source root to write the record to"))
        } else {
            self.merge_save_into(&root.join(record.on_medium()))
        };
        let replica = self.merge_save_into(&library_dir.join(record.replica()));
        match (on_medium, replica) {
            (Ok(()), _) => Ok(true),
            (Err(_), Ok(())) => Ok(false),
            (Err(medium), Err(replica)) => Err(medium.context(format!(
                "neither the medium nor the library replica could be written ({replica})"
            ))),
        }
    }

    /// One half of [`Self::merge_save_to_source`]: reload `path`, union
    /// `self` into it, write it back.
    fn merge_save_into(&self, path: &Path) -> anyhow::Result<()> {
        let mut merged = Self::load_from(path);
        merged.merge(self);
        merged.save_to(path)
    }

    // ── Core API ────────────────────────────────────────────────

    /// Insert a full 32-byte BLAKE3 content hash.
    ///
    /// Inserting the same hash twice is a no-op: the set stores each hash
    /// once, so [`Self::len`] counts distinct images. The all-zero
    /// [`UNHASHED`] placeholder is refused outright — see its comment.
    pub fn insert(&mut self, hash: &[u8; 32]) {
        if *hash == UNHASHED || !self.hashes.insert(*hash) {
            return;
        }
        // Resize the bloom filter if the load factor is getting too high.
        if self.hashes.len() * 10 > self.num_bits {
            self.rebuild_bloom();
        } else {
            self.bloom_insert(hash);
        }
    }

    /// Check whether a hash is in the set. **Exact** in both directions.
    ///
    /// The bloom filter answers first: if it says "not present" that is
    /// definitive and we return immediately. A bloom *maybe* — which includes
    /// its false positives (up to ~1 %) — is then confirmed against the stored
    /// hashes, so a `true` result means this exact 32-byte hash was inserted.
    pub fn contains(&self, hash: &[u8; 32]) -> bool {
        self.bloom_maybe_contains(hash) && self.hashes.contains(hash)
    }

    /// Union `other` into this set.
    ///
    /// The only combining operation the format needs: the set is grow-only,
    /// so a union can never lose information and never has to choose a
    /// winner.
    pub fn merge(&mut self, other: &Self) {
        for hash in &other.hashes {
            self.insert(hash);
        }
    }

    /// Number of hashes stored.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    // ── Internals ───────────────────────────────────────────────

    /// Bloom fast-reject stage.
    ///
    /// `false` → definitely not in the set. `true` → maybe; the caller must
    /// confirm against `hashes`.
    fn bloom_maybe_contains(&self, hash: &[u8; 32]) -> bool {
        if self.num_bits == 0 {
            return false;
        }
        for i in 0..K {
            let pos = bloom_pos(hash, i, self.num_bits);
            if self.bits[pos / 64] & (1u64 << (pos % 64)) == 0 {
                return false;
            }
        }
        true
    }

    fn bloom_insert(&mut self, hash: &[u8; 32]) {
        for i in 0..K {
            let pos = bloom_pos(hash, i, self.num_bits);
            self.bits[pos / 64] |= 1u64 << (pos % 64);
        }
    }

    fn rebuild_bloom(&mut self) {
        let num_bits = optimal_bits(self.hashes.len()).max(MIN_BITS);
        let words = num_bits.div_ceil(64);
        let num_bits = words * 64;
        let mut bits = vec![0u64; words];
        for h in &self.hashes {
            for i in 0..K {
                let pos = bloom_pos(h, i, num_bits);
                bits[pos / 64] |= 1u64 << (pos % 64);
            }
        }
        self.bits = bits;
        self.num_bits = num_bits;
    }

    /// Binary format: `version(u32 LE) | count(u32 LE) | hashes(count × 32 bytes)`.
    ///
    /// Storage: 8 B header + 32 B/image.  100 k images ≈ 3.2 MB.
    ///
    /// Hashes are written in sorted order. Their order carries no meaning, but
    /// `HashSet` iteration is arbitrary, and sorting keeps the file byte-stable
    /// across saves of the same content.
    fn to_bytes(&self) -> Vec<u8> {
        let count = self.hashes.len() as u32;
        let mut sorted: Vec<&[u8; 32]> = self.hashes.iter().collect();
        sorted.sort_unstable();
        let mut buf = Vec::with_capacity(8 + self.hashes.len() * 32);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        for h in sorted {
            buf.extend_from_slice(h);
        }
        buf
    }

    /// Parse the binary format, or `None` when these bytes are not a set
    /// this version can read.
    fn parse(data: &[u8]) -> Option<Self> {
        if data.len() < 8 {
            return None;
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if version != VERSION {
            return None;
        }
        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < 8 + count * 32 {
            return None;
        }
        let mut hashes = HashSet::with_capacity(count);
        for i in 0..count {
            let off = 8 + i * 32;
            let h: [u8; 32] = data[off..off + 32].try_into().unwrap();
            // Drop the sentinel on read as well as on write: a file written
            // before `insert` started refusing it is already poisoned, and
            // this is what un-poisons it on the next save.
            if h != UNHASHED {
                hashes.insert(h);
            }
        }
        let mut set = Self {
            bits: Vec::new(),
            num_bits: 0,
            hashes,
        };
        set.rebuild_bloom();
        Some(set)
    }
}

impl Default for SeenSet {
    fn default() -> Self {
        Self::new()
    }
}

/// Scratch filename to stage an atomic replacement of `path` in.
///
/// Beside the target, so the rename stays within one filesystem, and unique
/// per process *and* per call: concurrent importers must not stage into the
/// same file, or one would rename the other's half-written bytes into place.
fn scratch_path(path: &Path) -> PathBuf {
    let name = path.file_name().unwrap_or_default().to_string_lossy();
    let seq = TEMP_SEQ.fetch_add(1, Ordering::Relaxed);
    path.with_file_name(format!(".{name}.{}.{seq}.tmp", std::process::id()))
}

// ── Bloom filter math ────────────────────────────────────────────

/// Bits for `n` items targeting a ~1 % bloom false-positive rate.
///
/// Rounding up to a power of two means the realised rate is usually better
/// than the target — it swings between ~0.06 % just after a resize and ~1 %
/// just before the next one. Either way it only costs a wasted hash-set
/// lookup; [`SeenSet::contains`] is exact regardless.
fn optimal_bits(n: usize) -> usize {
    if n == 0 {
        return MIN_BITS;
    }
    // m = -n · ln(0.01) / (ln 2)²  ≈  n × 9.585
    ((n as f64 * 9.585).ceil() as usize).next_power_of_two()
}

/// Bit position for hash function `i` using double hashing on the
/// first 16 bytes of the BLAKE3 output.
///
/// `h1 = low64(hash)`, `h2 = high64(hash[8..16]) | 1` (non-zero).
fn bloom_pos(hash: &[u8; 32], i: u32, num_bits: usize) -> usize {
    let h1 = u64::from_le_bytes(hash[0..8].try_into().unwrap());
    let h2 = u64::from_le_bytes(hash[8..16].try_into().unwrap()) | 1;
    let combined = h1.wrapping_add((i as u64).wrapping_mul(h2));
    (combined % num_bits as u64) as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a pseudorandom 32-byte hash from a u64 seed.
    ///
    /// Uses multiply-xorshift across four 8-byte blocks so every bit of the
    /// output depends on `seed`, giving good bloom-filter coverage without
    /// requiring blake3 as a test dependency.
    fn fake_hash(seed: u64) -> [u8; 32] {
        let mut state = seed ^ 0xdead_beef_cafe_babe;
        let mut h = [0u8; 32];
        for i in 0..4u64 {
            state = state
                .wrapping_mul(0x517c_c1b7_2722_0a95)
                .wrapping_add(i.wrapping_mul(0x6c62_272e_07bb_0142));
            state ^= state >> 32;
            let start = i as usize * 8;
            h[start..start + 8].copy_from_slice(&state.to_le_bytes());
        }
        h
    }

    /// Replace the bloom filter with one whose bits are all set, so the
    /// fast-reject stage answers "maybe" for *every* hash. Membership then
    /// rests entirely on the exact hash set — a deterministic worst case,
    /// rather than waiting for a bloom collision to happen by chance.
    fn saturate_bloom(set: &mut SeenSet) {
        set.bits = vec![u64::MAX; 1];
        set.num_bits = 64;
    }

    #[test]
    fn insert_and_query() {
        let mut set = SeenSet::new();
        let h = fake_hash(1);
        assert!(!set.contains(&h));
        set.insert(&h);
        assert!(set.contains(&h));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn exact_under_total_bloom_saturation() {
        let mut set = SeenSet::new();
        for i in 0..1000u64 {
            set.insert(&fake_hash(i));
        }
        saturate_bloom(&mut set);

        // Precondition: the fast-reject stage now rejects nothing, so every
        // query below reaches the exact check.
        for i in 100_000..101_000u64 {
            assert!(set.bloom_maybe_contains(&fake_hash(i)));
        }

        // Not one of the never-inserted hashes may report as contained.
        let mut false_positives = 0;
        for i in 100_000..101_000u64 {
            if set.contains(&fake_hash(i)) {
                false_positives += 1;
            }
        }
        assert_eq!(false_positives, 0, "membership is not exact");

        // Everything actually inserted still matches.
        for i in 0..1000u64 {
            assert!(set.contains(&fake_hash(i)));
        }
    }

    #[test]
    fn exactness_survives_roundtrip() {
        let mut set = SeenSet::new();
        for i in 0..500u64 {
            set.insert(&fake_hash(i));
        }

        // `from_bytes` rebuilds the bloom from scratch; it must rebuild the
        // exact index too.
        let mut loaded = SeenSet::parse(&set.to_bytes()).unwrap();
        assert_eq!(loaded.len(), 500);
        saturate_bloom(&mut loaded);

        for i in 0..500u64 {
            assert!(loaded.contains(&fake_hash(i)));
        }
        for i in 100_000..100_500u64 {
            assert!(!loaded.contains(&fake_hash(i)));
        }
    }

    #[test]
    fn bloom_rejects_before_the_exact_check() {
        let mut set = SeenSet::new();
        for i in 0..1000u64 {
            set.insert(&fake_hash(i));
        }
        // At the designed ~1 % rate the vast majority of non-members are
        // rejected by the bloom alone, never touching the hash set.
        let rejected = (100_000..101_000u64)
            .filter(|i| !set.bloom_maybe_contains(&fake_hash(*i)))
            .count();
        assert!(rejected > 900, "bloom rejected only {rejected}/1000");
    }

    #[test]
    fn duplicate_insert_is_deduped() {
        let mut set = SeenSet::new();
        let h = fake_hash(7);
        set.insert(&h);
        set.insert(&h);
        set.insert(&h);
        assert_eq!(set.len(), 1);
        assert!(set.contains(&h));

        // And the duplicates do not survive a save/load cycle either.
        let loaded = SeenSet::parse(&set.to_bytes()).unwrap();
        assert_eq!(loaded.len(), 1);
        assert!(loaded.contains(&h));
    }

    #[test]
    fn roundtrip_bytes() {
        let mut set = SeenSet::new();
        let h1 = fake_hash(10);
        let h2 = fake_hash(20);
        let h3 = fake_hash(30);
        set.insert(&h1);
        set.insert(&h2);
        set.insert(&h3);

        let bytes = set.to_bytes();
        let loaded = SeenSet::parse(&bytes).unwrap();

        assert_eq!(loaded.len(), 3);
        assert!(loaded.contains(&h1));
        assert!(loaded.contains(&h2));
        assert!(loaded.contains(&h3));
        assert!(!loaded.contains(&fake_hash(999_999)));
    }

    #[test]
    fn save_and_load_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("seen.bin");

        let mut set = SeenSet::new();
        let h = fake_hash(42);
        set.insert(&h);
        set.save_to(&path).unwrap();

        let loaded = SeenSet::load_from(&path);
        assert!(loaded.contains(&h));
        assert!(!loaded.contains(&fake_hash(999_999)));
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let set = SeenSet::load_from(Path::new("/nonexistent/seen.bin"));
        assert!(set.is_empty());
    }

    #[test]
    fn bad_data_is_rejected_rather_than_read_as_empty() {
        // `None`, not an empty set: `load_for_source` tells the two apart,
        // and reading a corrupt medium record as "nothing imported yet"
        // would send the user re-importing a whole card.
        assert!(SeenSet::parse(&[]).is_none());
        assert!(SeenSet::parse(&[0; 4]).is_none());
        assert!(SeenSet::parse(&[99, 0, 0, 0, 0, 0, 0, 0]).is_none(), "wrong version");
        // A count the payload cannot cover — a truncated write.
        assert!(SeenSet::parse(&[1, 0, 0, 0, 9, 0, 0, 0]).is_none());
    }

    // ── The record on the import medium ──────────────────────────

    #[test]
    fn the_medium_carries_the_record_to_a_library_that_never_saw_it() {
        let card = tempfile::tempdir().unwrap();
        let first_library = tempfile::tempdir().unwrap();
        let other_library = tempfile::tempdir().unwrap();

        let mut set = SeenSet::new();
        set.insert(&fake_hash(1));
        set.merge_save_to_source(card.path(), first_library.path(), Record::Imported).unwrap();

        // Plugged into a different machine: the card still knows.
        let loaded = SeenSet::load_for_source(card.path(), other_library.path(), Record::Imported);
        assert!(loaded.contains(&fake_hash(1)));
    }

    #[test]
    fn a_source_with_no_record_falls_back_to_the_library_replica() {
        let card = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();

        let mut set = SeenSet::new();
        set.insert(&fake_hash(2));
        set.save_replica(library.path(), Record::Imported).unwrap();

        // Nothing was ever written to the card — a read-only one, say.
        assert!(!card.path().join(Record::Imported.on_medium()).exists());
        assert!(SeenSet::load_for_source(card.path(), library.path(), Record::Imported).contains(&fake_hash(2)));
    }

    #[test]
    fn a_corrupt_medium_record_falls_back_rather_than_forgetting() {
        let card = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();

        let mut set = SeenSet::new();
        set.insert(&fake_hash(3));
        set.save_replica(library.path(), Record::Imported).unwrap();
        std::fs::write(card.path().join(Record::Imported.on_medium()), b"not a seen set").unwrap();

        // Reading the garbage as an empty set would send the user
        // re-importing the whole card.
        assert!(SeenSet::load_for_source(card.path(), library.path(), Record::Imported).contains(&fake_hash(3)));
    }

    #[test]
    fn two_importers_saving_at_once_both_survive() {
        let card = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();

        // Both loaded the same (empty) record, then each copied its own
        // photos. Under the old load-modify-write the second save would
        // overwrite the first and lose `first`.
        let mut a = SeenSet::new();
        a.insert(&fake_hash(10));
        let mut b = SeenSet::new();
        b.insert(&fake_hash(20));

        a.merge_save_to_source(card.path(), library.path(), Record::Imported).unwrap();
        b.merge_save_to_source(card.path(), library.path(), Record::Imported).unwrap();

        let merged = SeenSet::load_for_source(card.path(), library.path(), Record::Imported);
        assert!(merged.contains(&fake_hash(10)), "the first importer's photos were lost");
        assert!(merged.contains(&fake_hash(20)));
        assert_eq!(merged.len(), 2);

        // And the replica saw the union too, not just the last writer.
        let replica = SeenSet::load_replica(library.path(), Record::Imported);
        assert!(replica.contains(&fake_hash(10)) && replica.contains(&fake_hash(20)));
    }

    #[test]
    fn merging_two_cards_unions_them() {
        let mut a = SeenSet::new();
        a.insert(&fake_hash(1));
        a.insert(&fake_hash(2));
        let mut b = SeenSet::new();
        b.insert(&fake_hash(2));
        b.insert(&fake_hash(3));

        a.merge(&b);
        assert_eq!(a.len(), 3, "the shared hash was counted twice");
        for i in 1..=3u64 {
            assert!(a.contains(&fake_hash(i)));
        }
    }

    #[test]
    fn the_unhashed_sentinel_never_enters_the_set() {
        let mut set = SeenSet::new();
        set.insert(&UNHASHED);
        assert!(set.is_empty());
        assert!(!set.contains(&UNHASHED), "one unreadable file would badge them all");
    }

    #[test]
    fn a_file_already_holding_the_sentinel_is_healed_on_load() {
        // version 1, two hashes: the sentinel plus a real one.
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&1u32.to_le_bytes());
        bytes.extend_from_slice(&2u32.to_le_bytes());
        bytes.extend_from_slice(&UNHASHED);
        bytes.extend_from_slice(&fake_hash(5));

        let set = SeenSet::parse(&bytes).unwrap();
        assert_eq!(set.len(), 1);
        assert!(!set.contains(&UNHASHED));
        assert!(set.contains(&fake_hash(5)));
    }

    #[test]
    fn a_save_leaves_no_scratch_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let mut set = SeenSet::new();
        set.insert(&fake_hash(1));
        set.save_to(&dir.path().join("seen.bin")).unwrap();

        let names: Vec<String> = std::fs::read_dir(dir.path())
            .unwrap()
            .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
            .collect();
        assert_eq!(names, vec!["seen.bin".to_string()]);
    }

    #[test]
    fn the_two_records_do_not_bleed_into_each_other() {
        let card = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();

        let mut imported = SeenSet::new();
        imported.insert(&fake_hash(1));
        imported
            .merge_save_to_source(card.path(), library.path(), Record::Imported)
            .unwrap();

        let mut skipped = SeenSet::new();
        skipped.insert(&fake_hash(2));
        skipped
            .merge_save_to_source(card.path(), library.path(), Record::Skipped)
            .unwrap();

        let imported = SeenSet::load_for_source(card.path(), library.path(), Record::Imported);
        let skipped = SeenSet::load_for_source(card.path(), library.path(), Record::Skipped);

        assert!(imported.contains(&fake_hash(1)) && !imported.contains(&fake_hash(2)));
        assert!(skipped.contains(&fake_hash(2)) && !skipped.contains(&fake_hash(1)));

        // Two files on the card, not one shared one.
        assert_ne!(Record::Imported.on_medium(), Record::Skipped.on_medium());
        assert!(card.path().join(Record::Imported.on_medium()).exists());
        assert!(card.path().join(Record::Skipped.on_medium()).exists());
    }

    #[test]
    fn a_skipped_record_survives_to_another_machine_like_an_imported_one() {
        let card = tempfile::tempdir().unwrap();
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();

        let mut skipped = SeenSet::new();
        skipped.insert(&fake_hash(3));
        skipped
            .merge_save_to_source(card.path(), first.path(), Record::Skipped)
            .unwrap();

        assert!(SeenSet::load_for_source(card.path(), second.path(), Record::Skipped)
            .contains(&fake_hash(3)));
    }

    #[test]
    fn an_empty_source_root_writes_nothing_to_the_working_directory() {
        let library = tempfile::tempdir().unwrap();
        let mut set = SeenSet::new();
        set.insert(&fake_hash(9));

        assert!(!set.merge_save_to_source(Path::new(""), library.path(), Record::Imported).unwrap());
        assert!(!Path::new(Record::Imported.on_medium()).exists(), "littered the cwd");
        assert!(SeenSet::load_replica(library.path(), Record::Imported).contains(&fake_hash(9)));
    }

    #[cfg(unix)]
    #[test]
    fn a_read_only_medium_still_records_in_the_library() {
        use std::os::unix::fs::PermissionsExt;

        let card = tempfile::tempdir().unwrap();
        let library = tempfile::tempdir().unwrap();
        std::fs::set_permissions(card.path(), std::fs::Permissions::from_mode(0o555)).unwrap();

        let mut set = SeenSet::new();
        set.insert(&fake_hash(7));
        let on_medium = set.merge_save_to_source(card.path(), library.path(), Record::Imported).unwrap();

        assert!(!on_medium, "a read-only card cannot have taken the write");
        assert!(SeenSet::load_replica(library.path(), Record::Imported).contains(&fake_hash(7)));
        // …and the card falls back to that replica next time.
        assert!(SeenSet::load_for_source(card.path(), library.path(), Record::Imported).contains(&fake_hash(7)));

        std::fs::set_permissions(card.path(), std::fs::Permissions::from_mode(0o755)).unwrap();
    }

    #[test]
    fn auto_resizes_bloom() {
        let mut set = SeenSet::new();
        let initial_bits = set.num_bits;
        for i in 0u64..2000 {
            set.insert(&fake_hash(i));
        }
        assert!(set.num_bits >= initial_bits);
    }
}
