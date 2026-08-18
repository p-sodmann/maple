//! Persistent sets for tracking previously imported and rejected images.
//!
//! Two separate files are maintained:
//!
//! * `seen_imported.bin` — images copied to the destination.
//! * `seen_rejected.bin` — images explicitly skipped by the user.
//!
//! Each stores the full 32-byte BLAKE3 content hashes of its members, both in
//! memory and on disk.
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
use std::path::Path;

/// Number of hash functions for the bloom filter.
const K: u32 = 7;

/// Minimum bloom filter size in bits.
const MIN_BITS: usize = 8192;

/// File format version.
const VERSION: u32 = 1;

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

    /// Load the imported-images set from `dir/seen_imported.bin`.
    pub fn load_imported(dir: &Path) -> Self {
        Self::load_from(&dir.join("seen_imported.bin"))
    }

    /// Load the rejected-images set from `dir/seen_rejected.bin`.
    pub fn load_rejected(dir: &Path) -> Self {
        Self::load_from(&dir.join("seen_rejected.bin"))
    }

    // ── Load / Save ─────────────────────────────────────────────

    /// Load from a specific file (`Self::new()` on any error).
    pub fn load_from(path: &Path) -> Self {
        match std::fs::read(path) {
            Ok(data) => Self::from_bytes(&data),
            Err(_) => Self::new(),
        }
    }

    /// Save the imported-images set to `dir/seen_imported.bin`.
    pub fn save_imported(&self, dir: &Path) -> anyhow::Result<()> {
        self.save_to(&dir.join("seen_imported.bin"))
    }

    /// Save the rejected-images set to `dir/seen_rejected.bin`.
    pub fn save_rejected(&self, dir: &Path) -> anyhow::Result<()> {
        self.save_to(&dir.join("seen_rejected.bin"))
    }

    /// Save to a specific file.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    // ── Core API ────────────────────────────────────────────────

    /// Insert a full 32-byte BLAKE3 content hash.
    ///
    /// Inserting the same hash twice is a no-op: the set stores each hash
    /// once, so [`Self::len`] counts distinct images.
    pub fn insert(&mut self, hash: &[u8; 32]) {
        if !self.hashes.insert(*hash) {
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

    fn from_bytes(data: &[u8]) -> Self {
        if data.len() < 8 {
            return Self::new();
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if version != VERSION {
            return Self::new();
        }
        let count = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < 8 + count * 32 {
            return Self::new();
        }
        let mut hashes = HashSet::with_capacity(count);
        for i in 0..count {
            let off = 8 + i * 32;
            let h: [u8; 32] = data[off..off + 32].try_into().unwrap();
            hashes.insert(h);
        }
        let mut set = Self {
            bits: Vec::new(),
            num_bits: 0,
            hashes,
        };
        set.rebuild_bloom();
        set
    }
}

impl Default for SeenSet {
    fn default() -> Self {
        Self::new()
    }
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
        let mut loaded = SeenSet::from_bytes(&set.to_bytes());
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
        let loaded = SeenSet::from_bytes(&set.to_bytes());
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
        let loaded = SeenSet::from_bytes(&bytes);

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
    fn bad_data_returns_empty() {
        assert!(SeenSet::from_bytes(&[]).is_empty());
        assert!(SeenSet::from_bytes(&[0; 4]).is_empty());
        assert!(SeenSet::from_bytes(&[99, 0, 0, 0, 0, 0, 0, 0]).is_empty());
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
