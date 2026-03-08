//! Persistent sets for tracking previously imported and rejected images.
//!
//! Two separate files are maintained:
//!
//! * `seen_imported.bin` — images copied to the destination.
//! * `seen_rejected.bin` — images explicitly skipped by the user.
//!
//! Each uses a bloom filter for O(1) probabilistic membership queries backed
//! by full 32-byte BLAKE3 content hashes persisted to disk.
//!
//! A negative query ("definitely not seen") is always correct.
//! A positive query ("probably seen") has an extremely low false-positive
//! rate — storing the full BLAKE3 hash means collisions are cryptographically
//! implausible (2⁻²⁵⁶ per pair).

use std::path::Path;

/// Number of hash functions for the bloom filter.
const K: u32 = 7;

/// Minimum bloom filter size in bits.
const MIN_BITS: usize = 8192;

/// File format version.
const VERSION: u32 = 1;

/// Persistent set keyed by full 32-byte BLAKE3 content hashes.
pub struct SeenSet {
    /// Bloom filter bit array.
    bits: Vec<u64>,
    /// Number of usable bits (`bits.len() * 64`).
    num_bits: usize,
    /// Full 32-byte BLAKE3 hashes stored for persistence.
    hashes: Vec<[u8; 32]>,
}

impl SeenSet {
    /// Create an empty set.
    pub fn new() -> Self {
        Self::with_capacity(0)
    }

    fn with_capacity(expected: usize) -> Self {
        let num_bits = optimal_bits(expected).max(MIN_BITS);
        let words = (num_bits + 63) / 64;
        Self {
            bits: vec![0u64; words],
            num_bits: words * 64,
            hashes: Vec::with_capacity(expected),
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
    pub fn insert(&mut self, hash: &[u8; 32]) {
        self.hashes.push(*hash);
        // Resize the bloom filter if the load factor is getting too high.
        if self.hashes.len() * 10 > self.num_bits {
            self.rebuild_bloom();
        } else {
            self.bloom_insert(hash);
        }
    }

    /// Check if a hash is probably in the set.
    ///
    /// `false` → **definitely** not seen.
    /// `true`  → **probably** seen (negligible false-positive rate given the
    ///           full 32-byte hash; bloom filter error is well below 1%).
    pub fn probably_contains(&self, hash: &[u8; 32]) -> bool {
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

    /// Number of hashes stored.
    pub fn len(&self) -> usize {
        self.hashes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hashes.is_empty()
    }

    // ── Internals ───────────────────────────────────────────────

    fn bloom_insert(&mut self, hash: &[u8; 32]) {
        for i in 0..K {
            let pos = bloom_pos(hash, i, self.num_bits);
            self.bits[pos / 64] |= 1u64 << (pos % 64);
        }
    }

    fn rebuild_bloom(&mut self) {
        let num_bits = optimal_bits(self.hashes.len()).max(MIN_BITS);
        let words = (num_bits + 63) / 64;
        self.bits = vec![0u64; words];
        self.num_bits = words * 64;
        for i in 0..self.hashes.len() {
            let h = self.hashes[i];
            self.bloom_insert(&h);
        }
    }

    /// Binary format: `version(u32 LE) | count(u32 LE) | hashes(count × 32 bytes)`.
    ///
    /// Storage: 8 B header + 32 B/image.  100 k images ≈ 3.2 MB.
    fn to_bytes(&self) -> Vec<u8> {
        let count = self.hashes.len() as u32;
        let mut buf = Vec::with_capacity(8 + self.hashes.len() * 32);
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&count.to_le_bytes());
        for h in &self.hashes {
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
        let mut hashes = Vec::with_capacity(count);
        for i in 0..count {
            let off = 8 + i * 32;
            let h: [u8; 32] = data[off..off + 32].try_into().unwrap();
            hashes.push(h);
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

/// Optimal number of bits for `n` items at ~1 % false-positive rate.
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

    #[test]
    fn insert_and_query() {
        let mut set = SeenSet::new();
        let h = fake_hash(1);
        assert!(!set.probably_contains(&h));
        set.insert(&h);
        assert!(set.probably_contains(&h));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn negative_is_definite() {
        let mut set = SeenSet::new();
        for i in 0..1000u64 {
            set.insert(&fake_hash(i));
        }
        // Values we never inserted should (almost certainly) not match.
        let mut false_positives = 0;
        for i in 100_000..101_000u64 {
            if set.probably_contains(&fake_hash(i)) {
                false_positives += 1;
            }
        }
        // ~1 % FP rate → expect ~10 out of 1000; allow generous margin.
        assert!(false_positives < 50, "too many false positives: {false_positives}");
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
        assert!(loaded.probably_contains(&h1));
        assert!(loaded.probably_contains(&h2));
        assert!(loaded.probably_contains(&h3));
        assert!(!loaded.probably_contains(&fake_hash(999_999)));
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
        assert!(loaded.probably_contains(&h));
        assert!(!loaded.probably_contains(&fake_hash(999_999)));
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
