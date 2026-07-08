//! Persistent cache of DINOv2 image embeddings, keyed by BLAKE3 content hash.
//!
//! Written back to the scanned source directory (e.g. an SD card) so that
//! re-scanning the same source before it's reused/formatted doesn't require
//! recomputing embeddings via ONNX inference. Keyed by content hash rather
//! than path, so renamed/reorganized files still hit the cache.
//!
//! Same on-disk idiom as `maple_state::SeenSet`: a small custom binary
//! format rather than serde/bincode. Scoped to one embedding model at a
//! time — the whole cache is discarded and rebuilt if the algorithm key
//! (which encodes the model repo) changes.

use std::collections::HashMap;
use std::path::Path;

/// File format version.
const VERSION: u32 = 1;

/// Cache of image embeddings, keyed by 32-byte BLAKE3 content hash.
pub struct EmbeddingCache {
    /// Identifies the model these embeddings were computed with (e.g.
    /// `"onnx:onnx-community/dinov2-small"`). A cache loaded under a
    /// different key is discarded rather than mixed with stale vectors.
    algorithm_key: String,
    embeddings: HashMap<[u8; 32], Vec<f32>>,
}

impl EmbeddingCache {
    /// Create an empty cache scoped to `algorithm_key`.
    pub fn new(algorithm_key: impl Into<String>) -> Self {
        Self {
            algorithm_key: algorithm_key.into(),
            embeddings: HashMap::new(),
        }
    }

    /// Load from `path`, scoped to `algorithm_key`. Returns an empty cache
    /// (scoped to `algorithm_key`) if the file is missing, corrupt, or was
    /// written under a different algorithm key.
    pub fn load_from(path: &Path, algorithm_key: &str) -> Self {
        match std::fs::read(path) {
            Ok(data) => Self::from_bytes(&data, algorithm_key),
            Err(_) => Self::new(algorithm_key),
        }
    }

    /// Save to `path`. Best-effort — callers should log and ignore failures
    /// (e.g. a read-only source card) rather than treat this as fatal.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(path, self.to_bytes())?;
        Ok(())
    }

    /// Look up a cached embedding by content hash.
    pub fn get(&self, hash: &[u8; 32]) -> Option<&[f32]> {
        self.embeddings.get(hash).map(|v| v.as_slice())
    }

    /// Insert (or replace) the embedding for `hash`.
    pub fn insert(&mut self, hash: [u8; 32], embedding: Vec<f32>) {
        self.embeddings.insert(hash, embedding);
    }

    /// Number of cached embeddings.
    pub fn len(&self) -> usize {
        self.embeddings.len()
    }

    pub fn is_empty(&self) -> bool {
        self.embeddings.is_empty()
    }

    /// Binary format:
    /// `version(u32 LE) | key_len(u32 LE) | key(utf8) | dim(u32 LE) |
    ///  count(u32 LE) | records(count × (32-byte hash + dim×4-byte f32 LE))`
    ///
    /// `dim` is read off an arbitrary entry — every entry in one cache
    /// comes from the same model and therefore shares a dimension.
    fn to_bytes(&self) -> Vec<u8> {
        let key_bytes = self.algorithm_key.as_bytes();
        let dim = self.embeddings.values().next().map(|v| v.len()).unwrap_or(0);
        let count = self.embeddings.len();

        let mut buf = Vec::with_capacity(16 + key_bytes.len() + count * (32 + dim * 4));
        buf.extend_from_slice(&VERSION.to_le_bytes());
        buf.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
        buf.extend_from_slice(key_bytes);
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        buf.extend_from_slice(&(count as u32).to_le_bytes());
        for (hash, embedding) in &self.embeddings {
            buf.extend_from_slice(hash);
            for f in embedding {
                buf.extend_from_slice(&f.to_le_bytes());
            }
        }
        buf
    }

    fn from_bytes(data: &[u8], algorithm_key: &str) -> Self {
        let empty = || Self::new(algorithm_key);

        if data.len() < 8 {
            return empty();
        }
        let version = u32::from_le_bytes(data[0..4].try_into().unwrap());
        if version != VERSION {
            return empty();
        }
        let key_len = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
        if data.len() < 8 + key_len {
            return empty();
        }
        let Ok(stored_key) = std::str::from_utf8(&data[8..8 + key_len]) else {
            return empty();
        };
        if stored_key != algorithm_key {
            return empty();
        }

        let mut offset = 8 + key_len;
        if data.len() < offset + 8 {
            return empty();
        }
        let dim = u32::from_le_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        let count = u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        offset += 8;

        let record_size = 32 + dim * 4;
        if data.len() < offset + count * record_size {
            return empty();
        }

        let mut embeddings = HashMap::with_capacity(count);
        for i in 0..count {
            let rec = &data[offset + i * record_size..offset + (i + 1) * record_size];
            let hash: [u8; 32] = rec[0..32].try_into().unwrap();
            let embedding: Vec<f32> = rec[32..]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            embeddings.insert(hash, embedding);
        }

        Self {
            algorithm_key: algorithm_key.to_string(),
            embeddings,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fake_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        h[0] = seed;
        h
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let mut cache = EmbeddingCache::new("onnx:test-model");
        let h = fake_hash(1);
        assert!(cache.get(&h).is_none());
        cache.insert(h, vec![1.0, 2.0, 3.0]);
        assert_eq!(cache.get(&h), Some(&[1.0, 2.0, 3.0][..]));
    }

    #[test]
    fn save_and_load_file_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.bin");

        let mut cache = EmbeddingCache::new("onnx:test-model");
        let h1 = fake_hash(1);
        let h2 = fake_hash(2);
        cache.insert(h1, vec![0.5, -0.5]);
        cache.insert(h2, vec![1.0, 1.0]);
        cache.save_to(&path).unwrap();

        let loaded = EmbeddingCache::load_from(&path, "onnx:test-model");
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded.get(&h1), Some(&[0.5, -0.5][..]));
        assert_eq!(loaded.get(&h2), Some(&[1.0, 1.0][..]));
    }

    #[test]
    fn algorithm_key_mismatch_discards_cache() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("cache.bin");

        let mut cache = EmbeddingCache::new("onnx:old-model");
        cache.insert(fake_hash(1), vec![1.0]);
        cache.save_to(&path).unwrap();

        let loaded = EmbeddingCache::load_from(&path, "onnx:new-model");
        assert!(loaded.is_empty(), "cache under a different model must be discarded");
    }

    #[test]
    fn load_missing_file_returns_empty() {
        let cache = EmbeddingCache::load_from(Path::new("/nonexistent/cache.bin"), "onnx:test-model");
        assert!(cache.is_empty());
    }

    #[test]
    fn bad_data_returns_empty() {
        assert!(EmbeddingCache::from_bytes(&[], "onnx:test-model").is_empty());
        assert!(EmbeddingCache::from_bytes(&[1, 0, 0, 0], "onnx:test-model").is_empty());
    }
}
