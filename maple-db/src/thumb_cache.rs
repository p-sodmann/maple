//! `redb`-backed thumbnail cache.
//!
//! Keys are 32-byte BLAKE3 content hashes; values are lossy WebP bytes.
//! Because the key is the content hash, cache entries are automatically
//! invalidated whenever a file changes on disk (EXIF rotation rewrites bytes
//! → new hash → cache miss → fresh thumbnail generated).
//!
//! `redb` is a pure-Rust embedded key-value store that grows the backing file
//! on demand — no upfront map-size reservation, no unsafe resize, and no
//! Windows pre-allocation issues (P2 fix replacing the previous LMDB backend).

use std::path::Path;

use redb::{Database, TableDefinition};

const THUMBS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("thumbs");

pub struct ThumbnailCache {
    db: Database,
}

impl ThumbnailCache {
    /// Open (or create) the cache at `dir/thumbs.redb`.
    ///
    /// `dir` is created automatically if it does not exist.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let db = Database::create(dir.join("thumbs.redb"))?;
        {
            let wtxn = db.begin_write()?;
            wtxn.open_table(THUMBS)?;
            wtxn.commit()?;
        }
        Ok(Self { db })
    }

    /// Return the cached WebP bytes for `hash`, or `None` on a miss.
    pub fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>> {
        let rtxn = self.db.begin_read().ok()?;
        let table = rtxn.open_table(THUMBS).ok()?;
        table
            .get(hash.as_slice())
            .ok()?
            .map(|v| v.value().to_vec())
    }

    /// Store `webp` bytes under `hash`.
    pub fn insert(&self, hash: &[u8; 32], webp: &[u8]) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(THUMBS)?;
            table.insert(hash.as_slice(), webp)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Remove the cache entry for `hash` (no-op if not present).
    pub fn remove(&self, hash: &[u8; 32]) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        {
            let mut table = wtxn.open_table(THUMBS)?;
            table.remove(hash.as_slice())?;
        }
        wtxn.commit()?;
        Ok(())
    }

    /// Delete every entry in the cache.
    pub fn clear(&self) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        wtxn.delete_table(THUMBS)?;
        wtxn.open_table(THUMBS)?;
        wtxn.commit()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hash(b: u8) -> [u8; 32] {
        [b; 32]
    }

    fn total_dir_bytes(dir: &std::path::Path) -> u64 {
        std::fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .filter_map(|e| e.metadata().ok())
            .map(|m| m.len())
            .sum()
    }

    #[test]
    fn miss_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        assert!(cache.get(&hash(0)).is_none());
    }

    #[test]
    fn insert_and_get_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(1), b"webp-bytes").unwrap();
        assert_eq!(cache.get(&hash(1)).unwrap(), b"webp-bytes");
    }

    #[test]
    fn insert_overwrites_existing() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(2), b"v1").unwrap();
        cache.insert(&hash(2), b"v2").unwrap();
        assert_eq!(cache.get(&hash(2)).unwrap(), b"v2");
    }

    #[test]
    fn remove_entry() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(3), b"data").unwrap();
        cache.remove(&hash(3)).unwrap();
        assert!(cache.get(&hash(3)).is_none());
    }

    #[test]
    fn remove_nonexistent_is_noop() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        assert!(cache.remove(&hash(4)).is_ok());
    }

    #[test]
    fn clear_empties_cache() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(5), b"a").unwrap();
        cache.insert(&hash(6), b"b").unwrap();
        cache.clear().unwrap();
        assert!(cache.get(&hash(5)).is_none());
        assert!(cache.get(&hash(6)).is_none());
    }

    #[test]
    fn insert_works_after_clear() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(7), b"before").unwrap();
        cache.clear().unwrap();
        cache.insert(&hash(7), b"after").unwrap();
        assert_eq!(cache.get(&hash(7)).unwrap(), b"after");
    }

    #[test]
    fn file_size_is_bounded() {
        // P2: the cache must not pre-allocate large disk space.
        // LMDB on Windows extended the file to the full map size (1 GiB) at open;
        // redb grows lazily so this holds on all platforms.
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        for i in 0u8..10 {
            cache.insert(&hash(i), b"thumbnail-data").unwrap();
        }
        drop(cache);
        let size = total_dir_bytes(dir.path());
        assert!(
            size < 10 * 1024 * 1024,
            "cache files total {size} bytes — P2: should stay <10 MiB on all platforms"
        );
    }
}
