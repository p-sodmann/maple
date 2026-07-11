//! `redb`-backed thumbnail cache.
//!
//! Three tables share one backing file:
//! - `thumbs` — library-grid thumbnails, keyed by 32-byte BLAKE3 content
//!   hash. Because the key is the content hash, entries are automatically
//!   invalidated whenever a file changes on disk (EXIF rotation rewrites
//!   bytes → new hash → cache miss → fresh thumbnail generated).
//! - `face_crops` — a person's representative face crop, keyed by
//!   `face_detections.id` (big-endian `i64`). Overwritten in place when the
//!   representative face changes.
//! - `covers` — a collection's representative (cover) image crop, keyed by
//!   `collections.id` (big-endian `i64`). Same overwrite-in-place behaviour.
//!
//! All values are lossy WebP bytes.
//!
//! `redb` is a pure-Rust embedded key-value store that grows the backing file
//! on demand — no upfront map-size reservation, no unsafe resize, and no
//! Windows pre-allocation issues (P2 fix replacing the previous LMDB backend).

use std::path::Path;

use redb::{Database, ReadableDatabase, TableDefinition};

const THUMBS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("thumbs");
const FACE_CROPS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("face_crops");
const COVERS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("covers");

const ALL_TABLES: [TableDefinition<&[u8], &[u8]>; 3] = [THUMBS, FACE_CROPS, COVERS];

pub struct ThumbnailCache {
    db: Database,
}

impl ThumbnailCache {
    /// Open (or create) the cache at `dir/thumbs.redb`.
    ///
    /// `dir` is created automatically if it does not exist. A file left behind
    /// by an older, file-format-incompatible `redb` release (e.g. the v2→v3
    /// format break between redb 2.x and 3.x) is treated as disposable — since
    /// entries are keyed by content hash, discarding it just means the next
    /// thumbnail requests are cache misses — so it is deleted and recreated
    /// rather than permanently redirecting this session to a fallback path.
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        let path = dir.join("thumbs.redb");
        let db = match Database::create(&path) {
            Err(redb::DatabaseError::UpgradeRequired(_)) => {
                std::fs::remove_file(&path)?;
                Database::create(&path)?
            }
            other => other?,
        };
        {
            let wtxn = db.begin_write()?;
            for table in ALL_TABLES {
                wtxn.open_table(table)?;
            }
            wtxn.commit()?;
        }
        Ok(Self { db })
    }

    // ── Generic table access (DRY core shared by all three tables) ────

    fn get_table(&self, table: TableDefinition<&[u8], &[u8]>, key: &[u8]) -> Option<Vec<u8>> {
        let rtxn = self.db.begin_read().ok()?;
        let t = rtxn.open_table(table).ok()?;
        t.get(key).ok()?.map(|v| v.value().to_vec())
    }

    fn put_table(&self, table: TableDefinition<&[u8], &[u8]>, key: &[u8], value: &[u8]) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        {
            let mut t = wtxn.open_table(table)?;
            t.insert(key, value)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    fn remove_table(&self, table: TableDefinition<&[u8], &[u8]>, key: &[u8]) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        {
            let mut t = wtxn.open_table(table)?;
            t.remove(key)?;
        }
        wtxn.commit()?;
        Ok(())
    }

    // ── Library-grid thumbnails (content-hash keyed) ───────────────────

    /// Return the cached WebP bytes for `hash`, or `None` on a miss.
    pub fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>> {
        self.get_table(THUMBS, hash.as_slice())
    }

    /// Store `webp` bytes under `hash`.
    pub fn insert(&self, hash: &[u8; 32], webp: &[u8]) -> anyhow::Result<()> {
        self.put_table(THUMBS, hash.as_slice(), webp)
    }

    /// Remove the cache entry for `hash` (no-op if not present).
    pub fn remove(&self, hash: &[u8; 32]) -> anyhow::Result<()> {
        self.remove_table(THUMBS, hash.as_slice())
    }

    // ── Person representative face crops (face-id keyed) ───────────────

    /// Return the cached WebP bytes for a person's representative face crop.
    pub fn get_face_crop(&self, face_id: i64) -> Option<Vec<u8>> {
        self.get_table(FACE_CROPS, &face_id.to_be_bytes())
    }

    /// Store a representative face crop's WebP bytes, keyed by `face_id`.
    pub fn insert_face_crop(&self, face_id: i64, webp: &[u8]) -> anyhow::Result<()> {
        self.put_table(FACE_CROPS, &face_id.to_be_bytes(), webp)
    }

    /// Remove a cached face crop (no-op if not present). Called when a
    /// person is deleted so a stale crop doesn't linger under a dead id.
    pub fn remove_face_crop(&self, face_id: i64) -> anyhow::Result<()> {
        self.remove_table(FACE_CROPS, &face_id.to_be_bytes())
    }

    // ── Collection representative (cover) image crops (collection-id keyed) ──

    /// Return the cached WebP bytes for a collection's cover crop.
    pub fn get_cover(&self, collection_id: i64) -> Option<Vec<u8>> {
        self.get_table(COVERS, &collection_id.to_be_bytes())
    }

    /// Store a cover crop's WebP bytes, keyed by `collection_id`.
    pub fn insert_cover(&self, collection_id: i64, webp: &[u8]) -> anyhow::Result<()> {
        self.put_table(COVERS, &collection_id.to_be_bytes(), webp)
    }

    /// Remove a cached cover crop (no-op if not present). Called when a
    /// collection is deleted.
    pub fn remove_cover(&self, collection_id: i64) -> anyhow::Result<()> {
        self.remove_table(COVERS, &collection_id.to_be_bytes())
    }

    // ── Whole-cache maintenance ──────────────────────────────────────

    /// Delete every entry in every table.
    pub fn clear(&self) -> anyhow::Result<()> {
        let wtxn = self.db.begin_write()?;
        for table in ALL_TABLES {
            wtxn.delete_table(table)?;
            wtxn.open_table(table)?;
        }
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
    fn face_crop_roundtrip_and_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        assert!(cache.get_face_crop(42).is_none());
        cache.insert_face_crop(42, b"face-v1").unwrap();
        assert_eq!(cache.get_face_crop(42).unwrap(), b"face-v1");
        cache.insert_face_crop(42, b"face-v2").unwrap();
        assert_eq!(cache.get_face_crop(42).unwrap(), b"face-v2");
    }

    #[test]
    fn face_crop_remove() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert_face_crop(1, b"data").unwrap();
        cache.remove_face_crop(1).unwrap();
        assert!(cache.get_face_crop(1).is_none());
    }

    #[test]
    fn cover_roundtrip_and_overwrite() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        assert!(cache.get_cover(7).is_none());
        cache.insert_cover(7, b"cover-v1").unwrap();
        assert_eq!(cache.get_cover(7).unwrap(), b"cover-v1");
        cache.insert_cover(7, b"cover-v2").unwrap();
        assert_eq!(cache.get_cover(7).unwrap(), b"cover-v2");
    }

    #[test]
    fn cover_remove() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert_cover(1, b"data").unwrap();
        cache.remove_cover(1).unwrap();
        assert!(cache.get_cover(1).is_none());
    }

    #[test]
    fn face_crop_and_cover_and_thumb_keys_dont_collide() {
        // Same numeric id/key bytes used across all three tables shouldn't
        // cross-contaminate — they're separate redb tables.
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert(&hash(9), b"thumb").unwrap();
        cache.insert_face_crop(9, b"face").unwrap();
        cache.insert_cover(9, b"cover").unwrap();
        assert_eq!(cache.get(&hash(9)).unwrap(), b"thumb");
        assert_eq!(cache.get_face_crop(9).unwrap(), b"face");
        assert_eq!(cache.get_cover(9).unwrap(), b"cover");
    }

    #[test]
    fn clear_empties_face_crops_and_covers_too() {
        let dir = tempfile::tempdir().unwrap();
        let cache = ThumbnailCache::open(dir.path()).unwrap();
        cache.insert_face_crop(1, b"face").unwrap();
        cache.insert_cover(1, b"cover").unwrap();
        cache.clear().unwrap();
        assert!(cache.get_face_crop(1).is_none());
        assert!(cache.get_cover(1).is_none());
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
