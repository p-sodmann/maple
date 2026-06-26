//! LMDB-backed thumbnail cache.
//!
//! Keys are 32-byte BLAKE3 content hashes; values are lossy WebP bytes.
//! Because the key is the content hash, cache entries are automatically
//! invalidated whenever a file changes on disk (EXIF rotation rewrites bytes
//! → new hash → cache miss → fresh thumbnail generated).
//!
//! # Auto-grow
//!
//! LMDB requires a fixed map size set at open time.  `ThumbnailCache` starts
//! at 1 GiB (a sparse file on Linux — actual disk usage matches stored data)
//! and doubles the map on `MDB_MAP_FULL`.  Resizing requires no active
//! transactions, so operations hold a shared `RwLock` while resize takes the
//! exclusive write lock, draining in-flight transactions before growing.

use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::RwLock;

use heed::types::Bytes;
use heed::{Database, Env, EnvOpenOptions};

const INITIAL_MAP_SIZE: u64 = 1 << 30; // 1 GiB

struct Inner {
    env: Env,
    db: Database<Bytes, Bytes>,
}

pub struct ThumbnailCache {
    inner: RwLock<Inner>,
    /// Tracked separately so we can double without querying the env.
    map_size: AtomicU64,
}

impl ThumbnailCache {
    /// Open (or create) the LMDB cache at `dir`.
    ///
    /// `dir` is created automatically if it does not exist.
    /// The initial map size is 1 GiB (sparse file on Linux).
    pub fn open(dir: &Path) -> anyhow::Result<Self> {
        std::fs::create_dir_all(dir)?;
        // SAFETY: standard LMDB safety requirements — single process, no
        // concurrent mmap writes from outside Rust.  All access goes through
        // `inner` which is protected by `RwLock`.
        let env = unsafe {
            EnvOpenOptions::new()
                .map_size(INITIAL_MAP_SIZE as usize)
                .max_dbs(1)
                .open(dir)?
        };
        let db: Database<Bytes, Bytes> = {
            let mut wtxn = env.write_txn()?;
            let db = env.create_database(&mut wtxn, Some("thumbs"))?;
            wtxn.commit()?;
            db
        };
        Ok(Self {
            inner: RwLock::new(Inner { env, db }),
            map_size: AtomicU64::new(INITIAL_MAP_SIZE),
        })
    }

    /// Return the cached WebP bytes for `hash`, or `None` on a miss.
    pub fn get(&self, hash: &[u8; 32]) -> Option<Vec<u8>> {
        let guard = self.inner.read().ok()?;
        let rtxn = guard.env.read_txn().ok()?;
        guard
            .db
            .get(&rtxn, hash.as_slice())
            .ok()?
            .map(|b| b.to_vec())
    }

    /// Store `webp` bytes under `hash`.
    ///
    /// Automatically doubles the LMDB map size on `MDB_MAP_FULL` and retries.
    pub fn insert(&self, hash: &[u8; 32], webp: &[u8]) -> anyhow::Result<()> {
        loop {
            let map_full = {
                let guard = self
                    .inner
                    .read()
                    .map_err(|_| anyhow::anyhow!("thumb cache lock poisoned"))?;
                match Self::try_insert_locked(&guard, hash, webp) {
                    Ok(()) => return Ok(()),
                    Err(heed::Error::Mdb(heed::MdbError::MapFull)) => true,
                    Err(e) => return Err(e.into()),
                }
                // `guard` (read lock) dropped here — all transactions finished
            };

            if map_full {
                self.grow()?;
            }
        }
    }

    /// Remove the cache entry for `hash` (no-op if not present).
    pub fn remove(&self, hash: &[u8; 32]) -> anyhow::Result<()> {
        let guard = self
            .inner
            .read()
            .map_err(|_| anyhow::anyhow!("thumb cache lock poisoned"))?;
        let mut wtxn = guard.env.write_txn()?;
        guard.db.delete(&mut wtxn, hash.as_slice())?;
        wtxn.commit()?;
        Ok(())
    }

    /// Delete every entry in the cache.
    pub fn clear(&self) -> anyhow::Result<()> {
        let guard = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("thumb cache lock poisoned"))?;
        let mut wtxn = guard.env.write_txn()?;
        guard.db.clear(&mut wtxn)?;
        wtxn.commit()?;
        Ok(())
    }

    // ── Internals ────────────────────────────────────────────────

    fn try_insert_locked(
        guard: &Inner,
        hash: &[u8; 32],
        webp: &[u8],
    ) -> Result<(), heed::Error> {
        let mut wtxn = guard.env.write_txn()?;
        guard.db.put(&mut wtxn, hash.as_slice(), webp)?;
        wtxn.commit()
    }

    /// Double the LMDB map size.
    ///
    /// Takes the exclusive write lock so all in-flight transactions complete
    /// before `env.resize` is called (LMDB requires no active transactions).
    fn grow(&self) -> anyhow::Result<()> {
        let guard = self
            .inner
            .write()
            .map_err(|_| anyhow::anyhow!("thumb cache lock poisoned"))?;
        let current = self.map_size.load(Ordering::Relaxed);
        let new_size = current.saturating_mul(2);
        // SAFETY: exclusive write lock guarantees no active transactions.
        unsafe { guard.env.resize(new_size as usize)? };
        self.map_size.store(new_size, Ordering::Relaxed);
        tracing::info!(
            "Thumbnail cache grown: {} MiB → {} MiB",
            current / 1024 / 1024,
            new_size / 1024 / 1024,
        );
        Ok(())
    }
}
