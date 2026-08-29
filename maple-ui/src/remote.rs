//! Where the UI fetches pixels it does not have on disk.
//!
//! A relay servant's library rows are real — they page, search and sort like
//! any other — but their bytes live on the master. Two places have to notice:
//! [`crate::grid::load_thumbnail`] and [`crate::image_loader`]. Both run on
//! background threads that own nothing but an `Arc`, which is what shapes
//! this module.
//!
//! # Why this is a process-wide handle and not a field
//!
//! The obvious home is `AppCtx`, but neither consumer can reach it. Grid
//! thumbnails are decoded on scoped worker threads that capture exactly one
//! thing (`Arc<ThumbnailCache>`); `image_loader::load_full_image` spawns its
//! own pair of threads from a `PathBuf`. Threading a handle to both would
//! mean widening two constructors, three call sites and a `slint::Weak`
//! boundary to carry something there is exactly one of per process — which
//! master this device is a servant of.
//!
//! So the instance lives here and [`SyncSupervisor::restart`] writes it. The
//! *clearing* half matters as much as the setting half: when the role goes
//! back to Off, or the master is unpaired, a stale client would keep dialling
//! a device this machine no longer has a key for, and every remote thumbnail
//! would fail slowly instead of failing immediately.
//!
//! # A master has no master
//!
//! `restart` sets the handle only for a servant, so on a master every fetch
//! fails — correctly, and permanently: a master runs no client and has no
//! route back to a servant behind a NAT (see `maple_sync::worker`). But a
//! master paired with a servant still *replicates that servant's metadata*,
//! so its grid holds a `remote` row for every photo the servant has, and
//! nothing on the master will ever fill one in. Failing those with a generic
//! error read as "the image is broken"; [`NoMaster`] and [`RemoteBlobs::peer_name`]
//! are what let the grid say "held on Laptop" instead, and the fix for the
//! underlying emptiness is on the servant — it has to be in a mode that
//! uploads.
//!
//! # What it deliberately does not do
//!
//! Originals are never written to disk. That is the relay contract (§3.6):
//! thumbnails are cached — they are ~10 KB and are what makes a grid usable —
//! and full-res pixels live in memory for as long as the detail window shows
//! them.

use std::collections::HashMap;
use std::sync::{Arc, LazyLock, Mutex};

use maple_sync::{PeerKey, SyncClient, SyncFailure};

/// Why a remote fetch failed before it was even attempted.
///
/// A distinct type rather than another `anyhow!` string because the UI has to
/// *render* this one. It is not a transient network failure that a later
/// scroll might resolve: this device has no master to ask, which on a master
/// is permanent (a master runs no client and has no route to a servant) and
/// on an unpaired device lasts until the user pairs. The grid draws a
/// "held on …" placeholder for it instead of a broken tile, and the caller
/// recognises it with `downcast_ref::<NoMaster>()`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NoMaster;

impl std::fmt::Display for NoMaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("this device has no master to fetch photos from")
    }
}

impl std::error::Error for NoMaster {}

/// The master to fetch from, when there is one.
struct Source {
    client: Arc<SyncClient>,
    /// The pairing key. Held here rather than re-read from the trust store
    /// per request: a thumbnail fetch is on the render path, and the store is
    /// behind the same mutex that pairing writes a file under.
    key: PeerKey,
    device_id: String,
}

/// A clonable handle to whatever master this device currently relays from.
///
/// Cheap to clone and `Send + Sync`, so a worker thread can hold one.
#[derive(Clone)]
pub struct RemoteBlobs {
    source: Arc<Mutex<Option<Source>>>,
    /// What this device calls each paired device, `device_id` → display name.
    ///
    /// Kept beside the master handle because it has the same shape of problem
    /// and the same one instance per process: the grid's decode threads own an
    /// `Arc` and nothing else, and a tile that says "held on Laptop" needs the
    /// name behind `images.origin_device`, which lives in `sync_peers`.
    /// Written by
    /// [`SyncSupervisor::restart`](crate::sync_supervisor::SyncSupervisor::restart),
    /// which already reads that table — so it goes stale only if a peer is
    /// renamed mid-session, and the next restart refreshes it.
    peer_names: Arc<Mutex<HashMap<String, String>>>,
}

impl RemoteBlobs {
    fn new() -> Self {
        Self {
            source: Arc::new(Mutex::new(None)),
            peer_names: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Whether there is a master to fetch from at all.
    ///
    /// Asked *before* a fetch, not after one fails: on a master the answer is
    /// no and always will be, so dialling and waiting would spend a network
    /// timeout to learn something already known.
    pub fn has_master(&self) -> bool {
        lock(&self.source).is_some()
    }

    /// Record what this device calls each paired device.
    pub fn set_peer_names(&self, names: HashMap<String, String>) {
        *lock(&self.peer_names) = names;
    }

    /// What to call the device a remote row came from.
    ///
    /// Falls back to the raw device id, and then to "another device": a row
    /// whose origin is unknown still has to say *something*, and a blank
    /// placeholder is the failure this exists to replace.
    pub fn peer_name(&self, device_id: Option<&str>) -> String {
        let names = lock(&self.peer_names);
        match device_id {
            Some(id) => names.get(id).cloned().unwrap_or_else(|| id.to_owned()),
            None => "another device".to_owned(),
        }
    }

    /// Point at `client`, replacing whatever was there.
    pub fn set(&self, device_id: String, client: SyncClient, key: PeerKey) {
        *lock(&self.source) = Some(Source {
            client: Arc::new(client),
            key,
            device_id,
        });
    }

    /// Forget the master. Every later fetch fails immediately instead of
    /// dialling a device we no longer have a key for.
    pub fn clear(&self) {
        *lock(&self.source) = None;
    }

    /// A thumbnail as WebP bytes, sized and encoded by the master.
    pub fn thumb(&self, hash: &[u8; 32]) -> anyhow::Result<Vec<u8>> {
        self.fetch(|client, key| client.blob_thumb(key, hash))
    }

    /// An original's bytes. `raw` asks for the companion raw file.
    pub fn original(&self, hash: &[u8; 32], raw: bool) -> anyhow::Result<Vec<u8>> {
        self.fetch(|client, key| client.blob_orig(key, hash, raw))
    }

    /// Run `f` against the current master, holding the lock only long enough
    /// to clone the handle out of it — the request itself is blocking HTTP,
    /// and holding this mutex across it would serialise every thumbnail in
    /// the grid behind one slow fetch.
    fn fetch<F>(&self, f: F) -> anyhow::Result<Vec<u8>>
    where
        F: FnOnce(&SyncClient, &PeerKey) -> Result<Vec<u8>, SyncFailure>,
    {
        let (client, key, device_id) = {
            let guard = lock(&self.source);
            let source = guard.as_ref().ok_or(NoMaster)?;
            (
                source.client.clone(),
                source.key.clone(),
                source.device_id.clone(),
            )
        };
        f(&client, &key).map_err(|e| anyhow::anyhow!("master {device_id}: {e}"))
    }
}

/// The one instance. See the module docs for why it is a static.
static BLOBS: LazyLock<RemoteBlobs> = LazyLock::new(RemoteBlobs::new);

pub fn blobs() -> RemoteBlobs {
    BLOBS.clone()
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
