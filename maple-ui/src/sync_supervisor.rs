//! Starts and stops the sync server and worker as the user's role changes.
//!
//! Exactly one of the two runs at a time, because the role column allows
//! exactly one: a master listens and a servant dials, and a machine doing
//! both would sync its own writes back to itself through a peer.
//!
//! # Why a supervisor rather than starting them at boot
//!
//! Role, pairing and address all change while the app is running — that is
//! what the Sync card is for. Every one of those changes invalidates whatever
//! is currently running: switching to master means the worker's master is now
//! itself, pairing means the servant finally has an address to dial, and
//! unpairing means the running worker holds a key the peer has forgotten.
//! [`SyncSupervisor::restart`] is the single funnel for all of them, so no
//! call site has to work out which parts to tear down.
//!
//! # Discovery
//!
//! The two mDNS halves have opposite lifetimes, so they are owned here
//! rather than started once at boot. A **master** advertises only while its
//! listener is up, and only after it is up — the record has to carry the
//! port the listener actually bound, which `listen_addr` need not name. A
//! **servant** browses for as long as it is a servant, because the worker
//! re-resolves on every failed pass; a device that is Off or a master keeps
//! no browser, except for the one the pairing modal starts on demand, which
//! the next `restart` clears.
//!
//! # Threading
//!
//! Lives on the UI thread and is not `Send`. The handles it owns each carry
//! their own thread and stop on drop, so `restart` needs only to drop them.
//! The worker's change callback marshals back through
//! `slint::Weak::upgrade_in_event_loop` — it fires on the worker thread, and
//! touching a Slint property from there would be undefined behaviour.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_state::SyncRole;
use maple_sync::discovery::{Advertiser, Browser, DeviceSource};
use maple_sync::server::{ServerConfig, ServerDeps, ThumbRenderer};
use maple_sync::worker::{WorkerConfig, WorkerDeps};
use maple_sync::{
    Clock, LibraryLayout, PairingSlot, SharedRandom, StatusCell, SyncClient, SyncServer, SyncStatus,
    SyncWorker, TrustStore,
};

use crate::remote::RemoteBlobs;

/// How stale a peer's `last_seen_at` may be before it stops counting as
/// connected. Two default sync intervals, so one missed pass does not
/// flicker the pill.
const ONLINE_WINDOW_MS: i64 = 10 * 60 * 1000;

pub struct SyncSupervisor {
    db: Arc<Mutex<maple_db::Database>>,
    trust: Arc<Mutex<TrustStore>>,
    pairing: PairingSlot,
    status: StatusCell,
    clock: Clock,
    rng: SharedRandom,
    server: RefCell<Option<SyncServer>>,
    worker: RefCell<Option<SyncWorker>>,
    /// The master's mDNS registration. Alive exactly as long as the listener
    /// it names.
    advertiser: RefCell<Option<Advertiser>>,
    /// The servant's mDNS browse. Started on demand by [`Self::discovery`],
    /// because a master never needs one and an `Off` device needs one only
    /// while its pairing modal is open.
    browser: RefCell<Option<Arc<Browser>>>,
    /// The master's own thumbnail store, so a servant's `/blob/thumb` request
    /// is answered from cache when the master has already rendered it.
    thumbs: Arc<maple_db::ThumbnailCache>,
    /// Longest edge and WebP quality for a thumbnail this master renders on
    /// behalf of a servant. Its own settings — the servant asks for a photo,
    /// not for a size.
    thumb_px: u32,
    thumb_quality: u8,
    /// Where the grid and the detail view fetch remote pixels from. Written
    /// by [`restart`](Self::restart), which is the only place that knows
    /// whether there is a master to fetch from at all.
    blobs: RemoteBlobs,
}

impl SyncSupervisor {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db: Arc<Mutex<maple_db::Database>>,
        trust: Arc<Mutex<TrustStore>>,
        pairing: PairingSlot,
        status: StatusCell,
        rng: SharedRandom,
        thumbs: Arc<maple_db::ThumbnailCache>,
        thumb_px: u32,
        thumb_quality: u8,
    ) -> Rc<Self> {
        Rc::new(Self {
            db,
            trust,
            pairing,
            status,
            clock: Arc::new(maple_sync::now_ms),
            rng,
            server: RefCell::new(None),
            worker: RefCell::new(None),
            advertiser: RefCell::new(None),
            browser: RefCell::new(None),
            thumbs,
            thumb_px,
            thumb_quality,
            blobs: crate::remote::blobs(),
        })
    }

    pub fn trust(&self) -> &Arc<Mutex<TrustStore>> {
        &self.trust
    }

    pub fn pairing_slot(&self) -> PairingSlot {
        self.pairing.clone()
    }

    pub fn rng(&self) -> SharedRandom {
        self.rng.clone()
    }

    pub fn clock(&self) -> Clock {
        self.clock.clone()
    }

    /// Stop whatever is running and start whatever the current role calls for.
    ///
    /// Safe to call after any change — role, pairing, unpairing, a settings
    /// edit. Stopping first is not optional: two listeners cannot share a
    /// port, and two workers would race each other's watermarks.
    pub fn restart(self: &Rc<Self>) {
        *self.worker.borrow_mut() = None;
        *self.server.borrow_mut() = None;
        // Dropped before the listener is rebuilt, so no window exists where
        // a record points at a port nothing answers on.
        *self.advertiser.borrow_mut() = None;
        // Cleared unconditionally, then re-set below only if this device is
        // still a servant with a paired master. Leaving a stale client in
        // place would have every remote thumbnail keep dialling a device we
        // may no longer hold a key for, failing slowly instead of at once.
        self.blobs.clear();

        let settings = maple_state::Settings::load();
        let role = {
            let db = maple_db::lock_db(&self.db);
            // Published for *every* role, unlike the master handle above. A
            // master never fetches anything, but its grid is the one full of
            // rows it can only label: `images.origin_device` is a device id,
            // and the name beside it lives in `sync_peers`, which the grid's
            // decode threads cannot reach.
            self.blobs.set_peer_names(
                db.list_sync_peers()
                    .unwrap_or_default()
                    .into_iter()
                    .map(|peer| (peer.device_id.clone(), peer.display_name()))
                    .collect(),
            );
            db.sync_role().unwrap_or(SyncRole::Off)
        };

        // A browse is only ever useful to a servant. Dropping it here also
        // reaps the one a pairing modal started on a device that has since
        // become a master.
        if role != SyncRole::Servant {
            *self.browser.borrow_mut() = None;
        }

        match role {
            SyncRole::Off => {
                set_status(&self.status, SyncStatus::for_role(SyncRole::Off));
            }
            SyncRole::Master => self.start_master(&settings),
            SyncRole::Servant => self.start_servant(&settings),
        }
    }

    /// The device list the pairing modal and the sync worker read.
    ///
    /// Starts a browse on first use and keeps it: mDNS answers arrive over
    /// the following seconds, so a browser created per lookup would report
    /// an empty network every time. A failure to start is logged once and
    /// then simply means no discovery — every caller treats `None` as "type
    /// the address by hand", which is the documented fallback (§2.4).
    pub fn discovery(&self) -> Option<Arc<dyn DeviceSource>> {
        if self.browser.borrow().is_none() {
            match Browser::start() {
                Ok(browser) => *self.browser.borrow_mut() = Some(Arc::new(browser)),
                Err(e) => {
                    tracing::warn!("sync: mDNS discovery unavailable: {e}");
                    return None;
                }
            }
        }
        self.browser
            .borrow()
            .as_ref()
            .map(|browser| browser.clone() as Arc<dyn DeviceSource>)
    }

    /// Ask a sleeping worker to run a pass now rather than waiting out its
    /// retry delay.
    ///
    /// Returns whether there was a worker to ask. `false` on a master (which
    /// dials nobody), on an `Off` device, and — the case worth naming — after
    /// an auth failure, where the worker has already returned: a rejected
    /// credential is not something a retry fixes, and the pill's `retryable`
    /// flag keeps the button off that state for the same reason.
    ///
    /// Deliberately *not* a `restart`, which would also be a way to get a
    /// fresh pass. `restart` joins the worker thread, and a pass in flight
    /// can be a two-minute round trip — on the UI thread, that is a frozen
    /// window. Waking the existing one is a flag and a notify.
    pub fn retry_now(&self) -> bool {
        match self.worker.borrow().as_ref() {
            Some(worker) => {
                worker.retry_now();
                true
            }
            None => false,
        }
    }

    /// Stop everything. Called when the app closes.
    pub fn stop(&self) {
        *self.worker.borrow_mut() = None;
        *self.server.borrow_mut() = None;
        *self.advertiser.borrow_mut() = None;
        *self.browser.borrow_mut() = None;
        self.blobs.clear();
    }

    fn start_master(self: &Rc<Self>, settings: &maple_state::Settings) {
        // Amber until the listener's first status refresh, rather than
        // whatever the previous role left behind.
        set_status(&self.status, SyncStatus::for_role(SyncRole::Master));

        let result = SyncServer::spawn(
            ServerConfig {
                listen_addr: settings.sync.listen_addr.clone(),
                max_revs: maple_db::sync::DEFAULT_MAX_REVS,
                online_window_ms: ONLINE_WINDOW_MS,
            },
            ServerDeps {
                db: self.db.clone(),
                trust: self.trust.clone(),
                pairing: self.pairing.clone(),
                status: self.status.clone(),
                clock: self.clock.clone(),
                rng: self.rng.clone(),
                thumbs: self.thumbs.clone(),
                render_thumb: self.thumb_renderer(),
                layout: layout(settings),
                on_change: Arc::new(on_change),
            },
        );

        match result {
            Ok(server) => {
                tracing::info!("sync: listening on {}", server.local_addr());
                self.advertise(server.local_addr());
                *self.server.borrow_mut() = Some(server);
            }
            Err(e) => {
                // A port already in use is the common case, and it is the
                // user's to fix — surfaced on the pill's hover rather than
                // only in a log they will not read.
                tracing::error!("sync: could not start listener: {e}");
                let mut status = SyncStatus::for_role(SyncRole::Master);
                status.last_error = Some(e.to_string());
                set_status(&self.status, status);
            }
        }
    }

    /// Re-publish the mDNS record under a new device name.
    ///
    /// Deliberately not a [`Self::restart`]: the name is the *only* thing
    /// that went stale. `/sync/hello` reads it from the database on every
    /// request, so the listener is already correct, and tearing it down
    /// would drop whatever a servant had in flight for a cosmetic change.
    /// A no-op when this device is not advertising.
    pub fn renamed(&self) {
        let Some(bound) = self
            .server
            .borrow()
            .as_ref()
            .map(|server| server.local_addr())
        else {
            return;
        };
        // The device-name field commits on Enter, which the user may press
        // without having changed anything. Re-publishing then would take the
        // record off every browser on the network and put an identical one
        // back, for nothing.
        if self.advertising_current_name() {
            return;
        }
        // Dropped first, so the old instance name gets its goodbye instead
        // of sitting beside the new one on every browser until it expires.
        *self.advertiser.borrow_mut() = None;
        self.advertise(bound);
    }

    /// Whether the live registration already carries the stored name.
    fn advertising_current_name(&self) -> bool {
        let Some(advertiser) = self.advertiser.borrow().as_ref().map(|a| a.fullname().to_owned())
        else {
            return false;
        };
        let (device_id, name) = {
            let db = maple_db::lock_db(&self.db);
            (
                db.device_id().to_owned(),
                db.device_name().unwrap_or_default(),
            )
        };
        advertiser.starts_with(&maple_sync::discovery::instance_name(&device_id, &name))
    }

    /// Publish this master's `_maple-sync._tcp` record.
    ///
    /// Not fatal if it fails: a servant that already knows the address keeps
    /// working, and one that does not can still be given it by hand. The
    /// listener is the feature; discovery is how it gets found.
    fn advertise(&self, bound: std::net::SocketAddr) {
        let (device_id, name) = {
            let db = maple_db::lock_db(&self.db);
            (
                db.device_id().to_owned(),
                db.device_name().unwrap_or_default(),
            )
        };
        match Advertiser::start(&device_id, &name, bound) {
            Ok(advertiser) => *self.advertiser.borrow_mut() = Some(advertiser),
            Err(e) => tracing::warn!("sync: could not advertise over mDNS: {e}"),
        }
    }

    fn start_servant(self: &Rc<Self>, settings: &maple_state::Settings) {
        set_status(&self.status, SyncStatus::for_role(SyncRole::Servant));

        let Some((master_device_id, address)) = self.master_endpoint() else {
            // Amber `Connecting…` with the reason on hover: a servant with no
            // master is "not set up", which §1.3 deliberately colours
            // differently from "was working, now broken".
            tracing::info!("sync: servant has no paired master yet");
            let mut status = SyncStatus::for_role(SyncRole::Servant);
            status.last_error = Some("no master paired yet".into());
            set_status(&self.status, status);
            return;
        };

        // The blob client is a servant-side thing and belongs to the same
        // (device, key) pair the worker signs with, so it is built from the
        // same trust-store lookup rather than a second, separately-stale one.
        self.point_blobs_at(&master_device_id, &address);

        // …and it has to follow the master when discovery moves it, or a
        // relay servant would keep syncing while every tile on screen stayed
        // blank. Everything this closure captures is `Send + Sync`, which is
        // why it can be built here and called from the worker thread; the
        // supervisor itself is `Rc` and could not be.
        let on_relocate: maple_sync::worker::RelocateHook = {
            let blobs = self.blobs.clone();
            let trust = self.trust.clone();
            let clock = self.clock.clone();
            let rng = self.rng.clone();
            let device_id = {
                let db = maple_db::lock_db(&self.db);
                db.device_id().to_owned()
            };
            let master = master_device_id.clone();
            Arc::new(move |address: &str| {
                point_blobs_at(&blobs, &trust, &clock, &rng, &device_id, &master, address);
            })
        };

        let worker = maple_sync::worker::spawn(
            WorkerConfig {
                address,
                master_device_id,
                interval: Duration::from_secs(settings.sync.interval_secs.max(1)),
                max_revs: maple_db::sync::DEFAULT_MAX_REVS,
                layout: layout(settings),
            },
            WorkerDeps {
                db: self.db.clone(),
                trust: self.trust.clone(),
                status: self.status.clone(),
                clock: self.clock.clone(),
                rng: self.rng.clone(),
                on_change: Arc::new(on_change),
                discovery: self.discovery(),
                on_relocate,
            },
        );
        *self.worker.borrow_mut() = Some(worker);
    }

    /// Point the grid and detail view at `address` for remote pixels.
    fn point_blobs_at(&self, master_device_id: &str, address: &str) {
        let device_id = {
            let db = maple_db::lock_db(&self.db);
            db.device_id().to_owned()
        };
        point_blobs_at(
            &self.blobs,
            &self.trust,
            &self.clock,
            &self.rng,
            &device_id,
            master_device_id,
            address,
        );
    }

    /// The closure the listener renders thumbnails through.
    ///
    /// See [`ThumbRenderer`] for why this is injected: the codec lives in
    /// `maple_ui::thumbnail`, and `maple-ui` depends on `maple-sync`, so the
    /// transport crate cannot call it directly.
    fn thumb_renderer(&self) -> ThumbRenderer {
        let (px, quality) = (self.thumb_px, self.thumb_quality);
        Arc::new(move |path: &std::path::Path| crate::thumbnail::generate_thumbnail(path, px, quality))
    }

    /// The master this servant should dial: its device id and last-known
    /// address.
    ///
    /// P5 supports exactly one master, which is what a star topology means
    /// from the servant's side. The address comes from the trust file, where
    /// pairing recorded it, and an **empty** one is returned rather than
    /// nothing at all: a device paired from the other side has a master it
    /// has never dialled, and since P8 that is a job for discovery rather
    /// than a dead end. The peer with an address wins over one without, so a
    /// known master is never passed over for an unknown one.
    fn master_endpoint(&self) -> Option<(String, String)> {
        let trust = lock(&self.trust);
        let peers = trust.peers();
        let peer = peers
            .iter()
            .find(|peer| peer.address.is_some())
            .or_else(|| peers.first())?;
        Some((
            peer.device_id.clone(),
            peer.address.clone().unwrap_or_default(),
        ))
    }
}

/// Build the blob client and hand it to [`RemoteBlobs`].
///
/// A free function because it is called from two places that cannot share a
/// receiver: the supervisor on the UI thread, and the sync worker's relocate
/// hook on its own. Every argument is `Send + Sync` for that reason.
///
/// A missing key is not an error worth surfacing: the worker is about to hit
/// the same gap and report it through the status pill, and two messages for
/// one cause is one too many.
#[allow(clippy::too_many_arguments)]
fn point_blobs_at(
    blobs: &RemoteBlobs,
    trust: &Arc<Mutex<TrustStore>>,
    clock: &Clock,
    rng: &SharedRandom,
    device_id: &str,
    master_device_id: &str,
    address: &str,
) {
    let key = {
        let trust = lock(trust);
        trust.peer(master_device_id).map(|p| p.key.clone())
    };
    let Some(key) = key else {
        tracing::warn!("sync: no key for master {master_device_id}, remote photos unavailable");
        return;
    };
    // An address discovery has not found yet would build a client that can
    // only fail; leaving the handle cleared makes a fetch fail at once and
    // say why, instead of after a connect timeout per tile.
    if address.is_empty() {
        return;
    }
    let client = SyncClient::new(address, device_id.to_owned(), clock.clone(), rng.clone());
    blobs.set(master_device_id.to_owned(), client, key);
}

/// Where this device files a photo that arrives over the wire.
///
/// The import templates, not a second set: a library organised one way by the
/// card importer and another by sync would be worse than either. Read on each
/// restart rather than cached, so editing the template in Settings takes
/// effect on the next role change like every other sync setting.
fn layout(settings: &maple_state::Settings) -> LibraryLayout {
    LibraryLayout {
        library_dir: settings.library_dir.clone(),
        folder_template: settings.path_template.folder.clone(),
        filename_template: settings.path_template.filename.clone(),
    }
}

/// Reload whatever the UI is showing, from the worker thread.
fn on_change() {
    // `request_reload` touches Slint models, so it has to run on the event
    // loop. This callback does not.
    let _ = slint::invoke_from_event_loop(|| {
        crate::grid::request_reload();
    });
}

fn set_status(cell: &StatusCell, status: SyncStatus) {
    crate::sync_status::set_status(cell, status);
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
