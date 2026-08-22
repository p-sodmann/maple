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
use maple_sync::server::{ServerConfig, ServerDeps};
use maple_sync::worker::{WorkerConfig, WorkerDeps};
use maple_sync::{
    Clock, PairingSlot, SharedRandom, StatusCell, SyncServer, SyncStatus, SyncWorker, TrustStore,
};

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
}

impl SyncSupervisor {
    pub fn new(
        db: Arc<Mutex<maple_db::Database>>,
        trust: Arc<Mutex<TrustStore>>,
        pairing: PairingSlot,
        status: StatusCell,
        rng: SharedRandom,
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

        let settings = maple_state::Settings::load();
        let role = {
            let db = maple_db::lock_db(&self.db);
            db.sync_role().unwrap_or(SyncRole::Off)
        };

        match role {
            SyncRole::Off => {
                set_status(&self.status, SyncStatus::for_role(SyncRole::Off));
            }
            SyncRole::Master => self.start_master(&settings),
            SyncRole::Servant => self.start_servant(&settings),
        }
    }

    /// Stop everything. Called when the app closes.
    pub fn stop(&self) {
        *self.worker.borrow_mut() = None;
        *self.server.borrow_mut() = None;
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
            },
        );

        match result {
            Ok(server) => {
                tracing::info!("sync: listening on {}", server.local_addr());
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

        let worker = maple_sync::worker::spawn(
            WorkerConfig {
                address,
                master_device_id,
                interval: Duration::from_secs(settings.sync.interval_secs.max(1)),
                max_revs: maple_db::sync::DEFAULT_MAX_REVS,
            },
            WorkerDeps {
                db: self.db.clone(),
                trust: self.trust.clone(),
                status: self.status.clone(),
                clock: self.clock.clone(),
                rng: self.rng.clone(),
                on_change: Arc::new(on_change),
            },
        );
        *self.worker.borrow_mut() = Some(worker);
    }

    /// The master this servant should dial: its device id and last-known
    /// address.
    ///
    /// P5 supports exactly one master, which is what a star topology means
    /// from the servant's side. The address comes from the trust file, where
    /// pairing recorded it — mDNS re-resolution (P8) is what will eventually
    /// heal a DHCP lease change; until then a moved master needs re-pairing
    /// or a hand-edited address.
    fn master_endpoint(&self) -> Option<(String, String)> {
        let trust = lock(&self.trust);
        trust
            .peers()
            .iter()
            .find_map(|peer| Some((peer.device_id.clone(), peer.address.clone()?)))
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
