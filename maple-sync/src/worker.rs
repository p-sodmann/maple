//! The servant's sync loop.
//!
//! `std::thread` + `mpsc`, following the same spawn→work→sleep→check-stop
//! shape as `maple_db::worker::spawn_db_worker`. It cannot reuse that helper
//! directly — that one fetches a `Vec` of items and processes them one by
//! one, whereas a sync pass is a single indivisible round trip — but the stop
//! semantics are identical: `recv_timeout` is the sleep, so a quit is noticed
//! immediately instead of after the full interval.
//!
//! # One pass
//!
//! ```text
//! hello            → is anything there, and can we merge with it
//! pull  → apply    → repeat while the master still has more
//! collect → push   → repeat while we still have more
//! transfer         → move originals, if this link's mode moves any
//! ```
//!
//! Pull runs before push so that a conflicting edit is resolved against the
//! master's current state rather than one round stale. Both directions loop
//! rather than shipping one batch per interval: the batch boundary is a stamp
//! count (`DEFAULT_MAX_REVS`), so a first sync of a real library needs many
//! of them, and doing one per interval would take days.
//!
//! # Watermarks and the one rule that matters
//!
//! A watermark advances only when [`may_advance`] says so. A deferred row —
//! one whose mandatory parent has not arrived — is re-sent next pass *only*
//! if the sender still thinks it unsent. Advancing past it loses it silently
//! and permanently, and nothing later notices, because the row's stamp is
//! below the watermark forever after.
//!
//! # A master that moved
//!
//! §1.4 asks for the address to be re-resolved on reconnect, so a DHCP lease
//! change heals itself. The loop does that on the *failure* path and only
//! there: a pass that worked proves the address is right, and asking mDNS
//! anyway would let an unauthenticated record pull a working link somewhere
//! else. What discovery may do is narrow — move an **already paired** peer,
//! identified by the device id in its TXT record, to a new address. The
//! credential does not travel with the record, so a machine advertising
//! someone else's id gains nothing but a connection that fails its MAC.
//!
//! # Why an auth failure ends the thread
//!
//! §1.4: a rejected credential is not a network problem and will not fix
//! itself. The loop sets [`SyncState::Unauthorized`] and *returns*, rather
//! than sleeping and trying again, so nothing keeps a dead credential warm.
//! Re-pairing spawns a new worker.

use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_db::Database;

use crate::backoff::{Backoff, FailureKind, Retry};
use crate::client::{SyncClient, SyncFailure};
use crate::discovery::DeviceSource;
use crate::merge::{apply_and_refresh, may_advance};
use crate::server::Clock;
use crate::transfer::{self, LibraryLayout};
use crate::status::{StatusCell, SyncState};
use crate::trust::{PeerKey, TrustStore};
use maple_state::{PeerMode, SyncRole};

/// Most pull-or-push rounds in a single pass.
///
/// A bound rather than "loop until caught up" so a master that keeps
/// producing changes cannot hold this thread past a quit request forever.
/// Whatever is left is picked up on the next pass, one interval later.
const MAX_ROUNDS: usize = 64;

/// Told where the master moved to, so the rest of the app can follow.
pub type RelocateHook = Arc<dyn Fn(&str) + Send + Sync>;

/// The handles the worker borrows from the app.
pub struct WorkerDeps {
    pub db: Arc<Mutex<Database>>,
    pub trust: Arc<Mutex<TrustStore>>,
    pub status: StatusCell,
    pub clock: Clock,
    pub rng: crate::random::SharedRandom,
    /// Run after any pass that altered the local library, so the caller can
    /// reload whatever it is showing. Called from the worker thread, so a UI
    /// caller must marshal (`slint::Weak::upgrade_in_event_loop`).
    pub on_change: Arc<dyn Fn() + Send + Sync>,
    /// Where to look when the master stops answering. `None` on a network
    /// with no mDNS, or before the browser has started — the loop then does
    /// exactly what it did before P8 and retries the stored address.
    pub discovery: Option<Arc<dyn DeviceSource>>,
    /// Called with the new `host:port` whenever discovery moves the master.
    ///
    /// The worker is not the only thing dialling it: a relay servant's grid
    /// and detail view fetch pixels through their own client, and one built
    /// for the old address would keep failing long after sync had healed.
    /// Called from the worker thread, like [`Self::on_change`].
    pub on_relocate: RelocateHook,
}

/// Which master to talk to and how often.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// `host:port` of the master, as last recorded. May be **empty**: a
    /// device paired from the other side has a master it has never dialled,
    /// and discovery is what finds it. The loop treats that as a failed pass
    /// and goes looking.
    pub address: String,
    /// The master's device id — the key in the trust file, and the
    /// `sync_peers` row that holds both watermarks.
    pub master_device_id: String,
    /// Idle cadence between passes. A *failed* pass retries sooner, on the
    /// backoff schedule.
    pub interval: Duration,
    pub max_revs: usize,
    /// Where a downloaded original is filed, when the mode says to keep one.
    pub layout: LibraryLayout,
}

/// How long to wait before a pass that still has files queued.
///
/// A first sync in **full** mode is thousands of photos and each pass moves
/// at most [`transfer::MAX_TRANSFERS_PER_PASS`] of them; waiting out the
/// five-minute idle interval between batches would turn an afternoon into a
/// fortnight. Not zero, so the loop still yields and a quit is still noticed
/// promptly.
const CATCHUP_INTERVAL: Duration = Duration::from_secs(2);

/// How long to wait before the first attempt at a *newly discovered*
/// address.
///
/// Short on purpose. The backoff schedule measures how long an endpoint has
/// been failing, and a master that just turned up at a different address is
/// not that endpoint — waiting out a 60-second step to try it would be
/// answering the wrong question.
const RELOCATE_DELAY: Duration = Duration::from_secs(1);

/// A running servant loop. Dropping it stops the thread.
pub struct SyncWorker {
    stop: SyncSender<()>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SyncWorker {
    pub fn stop(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        let _ = self.stop.send(());
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for SyncWorker {
    fn drop(&mut self) {
        // Same reasoning as `SyncServer`: this thread holds the database
        // mutex during `apply_batch`, and letting it outlive the app means
        // it can be mid-transaction while the connection is torn down.
        self.stop_and_join();
    }
}

/// What one pass accomplished, for the status cell.
#[derive(Debug, Default, Clone, Copy)]
struct PassOutcome {
    merged: usize,
    pushed: usize,
    /// Rows still waiting to go out when the pass ended.
    pending: usize,
    /// Whether anything landed locally — the signal to refresh the UI.
    changed: bool,
    /// Whether the file half of the pass stopped with work still queued.
    more_files: bool,
}

/// Start syncing with a master.
pub fn spawn(config: WorkerConfig, deps: WorkerDeps) -> SyncWorker {
    let (stop_tx, stop_rx) = mpsc::sync_channel(1);
    let WorkerDeps {
        db,
        trust,
        status,
        clock,
        rng,
        on_change,
        discovery,
        on_relocate,
    } = deps;

    let thread = std::thread::Builder::new()
        .name("maple-sync-worker".into())
        .spawn(move || {
            let device_id = {
                let guard = lock(&db);
                guard.device_id().to_owned()
            };
            // Both mutable, because discovery may move the master: see the
            // module docs. `address` is the `host:port` form the trust file
            // and the mDNS record use, so it can be compared with what a
            // browse returns; the client holds the URL built from it.
            let mut address = config.address.clone();
            let mut client =
                SyncClient::new(&address, device_id.clone(), clock.clone(), rng.clone());
            let mut backoff = Backoff::new();

            tracing::info!(
                "sync worker: started, master {} every {:?}",
                if address.is_empty() { "(not located yet)" } else { &address },
                config.interval
            );
            set_state(&status, SyncState::Connecting, None);

            // Whether the last pass failed. A pass that merges nothing is
            // still worth reporting when it is the first one after an outage:
            // every remote thumbnail the grid tried to fetch while the master
            // was unreachable is still a blank tile, and nothing else ever
            // retries one. "The master answered again" is the moment they can
            // succeed, and only this loop knows when that happened.
            let mut recovering = false;

            loop {
                // An empty address is a paired master this device has never
                // reached — the far side did the dialling when they paired,
                // so nothing recorded where it lives. Rather than a special
                // startup path, it enters the loop as a failed pass, and the
                // retry branch below is what goes looking for it.
                let pass = if address.is_empty() {
                    Err(PassError::Failed(SyncFailure {
                        kind: FailureKind::Unreachable,
                        code: None,
                        message: "no address for the master yet".into(),
                    }))
                } else {
                    run_pass(&db, &trust, &status, &client, &config, &stop_rx)
                };
                match pass {
                    Ok(outcome) => {
                        backoff.on_success();
                        publish_success(&status, &outcome, (clock)());
                        if outcome.changed || recovering {
                            on_change();
                        }
                        recovering = false;
                        let wait = if outcome.more_files {
                            CATCHUP_INTERVAL
                        } else {
                            config.interval
                        };
                        match stop_rx.recv_timeout(wait) {
                            Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                            Err(RecvTimeoutError::Timeout) => {}
                        }
                    }
                    Err(PassError::Stopped) => break,
                    Err(PassError::Failed(failure)) => {
                        let retry = backoff.on_failure(failure.kind);
                        match retry {
                            Retry::Never => {
                                tracing::error!(
                                    "sync worker: {failure} — not retrying, this needs re-pairing"
                                );
                                set_state(
                                    &status,
                                    SyncState::Unauthorized,
                                    Some(failure.to_string()),
                                );
                                break;
                            }
                            Retry::After(mut delay) => {
                                recovering = true;
                                // Before sleeping: has the master turned up
                                // somewhere else? Only reachable from the
                                // retryable branch, so a latched auth halt
                                // cannot be cleared by the reset below.
                                if let Some(found) =
                                    relocate(&discovery, &config.master_device_id, &address)
                                {
                                    tracing::info!(
                                        "sync worker: master moved from {address} to {found}"
                                    );
                                    note_address(&trust, &config.master_device_id, &found);
                                    address = found;
                                    client = SyncClient::new(
                                        &address,
                                        device_id.clone(),
                                        clock.clone(),
                                        rng.clone(),
                                    );
                                    backoff.reset();
                                    delay = RELOCATE_DELAY;
                                    on_relocate(&address);
                                }
                                tracing::warn!(
                                    "sync worker: {failure} — retrying in {}s",
                                    delay.as_secs()
                                );
                                set_state(
                                    &status,
                                    SyncState::Offline {
                                        retry_secs: delay.as_secs(),
                                    },
                                    Some(failure.to_string()),
                                );
                                match stop_rx.recv_timeout(delay) {
                                    Ok(_) | Err(RecvTimeoutError::Disconnected) => break,
                                    Err(RecvTimeoutError::Timeout) => {}
                                }
                            }
                        }
                    }
                }
            }

            tracing::info!("sync worker: stopped");
        })
        .expect("failed to spawn sync worker thread");

    SyncWorker {
        stop: stop_tx,
        thread: Some(thread),
    }
}

/// A pass either failed or was interrupted by a stop request.
enum PassError {
    Failed(SyncFailure),
    Stopped,
}

impl From<SyncFailure> for PassError {
    fn from(failure: SyncFailure) -> Self {
        Self::Failed(failure)
    }
}

fn run_pass(
    db: &Arc<Mutex<Database>>,
    trust: &Arc<Mutex<TrustStore>>,
    status: &StatusCell,
    client: &SyncClient,
    config: &WorkerConfig,
    stop_rx: &mpsc::Receiver<()>,
) -> Result<PassOutcome, PassError> {
    let hello = client.hello()?;
    SyncClient::check_compatible(&hello)?;
    if hello.role != SyncRole::Master.as_str() {
        // Not a transport failure and not an auth failure: the peer is
        // reachable and would happily answer, but a servant syncing to a
        // servant is not a topology the merge engine is built for and would
        // form a loop. Retryable, because the user may simply not have
        // switched the other machine to master yet.
        return Err(PassError::Failed(SyncFailure {
            kind: FailureKind::Unreachable,
            code: None,
            message: format!("peer at {} is '{}', not a master", client.address(), hello.role),
        }));
    }

    let key = peer_key(trust, &config.master_device_id)?;
    let mode = peer_mode(db, &config.master_device_id);
    let mut outcome = PassOutcome::default();

    // ── Pull ────────────────────────────────────────────────────
    let mut seen = 0usize;
    for _ in 0..MAX_ROUNDS {
        check_stop(stop_rx)?;
        let since = watermark(db, &config.master_device_id).0;
        let batch = client.pull(&key, since, config.max_revs, mode)?;
        if batch.next_rev <= since {
            break;
        }

        seen += batch.len();
        set_state(
            status,
            SyncState::Running {
                done: outcome.merged as u32,
                total: seen as u32,
            },
            None,
        );

        let guard = lock(db);
        let report =
            apply_and_refresh(&guard, &batch, &config.master_device_id).map_err(internal)?;
        if may_advance(&report) {
            guard
                .set_sync_peer_pull_rev(&config.master_device_id, batch.next_rev)
                .map_err(internal)?;
        } else {
            // The watermark is pinned until the missing parents arrive; a
            // further round would re-fetch the same rows forever.
            tracing::info!("sync worker: {} rows deferred, holding watermark", report.deferred);
            outcome.merged += report.inserted + report.updated + report.deleted;
            outcome.changed |= report.changed();
            break;
        }
        outcome.merged += report.inserted + report.updated + report.deleted + report.resurrected;
        outcome.changed |= report.changed();
        drop(guard);
    }

    // ── Push ────────────────────────────────────────────────────
    for _ in 0..MAX_ROUNDS {
        check_stop(stop_rx)?;
        let since = watermark(db, &config.master_device_id).1;
        let batch = {
            let guard = lock(db);
            guard.collect_changes(since, config.max_revs).map_err(internal)?
        };
        if batch.next_rev <= since {
            outcome.pending = 0;
            break;
        }
        outcome.pending = batch.len();
        set_state(
            status,
            SyncState::Running {
                done: outcome.pushed as u32,
                total: (outcome.pushed + batch.len()) as u32,
            },
            None,
        );

        let response = client.push(&key, &batch)?;
        outcome.pushed += response.applied;
        if response.deferred > 0 {
            // The master is missing parents for some of these. Leaving our
            // watermark where it is means we re-send the batch next pass,
            // which is wasteful but is the only thing that does not lose the
            // deferred rows.
            tracing::info!(
                "sync worker: master deferred {} rows, holding watermark",
                response.deferred
            );
            break;
        }
        let guard = lock(db);
        guard
            .set_sync_peer_push_rev(&config.master_device_id, batch.next_rev)
            .map_err(internal)?;
        outcome.pending = 0;
    }

    // ── Files ───────────────────────────────────────────────────
    //
    // After metadata, never before: a photo can only be downloaded once its
    // row exists to be filled in, and can only be uploaded once the master
    // has a row to want it. On a first sync the two happen in the same pass,
    // in this order.
    check_stop(stop_rx)?;
    let files = transfer::transfer(
        db,
        client,
        &key,
        mode,
        &config.layout,
        &|| matches!(stop_rx.try_recv(), Ok(_) | Err(TryRecvError::Disconnected)),
        &|done, total| {
            set_state(
                status,
                SyncState::Running {
                    done: done as u32,
                    total: total as u32,
                },
                None,
            );
        },
    )?;
    if files.moved() > 0 || files.skipped > 0 {
        tracing::info!(
            "sync worker: {} downloaded, {} uploaded, {} skipped",
            files.downloaded,
            files.uploaded,
            files.skipped
        );
    }
    outcome.more_files = files.more_pending;
    // A downloaded photo stops being a relayed one: its tile now has a file
    // behind it, so the grid has to re-read the row it is drawing.
    outcome.changed |= files.downloaded > 0;

    Ok(outcome)
}

/// Where discovery says the master is, if that is somewhere new.
///
/// `None` covers three cases that all mean the same thing to the caller —
/// no discovery running, the master not currently on the network, and the
/// master exactly where we already thought it was. Only a genuine move is
/// worth rebuilding a client for.
fn relocate(
    discovery: &Option<Arc<dyn DeviceSource>>,
    master_device_id: &str,
    current: &str,
) -> Option<String> {
    let found = discovery.as_ref()?.address_of(master_device_id)?;
    (found != current).then_some(found)
}

/// Remember where the master answered, so the next launch dials the right
/// place before discovery has found anything.
///
/// Advisory, and a failure to write it is not worth failing a pass over: the
/// worst case is that the next start needs one more round of discovery.
fn note_address(trust: &Arc<Mutex<TrustStore>>, master_device_id: &str, address: &str) {
    if let Err(e) = lock(trust).note_address(master_device_id, address) {
        tracing::warn!("sync worker: could not record the master's address: {e}");
    }
}

/// This link's file mode, as the *servant* has it set.
///
/// The servant's disk is the one that fills, so the servant's setting governs
/// — and it is the servant's settings card that offers the choice. An
/// unreadable or absent peer row reads as [`PeerMode::Relay`], the mode that
/// moves nothing: a missing setting must not start filling a disk.
fn peer_mode(db: &Arc<Mutex<Database>>, master_device_id: &str) -> PeerMode {
    let guard = lock(db);
    match guard.sync_peer(master_device_id) {
        Ok(Some(peer)) => peer.mode,
        Ok(None) => PeerMode::Relay,
        Err(e) => {
            tracing::warn!("sync worker: could not read the peer's mode: {e}");
            PeerMode::Relay
        }
    }
}

/// `(last_pull_rev, last_push_rev)` for the master, defaulting to zero.
///
/// Zero is "never synced", and a pull asks for `rev > 0`, so an unknown peer
/// correctly asks for everything rather than nothing.
fn watermark(db: &Arc<Mutex<Database>>, master_device_id: &str) -> (i64, i64) {
    let guard = lock(db);
    match guard.sync_peer(master_device_id) {
        Ok(Some(peer)) => (peer.last_pull_rev, peer.last_push_rev),
        Ok(None) => (0, 0),
        Err(e) => {
            tracing::warn!("sync worker: could not read watermarks: {e}");
            (0, 0)
        }
    }
}

/// The long-term key for this master.
///
/// A missing key is [`FailureKind::Auth`], not a transport error: it means
/// this device was unpaired (here or on the master), and no amount of
/// retrying produces one. This is the case §1.4 exists for.
fn peer_key(trust: &Arc<Mutex<TrustStore>>, master_device_id: &str) -> Result<PeerKey, PassError> {
    let guard = lock(trust);
    guard
        .peer(master_device_id)
        .map(|peer| peer.key.clone())
        .ok_or_else(|| {
            PassError::Failed(SyncFailure {
                kind: FailureKind::Auth,
                code: Some(crate::protocol::ErrorCode::Unauthorized),
                message: format!("no stored key for master {master_device_id}"),
            })
        })
}

fn check_stop(stop_rx: &mpsc::Receiver<()>) -> Result<(), PassError> {
    match stop_rx.try_recv() {
        Ok(_) | Err(TryRecvError::Disconnected) => Err(PassError::Stopped),
        Err(TryRecvError::Empty) => Ok(()),
    }
}

/// A local failure (SQL, mostly) is retryable: the master is fine, and the
/// next pass may find the lock free or the disk unwedged.
fn internal(error: impl std::fmt::Display) -> PassError {
    PassError::Failed(SyncFailure {
        kind: FailureKind::Unreachable,
        code: None,
        message: error.to_string(),
    })
}

fn publish_success(status: &StatusCell, outcome: &PassOutcome, now_ms: i64) {
    let mut guard = lock(status);
    // "Synced" only when there is genuinely nothing left. A first full sync
    // finishes its metadata in seconds and its photographs in hours, and a
    // green idle pill through all of that would be a lie the user has no way
    // to check. With files still queued the state is left exactly as the last
    // transfer set it — rewriting it to a fresh `0/0` every two seconds would
    // make the count flicker rather than climb.
    if !outcome.more_files {
        guard.state = SyncState::Idle;
    }
    guard.peers_online = 1;
    guard.last_sync_ms = Some(now_ms);
    guard.pending = outcome.pending as u32;
    guard.last_error = None;
}

fn set_state(status: &StatusCell, state: SyncState, error: Option<String>) {
    let mut guard = lock(status);
    guard.state = state;
    if matches!(guard.state, SyncState::Offline { .. } | SyncState::Unauthorized) {
        guard.peers_online = 0;
    }
    guard.last_error = error;
}

/// Recover from a poisoned lock rather than propagating the panic — see the
/// same helper in [`crate::server`].
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
