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
use crate::merge::{apply_and_refresh, may_advance};
use crate::server::Clock;
use crate::status::{StatusCell, SyncState};
use crate::trust::{PeerKey, TrustStore};
use maple_state::SyncRole;

/// Most pull-or-push rounds in a single pass.
///
/// A bound rather than "loop until caught up" so a master that keeps
/// producing changes cannot hold this thread past a quit request forever.
/// Whatever is left is picked up on the next pass, one interval later.
const MAX_ROUNDS: usize = 64;

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
}

/// Which master to talk to and how often.
#[derive(Debug, Clone)]
pub struct WorkerConfig {
    /// `host:port` of the master.
    pub address: String,
    /// The master's device id — the key in the trust file, and the
    /// `sync_peers` row that holds both watermarks.
    pub master_device_id: String,
    /// Idle cadence between passes. A *failed* pass retries sooner, on the
    /// backoff schedule.
    pub interval: Duration,
    pub max_revs: usize,
}

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
    } = deps;

    let thread = std::thread::Builder::new()
        .name("maple-sync-worker".into())
        .spawn(move || {
            let device_id = {
                let guard = lock(&db);
                guard.device_id().to_owned()
            };
            let client = SyncClient::new(&config.address, device_id, clock.clone(), rng);
            let mut backoff = Backoff::new();

            tracing::info!(
                "sync worker: started, master {} every {:?}",
                config.address,
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
                match run_pass(&db, &trust, &status, &client, &config, &stop_rx) {
                    Ok(outcome) => {
                        backoff.on_success();
                        publish_success(&status, &outcome, (clock)());
                        if outcome.changed || recovering {
                            on_change();
                        }
                        recovering = false;
                        match stop_rx.recv_timeout(config.interval) {
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
                            Retry::After(delay) => {
                                recovering = true;
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
            message: format!("peer at {} is '{}', not a master", config.address, hello.role),
        }));
    }

    let key = peer_key(trust, &config.master_device_id)?;
    let mut outcome = PassOutcome::default();

    // ── Pull ────────────────────────────────────────────────────
    let mut seen = 0usize;
    for _ in 0..MAX_ROUNDS {
        check_stop(stop_rx)?;
        let since = watermark(db, &config.master_device_id).0;
        let batch = client.pull(&key, since, config.max_revs)?;
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

    Ok(outcome)
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
    guard.state = SyncState::Idle;
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
