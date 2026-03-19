//! Shared background worker pattern for DB-polling loops.
//!
//! Both the AI tagger and face tagger follow the same structure:
//!   1. Fetch a batch of work items from the database.
//!   2. If empty, sleep for `interval` (checking the stop signal).
//!   3. Process each item, checking the stop signal between items.
//!   4. Repeat.
//!
//! This module extracts that loop so new workers can be added without
//! duplicating the sleep / stop-signal / error-handling boilerplate.

use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crate::Database;

/// Handle to a running background worker.  Call [`stop`](WorkerHandle::stop)
/// to request a graceful shutdown (the thread finishes its current item first).
pub struct WorkerHandle {
    stop_tx: SyncSender<()>,
}

impl WorkerHandle {
    /// Signal the worker thread to stop after the current item.
    pub fn stop(&self) {
        let _ = self.stop_tx.send(());
    }
}

/// Spawn a background worker thread that polls the database for work.
///
/// - `name` — thread name (shown in logs and OS thread lists).
/// - `db` — shared database handle.
/// - `state` — mutable processor state (e.g. the AI describer or face detector).
/// - `interval` — how long to sleep when the work queue is empty.
/// - `fetch` — called with a locked `&Database` to retrieve the next batch of
///   work items.  Return an empty vec to trigger a sleep.
/// - `process` — called for each work item.  Receives `(&mut state, &Database)`
///   so it can both use the processor and write results back.
pub fn spawn_db_worker<S, I, F, P>(
    name: &str,
    db: Arc<Mutex<Database>>,
    state: S,
    interval: Duration,
    fetch: F,
    process: P,
) -> WorkerHandle
where
    S: Send + 'static,
    I: Send + 'static,
    F: Fn(&Database) -> Vec<I> + Send + 'static,
    P: Fn(&mut S, &Arc<Mutex<Database>>, I) + Send + 'static,
{
    let (stop_tx, stop_rx) = mpsc::sync_channel(1);

    let thread_name = name.to_owned();
    let panic_name = thread_name.clone();
    std::thread::Builder::new()
        .name(thread_name.clone())
        .spawn(move || {
            let mut state = state;
            tracing::info!("{thread_name}: started");

            'outer: loop {
                // ── Fetch work ────────────────────────────────
                let items = {
                    let guard = crate::lock_db(&db);
                    fetch(&guard)
                };

                if items.is_empty() {
                    tracing::info!("{thread_name}: no pending work, sleeping");
                    match stop_rx.recv_timeout(interval) {
                        Ok(_) | Err(RecvTimeoutError::Disconnected) => break 'outer,
                        Err(RecvTimeoutError::Timeout) => continue 'outer,
                    }
                }

                tracing::info!("{thread_name}: processing {} items", items.len());

                // ── Process each item ─────────────────────────
                for item in items {
                    match stop_rx.try_recv() {
                        Ok(_) | Err(TryRecvError::Disconnected) => break 'outer,
                        Err(TryRecvError::Empty) => {}
                    }
                    process(&mut state, &db, item);
                }
            }

            tracing::info!("{thread_name}: stopped");
        })
        .unwrap_or_else(|_| panic!("failed to spawn {panic_name} thread"));

    WorkerHandle { stop_tx }
}
