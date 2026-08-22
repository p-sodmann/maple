//! The master's HTTP listener.
//!
//! One blocking `tiny_http` server on one background thread — no tokio, no
//! async, matching the architecture the rest of the workspace commits to.
//! Concurrency is not the point here: a master serves a handful of servants
//! that poll every few minutes, and every route ends up holding the database
//! mutex anyway.
//!
//! # Routes
//!
//! ```text
//! GET  /sync/hello    unsigned — reachability probe (see `protocol::Hello`)
//! POST /pair/claim    unsigned — the §2.1 handshake
//! POST /sync/pull     signed   — "everything stamped above my watermark"
//! POST /sync/push     signed   — merge the caller's batch
//! ```
//!
//! # What "signed" costs, in order
//!
//! For the two signed routes the body is read *before* anything else looks at
//! it, because the MAC covers a hash of it — which means an unauthenticated
//! caller chooses how much the server allocates. [`MAX_BODY_BYTES`] is the
//! bound on that, and it is checked against the declared length before the
//! read rather than after.
//!
//! The signature is then checked before the JSON is parsed. Parsing first
//! would run a serde deserializer over attacker-controlled bytes for no
//! reason, and would let an unpaired caller learn which shapes the server
//! accepts by watching which ones 400 and which ones 401.
//!
//! # Determinism
//!
//! Time and randomness arrive as a [`Clock`] and a
//! [`SharedRandom`](crate::random::SharedRandom), never sampled here, so a
//! loopback test can pin both and assert exact bytes — the same property P3
//! established for the handshake.

use std::net::SocketAddr;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_db::{Database, SyncBatch};
use maple_state::{PeerMode, SyncRole};

use crate::auth::{NonceRing, SignedRequest};
use crate::merge::{apply_and_refresh, may_advance};
use crate::pairing::{ClaimRequest, PairingSlot};
use crate::protocol::{
    route, ErrorBody, ErrorCode, Hello, PullRequest, PushResponse, MAX_BODY_BYTES,
    PROTOCOL_VERSION,
};
use crate::random::SharedRandom;
use crate::status::StatusCell;
use crate::trust::{TrustStore, TrustedPeer};

/// Wall-clock source, injected so tests can pin it.
pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// How long the accept loop blocks before re-checking the stop flag.
///
/// Short enough that quitting the app feels instant, long enough that an idle
/// master is not spinning a thread. `tiny_http` has no way to interrupt a
/// blocking `recv`, so this poll is the shutdown mechanism.
const ACCEPT_POLL: Duration = Duration::from_millis(200);

/// Where to listen and how much to ship.
#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// `host:port`. `0.0.0.0:7645` in production; `127.0.0.1:0` in tests,
    /// which is how they get an ephemeral port without racing each other.
    pub listen_addr: String,
    /// Stamp-group cap per pull batch.
    pub max_revs: usize,
    /// How stale a peer's `last_seen_at` may be before the pill stops
    /// counting it as connected.
    pub online_window_ms: i64,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:7645".into(),
            max_revs: maple_db::sync::DEFAULT_MAX_REVS,
            // Two default sync intervals' grace, so one missed pass does not
            // flicker the pill.
            online_window_ms: 10 * 60 * 1000,
        }
    }
}

/// The handles the listener borrows from the app.
pub struct ServerDeps {
    pub db: Arc<Mutex<Database>>,
    pub trust: Arc<Mutex<TrustStore>>,
    pub pairing: PairingSlot,
    /// The pill's cell. A master is passive, so it has no worker of its own
    /// to write this — the accept loop does it, which also makes the reading
    /// honest: it reports what the listener has actually seen.
    pub status: StatusCell,
    pub clock: Clock,
    pub rng: SharedRandom,
}

/// Everything a request handler needs.
struct Ctx {
    db: Arc<Mutex<Database>>,
    /// The key store. Held in memory and rewritten on each pairing rather
    /// than re-read per request — a request that has to hit the disk to
    /// check a MAC gives an unpaired caller a way to generate disk traffic.
    trust: Arc<Mutex<TrustStore>>,
    pairing: PairingSlot,
    clock: Clock,
    rng: SharedRandom,
    status: StatusCell,
    nonces: Mutex<NonceRing>,
    config: ServerConfig,
}

/// A running master listener. Dropping it stops the thread.
pub struct SyncServer {
    addr: SocketAddr,
    stop: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
}

impl SyncServer {
    /// Bind `listen_addr` and start serving.
    ///
    /// Binding happens on *this* thread so a port already in use is reported
    /// to the caller as an error it can show the user, rather than vanishing
    /// into a background thread that quietly never serves anything.
    pub fn spawn(config: ServerConfig, deps: ServerDeps) -> anyhow::Result<Self> {
        let server = tiny_http::Server::http(&config.listen_addr)
            .map_err(|e| anyhow::anyhow!("could not bind {}: {e}", config.listen_addr))?;
        let addr = server.server_addr().to_ip().ok_or_else(|| {
            anyhow::anyhow!("{} did not resolve to an IP socket", config.listen_addr)
        })?;

        let ctx = Arc::new(Ctx {
            db: deps.db,
            trust: deps.trust,
            pairing: deps.pairing,
            clock: deps.clock,
            rng: deps.rng,
            status: deps.status,
            nonces: Mutex::new(NonceRing::new()),
            config,
        });

        let stop = Arc::new(AtomicBool::new(false));
        let thread = std::thread::Builder::new()
            .name("maple-sync-server".into())
            .spawn({
                let stop = stop.clone();
                move || {
                    tracing::info!("sync server: listening on {addr}");
                    refresh_status(&ctx);
                    let mut last_status = (ctx.clock)();
                    while !stop.load(Ordering::Relaxed) {
                        match server.recv_timeout(ACCEPT_POLL) {
                            Ok(Some(request)) => {
                                handle(&ctx, request);
                                // A served request may have changed which
                                // peers count as connected, so refresh before
                                // waiting again rather than up to a second
                                // later.
                                refresh_status(&ctx);
                                last_status = (ctx.clock)();
                            }
                            Ok(None) => {
                                // Idle: peers still time *out* of the online
                                // window with nothing happening, so the pill
                                // has to be re-derived even when nobody calls.
                                let now = (ctx.clock)();
                                if now - last_status >= 1_000 {
                                    refresh_status(&ctx);
                                    last_status = now;
                                }
                            }
                            Err(e) => {
                                tracing::warn!("sync server: accept failed: {e}");
                                break;
                            }
                        }
                    }
                    tracing::info!("sync server: stopped");
                }
            })?;

        Ok(Self {
            addr,
            stop,
            thread: Some(thread),
        })
    }

    /// The address actually bound. Not the configured one: a `:0` port is
    /// how the tests get an ephemeral port without racing each other for a
    /// fixed one.
    pub fn local_addr(&self) -> SocketAddr {
        self.addr
    }

    /// Ask the listener to stop and wait for it.
    pub fn shutdown(mut self) {
        self.stop_and_join();
    }

    fn stop_and_join(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for SyncServer {
    fn drop(&mut self) {
        // Joining in `Drop` rather than detaching: the thread holds an
        // `Arc<Mutex<Database>>`, and letting it outlive the app means it can
        // be mid-`apply_batch` while the process tears the connection down.
        self.stop_and_join();
    }
}

// ── Dispatch ────────────────────────────────────────────────────

fn handle(ctx: &Ctx, mut request: tiny_http::Request) {
    let method = request.method().as_str().to_owned();
    // Signed exactly as sent, query string included, so a route that grows
    // parameters later cannot have them stripped in flight.
    let url = request.url().to_owned();
    let path = url.split('?').next().unwrap_or("").to_owned();

    let result = match (method.as_str(), path.as_str()) {
        ("GET", route::HELLO) => hello(ctx),
        ("POST", route::PAIR_CLAIM) => read_body(&mut request).and_then(|body| claim(ctx, &body)),
        ("POST", route::PULL) => signed(ctx, &mut request, &method, &url, pull),
        ("POST", route::PUSH) => signed(ctx, &mut request, &method, &url, push),
        _ => Err(ErrorBody::new(
            ErrorCode::BadRequest,
            format!("no route for {method} {path}"),
        )),
    };

    let response = match result {
        Ok(json) => json_response(200, &json),
        Err(error) => {
            tracing::debug!("sync server: {method} {path} → {} ({})", error.code, error.message);
            let body = serde_json::to_string(&error)
                .unwrap_or_else(|_| r#"{"code":"internal","message":"?"}"#.to_owned());
            json_response(error.code.http_status(), &body)
        }
    };
    if let Err(e) = request.respond(response) {
        tracing::debug!("sync server: could not send response: {e}");
    }
}

/// Read the body, verify the signature over it, then run `f`.
fn signed<F>(
    ctx: &Ctx,
    request: &mut tiny_http::Request,
    method: &str,
    url: &str,
    f: F,
) -> Result<String, ErrorBody>
where
    F: FnOnce(&Ctx, &str, &[u8]) -> Result<String, ErrorBody>,
{
    let header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_owned())
        .ok_or_else(|| ErrorBody::new(ErrorCode::Malformed, "no Authorization header"))?;

    let credential = SignedRequest::parse(&header)
        .map_err(|e| ErrorBody::new(ErrorCode::from(e), e.to_string()))?;

    let body = read_body(request)?;

    // Look the key up and release the trust lock before verifying: the MAC is
    // cheap but the lock is also taken by pairing, which writes a file.
    let key = {
        let trust = lock(&ctx.trust);
        trust.peer(&credential.device_id).map(|p| p.key.clone())
    };
    let Some(key) = key else {
        // Deliberately the same code as a bad MAC. An attacker probing which
        // device ids are paired learns nothing, and the servant's reaction is
        // identical either way: this credential is dead, re-pair.
        return Err(ErrorBody::new(
            ErrorCode::Unauthorized,
            "no key for this device",
        ));
    };

    let now = (ctx.clock)();
    {
        let mut nonces = lock(&ctx.nonces);
        credential
            .verify(&key, method, url, &body, now, &mut nonces)
            .map_err(|e| ErrorBody::new(ErrorCode::from(e), e.to_string()))?;
    }

    // Only now is `device_id` more than a claim.
    f(ctx, &credential.device_id, &body)
}

// ── Handlers ────────────────────────────────────────────────────

fn hello(ctx: &Ctx) -> Result<String, ErrorBody> {
    let db = lock(&ctx.db);
    let hello = Hello {
        device_id: db.device_id().to_owned(),
        name: db.device_name().unwrap_or_default(),
        role: db.sync_role().unwrap_or(SyncRole::Off).as_str().to_owned(),
        protocol: PROTOCOL_VERSION,
        schema_version: db.schema_version().map_err(internal)?,
    };
    encode(&hello)
}

fn claim(ctx: &Ctx, body: &[u8]) -> Result<String, ErrorBody> {
    let request: ClaimRequest = parse(body)?;
    let now = (ctx.clock)();

    // Notice expiry before answering, so a claim arriving one millisecond
    // late is refused rather than served by a window nobody is watching.
    ctx.pairing.expire_if_due(now);

    let response = ctx
        .pairing
        .handle_claim(&request, now, &ctx.rng)
        .map_err(internal)?
        .map_err(|e| ErrorBody::new(ErrorCode::from(e), e.to_string()))?;

    // Persist here rather than leaving it for the UI to drain. The response
    // can be lost in flight; if it is, the client retries and gets a *fresh*
    // key, and the master must already have stored the previous one or the
    // two would disagree about which pairing is current. Storing eagerly and
    // letting the retry overwrite converges; storing lazily does not.
    persist_pairing(ctx, &request)?;

    encode(&response)
}

/// Record a completed pairing in both stores.
fn persist_pairing(ctx: &Ctx, request: &ClaimRequest) -> Result<(), ErrorBody> {
    let Some(outcome) = ctx.pairing.take_outcome() else {
        return Ok(());
    };

    {
        let mut trust = lock(&ctx.trust);
        trust
            .upsert_peer(TrustedPeer {
                device_id: outcome.device_id.clone(),
                key: outcome.key,
                // The master does not learn the servant's listen address — the
                // servant is the one that dials. Left empty rather than guessed
                // from the request's source address, which is the NAT's problem
                // to get wrong, not ours.
                address: None,
            })
            .map_err(internal)?;
    }

    let db = lock(&ctx.db);
    // Relay by default: it stores nothing, so a pairing completed before the
    // user has chosen a mode cannot start filling a disk.
    db.upsert_sync_peer(
        &outcome.device_id,
        Some(request.name.as_str()),
        PeerMode::Relay,
    )
    .map_err(internal)?;
    tracing::info!("sync server: paired with {} ({})", request.name, outcome.device_id);
    Ok(())
}

fn pull(ctx: &Ctx, device_id: &str, body: &[u8]) -> Result<String, ErrorBody> {
    let request: PullRequest = parse(body)?;
    let max_revs = if request.max_revs == 0 {
        ctx.config.max_revs
    } else {
        request.max_revs.min(ctx.config.max_revs)
    };

    let db = lock(&ctx.db);
    let batch = db.collect_changes(request.since, max_revs).map_err(internal)?;

    // A pull at watermark N is the peer telling us it holds everything we
    // stamped at or below N. That is the only signal the master gets in this
    // direction, and §3.3's tombstone pruning needs it.
    if let Err(e) = db.set_sync_peer_push_rev(device_id, request.since) {
        tracing::warn!("sync server: could not record {device_id}'s watermark: {e}");
    }
    if let Err(e) = db.touch_sync_peer(device_id, (ctx.clock)()) {
        tracing::warn!("sync server: could not touch {device_id}: {e}");
    }

    encode(&batch)
}

fn push(ctx: &Ctx, device_id: &str, body: &[u8]) -> Result<String, ErrorBody> {
    let batch: SyncBatch = parse(body)?;
    let db = lock(&ctx.db);
    let report = apply_and_refresh(&db, &batch).map_err(internal)?;

    if may_advance(&report) {
        if let Err(e) = db.set_sync_peer_pull_rev(device_id, batch.next_rev) {
            tracing::warn!("sync server: could not record {device_id}'s watermark: {e}");
        }
    }
    if let Err(e) = db.touch_sync_peer(device_id, (ctx.clock)()) {
        tracing::warn!("sync server: could not touch {device_id}: {e}");
    }

    encode(&PushResponse {
        applied: report.inserted + report.updated + report.deleted + report.resurrected,
        deferred: report.deferred,
        acked_rev: batch.next_rev,
    })
}

/// Re-derive the pill from what the listener has actually seen.
///
/// A master has no worker to write the status cell, and counting rows in
/// `sync_peers` from the UI thread once a second would put a database lock on
/// the render path. Doing it here instead keeps that off the UI thread and
/// makes the number mean something concrete: peers that have made a signed
/// request recently, not peers that exist in a table.
fn refresh_status(ctx: &Ctx) {
    let now = (ctx.clock)();
    let online = {
        let db = lock(&ctx.db);
        match db.list_sync_peers() {
            Ok(peers) => peers
                .into_iter()
                .filter(|p| {
                    p.last_seen_at
                        .is_some_and(|seen| now - seen < ctx.config.online_window_ms)
                })
                .count() as u32,
            Err(e) => {
                tracing::warn!("sync server: could not count peers: {e}");
                0
            }
        }
    };

    let mut status = lock(&ctx.status);
    status.peers_online = online;
    // `display()` renders a master with zero peers as amber "Listening · no
    // devices" whichever of these it is, so the distinction here is only for
    // a future reader of the cell.
    status.state = if online == 0 {
        crate::status::SyncState::Connecting
    } else {
        crate::status::SyncState::Idle
    };
    status.last_error = None;
}

// ── Plumbing ────────────────────────────────────────────────────

/// Read the whole body, refusing anything over [`MAX_BODY_BYTES`].
///
/// The declared length is checked first so an oversized body is rejected
/// without reading it; the reader is then capped anyway, because a chunked
/// request declares no length at all and would otherwise be unbounded.
fn read_body(request: &mut tiny_http::Request) -> Result<Vec<u8>, ErrorBody> {
    if let Some(declared) = request.body_length() {
        if declared as u64 > MAX_BODY_BYTES {
            return Err(ErrorBody::new(
                ErrorCode::BadRequest,
                format!("body of {declared} bytes exceeds the {MAX_BODY_BYTES}-byte limit"),
            ));
        }
    }
    let mut body = Vec::new();
    let mut capped = std::io::Read::take(request.as_reader(), MAX_BODY_BYTES + 1);
    std::io::Read::read_to_end(&mut capped, &mut body)
        .map_err(|e| ErrorBody::new(ErrorCode::BadRequest, format!("could not read body: {e}")))?;
    if body.len() as u64 > MAX_BODY_BYTES {
        return Err(ErrorBody::new(
            ErrorCode::BadRequest,
            "body exceeds the size limit",
        ));
    }
    Ok(body)
}

fn parse<T: serde::de::DeserializeOwned>(body: &[u8]) -> Result<T, ErrorBody> {
    serde_json::from_slice(body)
        .map_err(|e| ErrorBody::new(ErrorCode::BadRequest, format!("could not parse body: {e}")))
}

fn encode<T: serde::Serialize>(value: &T) -> Result<String, ErrorBody> {
    serde_json::to_string(value).map_err(internal)
}

/// Turn any internal failure into a retryable error, logging the cause.
///
/// The message reaches the peer, so it says what failed without echoing
/// anything the peer did not already send.
fn internal(error: impl std::fmt::Display) -> ErrorBody {
    tracing::error!("sync server: {error}");
    ErrorBody::new(ErrorCode::Internal, "the master could not complete the request")
}

fn json_response(status: u16, body: &str) -> tiny_http::Response<std::io::Cursor<Vec<u8>>> {
    tiny_http::Response::from_string(body)
        .with_status_code(status)
        .with_header(
            tiny_http::Header::from_bytes(&b"Content-Type"[..], &b"application/json"[..])
                .expect("static header is well-formed"),
        )
}

/// Recover from a poisoned lock rather than propagating the panic.
///
/// A panicked handler leaves no half-written invariant behind — every mutation
/// path here is either a SQLite transaction or a whole-value replacement — so
/// taking the master's whole listener down with one bad request would be a
/// worse outcome than continuing.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
