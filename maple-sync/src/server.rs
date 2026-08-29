//! The master's HTTP listener.
//!
//! One blocking `tiny_http` server on one background thread — no tokio, no
//! async, matching the architecture the rest of the workspace commits to.
//! Requests are handled one at a time: a master serves a handful of servants,
//! and every route ends up holding the database mutex anyway.
//!
//! Serving serially is a choice; *accepting* serially is not. `tiny_http`
//! reads each connection on a thread from a pool, and a task holding a
//! keep-alive connection parks there waiting for a second request that a
//! one-shot client never sends. A burst of connections arriving while one
//! thread is still idle queues behind it, and it pins itself on the first —
//! leaving the rest accepted and never parsed, with their callers waiting out
//! a timeout on requests this loop never sees. The fix has to be on the
//! caller: [`crate::client`] asks for `Connection: close`, so no task ever
//! parks. Anything else pointed at this listener has to do the same.
//!
//! # Routes
//!
//! ```text
//! GET  /sync/hello         unsigned — reachability probe (`protocol::Hello`)
//! POST /pair/claim         unsigned — the §2.1 handshake
//! POST /sync/pull          signed   — "everything stamped above my watermark"
//! POST /sync/push          signed   — merge the caller's batch
//! POST /sync/wanted        signed   — hashes this device lists but lacks
//! GET  /blob/thumb/{hash}  signed   — WebP thumbnail, rendered on a miss
//! GET  /blob/orig/{hash}   signed   — the original file, streamed
//! POST /blob/orig/{hash}   signed   — receive an original (§3.8)
//! ```
//!
//! The blob routes are what make **relay** possible: a servant that stores no
//! originals still renders a grid and a detail view by fetching them here.
//! They are signed like everything else — an unpaired machine on the LAN must
//! not be able to read the library one photo at a time.
//!
//! The last two are P7's, and they are the master's entire part in moving
//! files *to* it: it cannot dial a servant, so a servant asks what is wanted
//! and posts it. What that admits is deliberately narrow — a paired peer can
//! only supply bytes for a row this library already holds the metadata of,
//! and only bytes that hash to what that row already says. See
//! [`crate::transfer`].
//!
//! # What "signed" costs, in order
//!
//! For the JSON routes the body is read *before* anything else looks at it,
//! because the MAC covers a hash of it — which means an unauthenticated
//! caller chooses how much the server allocates. [`MAX_BODY_BYTES`] is the
//! bound on that, and it is checked against the declared length before the
//! read rather than after.
//!
//! The upload route is the one exception, and has to be: its bodies are
//! photographs, and buffering one to check a signature would mean holding a
//! whole raw file in memory before knowing whether the caller is anyone. It
//! verifies the signature over the *path* with an empty body and streams what
//! follows to disk — safe only because the path names the content hash, which
//! is checked against the bytes as they land. See
//! [`SyncClient::upload_orig`](crate::client::SyncClient::upload_orig).
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
use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_db::{Database, SyncBatch, ThumbnailCache};
use maple_state::{PeerMode, SyncRole};

use crate::auth::{NonceRing, SignedRequest};
use crate::merge::{apply_and_refresh, may_advance};
use crate::pairing::{ClaimRequest, PairingSlot};
use crate::protocol::{
    route, ErrorBody, ErrorCode, Hello, PullRequest, PushResponse, UploadResponse, WantedRequest,
    WantedResponse, MAX_BODY_BYTES, PROTOCOL_VERSION,
};
use crate::transfer::{receive_to_file, LibraryLayout, MAX_UPLOAD_BYTES};
use crate::random::SharedRandom;
use crate::status::StatusCell;
use crate::trust::{TrustStore, TrustedPeer};

/// Wall-clock source, injected so tests can pin it.
pub type Clock = Arc<dyn Fn() -> i64 + Send + Sync>;

/// Renders a thumbnail for a file on disk, as WebP bytes.
///
/// Injected rather than called directly, for a structural reason: the codec
/// lives in `maple_ui::thumbnail`, and `maple-ui` depends on *this* crate, so
/// the arrow cannot be turned round. The alternatives were to move thumbnail
/// rendering down into `maple-db`, or to serve only what the master happens
/// to have cached and 404 the rest — the second would make a servant's grid
/// depend on what the master's own user had recently scrolled past. Injection
/// matches how [`Clock`] and [`SharedRandom`] already arrive, keeps image
/// decoding out of a transport crate, and the size and quality are the
/// caller's settings baked into the closure.
pub type ThumbRenderer = Arc<dyn Fn(&Path) -> anyhow::Result<Vec<u8>> + Send + Sync>;

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
    /// The master's own thumbnail store, consulted before rendering anything.
    pub thumbs: Arc<ThumbnailCache>,
    pub render_thumb: ThumbRenderer,
    /// Where an uploaded original is filed. The same templates the master's
    /// own imports use, so a photo that arrives from a servant is organised
    /// as if it had been imported here.
    pub layout: LibraryLayout,
    /// Run after a request changed this library, so the caller can reload
    /// what it is showing.
    ///
    /// A master is passive in every other respect, which is exactly why it
    /// needs this: nothing on this side polls, so a photo a servant pushed
    /// would sit in the database unseen until the app was restarted. Called
    /// from the listener thread — a UI caller must marshal
    /// (`slint::Weak::upgrade_in_event_loop`).
    pub on_change: Arc<dyn Fn() + Send + Sync>,
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
    thumbs: Arc<ThumbnailCache>,
    render_thumb: ThumbRenderer,
    layout: LibraryLayout,
    on_change: Arc<dyn Fn() + Send + Sync>,
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
            thumbs: deps.thumbs,
            render_thumb: deps.render_thumb,
            layout: deps.layout,
            on_change: deps.on_change,
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

/// What a handler produced.
///
/// The sync routes answer in JSON; the blob routes answer in bytes, and an
/// original can be a 60 MB raw file. `File` keeps that one streaming from
/// disk rather than buffering the whole photo in the master's memory just to
/// hand it to the socket — [`MAX_BODY_BYTES`] bounds what a *caller* can make
/// the server allocate, and serving blobs must not reintroduce the same
/// exposure from the other direction.
enum Payload {
    Json(String),
    Bytes {
        data: Vec<u8>,
        content_type: &'static str,
    },
    File {
        file: std::fs::File,
        len: u64,
        content_type: &'static str,
    },
}

/// One response type for all of them, so `handle` has a single exit.
type Reply = tiny_http::Response<Box<dyn std::io::Read + Send + 'static>>;

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
        ("POST", route::WANTED) => signed(ctx, &mut request, &method, &url, wanted),
        ("GET", p) if p.starts_with(route::BLOB_THUMB) => {
            let p = p.to_owned();
            signed(ctx, &mut request, &method, &url, move |ctx, _dev, _body| {
                blob_thumb(ctx, &p)
            })
        }
        ("GET", p) if p.starts_with(route::BLOB_ORIG) => {
            let (p, raw) = (p.to_owned(), wants_raw(&url));
            signed(ctx, &mut request, &method, &url, move |ctx, _dev, _body| {
                blob_orig(ctx, &p, raw)
            })
        }
        ("POST", p) if p.starts_with(route::BLOB_ORIG) => {
            let (p, raw, ext) = (p.to_owned(), wants_raw(&url), raw_ext(&url));
            signed_stream(ctx, &mut request, &method, &url, move |ctx, _dev, request| {
                blob_upload(ctx, request, &p, raw, ext.as_deref())
            })
        }
        _ => Err(ErrorBody::new(
            ErrorCode::BadRequest,
            format!("no route for {method} {path}"),
        )),
    };

    let response = match result {
        Ok(payload) => ok_response(payload),
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

/// `?raw=1` on a `/blob/orig/` URL asks for the companion raw file.
///
/// Matched on the exact token rather than a substring: the query is part of
/// the signed path, so a caller that wrote it differently has already failed
/// verification by the time this runs.
fn wants_raw(url: &str) -> bool {
    url.split_once('?')
        .map(|(_, query)| query.split('&').any(|p| p == "raw=1"))
        .unwrap_or(false)
}

/// The `ext=` parameter on a companion upload: what to file the raw as.
///
/// Sanitised by [`route::sanitise_ext`] rather than trusted — it becomes part
/// of a filename on this disk, and the sender is a peer.
fn raw_ext(url: &str) -> Option<String> {
    url.split_once('?')?
        .1
        .split('&')
        .find_map(|p| p.strip_prefix("ext="))
        .and_then(route::sanitise_ext)
}

/// Read the body, verify the signature over it, then run `f`.
fn signed<F>(
    ctx: &Ctx,
    request: &mut tiny_http::Request,
    method: &str,
    url: &str,
    f: F,
) -> Result<Payload, ErrorBody>
where
    F: FnOnce(&Ctx, &str, &[u8]) -> Result<Payload, ErrorBody>,
{
    let credential = credential(request)?;
    let body = read_body(request)?;
    verify(ctx, &credential, method, url, &body)?;
    // Only now is `device_id` more than a claim.
    f(ctx, &credential.device_id, &body)
}

/// Verify the signature over an **empty** body, then hand `f` the request
/// with its body still unread.
///
/// Only for the upload route, and only because the blob is content-addressed:
/// the hash is in the signed path, and `f` checks the bytes against it as it
/// writes them. Do not reach for this for a route whose body is not
/// self-verifying — there the MAC is the only thing standing between a paired
/// peer's request and a forged one.
fn signed_stream<F>(
    ctx: &Ctx,
    request: &mut tiny_http::Request,
    method: &str,
    url: &str,
    f: F,
) -> Result<Payload, ErrorBody>
where
    F: FnOnce(&Ctx, &str, &mut tiny_http::Request) -> Result<Payload, ErrorBody>,
{
    // Both rejections drain first — see `reject_streamed`. The sender is
    // already writing a photograph into this socket, and answering without
    // reading it closes the connection under a client that then reports a
    // broken pipe instead of the error it was actually sent.
    let credential = match credential(request) {
        Ok(credential) => credential,
        Err(error) => return Err(reject_streamed(request, error)),
    };
    if let Err(error) = verify(ctx, &credential, method, url, &[]) {
        return Err(reject_streamed(request, error));
    }
    let device_id = credential.device_id.clone();
    f(ctx, &device_id, request)
}

/// Answer a *streamed* request with an error, after consuming its body.
///
/// `tiny_http` closes the connection when a handler responds without reading
/// the request body — and on this one route the body is a photograph the
/// sender is still writing. So the client's write fails and it reports
/// `io: Broken pipe` rather than the JSON error it was actually sent.
///
/// Which would be a cosmetic problem if the two meant the same thing to the
/// caller, and they do not. `transfer::send_file` maps `NotFound` and
/// `BadRequest` to "skip this one file, keep the link" and *everything else*
/// to "the link is down" — so an ordinary per-file rejection came back
/// unrecognisable, ended the whole pass, and put the servant into backoff.
/// The next pass reached the same file and failed identically: sync stops
/// dead at the first photo the master declines, with nothing in the log but a
/// broken pipe and a growing retry delay.
///
/// Draining is bounded by the same cap the accepting path uses. Past that the
/// connection does drop, which is the right answer to a peer sending more
/// than the protocol allows — and unlike the rejections above, it is not a
/// case a well-behaved client can reach.
fn reject_streamed(request: &mut tiny_http::Request, error: ErrorBody) -> ErrorBody {
    let mut body = std::io::Read::take(request.as_reader(), MAX_UPLOAD_BYTES);
    match std::io::copy(&mut body, &mut std::io::sink()) {
        Ok(drained) => {
            tracing::debug!("sync server: drained {drained} bytes before answering {}", error.code)
        }
        // The sender gave up mid-body, or the socket died. Nothing to do —
        // the error still goes out, and if the connection is gone it simply
        // does not arrive.
        Err(e) => tracing::debug!("sync server: could not drain a rejected upload: {e}"),
    }
    error
}

fn credential(request: &tiny_http::Request) -> Result<SignedRequest, ErrorBody> {
    let header = request
        .headers()
        .iter()
        .find(|h| h.field.equiv("Authorization"))
        .map(|h| h.value.as_str().to_owned())
        .ok_or_else(|| ErrorBody::new(ErrorCode::Malformed, "no Authorization header"))?;

    SignedRequest::parse(&header).map_err(|e| ErrorBody::new(ErrorCode::from(e), e.to_string()))
}

fn verify(
    ctx: &Ctx,
    credential: &SignedRequest,
    method: &str,
    url: &str,
    body: &[u8],
) -> Result<(), ErrorBody> {
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
    let mut nonces = lock(&ctx.nonces);
    credential
        .verify(&key, method, url, body, now, &mut nonces)
        .map_err(|e| ErrorBody::new(ErrorCode::from(e), e.to_string()))
}

// ── Handlers ────────────────────────────────────────────────────

fn hello(ctx: &Ctx) -> Result<Payload, ErrorBody> {
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

fn claim(ctx: &Ctx, body: &[u8]) -> Result<Payload, ErrorBody> {
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

fn pull(ctx: &Ctx, device_id: &str, body: &[u8]) -> Result<Payload, ErrorBody> {
    let request: PullRequest = parse(body)?;
    let max_revs = if request.max_revs == 0 {
        ctx.config.max_revs
    } else {
        request.max_revs.min(ctx.config.max_revs)
    };

    let db = lock(&ctx.db);
    let batch = db.collect_changes(request.since, max_revs).map_err(internal)?;

    // The mode is the servant's to choose, so the master takes what it is
    // told. Recorded before the watermarks because it is what the settings
    // card renders, and a peer that never gets past the first pull should
    // still show as what it actually is.
    if let Some(mode) = request.mode.as_deref() {
        let reported = PeerMode::parse(mode);
        if let Err(e) = db.set_sync_peer_mode(device_id, reported) {
            tracing::warn!("sync server: could not record {device_id}'s mode: {e}");
        }
    }

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

fn push(ctx: &Ctx, device_id: &str, body: &[u8]) -> Result<Payload, ErrorBody> {
    let batch: SyncBatch = parse(body)?;
    let db = lock(&ctx.db);
    let report = apply_and_refresh(&db, &batch, device_id).map_err(internal)?;

    if may_advance(&report) {
        if let Err(e) = db.set_sync_peer_pull_rev(device_id, batch.next_rev) {
            tracing::warn!("sync server: could not record {device_id}'s watermark: {e}");
        }
    }
    if let Err(e) = db.touch_sync_peer(device_id, (ctx.clock)()) {
        tracing::warn!("sync server: could not touch {device_id}: {e}");
    }

    // The lock goes before the callback: a UI reload reads the same database,
    // and handing it the mutex we are still holding would deadlock it.
    let changed = report.changed();
    drop(db);
    if changed {
        (ctx.on_change)();
    }

    encode(&PushResponse {
        applied: report.inserted + report.updated + report.deleted + report.resurrected,
        deferred: report.deferred,
        acked_rev: batch.next_rev,
    })
}

/// `POST /sync/wanted` — the hashes this device lists but does not hold.
///
/// The master's whole part in receiving files. It cannot dial a servant, so
/// it publishes what it is short of and lets whoever can supply it do so.
/// Answering costs one indexed scan of the relayed rows, and it tells a
/// caller nothing it did not already learn from the metadata it pulled.
fn wanted(ctx: &Ctx, _device_id: &str, body: &[u8]) -> Result<Payload, ErrorBody> {
    let request: WantedRequest = parse(body)?;
    let limit = if request.limit == 0 {
        crate::transfer::MAX_TRANSFERS_PER_PASS
    } else {
        request.limit.min(crate::transfer::MAX_TRANSFERS_PER_PASS * 4)
    };

    let db = lock(&ctx.db);
    let hashes = db.wanted_hashes(limit).map_err(internal)?;
    encode(&WantedResponse {
        hashes: hashes.iter().map(route::hex).collect(),
    })
}

/// `POST /blob/orig/{hash}[?raw=1]` — take delivery of an original.
///
/// # What this is allowed to write
///
/// Only a file some row here is already waiting for. The hash must name a
/// `locality = 'remote'` row, and the bytes must hash to it — so a paired
/// peer can fill in a photo this library already knows about and cannot
/// invent a new one, replace an existing one, or choose a path. That is a
/// tighter grant than the peer already has over metadata, and it is what
/// makes accepting writes over the network defensible at all.
///
/// # Why a companion is staged rather than filed
///
/// A row names one display file and one companion, and adopting the display
/// file is what flips it to local. Filing the companion afterwards would find
/// nothing waiting for it, so a raw arrives *first*, waits in `.incoming`,
/// and is placed in the same breath as the display file it belongs to.
fn blob_upload(
    ctx: &Ctx,
    request: &mut tiny_http::Request,
    path: &str,
    raw: bool,
    ext: Option<&str>,
) -> Result<Payload, ErrorBody> {
    let hash = match blob_hash(path, route::BLOB_ORIG) {
        Ok(hash) => hash,
        Err(error) => return Err(reject_streamed(request, error)),
    };

    let row = {
        let db = lock(&ctx.db);
        db.row_wanting(&hash).map_err(internal)?
    };
    let Some(row) = row else {
        // Not an error on the sender's part: it asked what was wanted, and
        // between then and now another servant may have supplied this very
        // photo. `NotFound` is the code that has it drop this file and keep
        // the link — which it can only read if the body is drained first.
        return Err(reject_streamed(
            request,
            ErrorBody::new(ErrorCode::NotFound, "nothing here is waiting for that hash"),
        ));
    };
    if raw && row.raw_filename.is_none() {
        // This row has never heard of a companion, and the sender is holding
        // one. Believe it: `origin_raw_path` only arrived with P7, so a row
        // replicated by an earlier build is NULL and *nothing will ever fix
        // it* — `update_row` carries the origin's value, but only when
        // something else re-stamps the row, and a photograph nobody edits
        // again is never re-stamped. Refusing here meant the negative could
        // not cross on any pass, ever, and the servant re-offered it once
        // every pass forever.
        //
        // This does relax "only a row that is already waiting can be filled
        // in" (see `crate::transfer`) by one notch: a paired peer can now
        // attach a companion to a photo whose row did not declare one. The
        // blast radius is unchanged — `row_wanting` still gates it to a row
        // this library already replicated and is missing bytes for, and a
        // companion's bytes were *already* unverifiable and taken on the
        // pairing's word. An `images.raw_hash` column is what would close
        // both, and that is a schema change.
        let Some(ext) = ext else {
            return Err(reject_streamed(
                request,
                ErrorBody::new(
                    ErrorCode::BadRequest,
                    "a companion for a photo with none recorded must name its extension",
                ),
            ));
        };
        let noted = {
            let db = lock(&ctx.db);
            db.note_remote_companion(row.id, ext).map_err(internal)?
        };
        if !noted {
            // Lost a race with another servant filling the same row in.
            // Its companion is as good as this one.
            return Err(reject_streamed(
                request,
                ErrorBody::new(ErrorCode::NotFound, "that photo is no longer waiting"),
            ));
        }
        tracing::info!(
            "sync server: {} has a companion after all, filing it as .{ext}",
            row.filename
        );
    }

    let staged = ctx.layout.staged_path(&hash, raw);
    let written = receive_to_file(request.as_reader(), &staged, MAX_UPLOAD_BYTES).map_err(|e| {
        let _ = std::fs::remove_file(&staged);
        internal(e)
    })?;

    if raw {
        // Unverifiable — the schema hashes the display file, not its
        // companion — so this one is taken on the pairing's word. See the
        // `transfer` module docs.
        return encode(&UploadResponse {
            stored: false,
            path: None,
        });
    }
    if written != hash {
        let _ = std::fs::remove_file(&staged);
        return Err(ErrorBody::new(
            ErrorCode::BadRequest,
            "those bytes do not hash to the blob they were sent as",
        ));
    }

    // Filing and adopting under one lock, deliberately: this master's own
    // library scanner inserts any file no row claims, and `images.path` is
    // UNIQUE, so a scan landing between the two would take the path and leave
    // the adoption to fail on the constraint. See `crate::transfer`.
    let placed = {
        let db = lock(&ctx.db);
        let staged_raw = ctx.layout.staged_path(&hash, true);
        let companion = match (&row.raw_filename, staged_raw.exists()) {
            (Some(name), true) => Some((staged_raw.as_path(), name.as_str())),
            _ => None,
        };
        // One call for the pair: a companion filed anywhere but beside its
        // display file, under the same stem, is a second photograph as far as
        // this master's own library scanner is concerned. See
        // `maple_import::place_pair`.
        let (placed, placed_raw) = ctx
            .layout
            .place(&staged, &row.filename, companion)
            .map_err(internal)?;
        db.adopt_original(row.id, &placed, placed_raw.as_deref())
            .map_err(internal)?;
        placed
    };
    ctx.layout.discard(&hash);
    tracing::info!("sync server: received {} → {}", row.filename, placed.display());
    // The row stopped being relayed and now has a file behind it, which is a
    // different tile in the grid.
    (ctx.on_change)();

    encode(&UploadResponse {
        stored: true,
        path: Some(placed.display().to_string()),
    })
}

/// `GET /blob/thumb/{hash}` — the thumbnail for a photo, by content hash.
///
/// Cache first, render on a miss, and store what was rendered: a servant
/// scrolling the master's library warms the master's own cache, which is the
/// same work the master's grid would have done later anyway.
fn blob_thumb(ctx: &Ctx, path: &str) -> Result<Payload, ErrorBody> {
    let hash = blob_hash(path, route::BLOB_THUMB)?;

    if let Some(webp) = ctx.thumbs.get(&hash) {
        return Ok(webp_payload(webp));
    }

    // The database lock is released before rendering: decoding a 40-megapixel
    // JPEG takes long enough that holding it would stall every other route.
    let file = {
        let db = lock(&ctx.db);
        db.blob_path(&hash, false).map_err(internal)?
    };
    let Some(file) = file else {
        return Err(missing(&hash));
    };

    let webp = (ctx.render_thumb)(&file).map_err(|e| {
        // Not `internal`: the row exists and names a file this master cannot
        // decode (deleted under it, or a format it has no reader for). That is
        // a permanent property of this hash, so the caller should stop asking.
        tracing::warn!("sync server: could not render {}: {e}", file.display());
        ErrorBody::new(ErrorCode::NotFound, "could not render a thumbnail for that hash")
    })?;

    if let Err(e) = ctx.thumbs.insert(&hash, &webp) {
        tracing::warn!("sync server: thumbnail cache write failed: {e}");
    }
    Ok(webp_payload(webp))
}

/// `GET /blob/orig/{hash}[?raw=1]` — the original file's bytes, streamed.
///
/// Nothing is read into memory here: the file goes straight from disk to the
/// socket, so serving a raw file costs the master a file handle rather than
/// its size in RAM.
fn blob_orig(ctx: &Ctx, path: &str, raw: bool) -> Result<Payload, ErrorBody> {
    let hash = blob_hash(path, route::BLOB_ORIG)?;

    let found = {
        let db = lock(&ctx.db);
        db.blob_path(&hash, raw).map_err(internal)?
    };
    let Some(found) = found else {
        return Err(missing(&hash));
    };

    let file = std::fs::File::open(&found).map_err(|e| {
        // The row says the file is here and it is not — the scanner will mark
        // it missing within the minute. Until then, `not_found` is both true
        // and the answer that keeps the servant's link alive.
        tracing::warn!("sync server: could not open {}: {e}", found.display());
        ErrorBody::new(ErrorCode::NotFound, "that blob is no longer readable here")
    })?;
    let len = file.metadata().map_err(internal)?.len();

    Ok(Payload::File {
        file,
        len,
        content_type: "application/octet-stream",
    })
}

fn blob_hash(path: &str, prefix: &str) -> Result<[u8; 32], ErrorBody> {
    route::blob_hash(path, prefix).ok_or_else(|| {
        // A malformed hash is the caller's bug, not a miss: answering 404
        // would have it retry a URL that can never work.
        ErrorBody::new(ErrorCode::BadRequest, "blob path is not a 64-character hex hash")
    })
}

/// Deliberately says nothing about *why* the hash is unknown — whether the
/// library never had it or the row is remote here too is not the caller's
/// business, and both mean the same thing to it.
fn missing(hash: &[u8; 32]) -> ErrorBody {
    tracing::debug!("sync server: no local blob for {}", hex(hash));
    ErrorBody::new(ErrorCode::NotFound, "no blob with that hash here")
}

fn webp_payload(data: Vec<u8>) -> Payload {
    Payload::Bytes {
        data,
        content_type: "image/webp",
    }
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
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

fn encode<T: serde::Serialize>(value: &T) -> Result<Payload, ErrorBody> {
    serde_json::to_string(value)
        .map(Payload::Json)
        .map_err(internal)
}

/// Turn any internal failure into a retryable error, logging the cause.
///
/// The message reaches the peer, so it says what failed without echoing
/// anything the peer did not already send.
fn internal(error: impl std::fmt::Display) -> ErrorBody {
    tracing::error!("sync server: {error}");
    ErrorBody::new(ErrorCode::Internal, "the master could not complete the request")
}

fn json_response(status: u16, body: &str) -> Reply {
    let bytes = body.as_bytes().to_vec();
    let len = bytes.len() as u64;
    reply(status, "application/json", len, Box::new(std::io::Cursor::new(bytes)))
}

fn ok_response(payload: Payload) -> Reply {
    match payload {
        Payload::Json(body) => json_response(200, &body),
        Payload::Bytes { data, content_type } => {
            let len = data.len() as u64;
            reply(200, content_type, len, Box::new(std::io::Cursor::new(data)))
        }
        Payload::File { file, len, content_type } => {
            reply(200, content_type, len, Box::new(file))
        }
    }
}

fn reply(
    status: u16,
    content_type: &str,
    len: u64,
    body: Box<dyn std::io::Read + Send + 'static>,
) -> Reply {
    let header =
        tiny_http::Header::from_bytes(&b"Content-Type"[..], content_type.as_bytes())
            .expect("content types here are ASCII literals");
    tiny_http::Response::new(
        tiny_http::StatusCode(status),
        vec![header],
        body,
        Some(len as usize),
        None,
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
