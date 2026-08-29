//! Two real installations, one loopback socket.
//!
//! P3 built the handshake as plain functions and drove it in-process; the
//! loopback exercise was deferred to here because until P5 there was no
//! server to loop back to. These tests close that gap: a real `tiny_http`
//! listener on `127.0.0.1:0`, a real `ureq` client, two real SQLite
//! libraries in tempdirs, and the actual merge engine between them.
//!
//! Everything stays deterministic. The clock is an `AtomicI64` the test
//! advances by hand and the random source is a BLAKE3 XOF stream seeded by
//! integer, so a handshake replays byte for byte — the property P3
//! established, preserved across the network boundary.
//!
//! Ports are ephemeral (`:0`, then `local_addr()`), so these run in parallel
//! with each other and with anything else on the machine.

use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_db::Database;
use maple_state::PeerMode;
use maple_sync::auth::SignedRequest;
use maple_sync::backoff::FailureKind;
use maple_sync::discovery::{DeviceSource, DiscoveredDevice};
use maple_sync::pairing::{PairError, PairOutcome, MAX_ATTEMPTS};
use maple_sync::protocol::{route, ErrorCode};
use maple_sync::trust::TrustedPeer;
use maple_sync::{
    Clock, Initiator, PairCode, PairingSlot, RandomSource, SharedRandom, SyncClient, SyncServer,
    TrustStore,
};

/// A fixed instant that is a plausible Unix millisecond, so a stamp derived
/// from it is not mistaken for a counter.
const T0: i64 = 1_700_000_000_000;

// ── Harness ─────────────────────────────────────────────────────

/// A clock the test drives.
struct TestClock(Arc<AtomicI64>);

impl TestClock {
    fn new() -> Self {
        Self(Arc::new(AtomicI64::new(T0)))
    }
    fn handle(&self) -> Clock {
        let inner = self.0.clone();
        Arc::new(move || inner.load(Ordering::Relaxed))
    }
    fn advance(&self, ms: i64) {
        self.0.fetch_add(ms, Ordering::Relaxed);
    }
    fn now(&self) -> i64 {
        self.0.load(Ordering::Relaxed)
    }
}

/// A reproducible byte stream. Not `maple-sync`'s internal one — that is
/// private to the crate, deliberately, so a seeded generator cannot escape
/// into production.
fn seeded(seed: u64) -> SharedRandom {
    struct Xof(Mutex<blake3::OutputReader>);
    impl RandomSource for Xof {
        fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
            let mut reader = self.0.lock().map_err(|_| anyhow::anyhow!("poisoned"))?;
            reader.fill(buf);
            Ok(())
        }
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"maple-loopback");
    hasher.update(&seed.to_le_bytes());
    Arc::new(Xof(Mutex::new(hasher.finalize_xof())))
}

/// One installation: its database, its key store, and the tempdir both live
/// in. The tempdir is held so it outlives them.
struct Install {
    /// Held so the database, key store and any cache inside it outlive the
    /// test; also the root a test writes real photo files into.
    _dir: tempfile::TempDir,
    db: Arc<Mutex<Database>>,
    trust: Arc<Mutex<TrustStore>>,
    device_id: String,
}

impl Install {
    fn new(name: &str) -> Self {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open db");
        db.set_device_name(name).expect("name");
        let device_id = db.device_id().to_owned();
        let trust = TrustStore::open(dir.path().join("sync_trust.json"), &device_id, name)
            .expect("trust store");
        Self {
            _dir: dir,
            db: Arc::new(Mutex::new(db)),
            trust: Arc::new(Mutex::new(trust)),
            device_id,
        }
    }

    fn db(&self) -> std::sync::MutexGuard<'_, Database> {
        self.db.lock().expect("db lock")
    }

    /// This installation's photo directory. Created on demand, because most
    /// of these tests never put a file in one.
    fn library_dir(&self) -> std::path::PathBuf {
        let dir = self._dir.path().join("photos");
        std::fs::create_dir_all(&dir).expect("library dir");
        dir
    }

    /// Where this installation files a photo it receives. A flat destination
    /// and the source name kept, so a test asserting on a path is asserting
    /// on the transfer and not on the template renderer, which has its own
    /// tests in `maple-import`.
    fn layout(&self) -> maple_sync::LibraryLayout {
        self.layout_under("")
    }

    /// The same, under a real folder template.
    ///
    /// Worth one test on its own: a flat destination makes "the companion
    /// went where the display file went" true by construction, and the bug
    /// that shipped was precisely the two files being sent to *different*
    /// folders. See `a_companion_lands_beside_its_photo_under_a_real_template`.
    fn layout_under(&self, folder_template: &str) -> maple_sync::LibraryLayout {
        maple_sync::LibraryLayout {
            library_dir: self.library_dir(),
            folder_template: folder_template.into(),
            filename_template: "{original}".into(),
        }
    }

    /// Add a real file to this library, as an import would: bytes on disk and
    /// a `local` row that names them.
    fn import(&self, name: &str, bytes: &[u8]) -> std::path::PathBuf {
        let path = self.library_dir().join(name);
        std::fs::write(&path, bytes).expect("write photo");
        let hash = maple_import::content_hash(&path).expect("hash");
        self.db()
            .insert_image(&path, &hash, bytes.len() as u64)
            .expect("insert");
        path
    }

    fn trust(&self) -> std::sync::MutexGuard<'_, TrustStore> {
        self.trust.lock().expect("trust lock")
    }
}

/// A master with its listener already running.
struct Master {
    install: Install,
    slot: PairingSlot,
    server: SyncServer,
    status: maple_sync::StatusCell,
    /// The master's thumbnail store, kept so a test can assert what
    /// `/blob/thumb` put in it.
    thumbs: Arc<maple_db::ThumbnailCache>,
    /// How many times the injected renderer actually ran — the difference
    /// between "the master rendered this" and "the cache already had it".
    renders: Arc<AtomicI64>,
    /// How many times the listener asked its UI to reload. A master is
    /// passive and polls nothing, so this is the only thing standing between
    /// a photo a servant sent and a grid that shows it after a restart.
    changes: Arc<AtomicI64>,
}

impl Master {
    fn start(clock: &TestClock, rng: SharedRandom) -> Self {
        Self::start_filing_under(clock, rng, "")
    }

    /// A master that files what it receives under `folder_template`.
    fn start_filing_under(clock: &TestClock, rng: SharedRandom, folder_template: &str) -> Self {
        let install = Install::new("Workstation");
        install
            .db()
            .set_sync_role(maple_state::SyncRole::Master)
            .expect("role");
        let slot = PairingSlot::new();
        let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Master);
        let thumbs = Arc::new(
            maple_db::ThumbnailCache::open(&install._dir.path().join("thumbs"))
                .expect("thumbnail cache"),
        );
        // Stands in for `maple_ui::thumbnail::generate_thumbnail`, which this
        // crate cannot call: `maple-ui` depends on `maple-sync`, not the other
        // way round. What matters here is the plumbing — that a miss reaches a
        // renderer at all, and that its output is what comes back over the
        // wire — so the bytes are a marker rather than a real WebP.
        let renders = Arc::new(AtomicI64::new(0));
        let render_thumb: maple_sync::ThumbRenderer = {
            let renders = renders.clone();
            Arc::new(move |path: &std::path::Path| {
                renders.fetch_add(1, Ordering::Relaxed);
                let bytes = std::fs::read(path)?;
                Ok(format!("THUMB:{}", bytes.len()).into_bytes())
            })
        };
        let changes = Arc::new(AtomicI64::new(0));
        let server = SyncServer::spawn(
            maple_sync::server::ServerConfig {
                listen_addr: "127.0.0.1:0".into(),
                max_revs: 500,
                ..Default::default()
            },
            maple_sync::server::ServerDeps {
                db: install.db.clone(),
                trust: install.trust.clone(),
                pairing: slot.clone(),
                status: status.clone(),
                clock: clock.handle(),
                rng,
                thumbs: thumbs.clone(),
                render_thumb,
                layout: install.layout_under(folder_template),
                on_change: {
                    let changes = changes.clone();
                    Arc::new(move || {
                        changes.fetch_add(1, Ordering::Relaxed);
                    })
                },
            },
        )
        .expect("bind loopback");
        Self {
            install,
            slot,
            server,
            status,
            thumbs,
            renders,
            changes,
        }
    }

    fn address(&self) -> String {
        self.server.local_addr().to_string()
    }
}

fn client(master: &Master, servant: &Install, clock: &TestClock, rng: SharedRandom) -> SyncClient {
    SyncClient::new(&master.address(), &servant.device_id, clock.handle(), rng)
}

/// Run the full handshake over the socket and persist both halves, exactly as
/// the two UIs will. Returns the servant's view of the outcome.
fn pair(
    master: &Master,
    servant: &Install,
    client: &SyncClient,
    clock: &TestClock,
    master_code: &str,
    servant_code: &str,
) -> Result<PairOutcome, ErrorCode> {
    let master_code = PairCode::parse(master_code).expect("six digits");
    let servant_code = PairCode::parse(servant_code).expect("six digits");

    master.slot.open(
        &master.install.device_id,
        "Workstation",
        master_code.clone(),
        clock.now(),
    );
    // The human half: each user types what the other screen shows.
    master.slot.enter_peer_code(servant_code.clone());

    let initiator = Initiator::new(
        &servant.device_id,
        "Laptop",
        &servant_code,
        &master_code,
        &seeded(77),
    )
    .expect("initiator");

    let response = client
        .pair_claim(&initiator.claim())
        .map_err(|e| e.code.unwrap_or(ErrorCode::Internal))?;
    let outcome = initiator.accept(&response).map_err(ErrorCode::from)?;

    // The servant persists its half: the key, and the bookkeeping row that
    // carries the watermarks.
    servant
        .trust()
        .upsert_peer(TrustedPeer {
            device_id: outcome.device_id.clone(),
            key: outcome.key.clone(),
            address: Some(master.address()),
        })
        .expect("store key");
    servant
        .db()
        .upsert_sync_peer(&outcome.device_id, Some(&outcome.name), PeerMode::Relay)
        .expect("store peer");

    Ok(outcome)
}

/// A signed POST built by hand, so a test can send the *same* bytes twice.
fn raw_post(
    address: &str,
    path: &str,
    header: Option<&str>,
    body: &[u8],
) -> (u16, String) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .into();
    let mut request = agent
        .post(format!("http://{address}{path}"))
        .content_type("application/json");
    if let Some(header) = header {
        request = request.header("Authorization", header);
    }
    let mut response = request.send(body).expect("request reached the server");
    let status = response.status().as_u16();
    let text = response.body_mut().read_to_string().expect("read body");
    (status, text)
}

/// A `GET` built by hand, so a test can send a blob request with a header of
/// its choosing — or none at all.
fn raw_get(address: &str, path: &str, header: Option<&str>) -> (u16, Vec<u8>) {
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .http_status_as_error(false)
        .timeout_global(Some(Duration::from_secs(10)))
        .build()
        .into();
    let mut request = agent.get(format!("http://{address}{path}"));
    if let Some(header) = header {
        request = request.header("Authorization", header);
    }
    let mut response = request.call().expect("request reached the server");
    let status = response.status().as_u16();
    let body = response
        .body_mut()
        .with_config()
        .limit(8 * 1024 * 1024)
        .read_to_vec()
        .expect("read body");
    (status, body)
}

/// Put a real file in the master's library and a row pointing at it, so a
/// blob request has something to serve. Returns its content hash.
fn master_photo(master: &Master, name: &str, contents: &[u8]) -> [u8; 32] {
    let path = master.install._dir.path().join(name);
    std::fs::write(&path, contents).expect("write photo");
    let hash: [u8; 32] = blake3::hash(contents).into();
    master
        .install
        .db()
        .insert_image(&path, &hash, contents.len() as u64)
        .expect("insert");
    hash
}

// ── Hello ───────────────────────────────────────────────────────

#[test]
fn hello_answers_before_any_pairing() {
    // The whole reason hello is unsigned: an unpaired servant must be able to
    // tell "the master is there" from "nothing answered", or §1.3's amber and
    // red collapse into one indistinguishable timeout.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(1));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(2));

    let hello = client.hello().expect("hello answers unsigned");
    assert_eq!(hello.device_id, master.install.device_id);
    assert_eq!(hello.name, "Workstation");
    assert_eq!(hello.role, "master");
    assert_eq!(hello.protocol, maple_sync::PROTOCOL_VERSION);
    assert!(hello.schema_version >= 19);
    assert!(SyncClient::check_compatible(&hello).is_ok());
}

#[test]
fn an_unreachable_master_is_a_transport_failure_not_an_auth_one() {
    // A servant whose master is switched off must retry, not demand a
    // re-pair. Binding and immediately dropping the server gives us a port
    // nothing is listening on.
    let clock = TestClock::new();
    let address = {
        let master = Master::start(&clock, seeded(1));
        let address = master.address();
        master.server.shutdown();
        address
    };
    let servant = Install::new("Laptop");
    let client = SyncClient::new(&address, &servant.device_id, clock.handle(), seeded(2));

    let failure = client.hello().expect_err("nothing is listening");
    assert_eq!(failure.kind, FailureKind::Unreachable);
    assert_eq!(failure.code, None);
}

// ── Pairing over the wire ───────────────────────────────────────

#[test]
fn pairing_over_loopback_stores_the_same_key_on_both_sides() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(3));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(4));

    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159")
        .expect("the handshake completes");

    // The mutual half, end to end: both sides hold the *same* 32 bytes, and
    // neither sent them in the clear.
    let master_side = master
        .install
        .trust()
        .peer(&servant.device_id)
        .expect("master stored the servant")
        .key
        .clone();
    assert_eq!(master_side, outcome.key);
    assert_eq!(outcome.device_id, master.install.device_id);
    assert_eq!(outcome.name, "Workstation");

    // And the master recorded the bookkeeping row, defaulting to the mode
    // that stores nothing.
    let peer = master
        .install
        .db()
        .sync_peer(&servant.device_id)
        .expect("query")
        .expect("row exists");
    assert_eq!(peer.name.as_deref(), Some("Laptop"));
    assert_eq!(peer.mode, PeerMode::Relay);
}

#[test]
fn the_same_pair_of_codes_replays_byte_for_byte() {
    // Determinism across the socket, not just in-process: same seeds, same
    // codes, same sealed key. This is what lets the tests above assert on
    // exact values instead of "it didn't crash".
    let run = |seed: u64| {
        let clock = TestClock::new();
        let master = Master::start(&clock, seeded(seed));
        let servant = Install::new("Laptop");
        let client = client(&master, &servant, &clock, seeded(seed + 1));
        *pair(&master, &servant, &client, &clock, "482107", "314159")
            .expect("pairs")
            .key
            .as_bytes()
    };
    assert_eq!(run(11), run(11));
    assert_ne!(run(11), run(12));
}

#[test]
fn a_claim_before_the_master_typed_a_code_is_too_early() {
    // The normal state while one person walks to the other machine. It is a
    // 425 and it must not burn an attempt, or latency alone would abort the
    // window.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(5));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(6));

    master.slot.open(
        &master.install.device_id,
        "Workstation",
        PairCode::parse("482107").unwrap(),
        clock.now(),
    );
    // Note: no `enter_peer_code`.

    let initiator = Initiator::new(
        &servant.device_id,
        "Laptop",
        &PairCode::parse("314159").unwrap(),
        &PairCode::parse("482107").unwrap(),
        &seeded(7),
    )
    .unwrap();

    for _ in 0..8 {
        let failure = client.pair_claim(&initiator.claim()).expect_err("too early");
        assert_eq!(failure.code, Some(ErrorCode::TooEarly));
        assert_eq!(failure.kind, FailureKind::Unreachable, "polling is not fatal");
    }
    assert_eq!(
        master.slot.view(clock.now()).expect("still open").attempts,
        0,
        "polling must not consume attempts"
    );
}

#[test]
fn a_wrong_code_is_rejected_and_five_of_them_burn_the_window() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(8));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(9));

    master.slot.open(
        &master.install.device_id,
        "Workstation",
        PairCode::parse("482107").unwrap(),
        clock.now(),
    );
    master
        .slot
        .enter_peer_code(PairCode::parse("314159").unwrap());

    // A servant that guessed wrong: neither code matches what the master has.
    let guesser = Initiator::new(
        &servant.device_id,
        "Laptop",
        &PairCode::parse("111111").unwrap(),
        &PairCode::parse("222222").unwrap(),
        &seeded(10),
    )
    .unwrap();

    for attempt in 1..=MAX_ATTEMPTS {
        let failure = client.pair_claim(&guesser.claim()).expect_err("bad proof");
        let expected = if attempt < MAX_ATTEMPTS {
            ErrorCode::BadCode
        } else {
            ErrorCode::PairingClosed
        };
        assert_eq!(failure.code, Some(expected), "on attempt {attempt}");
    }

    assert!(
        master.slot.view(clock.now()).is_none(),
        "the window and both codes are gone"
    );
    assert!(
        master.install.trust().peer(&servant.device_id).is_none(),
        "no key was ever minted"
    );
}

#[test]
fn an_expired_window_stops_answering() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(12));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(13));

    master.slot.open(
        &master.install.device_id,
        "Workstation",
        PairCode::parse("482107").unwrap(),
        clock.now(),
    );
    master
        .slot
        .enter_peer_code(PairCode::parse("314159").unwrap());

    clock.advance(maple_sync::pairing::WINDOW_MS);

    let initiator = Initiator::new(
        &servant.device_id,
        "Laptop",
        &PairCode::parse("314159").unwrap(),
        &PairCode::parse("482107").unwrap(),
        &seeded(14),
    )
    .unwrap();
    let failure = client.pair_claim(&initiator.claim()).expect_err("expired");
    assert_eq!(failure.code, Some(ErrorCode::PairingClosed));
}

#[test]
fn a_claim_with_no_window_open_looks_the_same_as_an_expired_one() {
    // Deliberate: from the client's side both mean "stop asking", and
    // distinguishing them would tell an unpaired caller whether a human is
    // currently sitting at the master's pairing screen.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(15));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(16));

    let initiator = Initiator::new(
        &servant.device_id,
        "Laptop",
        &PairCode::parse("314159").unwrap(),
        &PairCode::parse("482107").unwrap(),
        &seeded(17),
    )
    .unwrap();
    let failure = client.pair_claim(&initiator.claim()).expect_err("no window");
    assert_eq!(failure.code, Some(ErrorCode::PairingClosed));
}

// ── Authentication on the signed routes ─────────────────────────

#[test]
fn an_unpaired_device_is_refused_and_the_client_calls_it_fatal() {
    // This is the §1.4 case the whole backoff exception exists for: the
    // master no longer holds our key, and no amount of retrying mints one.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(18));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(19));

    let stranger = maple_sync::PeerKey::from_bytes([0x5A; 32]);
    let failure = client
        .pull(&stranger, 0, 500, PeerMode::Relay)
        .expect_err("no key for this device");
    assert_eq!(failure.code, Some(ErrorCode::Unauthorized));
    assert_eq!(failure.kind, FailureKind::Auth, "must not retry");
}

#[test]
fn unpairing_on_the_master_turns_a_working_link_fatal() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(20));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(21));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    assert!(client.pull(&outcome.key, 0, 500, PeerMode::Relay).is_ok(), "paired link works");

    master
        .install
        .trust()
        .remove_peer(&servant.device_id)
        .expect("unpair");

    let failure = client.pull(&outcome.key, 0, 500, PeerMode::Relay).expect_err("key is gone");
    assert_eq!(failure.kind, FailureKind::Auth);
    assert_eq!(failure.code, Some(ErrorCode::Unauthorized));
}

#[test]
fn a_replayed_request_is_rejected_but_stays_retryable() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(22));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(23));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    // Built by hand so the *same* nonce goes out twice; `SyncClient` draws a
    // fresh one per call, which is exactly what stops this happening for real.
    let body = serde_json::to_vec(&maple_sync::PullRequest {
        mode: None,
        since: 0,
        max_revs: 500,
    })
    .unwrap();
    let credential = SignedRequest::sign(
        &outcome.key,
        &servant.device_id,
        "POST",
        route::PULL,
        &body,
        clock.now(),
        [0x11; 16],
    );

    let (first, _) = raw_post(
        &master.address(),
        route::PULL,
        Some(&credential.header()),
        &body,
    );
    assert_eq!(first, 200, "the first use is fine");

    let (second, text) = raw_post(
        &master.address(),
        route::PULL,
        Some(&credential.header()),
        &body,
    );
    assert_eq!(second, 401);
    assert!(text.contains("replay"), "{text}");

    // Retryable: a lost response looks exactly like this, and the retry
    // draws a fresh nonce.
    assert!(!ErrorCode::Replay.is_fatal());
}

#[test]
fn a_stale_timestamp_is_refused_without_demanding_a_re_pair() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(24));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(25));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let body = serde_json::to_vec(&maple_sync::PullRequest {
        mode: None,
        since: 0,
        max_revs: 500,
    })
    .unwrap();
    let credential = SignedRequest::sign(
        &outcome.key,
        &servant.device_id,
        "POST",
        route::PULL,
        &body,
        clock.now() - 6 * 60 * 1000,
        [0x22; 16],
    );

    let (status, text) = raw_post(
        &master.address(),
        route::PULL,
        Some(&credential.header()),
        &body,
    );
    assert_eq!(status, 401);
    assert!(text.contains("stale_timestamp"), "{text}");
    // A wrong clock is not a wrong key, and clocks get corrected.
    assert!(!ErrorCode::StaleTimestamp.is_fatal());
}

#[test]
fn a_body_rewritten_in_flight_does_not_verify() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(26));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(27));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let signed_body = serde_json::to_vec(&maple_sync::PullRequest {
        mode: None,
        since: 0,
        max_revs: 500,
    })
    .unwrap();
    let credential = SignedRequest::sign(
        &outcome.key,
        &servant.device_id,
        "POST",
        route::PULL,
        &signed_body,
        clock.now(),
        [0x33; 16],
    );

    // A man in the middle rewinds the watermark to re-harvest the library.
    let tampered = serde_json::to_vec(&maple_sync::PullRequest {
        mode: None,
        since: -1,
        max_revs: 999_999,
    })
    .unwrap();
    let (status, text) = raw_post(
        &master.address(),
        route::PULL,
        Some(&credential.header()),
        &tampered,
    );
    assert_eq!(status, 401);
    assert!(text.contains("unauthorized"), "{text}");
}

#[test]
fn a_signed_route_without_a_header_is_refused() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(28));
    let (status, text) = raw_post(&master.address(), route::PULL, None, b"{}");
    assert_eq!(status, 400);
    assert!(text.contains("malformed"), "{text}");
}

#[test]
fn an_unknown_route_is_a_bad_request_not_a_panic() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(29));
    let (status, _) = raw_post(&master.address(), "/sync/nope", None, b"{}");
    assert_eq!(status, 400);
}

// ── The link itself ─────────────────────────────────────────────

#[test]
fn metadata_flows_from_master_to_servant() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(30));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(31));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .db()
        .create_collection("Iceland 2024", "#3584e4", None)
        .expect("create");

    let batch = client.pull(&outcome.key, 0, 500, PeerMode::Relay).expect("pull");
    assert!(!batch.is_empty(), "the master had a change to send");
    let report = maple_sync::merge::apply_and_refresh(&servant.db(), &batch, &master.install.device_id).expect("apply");
    assert!(report.changed());

    let collections = servant.db().all_collections().expect("list");
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "Iceland 2024");
    assert_eq!(collections[0].color, "#3584e4");
}

#[test]
fn metadata_flows_from_servant_to_master() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(32));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(33));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    servant
        .db()
        .create_collection("Weekend", "#e74c3c", None)
        .expect("create");

    let batch = servant.db().collect_changes(0, 500).expect("collect");
    let response = client.push(&outcome.key, &batch).expect("push");
    assert!(response.applied > 0);
    assert_eq!(response.deferred, 0);
    assert_eq!(response.acked_rev, batch.next_rev);

    let collections = master.install.db().all_collections().expect("list");
    assert_eq!(collections.len(), 1);
    assert_eq!(collections[0].name, "Weekend");
}

#[test]
fn a_pull_advances_the_masters_record_of_what_the_servant_holds() {
    // §3.3's tombstone pruning needs this: the master only ever learns the
    // servant's watermark from the `since` on a pull.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(34));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(35));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .db()
        .create_collection("Iceland", "#3584e4", None)
        .expect("create");

    let batch = client.pull(&outcome.key, 0, 500, PeerMode::Relay).expect("first pull");
    assert!(batch.next_rev > 0);
    // The servant merges and comes back with the new watermark.
    let _ = client.pull(&outcome.key, batch.next_rev, 500, PeerMode::Relay).expect("second pull");

    let peer = master
        .install
        .db()
        .sync_peer(&servant.device_id)
        .unwrap()
        .unwrap();
    assert_eq!(peer.last_push_rev, batch.next_rev);
    assert_eq!(
        peer.last_seen_at,
        Some(clock.now()),
        "a served request is contact"
    );
}

#[test]
fn a_push_advances_the_masters_pull_watermark_for_that_peer() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(36));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(37));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    servant.db().create_collection("A", "#111111", None).unwrap();
    let batch = servant.db().collect_changes(0, 500).unwrap();
    client.push(&outcome.key, &batch).expect("push");

    let peer = master
        .install
        .db()
        .sync_peer(&servant.device_id)
        .unwrap()
        .unwrap();
    assert_eq!(peer.last_pull_rev, batch.next_rev);
}

#[test]
fn concurrent_renames_converge_on_the_same_winner() {
    // The merge engine's property, exercised across the socket rather than
    // in-process: whichever stamp is higher wins, and both sides agree.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(38));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(39));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    // One collection, replicated to both.
    let master_id = master
        .install
        .db()
        .create_collection("Trip", "#3584e4", None)
        .unwrap();
    let batch = client.pull(&outcome.key, 0, 500, PeerMode::Relay).unwrap();
    maple_sync::merge::apply_and_refresh(&servant.db(), &batch, &master.install.device_id).unwrap();
    let servant_id = servant.db().all_collections().unwrap()[0].id;

    // Both rename it, without talking first.
    master
        .install
        .db()
        .rename_collection(master_id, "Iceland")
        .unwrap();
    servant.db().rename_collection(servant_id, "Norway").unwrap();

    // Full exchange in both directions.
    let ours = servant.db().collect_changes(0, 500).unwrap();
    client.push(&outcome.key, &ours).unwrap();
    let theirs = client.pull(&outcome.key, batch.next_rev, 500, PeerMode::Relay).unwrap();
    maple_sync::merge::apply_and_refresh(&servant.db(), &theirs, &master.install.device_id).unwrap();

    let master_name = master.install.db().all_collections().unwrap()[0].name.clone();
    let servant_name = servant.db().all_collections().unwrap()[0].name.clone();
    assert_eq!(
        master_name, servant_name,
        "both devices must pick the same winner"
    );
    assert!(
        master_name == "Iceland" || master_name == "Norway",
        "the winner must be one of the two edits, got {master_name}"
    );
}

#[test]
fn a_pairing_error_maps_to_the_error_code_the_client_sees() {
    // Guards the seam between `PairError` and the wire vocabulary: a change
    // to one without the other would leave the modal unable to tell a wrong
    // code from a dead window.
    assert_eq!(ErrorCode::from(PairError::TooEarly), ErrorCode::TooEarly);
    assert_eq!(ErrorCode::from(PairError::BadProof), ErrorCode::BadCode);
    assert_eq!(ErrorCode::from(PairError::Expired), ErrorCode::PairingClosed);
    assert_eq!(ErrorCode::from(PairError::Aborted), ErrorCode::PairingClosed);
}

// ── The worker loop ─────────────────────────────────────────────

/// Poll `f` until it holds or the deadline passes.
///
/// The worker runs on its own thread with a real `recv_timeout`, so these
/// tests genuinely have to wait for it. Polling with a deadline rather than
/// sleeping a fixed amount keeps them fast when things work and gives a
/// useful failure — "never became true in 10s" — when they do not.
fn wait_until(what: &str, mut f: impl FnMut() -> bool) {
    let deadline = std::time::Instant::now() + Duration::from_secs(10);
    while std::time::Instant::now() < deadline {
        if f() {
            return;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    panic!("timed out waiting for: {what}");
}

fn spawn_worker(
    master: &Master,
    servant: &Install,
    clock: &TestClock,
    status: maple_sync::StatusCell,
    on_change: Arc<dyn Fn() + Send + Sync>,
) -> maple_sync::SyncWorker {
    maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: master.address(),
            master_device_id: master.install.device_id.clone(),
            // Far shorter than production's 300 s: these tests wait on real
            // wall time, and the interval is the only part of the worker that
            // deliberately is not on the injected clock — `recv_timeout` is
            // both the sleep and the stop check.
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status,
            clock: clock.handle(),
            rng: seeded(90),
            on_change,
            discovery: None,
            on_relocate: Arc::new(|_| {}),
        },
    )
}

/// A device list a test writes by hand, standing in for mDNS.
///
/// Discovery is the one part of P8 that has to touch a socket, and it is
/// deliberately thin for exactly this reason: everything downstream of it —
/// the worker's relocation, the pairing modal's pick-list — takes a
/// [`DeviceSource`] and is driven from data.
struct FakeDiscovery(Mutex<Vec<DiscoveredDevice>>);

impl FakeDiscovery {
    fn holding(device_id: &str, address: &str) -> Arc<Self> {
        Self::holding_all(device_id, &[address])
    }

    fn holding_all(device_id: &str, addresses: &[&str]) -> Arc<Self> {
        Arc::new(Self(Mutex::new(vec![DiscoveredDevice {
            device_id: device_id.to_owned(),
            name: "Workstation".into(),
            protocol: maple_sync::PROTOCOL_VERSION,
            addresses: addresses.iter().map(|a| (*a).to_owned()).collect(),
        }])))
    }
}

impl DeviceSource for FakeDiscovery {
    fn devices(&self) -> Vec<DiscoveredDevice> {
        self.0.lock().expect("not poisoned").clone()
    }
}

/// The key a pairing left behind.
///
/// Read into a local before the store is written again: `Install::trust`
/// takes the same mutex each time, so building an argument out of a second
/// call to it deadlocks against the first.
fn paired_key(install: &Install, peer: &str) -> maple_sync::PeerKey {
    install.trust().peer(peer).expect("paired").key.clone()
}

/// Where the worker is currently dialling, as its relocate hook reported it.
fn relocation_spy() -> (Arc<Mutex<Vec<String>>>, maple_sync::worker::RelocateHook) {
    let seen: Arc<Mutex<Vec<String>>> = Arc::default();
    let hook = {
        let seen = seen.clone();
        Arc::new(move |address: &str| {
            seen.lock().expect("not poisoned").push(address.to_owned());
        }) as maple_sync::worker::RelocateHook
    };
    (seen, hook)
}

#[test]
fn a_master_that_moved_is_found_again_and_the_link_heals() {
    // §1.4's DHCP story, and the reason discovery keeps running after
    // pairing: the address on file is stale, every pass fails against it,
    // and nothing but mDNS can say where the master went.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(120));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(121));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    // Nothing is listening on port 9 (discard), so the stored address is
    // wrong in the way a moved master is wrong: reachable syntax, dead host.
    let stale = "127.0.0.1:9".to_owned();
    let key = paired_key(&servant, &master.install.device_id);
    servant
        .trust()
        .upsert_peer(TrustedPeer {
            device_id: master.install.device_id.clone(),
            key,
            address: Some(stale.clone()),
        })
        .unwrap();

    master
        .install
        .db()
        .create_collection("From the master", "#3584e4", None)
        .unwrap();

    let (relocations, on_relocate) = relocation_spy();
    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: stale.clone(),
            master_device_id: master.install.device_id.clone(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status,
            clock: clock.handle(),
            rng: seeded(122),
            on_change: Arc::new(|| {}),
            discovery: Some(FakeDiscovery::holding(
                &master.install.device_id,
                &master.address(),
            )),
            on_relocate,
        },
    );

    wait_until("the servant to sync at the address it discovered", || {
        collection_names(&servant) == vec!["From the master".to_owned()]
    });
    worker.stop();

    assert_eq!(
        relocations.lock().unwrap().as_slice(),
        &[master.address()],
        "the blob client has to follow the master, or a relay servant syncs \
         while every tile on screen stays blank"
    );
    let recorded = servant
        .trust()
        .peer(&master.install.device_id)
        .expect("paired")
        .address
        .clone();
    assert_eq!(
        recorded,
        Some(master.address()),
        "the new address is written down, so the next launch starts there"
    );
}

#[test]
fn a_master_this_device_has_never_dialled_is_found_from_nothing() {
    // The case only discovery can serve: these two paired in the other
    // direction, so this side holds a key and a peer row but has never had
    // an address at all. Before P8 that was a servant that could not start.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(123));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(124));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();
    let key = paired_key(&servant, &master.install.device_id);
    servant
        .trust()
        .upsert_peer(TrustedPeer {
            device_id: master.install.device_id.clone(),
            key,
            address: None,
        })
        .unwrap();

    master
        .install
        .db()
        .create_collection("From the master", "#3584e4", None)
        .unwrap();

    let (relocations, on_relocate) = relocation_spy();
    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: String::new(),
            master_device_id: master.install.device_id.clone(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status,
            clock: clock.handle(),
            rng: seeded(125),
            on_change: Arc::new(|| {}),
            discovery: Some(FakeDiscovery::holding(
                &master.install.device_id,
                &master.address(),
            )),
            on_relocate,
        },
    );

    wait_until("a servant with no address to find its master", || {
        collection_names(&servant) == vec!["From the master".to_owned()]
    });
    worker.stop();
    assert_eq!(relocations.lock().unwrap().as_slice(), &[master.address()]);
}

#[test]
fn a_master_whose_first_address_is_unreachable_is_reached_at_its_second() {
    // A multi-homed master publishes every address it has, and this crate
    // cannot tell which one a given servant can reach — a Docker bridge on
    // `10.x` even sorts ahead of the `192.168.x` that works. Dialling only
    // the first would make one wrong guess permanent.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(130));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(131));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .db()
        .create_collection("From the master", "#3584e4", None)
        .unwrap();

    let (relocations, on_relocate) = relocation_spy();
    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: "127.0.0.1:9".into(),
            master_device_id: master.install.device_id.clone(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status,
            clock: clock.handle(),
            rng: seeded(132),
            on_change: Arc::new(|| {}),
            // Port 8 is discard, port 9 is discard: neither answers. The
            // real master is last in the list, so only rotation finds it.
            discovery: Some(FakeDiscovery::holding_all(
                &master.install.device_id,
                &["127.0.0.1:8", &master.address()],
            )),
            on_relocate,
        },
    );

    wait_until("the servant to work down the list to a live address", || {
        collection_names(&servant) == vec!["From the master".to_owned()]
    });
    worker.stop();

    let seen = relocations.lock().unwrap().clone();
    assert_eq!(
        seen.first().map(String::as_str),
        Some("127.0.0.1:8"),
        "the first candidate is tried first: {seen:?}"
    );
    assert!(
        seen.contains(&master.address()),
        "and the ring reaches the one that answers: {seen:?}"
    );
}

#[test]
fn a_worker_with_no_discovery_keeps_dialling_what_it_was_given() {
    // The fallback §2.4 promises: on a network where multicast is blocked,
    // `discovery: None` behaves exactly as P7 did — retry the stored
    // address, forever, and never invent a different one.
    let clock = TestClock::new();
    let servant = Install::new("Laptop");
    servant
        .db()
        .upsert_sync_peer("dev-ghost", Some("Workstation"), PeerMode::Relay)
        .unwrap();
    servant
        .trust()
        .upsert_peer(TrustedPeer {
            device_id: "dev-ghost".into(),
            key: maple_sync::PeerKey::from_bytes([9u8; 32]),
            address: Some("127.0.0.1:9".into()),
        })
        .unwrap();

    let (relocations, on_relocate) = relocation_spy();
    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: "127.0.0.1:9".into(),
            master_device_id: "dev-ghost".into(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status: status.clone(),
            clock: clock.handle(),
            rng: seeded(126),
            on_change: Arc::new(|| {}),
            discovery: None,
            on_relocate,
        },
    );

    wait_until("the pill to report a retry", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Offline { .. }
        )
    });
    worker.stop();

    assert!(relocations.lock().unwrap().is_empty());
    let recorded = servant.trust().peer("dev-ghost").expect("paired").address.clone();
    assert_eq!(
        recorded.as_deref(),
        Some("127.0.0.1:9"),
        "a failing link must not have its address rewritten"
    );
}

fn collection_names(install: &Install) -> Vec<String> {
    let mut names: Vec<String> = install
        .db()
        .all_collections()
        .expect("list")
        .into_iter()
        .map(|c| c.name)
        .collect();
    names.sort();
    names
}

#[test]
fn the_worker_carries_changes_in_both_directions() {
    // The end-to-end claim for P5: two libraries, one link, nobody pressing
    // a button.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(40));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(41));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .db()
        .create_collection("From the master", "#3584e4", None)
        .unwrap();
    servant
        .db()
        .create_collection("From the servant", "#e74c3c", None)
        .unwrap();

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let changes = Arc::new(AtomicI64::new(0));
    let worker = spawn_worker(&master, &servant, &clock, status.clone(), {
        let changes = changes.clone();
        Arc::new(move || {
            changes.fetch_add(1, Ordering::Relaxed);
        })
    });

    let expected = vec!["From the master".to_owned(), "From the servant".to_owned()];
    wait_until("both libraries to hold both collections", || {
        collection_names(&servant) == expected && collection_names(&master.install) == expected
    });

    // And the pill has something true to show.
    wait_until("the status to settle on idle", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Idle
        )
    });
    {
        let snapshot = status.lock().unwrap();
        assert_eq!(snapshot.peers_online, 1);
        assert_eq!(snapshot.pending, 0, "nothing left to send");
        assert!(snapshot.last_sync_ms.is_some());
        assert!(snapshot.last_error.is_none());
        assert_eq!(snapshot.display().label, "Synced");
    }
    assert!(
        changes.load(Ordering::Relaxed) > 0,
        "the UI must be told the library changed under it"
    );

    worker.stop();
}

#[test]
fn the_worker_keeps_its_watermarks_so_a_second_pass_is_quiet() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(42));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(43));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();
    master
        .install
        .db()
        .create_collection("Once", "#3584e4", None)
        .unwrap();

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let changes = Arc::new(AtomicI64::new(0));
    let worker = spawn_worker(&master, &servant, &clock, status, {
        let changes = changes.clone();
        Arc::new(move || {
            changes.fetch_add(1, Ordering::Relaxed);
        })
    });

    // Wait for the *callback*, not just the row: a pass applies the batch and
    // then reports it, and a snapshot taken between the two would count the
    // delivering pass's own report as if an idle pass had made it.
    wait_until("the collection to arrive and be reported", || {
        collection_names(&servant) == vec!["Once".to_owned()]
            && changes.load(Ordering::Relaxed) > 0
    });
    let after_first = changes.load(Ordering::Relaxed);

    // Several more passes go by with nothing to do.
    std::thread::sleep(Duration::from_millis(400));
    worker.stop();

    assert_eq!(
        changes.load(Ordering::Relaxed),
        after_first,
        "an idle pass must not report a change — the watermark held"
    );
    assert_eq!(
        collection_names(&servant),
        vec!["Once".to_owned()],
        "and nothing was applied twice"
    );
}

#[test]
fn a_worker_whose_key_was_revoked_stops_instead_of_retrying_forever() {
    // §1.4's exception, end to end. The pill must reach `Re-pair required`,
    // and the thread must actually stop rather than keep a dead credential
    // warm on a 60-second cycle.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(44));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(45));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .trust()
        .remove_peer(&servant.device_id)
        .unwrap();

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = spawn_worker(&master, &servant, &clock, status.clone(), Arc::new(|| {}));

    wait_until("the pill to demand a re-pair", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Unauthorized
        )
    });
    {
        let snapshot = status.lock().unwrap();
        assert_eq!(snapshot.display().label, "Re-pair required");
        assert_eq!(snapshot.peers_online, 0);
        assert!(
            snapshot.tooltip(clock.now()).contains("unauthorized"),
            "the reason belongs on hover: {}",
            snapshot.tooltip(clock.now())
        );
    }
    // `stop` joins the thread; it has already ended on its own.
    worker.stop();
}

#[test]
fn a_worker_with_no_master_listening_shows_a_retry_countdown() {
    let clock = TestClock::new();
    let address = {
        let master = Master::start(&clock, seeded(46));
        let address = master.address();
        master.server.shutdown();
        address
    };
    let servant = Install::new("Laptop");
    servant
        .db()
        .upsert_sync_peer("dev-ghost", Some("Workstation"), PeerMode::Relay)
        .unwrap();
    servant
        .trust()
        .upsert_peer(TrustedPeer {
            device_id: "dev-ghost".into(),
            key: maple_sync::PeerKey::from_bytes([3u8; 32]),
            address: Some(address.clone()),
        })
        .unwrap();

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address,
            master_device_id: "dev-ghost".into(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status: status.clone(),
            clock: clock.handle(),
            rng: seeded(47),
            on_change: Arc::new(|| {}),
            discovery: None,
            on_relocate: Arc::new(|_| {}),
        },
    );

    wait_until("the pill to report a retry", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Offline { .. }
        )
    });
    {
        let snapshot = status.lock().unwrap();
        // Red with a countdown, not a hang and not `Re-pair required`: the
        // master being off is a network fact, and it heals when it comes back.
        assert!(
            snapshot.display().label.starts_with("Offline · retry "),
            "{}",
            snapshot.display().label
        );
        assert!(snapshot.last_error.is_some());
    }
    worker.stop();
}

#[test]
fn a_master_that_comes_back_refreshes_the_ui_with_nothing_to_merge() {
    // The relay's own failure mode, and it is not a merge bug: a servant that
    // starts while the master is down fills its grid with thumbnails that
    // fail to fetch, and nothing ever retries one. The pill going green is
    // not enough — the tiles stay blank until something reloads them. So the
    // first pass after an outage refreshes the UI even though both libraries
    // are empty and it merged nothing at all.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(70));
    let address = master.address();
    let servant = Install::new("Laptop");

    // Trust written by hand rather than through `pair`, so nothing has
    // connected to the listener yet: the address below is rebound after a
    // shutdown, and an accepted connection would leave it in TIME_WAIT.
    let key = maple_sync::PeerKey::from_bytes([7u8; 32]);
    for (install, peer, addr) in [
        (&master.install, &servant.device_id, None),
        (&servant, &master.install.device_id, Some(address.clone())),
    ] {
        install
            .db()
            .upsert_sync_peer(peer, Some("Peer"), PeerMode::Relay)
            .unwrap();
        install
            .trust()
            .upsert_peer(TrustedPeer {
                device_id: peer.clone(),
                key: key.clone(),
                address: addr,
            })
            .unwrap();
    }

    let Master { install, slot, server, status: master_status, thumbs, .. } = master;
    server.shutdown();

    let changes = Arc::new(AtomicI64::new(0));
    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = maple_sync::worker::spawn(
        maple_sync::WorkerConfig {
            address: address.clone(),
            master_device_id: install.device_id.clone(),
            interval: Duration::from_millis(50),
            max_revs: 500,
            layout: servant.layout(),
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status: status.clone(),
            clock: clock.handle(),
            rng: seeded(71),
            on_change: {
                let changes = changes.clone();
                Arc::new(move || {
                    changes.fetch_add(1, Ordering::Relaxed);
                })
            },
            discovery: None,
            on_relocate: Arc::new(|_| {}),
        },
    );

    wait_until("the pill to report a retry", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Offline { .. }
        )
    });
    assert_eq!(
        changes.load(Ordering::Relaxed),
        0,
        "a failed pass has nothing to show"
    );

    // The master comes back on the same address, exactly as a machine that
    // was asleep does.
    let revived = SyncServer::spawn(
        maple_sync::server::ServerConfig {
            listen_addr: address,
            max_revs: 500,
            ..Default::default()
        },
        maple_sync::server::ServerDeps {
            db: install.db.clone(),
            trust: install.trust.clone(),
            pairing: slot,
            status: master_status,
            clock: clock.handle(),
            rng: seeded(72),
            thumbs,
            render_thumb: Arc::new(|_: &std::path::Path| Ok(Vec::new())),
            layout: install.layout(),
            on_change: Arc::new(|| {}),
        },
    )
    .expect("rebind the address the master just released");

    wait_until("the recovered pass to refresh the UI", || {
        changes.load(Ordering::Relaxed) > 0
    });
    worker.stop();
    revived.shutdown();
}

#[test]
fn a_servant_refuses_to_sync_to_another_servant() {
    // A star is the only topology the merge engine is built for. Reachable
    // but wrongly configured is retryable — the user may just not have
    // switched the other machine over yet.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(48));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(49));
    pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master
        .install
        .db()
        .set_sync_role(maple_state::SyncRole::Servant)
        .unwrap();

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = spawn_worker(&master, &servant, &clock, status.clone(), Arc::new(|| {}));

    wait_until("the worker to back off", || {
        matches!(
            status.lock().unwrap().state,
            maple_sync::status::SyncState::Offline { .. }
        )
    });
    let message = status.lock().unwrap().last_error.clone().unwrap_or_default();
    assert!(message.contains("not a master"), "{message}");
    worker.stop();
}

#[test]
fn the_masters_pill_counts_servants_that_actually_called() {
    // A master is passive and has no worker, so its listener writes the
    // status cell. The number therefore means "peers that made a signed
    // request recently", not "rows in sync_peers" — a servant that paired
    // last week and has been off since must not read as connected.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(60));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(61));

    wait_until("the master to settle on listening", || {
        master.status.lock().unwrap().display().label == "Listening · no devices"
    });

    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();
    // Pairing alone is not contact — nothing has been synced yet.
    assert_eq!(
        master.status.lock().unwrap().display().label,
        "Listening · no devices"
    );

    client.pull(&outcome.key, 0, 500, PeerMode::Relay).expect("pull");
    wait_until("the master to notice the servant", || {
        master.status.lock().unwrap().display().label == "1 device"
    });

    // And it drops back out of the window once the servant goes quiet for
    // longer than the grace period.
    clock.advance(11 * 60 * 1000);
    wait_until("the master to forget a silent servant", || {
        master.status.lock().unwrap().display().label == "Listening · no devices"
    });
}

// ── Blobs (P6: relay) ───────────────────────────────────────────

#[test]
fn a_signed_thumb_fetch_renders_on_the_master_and_caches_there() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(70));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(71));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let hash = master_photo(&master, "a.jpg", b"twelve bytes");

    let webp = client.blob_thumb(&outcome.key, &hash).expect("thumb");
    assert_eq!(webp, b"THUMB:12", "the renderer's bytes are what came back");
    assert_eq!(master.renders.load(Ordering::Relaxed), 1);
    assert_eq!(
        master.thumbs.get(&hash).as_deref(),
        Some(&b"THUMB:12"[..]),
        "the master keeps what it rendered"
    );

    // A servant scrolling past the same photo again must not make the master
    // re-decode it — that is the whole reason the cache is written above.
    let again = client.blob_thumb(&outcome.key, &hash).expect("thumb again");
    assert_eq!(again, webp);
    assert_eq!(master.renders.load(Ordering::Relaxed), 1, "served from cache");
}

#[test]
fn a_signed_orig_fetch_returns_the_files_bytes_verbatim() {
    // Verbatim matters beyond P6: P7 will verify the BLAKE3 of what it
    // downloads before writing it into a library.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(72));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(73));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let contents: Vec<u8> = (0..=255u8).cycle().take(200_000).collect();
    let hash = master_photo(&master, "big.jpg", &contents);

    let bytes = client.blob_orig(&outcome.key, &hash, false).expect("orig");
    assert_eq!(bytes.len(), contents.len());
    assert_eq!(blake3::hash(&bytes), blake3::hash(&contents));
    assert_eq!(master.renders.load(Ordering::Relaxed), 0, "no decoding involved");
}

#[test]
fn an_unknown_hash_is_a_404_that_does_not_break_the_pairing() {
    // A photo's hash changes when it is losslessly rotated, so a servant can
    // legitimately ask for one the master no longer has. Costing the link
    // over that would strand a working pairing behind a re-pair that fixes
    // nothing.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(74));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(75));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let failure = client
        .blob_thumb(&outcome.key, &[0x11; 32])
        .expect_err("nothing has that hash");
    assert_eq!(failure.code, Some(ErrorCode::NotFound));
    assert_eq!(failure.kind, FailureKind::Unreachable, "retryable, not fatal");

    // The link still works afterwards.
    let hash = master_photo(&master, "a.jpg", b"ok");
    assert!(client.blob_thumb(&outcome.key, &hash).is_ok());
}

#[test]
fn a_blob_request_is_refused_exactly_as_a_pull_is() {
    // The library must not be readable one photo at a time by anything on the
    // LAN that can guess a hash.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(76));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(77));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();
    let hash = master_photo(&master, "a.jpg", b"secret");
    let path = format!("/blob/thumb/{}", hex(&hash));

    let (status, body) = raw_get(&master.address(), &path, None);
    assert_eq!(status, 400, "no Authorization header at all");
    assert!(String::from_utf8_lossy(&body).contains("malformed"));

    // A well-formed credential from a device with no key: same answer a bad
    // MAC gets, so probing tells an attacker nothing.
    let forged = SignedRequest::sign_with(
        &outcome.key,
        "dev-nobody".to_owned(),
        "GET",
        &path,
        &[],
        clock.now(),
        &seeded(78),
    )
    .expect("sign");
    let (status, body) = raw_get(&master.address(), &path, Some(&forged.header()));
    assert_eq!(status, 401);
    assert!(String::from_utf8_lossy(&body).contains("unauthorized"));

    // And a signature over a *different* path does not transfer.
    let wrong_path = SignedRequest::sign_with(
        &outcome.key,
        servant.device_id.clone(),
        "GET",
        route::PULL,
        &[],
        clock.now(),
        &seeded(79),
    )
    .expect("sign");
    let (status, _) = raw_get(&master.address(), &path, Some(&wrong_path.header()));
    assert_eq!(status, 401, "the MAC covers the path");
}

#[test]
fn a_malformed_hash_is_a_400_not_a_404() {
    // 404 would have the servant retry a URL that can never work.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(80));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(81));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let path = "/blob/thumb/abcd";
    let credential = SignedRequest::sign_with(
        &outcome.key,
        servant.device_id.clone(),
        "GET",
        path,
        &[],
        clock.now(),
        &seeded(82),
    )
    .expect("sign");
    let (status, body) = raw_get(&master.address(), path, Some(&credential.header()));
    assert_eq!(status, 400);
    assert!(String::from_utf8_lossy(&body).contains("bad_request"));
}

#[test]
fn a_relay_servant_browses_the_masters_library_without_storing_a_file() {
    // The P6 acceptance test, end to end: metadata over the link, pixels over
    // the blob route, and nothing on the servant's disk when it is done.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(84));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(85));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let contents = b"a photo the servant will never own";
    let hash = master_photo(&master, "holiday.jpg", contents);

    // 1. The row arrives, and lands as a real library entry.
    let batch = client.pull(&outcome.key, 0, 500, PeerMode::Relay).expect("pull");
    let report =
        maple_sync::merge::apply_and_refresh(&servant.db(), &batch, &master.install.device_id)
            .expect("apply");
    assert_eq!(report.inserted, 1);

    // Read it back the way the grid does — a listing, not a hand-written
    // query — so this also pins that a relayed photo is *listed* at all.
    let listed = servant
        .db()
        .search_images(&maple_db::SearchQuery::default())
        .expect("list");
    assert_eq!(listed.len(), 1, "it belongs in the grid");
    let row = &listed[0];
    assert_eq!(row.status, maple_db::ImageStatus::Present);
    assert_eq!(row.locality, maple_db::Locality::Remote);
    assert_eq!(row.origin_device.as_deref(), Some(master.install.device_id.as_str()));
    assert_eq!(row.hash, Some(hash), "the blob key travelled with it");

    // 2. Both pixel seams work over the wire.
    let thumb = client.blob_thumb(&outcome.key, &hash).expect("thumbnail");
    assert_eq!(thumb, format!("THUMB:{}", contents.len()).into_bytes());
    let full = client.blob_orig(&outcome.key, &hash, false).expect("original");
    assert_eq!(full, contents);

    // 3. And the servant's own directory is still empty of photos. This is
    //    the relay contract; a servant that cached originals would have
    //    stopped being one.
    let strays: Vec<String> = std::fs::read_dir(servant._dir.path())
        .expect("read servant dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| name.ends_with(".jpg"))
        .collect();
    assert!(strays.is_empty(), "servant wrote originals: {strays:?}");
}

fn hex(hash: &[u8; 32]) -> String {
    hash.iter().map(|b| format!("{b:02x}")).collect()
}

// ── Originals (P7: full and partial) ────────────────────────────

/// One full metadata exchange, both directions, as a pass does it.
///
/// P7's transfers are all downstream of this: a photo can only be downloaded
/// once the row naming it exists here, and can only be uploaded once the
/// master has a row waiting for it.
fn sync_metadata(
    master: &Master,
    servant: &Install,
    client: &SyncClient,
    key: &maple_sync::PeerKey,
) {
    let batch = client.pull(key, 0, 500, PeerMode::Relay).expect("pull");
    maple_sync::merge::apply_and_refresh(&servant.db(), &batch, &master.install.device_id)
        .expect("apply");
    let ours = servant.db().collect_changes(0, 500).expect("collect");
    client.push(key, &ours).expect("push");
}

/// Run the file half of a pass, with nothing asking it to stop.
fn move_files(
    servant: &Install,
    client: &SyncClient,
    key: &maple_sync::PeerKey,
    mode: PeerMode,
) -> maple_sync::TransferOutcome {
    maple_sync::transfer::transfer(
        &servant.db,
        client,
        key,
        mode,
        &servant.layout(),
        &|| false,
        &|_, _| {},
    )
    .expect("transfer")
}

/// Photo files (not staging, not the database) directly inside a library.
fn photos_in(dir: &std::path::Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .expect("read library dir")
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    names
}

#[test]
fn a_full_servant_ends_up_holding_the_masters_photos() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(86));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(87));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let contents = b"the master's only photograph";
    let path = master.install.import("holiday.jpg", contents);
    let hash = maple_import::content_hash(&path).unwrap();
    sync_metadata(&master, &servant, &client, &outcome.key);
    assert_eq!(
        servant.db().originals_to_fetch(10).unwrap().len(),
        1,
        "P6 leaves it relayed; P7 is what fetches it"
    );

    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Full);
    assert_eq!(moved.downloaded, 1);
    assert_eq!(moved.skipped, 0);

    // The bytes are here, under this device's own path template.
    assert_eq!(photos_in(&servant.library_dir()), vec!["holiday.jpg"]);
    let landed = servant.library_dir().join("holiday.jpg");
    assert_eq!(std::fs::read(&landed).unwrap(), contents);

    // And the row now says so, which is what stops the grid fetching pixels
    // over the wire for a file that is sitting on this disk.
    assert!(servant.db().originals_to_fetch(10).unwrap().is_empty());
    assert_eq!(servant.db().holds_original(&hash).unwrap(), Some(landed));
    let listed = servant
        .db()
        .search_images(&maple_db::SearchQuery::default())
        .unwrap();
    assert_eq!(listed[0].locality, maple_db::Locality::Local);

    // A second pass has nothing left to do — the queue is what makes it
    // idempotent, and re-downloading every photo every five minutes would be
    // the worst possible bug to ship here.
    let again = move_files(&servant, &client, &outcome.key, PeerMode::Full);
    assert_eq!(again.downloaded, 0);
    assert_eq!(photos_in(&servant.library_dir()), vec!["holiday.jpg"]);
}

#[test]
fn a_partial_servant_pushes_its_photos_and_leaves_the_masters_alone() {
    // The mode's whole definition: the servant's originals end up on the
    // master, and master-only photos stay relayed on the servant.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(88));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(89));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let theirs = b"a photo only the master has";
    master.install.import("master.jpg", theirs);
    let mine = b"a photo taken on the laptop";
    let mine_path = servant.import("laptop.jpg", mine);
    let mine_hash = maple_import::content_hash(&mine_path).unwrap();
    sync_metadata(&master, &servant, &client, &outcome.key);

    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Partial);
    assert_eq!(moved.uploaded, 1);
    assert_eq!(moved.downloaded, 0, "partial does not pull originals down");

    // Up: the master holds it, filed under the *master's* template.
    assert!(master.install.library_dir().join("laptop.jpg").exists());
    assert_eq!(
        std::fs::read(master.install.library_dir().join("laptop.jpg")).unwrap(),
        mine
    );
    assert!(master.install.db().holds_original(&mine_hash).unwrap().is_some());
    assert!(master.install.db().originals_to_fetch(10).unwrap().is_empty());

    // Down: nothing. The master's photo is still one the servant relays.
    assert_eq!(photos_in(&servant.library_dir()), vec!["laptop.jpg"]);
    let queued = servant.db().originals_to_fetch(10).unwrap();
    assert_eq!(queued.len(), 1);
    assert_eq!(queued[0].filename, "master.jpg");
}

#[test]
fn a_relay_servant_moves_nothing_in_either_direction() {
    // P6's contract, now that there is machinery that could break it: relay
    // means no originals cross, not "originals cross more slowly".
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(90));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(91));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    master.install.import("master.jpg", b"master bytes");
    servant.import("laptop.jpg", b"laptop bytes");
    sync_metadata(&master, &servant, &client, &outcome.key);

    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Relay);
    assert_eq!(moved, maple_sync::TransferOutcome::default());
    assert_eq!(photos_in(&servant.library_dir()), vec!["laptop.jpg"]);
    assert_eq!(photos_in(&master.install.library_dir()), vec!["master.jpg"]);
    assert_eq!(servant.db().originals_to_fetch(10).unwrap().len(), 1);
    assert_eq!(master.install.db().originals_to_fetch(10).unwrap().len(), 1);
}

#[test]
fn a_companion_raw_travels_with_the_photo_it_belongs_to() {
    // The Fujifilm case: a JPEG and its RAF are one library row, and a full
    // sync that moved only the JPEG would silently drop the negative.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(92));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(93));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let jpeg = b"the embedded preview";
    let negative = b"the raw negative, much larger in real life";
    let display = servant.library_dir().join("DSCF0001.JPG");
    let raw = servant.library_dir().join("DSCF0001.RAF");
    std::fs::write(&display, jpeg).unwrap();
    std::fs::write(&raw, negative).unwrap();
    let hash = maple_import::content_hash(&display).unwrap();
    servant
        .db()
        .insert_image_with_raw(&display, &hash, jpeg.len() as u64, Some(&raw))
        .unwrap();

    sync_metadata(&master, &servant, &client, &outcome.key);
    // The master learned there *is* a companion — advisory, from the wire —
    // which is the only reason it will accept one.
    let queued = master.install.db().originals_to_fetch(10).unwrap();
    assert_eq!(queued[0].raw_filename.as_deref(), Some("DSCF0001.RAF"));

    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Partial);
    assert_eq!(moved.uploaded, 1, "one photo, even though two files crossed");

    assert_eq!(
        photos_in(&master.install.library_dir()),
        vec!["DSCF0001.JPG", "DSCF0001.RAF"]
    );
    assert_eq!(
        std::fs::read(master.install.library_dir().join("DSCF0001.RAF")).unwrap(),
        negative
    );
    // And the row points at the master's own copy of both.
    let raw_path = master
        .install
        .db()
        .blob_path(&hash, true)
        .unwrap()
        .expect("the master knows where its companion is");
    assert!(raw_path.starts_with(master.install.library_dir()), "{raw_path:?}");
    assert!(
        master.install.db().originals_to_fetch(10).unwrap().is_empty(),
        "nothing is still waiting"
    );
    assert!(
        master.changes.load(Ordering::Relaxed) > 0,
        "the master's own grid has to be told a photo landed — it polls nothing"
    );
}

/// Every photo file under a library, relative to it and `/`-separated, so an
/// assertion can name a folder the template built.
fn photo_paths_in(dir: &std::path::Path) -> Vec<String> {
    fn walk(dir: &std::path::Path, root: &std::path::Path, out: &mut Vec<String>) {
        let Ok(entries) = std::fs::read_dir(dir) else {
            return;
        };
        for entry in entries.filter_map(|e| e.ok()) {
            let name = entry.file_name().to_string_lossy().into_owned();
            if name.starts_with('.') {
                continue; // `.incoming`, and anything else internal
            }
            let path = entry.path();
            if path.is_dir() {
                walk(&path, root, out);
            } else {
                let relative = path.strip_prefix(root).unwrap_or(&path);
                out.push(
                    relative
                        .components()
                        .map(|c| c.as_os_str().to_string_lossy().into_owned())
                        .collect::<Vec<_>>()
                        .join("/"),
                );
            }
        }
    }
    let mut out = Vec::new();
    walk(dir, dir, &mut out);
    out.sort();
    out
}

/// The smallest JPEG carrying a `DateTimeOriginal`, so a folder template has
/// a real capture date to render — which is the whole point of the test
/// below. Mirrors `maple_import::copy`'s own fixture.
fn jpeg_taken_on(stamp: &str) -> Vec<u8> {
    let mut date = stamp.as_bytes().to_vec();
    date.push(0);
    let entry = |tag: u16, kind: u16, count: u32, value: u32| {
        let mut e = tag.to_le_bytes().to_vec();
        e.extend_from_slice(&kind.to_le_bytes());
        e.extend_from_slice(&count.to_le_bytes());
        e.extend_from_slice(&value.to_le_bytes());
        e
    };
    let ifd = |e: Vec<u8>| {
        let mut ifd = 1u16.to_le_bytes().to_vec();
        ifd.extend_from_slice(&e);
        ifd.extend_from_slice(&0u32.to_le_bytes());
        ifd
    };
    let mut tiff = b"II".to_vec();
    tiff.extend_from_slice(&42u16.to_le_bytes());
    tiff.extend_from_slice(&8u32.to_le_bytes());
    tiff.extend_from_slice(&ifd(entry(0x8769, 4, 1, 26)));
    tiff.extend_from_slice(&ifd(entry(0x9003, 2, date.len() as u32, 44)));
    tiff.extend_from_slice(&date);
    let mut app1 = b"Exif\0\0".to_vec();
    app1.extend_from_slice(&tiff);
    let mut jpeg = vec![0xFF, 0xD8, 0xFF, 0xE1];
    jpeg.extend_from_slice(&((app1.len() + 2) as u16).to_be_bytes());
    jpeg.extend_from_slice(&app1);
    jpeg.extend_from_slice(&[0xFF, 0xD9]);
    jpeg
}

/// The bug the flat-destination tests could not see.
///
/// With a real `{YYYY}/{MM}` template the two halves of one photo were filed
/// *independently*: the JPEG's own EXIF sent it to the month it was taken,
/// while the RAF — staged under a synthetic `<hash>.raw` name, so nothing
/// could tell it was a raw container — parsed as having no date at all and
/// landed under the month it arrived. The database still linked them; the
/// disk did not. And since the library scanner regroups by directory and
/// stem, the orphaned RAF was a photograph no row claimed, so the next scan
/// inserted a second `images` row for it — which then stamped and replicated
/// back to the servant.
#[test]
fn a_companion_lands_beside_its_photo_under_a_real_template() {
    let clock = TestClock::new();
    let master = Master::start_filing_under(&clock, seeded(120), "{YYYY}/{MM}");
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(121));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    // Taken in 2019; arriving today. A template that reads the two files
    // separately cannot put them in the same folder.
    let display = servant.library_dir().join("DSCF0001.JPG");
    let raw = servant.library_dir().join("DSCF0001.RAF");
    let jpeg = jpeg_taken_on("2019:07:04 10:11:12");
    std::fs::write(&display, &jpeg).unwrap();
    std::fs::write(&raw, b"a raw container this build has no reader for").unwrap();
    let hash = maple_import::content_hash(&display).unwrap();
    servant
        .db()
        .insert_image_with_raw(&display, &hash, jpeg.len() as u64, Some(&raw))
        .unwrap();

    sync_metadata(&master, &servant, &client, &outcome.key);
    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Partial);
    assert_eq!(moved.uploaded, 1);

    let library = master.install.library_dir();
    assert_eq!(
        photo_paths_in(&library),
        vec!["2019/07/DSCF0001.JPG", "2019/07/DSCF0001.RAF"],
        "the companion follows its display file, capture date and all"
    );

    // Said structurally, because that is the property the scanner depends on
    // and the literal paths above are only one instance of it.
    let placed = master.install.db().blob_path(&hash, false).unwrap().unwrap();
    let placed_raw = master.install.db().blob_path(&hash, true).unwrap().unwrap();
    assert_eq!(placed.parent(), placed_raw.parent(), "same directory");
    assert_eq!(placed.file_stem(), placed_raw.file_stem(), "same stem");

    // And so the master's own scanner has nothing to adopt: one photograph on
    // disk, one row, and no ghost minted from the orphaned negative.
    let before = master.install.db().count().unwrap();
    maple_db::LibraryScanner::new(master.install.db.clone(), library.clone(), None).run_scan();
    assert_eq!(
        master.install.db().count().unwrap(),
        before,
        "a scan after a companion transfer must insert nothing"
    );
    assert_eq!(before, 1, "one photo is one row");
}

/// A companion the servant only acquires *after* the photo first replicated
/// still crosses.
///
/// `Database::update_row` used to write every replicated column except
/// `raw_path`, which was set on INSERT and never again — reasonable-looking,
/// since `path` and `filename` beside it really are machine-local. But on a
/// `remote` row `raw_path` is not this machine's anything: it is the origin's
/// path, carried only so this device knows the photo *has* a negative worth
/// asking for. So a master that first heard about a JPEG on its own stayed
/// NULL forever, and `blob_upload` answers a `?raw=1` upload for such a row
/// with BadRequest — the raw was refused on every pass, permanently, with a
/// warning per attempt and nothing else to show for it.
#[test]
fn a_companion_found_after_the_photo_replicated_still_crosses() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(122));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(123));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    // First the JPEG alone — as if the RAF had not been copied off the card
    // yet, or the scanner had not got to it.
    let display = servant.library_dir().join("DSCF0001.JPG");
    std::fs::write(&display, b"the embedded preview").unwrap();
    let hash = maple_import::content_hash(&display).unwrap();
    servant.db().insert_image(&display, &hash, 20).unwrap();
    sync_metadata(&master, &servant, &client, &outcome.key);
    assert_eq!(
        master.install.db().originals_to_fetch(10).unwrap()[0].raw_filename,
        None,
        "the master cannot know about a companion nobody has mentioned"
    );

    // Now the negative turns up beside it and the servant records it — which
    // is exactly what its own 60-second scanner does.
    let raw = servant.library_dir().join("DSCF0001.RAF");
    let negative = b"the raw negative";
    std::fs::write(&raw, negative).unwrap();
    let image_id = servant.db().image_id_for_path(&display).unwrap().unwrap();
    servant.db().set_raw_path(image_id, &raw).expect("record the companion");
    // `set_raw_path` does not stamp, deliberately — `raw_path` is where *this*
    // machine keeps its copy, and every device discovers its own. So the fact
    // travels the way any machine-local column does: carried along by the next
    // edit that *is* replicated. Orientation stands in for that here; a real
    // library gets one from a rotation, an EXIF fill or a metadata edit.
    servant
        .db()
        .update_image_hash_and_orientation(image_id, &hash, 6)
        .expect("some later replicated edit");

    sync_metadata(&master, &servant, &client, &outcome.key);
    assert_eq!(
        master.install.db().originals_to_fetch(10).unwrap()[0].raw_filename.as_deref(),
        Some("DSCF0001.RAF"),
        "an update has to be able to teach the master about a companion"
    );

    let moved = move_files(&servant, &client, &outcome.key, PeerMode::Partial);
    assert_eq!(moved.uploaded, 1);
    assert_eq!(
        photos_in(&master.install.library_dir()),
        vec!["DSCF0001.JPG", "DSCF0001.RAF"]
    );
    assert_eq!(
        std::fs::read(master.install.library_dir().join("DSCF0001.RAF")).unwrap(),
        negative
    );
}

#[test]
fn bytes_that_do_not_hash_to_what_they_were_sent_as_are_refused() {
    // The upload route verifies as it writes, which is the only reason it can
    // stream a photograph without checking a MAC over it first.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(94));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(95));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let path = servant.import("laptop.jpg", b"the real photograph");
    let hash = maple_import::content_hash(&path).unwrap();
    sync_metadata(&master, &servant, &client, &outcome.key);
    assert!(master.install.db().row_wanting(&hash).unwrap().is_some());

    let failure = client
        .upload_orig(&outcome.key, &hash, false, &mut &b"something else entirely"[..])
        .expect_err("the master must not take these");
    assert_eq!(failure.code, Some(ErrorCode::BadRequest));
    assert_eq!(
        failure.kind,
        FailureKind::Unreachable,
        "one bad blob is not a broken pairing"
    );

    // Nothing was filed, and nothing was left behind in staging either.
    assert!(photos_in(&master.install.library_dir()).is_empty());
    assert!(!master
        .install
        .layout()
        .staged_path(&hash, false)
        .exists());
    assert!(
        master.install.db().row_wanting(&hash).unwrap().is_some(),
        "the row is still waiting for the real bytes"
    );
}

#[test]
fn a_master_will_not_take_a_blob_nothing_here_is_waiting_for() {
    // The admission rule that keeps this route from being "write a file into
    // my library": a paired peer can only fill in a row that already exists.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(96));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(97));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let contents = b"a photo the master has never heard of";
    let hash: [u8; 32] = blake3::hash(contents).into();
    let failure = client
        .upload_orig(&outcome.key, &hash, false, &mut &contents[..])
        .expect_err("unsolicited");
    assert_eq!(failure.code, Some(ErrorCode::NotFound));
    assert!(photos_in(&master.install.library_dir()).is_empty());
}

#[test]
fn an_unpaired_machine_cannot_upload_anything() {
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(98));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(99));
    let stranger = maple_sync::PeerKey::from_bytes([9u8; 32]);

    let failure = client
        .upload_orig(&stranger, &[0x11u8; 32], false, &mut &b"bytes"[..])
        .expect_err("no key for this device");
    assert_eq!(failure.code, Some(ErrorCode::Unauthorized));
    assert!(photos_in(&master.install.library_dir()).is_empty());
}

#[test]
fn the_servants_mode_reaches_the_masters_peer_list() {
    // Pairing defaults every peer to relay, because a mode chosen before the
    // user has picked one must not fill a disk. The servant then says what it
    // actually is, on every pull, or the master's card lies forever.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(100));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(101));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();

    let peer = master.install.db().sync_peer(&servant.device_id).unwrap().unwrap();
    assert_eq!(peer.mode, PeerMode::Relay, "the safe default");

    client.pull(&outcome.key, 0, 500, PeerMode::Full).expect("pull");

    let peer = master.install.db().sync_peer(&servant.device_id).unwrap().unwrap();
    assert_eq!(peer.mode, PeerMode::Full);
}

#[test]
fn the_worker_moves_originals_without_being_told_to_twice() {
    // End to end through the loop itself: the mode comes from the servant's
    // own `sync_peers` row, the transfer runs as part of a pass, and the
    // photo lands on disk with nothing driving it by hand.
    let clock = TestClock::new();
    let master = Master::start(&clock, seeded(102));
    let servant = Install::new("Laptop");
    let client = client(&master, &servant, &clock, seeded(103));
    let outcome = pair(&master, &servant, &client, &clock, "482107", "314159").unwrap();
    servant
        .db()
        .set_sync_peer_mode(&outcome.device_id, PeerMode::Full)
        .expect("full mode");

    let contents = b"a photo that should end up on both machines";
    master.install.import("shared.jpg", contents);

    let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Servant);
    let worker = spawn_worker(&master, &servant, &clock, status, Arc::new(|| {}));

    let landed = servant.library_dir().join("shared.jpg");
    wait_until("the worker to bring the photo across", || landed.exists());
    worker.stop();

    assert_eq!(std::fs::read(&landed).unwrap(), contents);
    assert!(servant.db().originals_to_fetch(10).unwrap().is_empty());
}
