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
}

impl Master {
    fn start(clock: &TestClock, rng: SharedRandom) -> Self {
        let install = Install::new("Workstation");
        install
            .db()
            .set_sync_role(maple_state::SyncRole::Master)
            .expect("role");
        let slot = PairingSlot::new();
        let status = maple_sync::SyncStatus::cell(maple_state::SyncRole::Master);
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
            },
        )
        .expect("bind loopback");
        Self {
            install,
            slot,
            server,
            status,
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
        .pull(&stranger, 0, 500)
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

    assert!(client.pull(&outcome.key, 0, 500).is_ok(), "paired link works");

    master
        .install
        .trust()
        .remove_peer(&servant.device_id)
        .expect("unpair");

    let failure = client.pull(&outcome.key, 0, 500).expect_err("key is gone");
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

    let batch = client.pull(&outcome.key, 0, 500).expect("pull");
    assert!(!batch.is_empty(), "the master had a change to send");
    let report = maple_sync::merge::apply_and_refresh(&servant.db(), &batch).expect("apply");
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

    let batch = client.pull(&outcome.key, 0, 500).expect("first pull");
    assert!(batch.next_rev > 0);
    // The servant merges and comes back with the new watermark.
    let _ = client.pull(&outcome.key, batch.next_rev, 500).expect("second pull");

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
    let batch = client.pull(&outcome.key, 0, 500).unwrap();
    maple_sync::merge::apply_and_refresh(&servant.db(), &batch).unwrap();
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
    let theirs = client.pull(&outcome.key, batch.next_rev, 500).unwrap();
    maple_sync::merge::apply_and_refresh(&servant.db(), &theirs).unwrap();

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
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status,
            clock: clock.handle(),
            rng: seeded(90),
            on_change,
        },
    )
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

    wait_until("the collection to arrive", || {
        collection_names(&servant) == vec!["Once".to_owned()]
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
        },
        maple_sync::worker::WorkerDeps {
            db: servant.db.clone(),
            trust: servant.trust.clone(),
            status: status.clone(),
            clock: clock.handle(),
            rng: seeded(47),
            on_change: Arc::new(|| {}),
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

    client.pull(&outcome.key, 0, 500).expect("pull");
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
