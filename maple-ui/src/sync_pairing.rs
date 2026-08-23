//! The pairing modal's state machine.
//!
//! Lives in Rust rather than in `settings.slint` for the usual reason: the
//! interesting behaviour is the *end* of a pairing window, and a deadline
//! that expires and an attempt counter that trips are exactly the things a
//! unit test needs to drive without an event loop.
//!
//! # The two halves are not symmetric
//!
//! §2.1's handshake is mutual, but the machines play different parts. The
//! **master** is passive: its listener answers `/pair/claim`, and the modal
//! here only shows the code and waits for [`PairingSlot`] to report that
//! something verified. The **servant** is active: it has to dial an address
//! and keep asking, because the master will answer `TooEarly` until its user
//! has typed the servant's code — which may be a walk down the hall away.
//!
//! That polling cannot happen on the UI thread. `pair_claim` is a blocking
//! HTTP call, and a modal that froze between attempts would stop its own
//! countdown. So the servant's half runs on a worker thread and reports back
//! through an `mpsc` channel that [`PairingController::tick`] drains — the
//! same shape as every other background job in this crate.
//!
//! # The pick-list is a shortcut, not a decision
//!
//! Since P8 a servant is shown the masters mDNS found (§2.4). Choosing one
//! fills in the address field and nothing else: the handshake that follows
//! is byte-for-byte the one a typed address gets, with the same two codes
//! and the same proof. A record is unauthenticated hearsay — see
//! [`maple_sync::discovery`] — so the list can only save typing, never
//! establish trust. The field stays editable underneath it, because plenty
//! of networks block multicast.
//!
//! # What "abort" has to mean
//!
//! When the three minutes run out or five wrong codes arrive, both codes are
//! discarded. Not greyed out, not left on screen with a disabled button —
//! discarded, because a code still visible after its window closed is a code
//! someone will try to use. The controller closes the shared slot, which
//! takes this device's code and the digits typed for the peer's with it.

use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use maple_sync::discovery::DeviceSource;
use maple_sync::pairing::{PairError, PairOutcome, MAX_ATTEMPTS};
use maple_sync::{
    Clock, Initiator, PairCode, PairingSlot, RandomSource, SharedRandom, SyncClient,
};

/// Per-request timeout while pairing. Short, because the modal retries every
/// second and a stuck request would eat the window.
const CLAIM_TIMEOUT: Duration = Duration::from_secs(8);

/// How long the servant waits between `TooEarly` answers.
const CLAIM_POLL: Duration = Duration::from_secs(1);

/// Most devices the pick-list will show.
///
/// The modal is a fixed-width card in a `VerticalLayout` with no scroll
/// area, so its height is the sum of its rows: on a network with twenty
/// masters an uncapped list would run off the bottom of the screen, taking
/// the Pair button with it. Anything not shown is still reachable by typing
/// its address — the same fallback a blocked-multicast network uses.
const MAX_DEVICES_SHOWN: usize = 6;

/// Which half of the handshake this device is playing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairingSide {
    /// Passive: the listener answers claims. No address needed.
    Master,
    /// Active: dials `address` and polls until the other user types.
    Servant,
}

/// Why a pairing window stopped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PairingEnd {
    /// The user pressed Cancel or closed the modal.
    Cancelled,
    /// The three-minute window elapsed.
    Expired,
    /// Five wrong codes.
    Aborted,
    /// A peer completed the handshake.
    Paired { peer_name: String },
    /// The servant could not reach the master, or it refused for a reason
    /// that is not a wrong code.
    Failed { reason: String },
}

impl PairingEnd {
    /// The line the settings card shows once the modal closes.
    pub fn message(&self) -> String {
        match self {
            Self::Cancelled => "Pairing cancelled.".into(),
            Self::Expired => "Pairing window expired — start again to get a fresh code.".into(),
            Self::Aborted => "Pairing aborted after too many wrong codes.".into(),
            Self::Paired { peer_name } => format!("Paired with {peer_name}."),
            Self::Failed { reason } => format!("Pairing failed: {reason}"),
        }
    }
}

/// One row of the discovered-devices pick-list.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceChoice {
    /// `Studio · 192.168.1.20:7645`, or with a `needs updating` tail.
    pub label: String,
    /// What choosing it writes into the address field.
    pub address: String,
    /// Whether the address field currently holds exactly this.
    pub chosen: bool,
}

/// Everything the modal renders, recomputed on each tick.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PairingView {
    /// This device's code, grouped as `482 107`. Empty when closed.
    pub own_code: String,
    /// Digits typed for the peer's code so far.
    pub entered: String,
    /// `2:41`. Empty when closed.
    pub countdown: String,
    /// Status line beside the countdown.
    pub message: String,
    /// Whether the Pair button is live.
    pub can_submit: bool,
    /// Whether the address field is shown and required.
    pub needs_address: bool,
    /// Masters mDNS has found, best-named first. Always empty on the master
    /// side, which dials nobody, and empty on a network with no discovery —
    /// in both cases the modal simply shows no list.
    pub devices: Vec<DeviceChoice>,
    /// Whether the modal should be on screen at all.
    pub open: bool,
}

/// What the servant's polling thread reports back.
enum ClaimMsg {
    Paired(Box<PairOutcome>),
    Failed(String),
}

/// A running servant-side claim poll. Dropping it asks the thread to stop.
struct ClaimDriver {
    stop: Arc<std::sync::atomic::AtomicBool>,
    rx: Receiver<ClaimMsg>,
}

impl Drop for ClaimDriver {
    fn drop(&mut self) {
        self.stop.store(true, std::sync::atomic::Ordering::Relaxed);
    }
}

/// Everything the controller needs that it cannot derive.
pub struct PairingDeps {
    pub device_id: String,
    pub device_name: String,
    pub side: PairingSide,
    pub clock: Clock,
    pub rng: SharedRandom,
    /// Where the pick-list comes from. `None` is a working modal with a
    /// typed address, which is what every pairing did before P8.
    pub discovery: Option<Arc<dyn DeviceSource>>,
}

/// Drives one pairing modal.
pub struct PairingController {
    slot: PairingSlot,
    /// Peer code digits, 0–6 of them. Held here rather than in the slot
    /// because a half-typed code is UI state the listener has no business
    /// seeing.
    entered: String,
    /// `host:port` of the master. Servant side only.
    address: String,
    /// Set once the peer's code has been committed.
    submitted: bool,
    side: PairingSide,
    /// Taken from the deps at `open`, because the modal outlives the call
    /// that built them and the list is re-read on every tick.
    discovery: Option<Arc<dyn DeviceSource>>,
    driver: Option<ClaimDriver>,
    ended: Option<PairingEnd>,
    /// A completed pairing waiting for the caller to persist. Servant side
    /// only — on the master the listener has already stored it.
    outcome: Option<PairOutcome>,
    status: Option<String>,
}

impl PairingController {
    /// Wraps the same [`PairingSlot`] the master's listener answers from.
    pub fn new(slot: PairingSlot) -> Self {
        Self {
            slot,
            entered: String::new(),
            address: String::new(),
            submitted: false,
            side: PairingSide::Master,
            discovery: None,
            driver: None,
            ended: None,
            outcome: None,
            status: None,
        }
    }

    /// Mint a fresh code and open the window.
    ///
    /// Any window already open is discarded rather than resumed — the user
    /// asked for a new pairing, and silently handing them a code that expires
    /// in nine seconds because it was minted three minutes ago is worse than
    /// starting over.
    pub fn open(&mut self, deps: &PairingDeps, now_ms: i64) -> anyhow::Result<()> {
        let code = PairCode::generate(&deps.rng)?;
        self.slot
            .open(&deps.device_id, &deps.device_name, code, now_ms);
        self.entered.clear();
        self.submitted = false;
        self.side = deps.side;
        self.discovery = deps.discovery.clone();
        self.driver = None;
        self.ended = None;
        self.outcome = None;
        self.status = None;
        Ok(())
    }

    pub fn is_open(&self) -> bool {
        self.slot.is_open()
    }

    /// How the last window ended, if one has.
    pub fn ended(&self) -> Option<&PairingEnd> {
        self.ended.as_ref()
    }

    /// Take the pairing this device completed as the *initiator*, for the
    /// caller to persist. Drains.
    pub fn take_outcome(&mut self) -> Option<PairOutcome> {
        self.outcome.take()
    }

    /// The master's address as typed. Servant side only.
    pub fn address(&self) -> &str {
        &self.address
    }

    pub fn set_address(&mut self, address: &str) {
        if !self.submitted {
            self.address = address.trim().to_owned();
        }
    }

    /// Take one of the discovered devices as the address to dial.
    ///
    /// Deliberately the same path as typing: it fills the field the user can
    /// still edit, rather than remembering a chosen device separately. One
    /// address, one place it lives, and the modal keeps working identically
    /// on a network where nothing is ever discovered.
    pub fn choose_device(&mut self, address: &str) {
        self.set_address(address);
    }

    /// The pick-list, and which row matches the current address.
    fn device_choices(&self) -> Vec<DeviceChoice> {
        if self.side != PairingSide::Servant {
            return Vec::new();
        }
        let Some(discovery) = self.discovery.as_ref() else {
            return Vec::new();
        };
        discovery
            .devices()
            .into_iter()
            .filter_map(|device| {
                let address = device.address()?.to_owned();
                Some(DeviceChoice {
                    label: device.label(),
                    chosen: address == self.address,
                    address,
                })
            })
            .take(MAX_DEVICES_SHOWN)
            .collect()
    }

    /// Close by user request.
    pub fn cancel(&mut self) {
        if self.slot.is_open() {
            self.finish(PairingEnd::Cancelled);
        }
    }

    /// Accept text from the peer-code field, returning the normalised value
    /// to write back into it.
    ///
    /// Non-digits are dropped rather than rejected so the grouped form the
    /// modal displays (`482 107`) can be pasted or typed with a space, and
    /// the field is capped at six so a stuck key cannot push the real digits
    /// out of view.
    pub fn set_entered(&mut self, text: &str) -> String {
        if !self.slot.is_open() {
            return String::new();
        }
        if self.submitted {
            // The code is committed; further typing would silently disagree
            // with the secret already derived from it.
            return self.entered.clone();
        }
        self.entered = text.chars().filter(char::is_ascii_digit).take(6).collect();
        self.entered.clone()
    }

    /// Commit the peer's code.
    ///
    /// On a master this arms the shared slot and the listener takes over. On
    /// a servant it starts the polling thread, because the servant is the one
    /// that dials.
    pub fn submit(&mut self, deps: &PairingDeps, now_ms: i64) -> Result<(), PairError> {
        if !self.slot.is_open() {
            return Err(PairError::Expired);
        }
        let peer_code = PairCode::parse(&self.entered)?;
        let Some(view) = self.slot.view(now_ms) else {
            return Err(PairError::Expired);
        };
        let own_code = PairCode::parse(&view.own_code)?;

        match self.side {
            PairingSide::Master => {
                self.slot.enter_peer_code(peer_code);
                self.status = Some("Waiting for the other device…".into());
            }
            PairingSide::Servant => {
                if self.address.is_empty() {
                    return Err(PairError::MalformedCode);
                }
                self.driver = Some(start_claim(
                    deps,
                    &self.address,
                    &own_code,
                    &peer_code,
                    self.slot.clone(),
                ));
                self.status = Some("Contacting the master…".into());
            }
        }
        self.submitted = true;
        Ok(())
    }

    /// Recompute what the modal shows, closing the window if its deadline
    /// passed, its attempts ran out, or a pairing completed.
    ///
    /// Called once a second by the modal's timer, so it is where all four of
    /// those are actually noticed — nothing else polls them.
    pub fn tick(&mut self, now_ms: i64) -> PairingView {
        self.drain_driver();
        self.slot.expire_if_due(now_ms);

        // The listener runs on its own thread and may have closed the window
        // between ticks — completed a pairing, or burnt it on a fifth wrong
        // code. Either way the modal finds out here, and `completed` is
        // checked first so a pairing that succeeded just before the deadline
        // is not reported as an expiry.
        if self.ended.is_none() {
            if let Some(peer_name) = self.slot.take_completed() {
                self.finish(PairingEnd::Paired { peer_name });
            } else if let Some(error) = self.slot.take_closed() {
                self.finish(match error {
                    PairError::Aborted => PairingEnd::Aborted,
                    _ => PairingEnd::Expired,
                });
            }
        }

        let Some(view) = self.slot.view(now_ms) else {
            return PairingView::default();
        };
        PairingView {
            own_code: group_code(&view.own_code),
            entered: self.entered.clone(),
            countdown: format_countdown(view.remaining_ms),
            message: self.message(&view),
            can_submit: !self.submitted
                && self.entered.len() == 6
                && (self.side == PairingSide::Master || !self.address.is_empty()),
            needs_address: self.side == PairingSide::Servant,
            devices: self.device_choices(),
            open: true,
        }
    }

    /// Read whatever the servant's polling thread has reported.
    fn drain_driver(&mut self) {
        let Some(driver) = self.driver.as_ref() else {
            return;
        };
        match driver.rx.try_recv() {
            Ok(ClaimMsg::Paired(outcome)) => {
                let peer_name = if outcome.name.trim().is_empty() {
                    outcome.device_id.clone()
                } else {
                    outcome.name.clone()
                };
                self.outcome = Some(*outcome);
                self.finish(PairingEnd::Paired { peer_name });
            }
            Ok(ClaimMsg::Failed(reason)) => {
                self.finish(PairingEnd::Failed { reason });
            }
            Err(TryRecvError::Empty) => {}
            Err(TryRecvError::Disconnected) => {
                // The thread ended without reporting — only reachable if it
                // panicked, which would otherwise leave the modal counting
                // down forever with nothing behind it.
                self.finish(PairingEnd::Failed {
                    reason: "the pairing attempt stopped unexpectedly".into(),
                });
            }
        }
    }

    fn message(&self, view: &maple_sync::pairing::SlotView) -> String {
        if view.attempts > 0 {
            let left = MAX_ATTEMPTS - view.attempts;
            return format!(
                "Wrong code — {left} attempt{} left",
                if left == 1 { "" } else { "s" }
            );
        }
        self.status
            .clone()
            .unwrap_or_else(|| "Enter the code shown on the other device.".into())
    }

    /// Close the window — and with it both codes — and record why.
    fn finish(&mut self, end: PairingEnd) {
        self.slot.close();
        self.driver = None;
        self.entered.clear();
        self.submitted = false;
        self.ended = Some(end);
    }
}

/// Spawn the servant's polling thread.
///
/// It keeps the *same* nonce across retries, per `Initiator::claim`: the
/// master may answer `TooEarly` several times before its user types the code,
/// and re-drawing would gain nothing while making each poll look like a
/// fresh attempt.
fn start_claim(
    deps: &PairingDeps,
    address: &str,
    own_code: &PairCode,
    peer_code: &PairCode,
    slot: PairingSlot,
) -> ClaimDriver {
    let (tx, rx) = mpsc::channel();
    let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));

    let initiator = match Initiator::new(
        &deps.device_id,
        &deps.device_name,
        own_code,
        peer_code,
        &deps.rng,
    ) {
        Ok(initiator) => initiator,
        Err(e) => {
            let _ = tx.send(ClaimMsg::Failed(e.to_string()));
            return ClaimDriver { stop, rx };
        }
    };

    let client = SyncClient::with_timeout(
        address,
        deps.device_id.clone(),
        deps.clock.clone(),
        deps.rng.clone(),
        CLAIM_TIMEOUT,
    );
    let clock = deps.clock.clone();

    std::thread::Builder::new()
        .name("maple-pair-claim".into())
        .spawn({
            let stop = stop.clone();
            move || {
                let claim = initiator.claim();
                while !stop.load(std::sync::atomic::Ordering::Relaxed) {
                    // The deadline is read from the shared window rather than
                    // kept here, so the thread and the modal cannot disagree
                    // about when pairing stopped working.
                    if !slot.view((clock)()).is_some_and(|v| v.remaining_ms > 0) {
                        return;
                    }
                    match client.pair_claim(&claim) {
                        Ok(response) => {
                            let message = match initiator.accept(&response) {
                                Ok(outcome) => ClaimMsg::Paired(Box::new(outcome)),
                                // The mutual half failing means whatever
                                // answered could not prove it knows both
                                // codes — a wrong address, or something on
                                // the LAN pretending to be the master.
                                Err(e) => ClaimMsg::Failed(format!(
                                    "the other device could not prove it knows both codes ({e})"
                                )),
                            };
                            let _ = tx.send(message);
                            return;
                        }
                        Err(failure) => {
                            if failure.code == Some(maple_sync::ErrorCode::TooEarly) {
                                std::thread::sleep(CLAIM_POLL);
                                continue;
                            }
                            let _ = tx.send(ClaimMsg::Failed(failure.to_string()));
                            return;
                        }
                    }
                }
            }
        })
        .expect("failed to spawn pairing thread");

    ClaimDriver { stop, rx }
}

/// `482107` → `482 107`. The grouping is only ever presentational; the code
/// is hashed as six bare digits.
fn group_code(code: &str) -> String {
    if code.len() == 6 {
        format!("{} {}", &code[..3], &code[3..])
    } else {
        code.to_owned()
    }
}

/// Milliseconds remaining as `M:SS`.
///
/// Rounds *up*, so a window with 400 ms left reads `0:01` rather than a
/// `0:00` that sits there looking broken for the last second of every
/// pairing.
fn format_countdown(remaining_ms: i64) -> String {
    let secs = (remaining_ms.max(0) as u64).div_ceil(1000);
    format!("{}:{:02}", secs / 60, secs % 60)
}

/// A [`SharedRandom`] backed by the library database — SQLite's `randomblob`,
/// the same source behind this device's id and every row guid. The workspace
/// deliberately has no `rand` dependency.
pub fn db_random(db: Arc<Mutex<maple_db::Database>>) -> SharedRandom {
    struct DbRandom(Arc<Mutex<maple_db::Database>>);
    impl RandomSource for DbRandom {
        fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
            let guard = maple_db::lock_db(&self.0);
            buf.copy_from_slice(&guard.random_bytes(buf.len())?);
            Ok(())
        }
    }
    Arc::new(DbRandom(db))
}

#[cfg(test)]
mod tests {
    use super::*;
    use maple_sync::pairing::WINDOW_MS;
    use maple_sync::{ClaimRequest, FnRandom};

    /// A reproducible byte stream. `maple-sync`'s own seeded source is
    /// private to that crate — same reason for existing: a pairing test that
    /// sampled a real RNG could only assert "it didn't crash".
    fn seeded(seed: u64) -> SharedRandom {
        struct Xof(Mutex<blake3::OutputReader>);
        impl RandomSource for Xof {
            fn fill(&self, buf: &mut [u8]) -> anyhow::Result<()> {
                self.0.lock().expect("not poisoned").fill(buf);
                Ok(())
            }
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(&seed.to_le_bytes());
        Arc::new(Xof(Mutex::new(hasher.finalize_xof())))
    }

    const T0: i64 = 1_700_000_000_000;

    fn deps(side: PairingSide) -> PairingDeps {
        PairingDeps {
            device_id: "dev-master".into(),
            device_name: "Workstation".into(),
            side,
            clock: Arc::new(|| T0),
            rng: seeded(1),
            discovery: None,
        }
    }

    fn opened(now_ms: i64) -> (PairingController, PairingSlot, PairingDeps) {
        let slot = PairingSlot::new();
        let mut controller = PairingController::new(slot.clone());
        let deps = deps(PairingSide::Master);
        controller.open(&deps, now_ms).expect("open");
        (controller, slot, deps)
    }

    /// A claim from a peer that guessed both codes wrong.
    fn wrong_claim() -> ClaimRequest {
        Initiator::new(
            "dev-servant",
            "Laptop",
            &PairCode::parse("111111").unwrap(),
            &PairCode::parse("222222").unwrap(),
            &FnRandom(|buf: &mut [u8]| {
                buf.fill(0xAB);
                Ok(())
            }),
        )
        .unwrap()
        .claim()
    }

    #[test]
    fn opening_shows_a_grouped_six_digit_code_and_a_full_countdown() {
        let (mut c, _slot, _deps) = opened(T0);
        let view = c.tick(T0);
        assert!(view.open);
        assert_eq!(view.countdown, "3:00");
        assert_eq!(view.own_code.len(), 7, "six digits and one space: {:?}", view.own_code);
        assert_eq!(&view.own_code[3..4], " ");
        assert!(view.own_code.replace(' ', "").chars().all(|c| c.is_ascii_digit()));
        assert!(!view.can_submit, "nothing typed yet");
        assert!(!view.needs_address, "a master does not dial");
    }

    #[test]
    fn the_countdown_is_derived_from_the_window_not_kept_separately() {
        // Reading `remaining_ms` from the shared slot rather than tracking a
        // second deadline is what stops the modal, the listener and the
        // servant's polling thread disagreeing about when pairing stopped
        // working.
        let (mut c, _slot, _deps) = opened(T0);
        assert_eq!(c.tick(T0 + 19_000).countdown, "2:41");
        assert_eq!(c.tick(T0 + 179_500).countdown, "0:01");
    }

    #[test]
    fn entering_the_peer_code_filters_and_caps_the_field() {
        let (mut c, _slot, _deps) = opened(T0);
        assert_eq!(c.set_entered("48"), "48");
        assert_eq!(c.set_entered("482 107"), "482107", "spaces are display only");
        assert_eq!(c.set_entered("48210789"), "482107", "capped at six");
        assert_eq!(c.set_entered("4a8-b2"), "482", "non-digits dropped");
    }

    #[test]
    fn submit_needs_six_digits_and_then_arms_the_shared_slot() {
        let (mut c, slot, deps) = opened(T0);
        c.set_entered("4821");
        assert!(!c.tick(T0).can_submit);
        assert_eq!(c.submit(&deps, T0), Err(PairError::MalformedCode));
        assert!(!slot.view(T0).unwrap().armed);

        c.set_entered("482107");
        assert!(c.tick(T0).can_submit);
        assert!(c.submit(&deps, T0).is_ok());

        // The listener thread can now verify a proof: the code reached the
        // slot, not just this struct.
        assert!(slot.view(T0).unwrap().armed);
        let view = c.tick(T0);
        assert!(!view.can_submit, "already committed");
        assert_eq!(view.message, "Waiting for the other device…");
        assert_eq!(c.set_entered("999999"), "482107", "a committed code is immutable");
    }

    #[test]
    fn a_servant_cannot_submit_without_an_address() {
        // It is the side that dials, and §1.2's modal has nowhere to dial to
        // until an address is picked or typed.
        let slot = PairingSlot::new();
        let mut c = PairingController::new(slot);
        let deps = deps(PairingSide::Servant);
        c.open(&deps, T0).unwrap();
        c.set_entered("482107");

        assert!(c.tick(T0).needs_address);
        assert!(!c.tick(T0).can_submit, "six digits are not enough on a servant");

        c.set_address("  192.168.1.20:7645 ");
        assert_eq!(c.address(), "192.168.1.20:7645", "trimmed");
        assert!(c.tick(T0).can_submit);
    }

    /// A stand-in for the mDNS browser.
    struct FakeDiscovery(Vec<maple_sync::DiscoveredDevice>);

    impl DeviceSource for FakeDiscovery {
        fn devices(&self) -> Vec<maple_sync::DiscoveredDevice> {
            self.0.clone()
        }
    }

    fn found(name: &str, address: &str) -> maple_sync::DiscoveredDevice {
        maple_sync::DiscoveredDevice {
            device_id: format!("dev-{name}"),
            name: name.to_owned(),
            protocol: maple_sync::PROTOCOL_VERSION,
            addresses: if address.is_empty() {
                Vec::new()
            } else {
                vec![address.to_owned()]
            },
        }
    }

    fn servant_with(devices: Vec<maple_sync::DiscoveredDevice>) -> (PairingController, PairingDeps) {
        let mut deps = deps(PairingSide::Servant);
        deps.discovery = Some(Arc::new(FakeDiscovery(devices)));
        let mut c = PairingController::new(PairingSlot::new());
        c.open(&deps, T0).unwrap();
        (c, deps)
    }

    #[test]
    fn choosing_a_discovered_device_fills_in_the_address() {
        // The whole point of §2.4 from the user's side: two clicks instead of
        // an IP address read off another screen.
        let (mut c, deps) = servant_with(vec![
            found("Studio", "192.168.1.20:7645"),
            found("Attic", "192.168.1.31:7645"),
        ]);
        c.set_entered("482107");

        let view = c.tick(T0);
        assert_eq!(view.devices.len(), 2);
        assert!(view.devices.iter().all(|d| !d.chosen), "nothing picked yet");
        assert!(!view.can_submit, "a servant still needs somewhere to dial");

        c.choose_device("192.168.1.31:7645");
        let view = c.tick(T0);
        assert_eq!(c.address(), "192.168.1.31:7645");
        assert!(view.can_submit);
        assert_eq!(
            view.devices.iter().filter(|d| d.chosen).map(|d| d.label.as_str()).collect::<Vec<_>>(),
            vec!["Attic · 192.168.1.31:7645"],
            "exactly the picked row is ticked"
        );

        // And a picked address is still just an address: typing over it wins,
        // which is what keeps the modal usable where multicast is blocked.
        c.set_address("10.0.0.9:7645");
        let view = c.tick(T0);
        assert!(view.devices.iter().all(|d| !d.chosen));
        assert!(view.can_submit);
        let _ = deps;
    }

    #[test]
    fn a_crowded_network_does_not_grow_the_modal_off_the_screen() {
        // The card has no scroll area, so its height is the sum of its rows.
        let crowd: Vec<_> = (0..20)
            .map(|i| found(&format!("Master{i}"), &format!("192.168.1.{i}:7645")))
            .collect();
        let (mut c, _deps) = servant_with(crowd);
        assert_eq!(c.tick(T0).devices.len(), MAX_DEVICES_SHOWN);
    }

    #[test]
    fn a_device_with_no_reachable_address_is_not_offered() {
        // Its record resolved without a usable address — link-local IPv6
        // only, say. Listing it would offer a row that cannot be picked.
        let (mut c, _deps) = servant_with(vec![found("Ghost", "")]);
        assert!(c.tick(T0).devices.is_empty());
    }

    #[test]
    fn a_master_is_offered_no_devices_at_all() {
        // It is claimed, not claiming. A pick-list on this side would invite
        // the user to dial a device that is about to dial them.
        let mut deps = deps(PairingSide::Master);
        deps.discovery = Some(Arc::new(FakeDiscovery(vec![found(
            "Studio",
            "192.168.1.20:7645",
        )])));
        let mut c = PairingController::new(PairingSlot::new());
        c.open(&deps, T0).unwrap();

        let view = c.tick(T0);
        assert!(view.devices.is_empty());
        assert!(!view.needs_address);
    }

    #[test]
    fn a_committed_pairing_ignores_a_later_pick() {
        // The secret is already derived from the code and the claim is in
        // flight; quietly repointing it at another machine would produce a
        // failure nobody could explain.
        let (mut c, deps) = servant_with(vec![found("Studio", "192.168.1.20:7645")]);
        c.set_entered("482107");
        c.choose_device("192.168.1.20:7645");
        c.submit(&deps, T0).unwrap();

        c.choose_device("10.0.0.9:7645");
        assert_eq!(c.address(), "192.168.1.20:7645");
    }

    #[test]
    fn expiry_closes_the_window_and_discards_both_codes() {
        let (mut c, slot, deps) = opened(T0);
        c.set_entered("482107");
        c.submit(&deps, T0).unwrap();

        let view = c.tick(T0 + WINDOW_MS);
        assert!(!view.open);
        assert!(view.own_code.is_empty(), "this device's code must not survive");
        assert!(view.entered.is_empty(), "the peer's code must not survive");
        assert_eq!(c.ended(), Some(&PairingEnd::Expired));
        assert!(!c.is_open());
        assert!(slot.view(T0 + WINDOW_MS).is_none(), "the listener sees it too");
    }

    #[test]
    fn five_wrong_codes_abort_and_discard_both_codes() {
        // The attempts arrive through the slot, exactly as the listener
        // thread delivers them.
        let (mut c, slot, deps) = opened(T0);
        c.set_entered("000000");
        c.submit(&deps, T0).unwrap();

        let claim = wrong_claim();
        for attempt in 1..=MAX_ATTEMPTS {
            let result = slot.handle_claim(&claim, T0 + 1_000, &seeded(2)).unwrap();
            if attempt < MAX_ATTEMPTS {
                assert_eq!(result.err(), Some(PairError::BadProof));
                let view = c.tick(T0 + 1_000);
                assert!(view.open);
                assert!(view.message.contains("attempt"), "{}", view.message);
            } else {
                assert_eq!(result.err(), Some(PairError::Aborted));
            }
        }

        let view = c.tick(T0 + 1_000);
        assert!(!view.open);
        assert!(view.own_code.is_empty());
        assert!(view.entered.is_empty());
        assert_eq!(c.ended(), Some(&PairingEnd::Aborted));
    }

    #[test]
    fn the_modal_notices_a_pairing_its_listener_completed() {
        // The master's half: the controller never sees the claim, only the
        // slot does, and the modal has to close anyway.
        let (mut c, slot, deps) = opened(T0);
        let own = c.tick(T0).own_code.replace(' ', "");
        let servant_code = PairCode::parse("314159").unwrap();
        c.set_entered(servant_code.as_str());
        c.submit(&deps, T0).unwrap();

        let initiator = Initiator::new(
            "dev-servant",
            "Laptop",
            &servant_code,
            &PairCode::parse(&own).unwrap(),
            &seeded(9),
        )
        .unwrap();
        let response = slot
            .handle_claim(&initiator.claim(), T0 + 5_000, &seeded(3))
            .unwrap()
            .expect("the proof verifies");

        // The mutual half: the servant unseals a key from that same response.
        let servant_side = initiator.accept(&response).expect("server proof verifies");
        let stored = slot.take_outcome().expect("the listener would persist this");
        assert_eq!(servant_side.key, stored.key);

        assert!(!c.tick(T0 + 5_000).open, "pairing closes the modal");
        assert_eq!(
            c.ended(),
            Some(&PairingEnd::Paired {
                peer_name: "Laptop".into()
            })
        );
    }

    #[test]
    fn a_claim_before_this_side_typed_a_code_is_too_early_not_a_failed_attempt() {
        // The normal state while one person walks to the other machine.
        // Counting it would burn the window on nothing but latency.
        let (mut c, slot, _deps) = opened(T0);
        let claim = wrong_claim();
        for _ in 0..8 {
            let result = slot.handle_claim(&claim, T0 + 1_000, &seeded(4)).unwrap();
            assert_eq!(result.err(), Some(PairError::TooEarly));
        }
        assert!(c.tick(T0 + 1_000).open, "polling must not abort the window");
        assert_eq!(slot.view(T0 + 1_000).unwrap().attempts, 0);
    }

    #[test]
    fn cancelling_closes_and_reopening_mints_a_fresh_code() {
        let (mut c, _slot, _deps) = opened(T0);
        let first = c.tick(T0).own_code;
        c.cancel();
        assert_eq!(c.ended(), Some(&PairingEnd::Cancelled));
        assert!(!c.tick(T0).open);

        // Reopening two minutes later must not resume the old window with
        // sixty seconds left on it.
        let deps = PairingDeps {
            rng: seeded(5),
            ..deps(PairingSide::Master)
        };
        c.open(&deps, T0 + 120_000).unwrap();
        let view = c.tick(T0 + 120_000);
        assert_eq!(view.countdown, "3:00");
        assert_ne!(view.own_code, first);
        assert!(c.ended().is_none(), "a fresh window clears the old outcome");
    }

    #[test]
    fn a_closed_controller_ignores_input() {
        let slot = PairingSlot::new();
        let mut c = PairingController::new(slot);
        let deps = deps(PairingSide::Master);
        assert_eq!(c.set_entered("482107"), "");
        assert_eq!(c.submit(&deps, T0), Err(PairError::Expired));
        assert_eq!(c.tick(T0), PairingView::default());
    }

    #[test]
    fn countdown_rounds_up_so_the_last_second_is_visible() {
        assert_eq!(format_countdown(180_000), "3:00");
        assert_eq!(format_countdown(400), "0:01");
        assert_eq!(format_countdown(0), "0:00");
        assert_eq!(format_countdown(-5_000), "0:00");
        assert_eq!(format_countdown(61_000), "1:01");
    }

    #[test]
    fn every_end_reason_says_something_useful() {
        // These strings are the only feedback a failed pairing gives, and an
        // empty one reads as the modal having closed for no reason.
        for end in [
            PairingEnd::Cancelled,
            PairingEnd::Expired,
            PairingEnd::Aborted,
            PairingEnd::Paired {
                peer_name: "Laptop".into(),
            },
            PairingEnd::Failed {
                reason: "connection refused".into(),
            },
        ] {
            assert!(!end.message().is_empty(), "{end:?}");
        }
        assert!(PairingEnd::Paired {
            peer_name: "Laptop".into()
        }
        .message()
        .contains("Laptop"));
    }
}
