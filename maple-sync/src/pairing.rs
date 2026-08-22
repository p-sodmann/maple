//! The mutual pairing handshake.
//!
//! Each machine displays a 6-digit code and the user types the other's into
//! it. The shared secret is derived from **both** codes, so neither side can
//! compute it until a human has physically seen both screens — that is what
//! makes the mutual step mean something rather than being decoration.
//!
//! ```text
//! secret = derive_key("maple-pair-v1", min(code_a, code_b) || max(code_a, code_b))
//!
//! client → /pair/claim
//!     { device_id, name, nonce_c, proof_c = keyed_hash(secret, "client" || nonce_c) }
//!
//! server:
//!     its user has not typed the client's code yet  → TooEarly (client retries)
//!     verify proof_c; five wrong attempts abort the window (3 minutes)
//!     key = 32 random bytes
//!   → { device_id, name, nonce_s,
//!       proof_s    = keyed_hash(secret, "server" || nonce_c || nonce_s),
//!       sealed_key = key XOR derive_key("maple-key-v1", secret || nonce_c || nonce_s) }
//!
//! client:
//!     verify proof_s   ← the mutual half; a fake server cannot produce it
//!     unseal key, both sides persist
//! ```
//!
//! # Why the codes are sorted
//!
//! Sorting makes the derivation symmetric, so both sides reach the same
//! secret without having to agree on who is "first". Either machine can be
//! the one that initiates, and the UI can let both users type at their own
//! pace.
//!
//! # Why the key is sealed rather than sent
//!
//! An eavesdropper on the LAN sees the whole handshake — there is no TLS.
//! Sending `key` in the clear would hand them every future request. XOR'ing
//! it with a pad derived from the pairing secret means recovering it requires
//! the secret, which requires both codes. The pad is safe as a one-time pad
//! because it is bound to two fresh nonces and therefore never reused.
//!
//! # What the window buys
//!
//! Combined entropy is ~40 bits (two 6-digit codes, but the attacker only has
//! to guess the one they cannot see, so ~20 bits per guess against a live
//! peer). Three minutes and five attempts is what keeps that from being
//! brute-forceable; the limits are not decoration either.

use serde::{Deserialize, Serialize};

use crate::random::RandomSource;
use crate::trust::PeerKey;

/// How long a pairing window stays open.
pub const WINDOW_MS: i64 = 3 * 60 * 1000;

/// How many wrong proofs abort the window.
pub const MAX_ATTEMPTS: u32 = 5;

/// BLAKE3 derivation contexts. Distinct strings keep the two derivations
/// domain-separated: the same secret must never produce both the pairing
/// secret and the sealing pad.
const PAIR_CONTEXT: &str = "maple-pair-v1";
const SEAL_CONTEXT: &str = "maple-key-v1";

// ── Codes ───────────────────────────────────────────────────────

/// A 6-digit pairing code, held as ASCII digits.
///
/// ASCII rather than a `u32` because the derivation hashes the code exactly
/// as the user sees it — a leading zero is part of the code, and `042107`
/// and `42107` must not derive the same secret.
#[derive(Clone, PartialEq, Eq)]
pub struct PairCode([u8; 6]);

impl PairCode {
    /// Draw a fresh code.
    ///
    /// Rejection-sampled rather than `% 1_000_000`: the modulo would make the
    /// first 4,967,296 values of a `u32` very slightly more likely, and while
    /// the bias is far too small to matter at this scale, "we did the biased
    /// thing because the bias looked small" is not a note worth leaving in a
    /// security path.
    pub fn generate(rng: &impl RandomSource) -> anyhow::Result<Self> {
        // Largest multiple of 1e6 that fits in a u32; anything at or above
        // it would wrap unevenly.
        const LIMIT: u32 = 4_294_000_000;
        for _ in 0..64 {
            let draw = u32::from_le_bytes(rng.array::<4>()?);
            if draw < LIMIT {
                return Ok(Self::from_number(draw % 1_000_000));
            }
        }
        // 64 consecutive rejections has probability ~(2.3e-7)^64. Reaching
        // here means the source is broken, and silently continuing with a
        // predictable code is worse than failing.
        anyhow::bail!("random source rejected 64 draws in a row — is it returning constants?")
    }

    fn from_number(value: u32) -> Self {
        let text = format!("{value:06}");
        let mut digits = [0u8; 6];
        digits.copy_from_slice(text.as_bytes());
        Self(digits)
    }

    /// Parse what a user typed. Spaces and dashes are ignored, since the UI
    /// displays the code grouped (`482 107`) and people type what they see.
    pub fn parse(input: &str) -> Result<Self, PairError> {
        let cleaned: Vec<u8> = input
            .bytes()
            .filter(|b| !matches!(b, b' ' | b'-' | b'\t'))
            .collect();
        let digits: [u8; 6] = cleaned
            .as_slice()
            .try_into()
            .map_err(|_| PairError::MalformedCode)?;
        if !digits.iter().all(|b| b.is_ascii_digit()) {
            return Err(PairError::MalformedCode);
        }
        Ok(Self(digits))
    }

    pub fn as_str(&self) -> &str {
        // Every constructor guarantees ASCII digits.
        std::str::from_utf8(&self.0).expect("pair codes are ASCII by construction")
    }
}

impl std::fmt::Display for PairCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Codes are secrets for the length of the window; don't let one land in a
/// log line because some enclosing struct derived `Debug`.
impl std::fmt::Debug for PairCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PairCode(<hidden>)")
    }
}

/// Derive the shared pairing secret from both codes.
///
/// Order-independent: `pair_secret(a, b) == pair_secret(b, a)`.
pub fn pair_secret(one: &PairCode, other: &PairCode) -> [u8; 32] {
    let (low, high) = if one.0 <= other.0 {
        (one, other)
    } else {
        (other, one)
    };
    // Both halves are exactly 6 bytes, so plain concatenation is unambiguous
    // — there is no pair of different code pairs that produces the same
    // 12-byte input.
    let mut material = [0u8; 12];
    material[..6].copy_from_slice(&low.0);
    material[6..].copy_from_slice(&high.0);
    blake3::derive_key(PAIR_CONTEXT, &material)
}

// ── Wire messages ───────────────────────────────────────────────

/// `POST /pair/claim` body.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimRequest {
    pub device_id: String,
    pub name: String,
    #[serde(with = "b64_array")]
    pub nonce_c: [u8; 16],
    #[serde(with = "b64_array")]
    pub proof_c: [u8; 32],
}

/// `POST /pair/claim` response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClaimResponse {
    pub device_id: String,
    pub name: String,
    #[serde(with = "b64_array")]
    pub nonce_s: [u8; 16],
    #[serde(with = "b64_array")]
    pub proof_s: [u8; 32],
    #[serde(with = "b64_array")]
    pub sealed_key: [u8; 32],
}

/// What both sides end up with: who the peer is, and the key shared with it.
#[derive(Debug, Clone)]
pub struct PairOutcome {
    pub device_id: String,
    pub name: String,
    pub key: PeerKey,
}

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PairError {
    /// The code was not six digits.
    MalformedCode,
    /// This side's user has not typed the peer's code yet. The client retries;
    /// this is the normal state while one person is still walking to the other
    /// machine, so it deliberately does **not** count as a failed attempt.
    TooEarly,
    /// The proof did not verify — wrong code, or someone guessing.
    BadProof,
    /// Too many wrong attempts; the window is dead and both codes are burnt.
    Aborted,
    /// The three-minute window elapsed.
    Expired,
}

impl std::fmt::Display for PairError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::MalformedCode => "pairing code must be six digits",
            Self::TooEarly => "the other device has not been given this device's code yet",
            Self::BadProof => "pairing code did not match",
            Self::Aborted => "pairing was aborted after too many wrong codes",
            Self::Expired => "the pairing window expired",
        };
        f.write_str(text)
    }
}

impl std::error::Error for PairError {}

// ── Client side ─────────────────────────────────────────────────

/// The side that sends `/pair/claim`.
///
/// Constructed once the user has typed the peer's code, since the secret
/// needs both halves.
pub struct Initiator {
    device_id: String,
    name: String,
    secret: [u8; 32],
    nonce_c: [u8; 16],
}

impl Initiator {
    pub fn new(
        device_id: impl Into<String>,
        name: impl Into<String>,
        own_code: &PairCode,
        peer_code: &PairCode,
        rng: &impl RandomSource,
    ) -> anyhow::Result<Self> {
        Ok(Self {
            device_id: device_id.into(),
            name: name.into(),
            secret: pair_secret(own_code, peer_code),
            nonce_c: rng.array()?,
        })
    }

    /// The request body. Stable across retries — the peer may answer
    /// [`PairError::TooEarly`] several times before its user types the code,
    /// and re-drawing the nonce each poll would gain nothing.
    pub fn claim(&self) -> ClaimRequest {
        ClaimRequest {
            device_id: self.device_id.clone(),
            name: self.name.clone(),
            nonce_c: self.nonce_c,
            proof_c: *keyed(&self.secret, b"client", &[&self.nonce_c]).as_bytes(),
        }
    }

    /// Verify the peer's proof and unseal the long-term key.
    ///
    /// This is the half that makes the handshake mutual. Without it, anything
    /// on the LAN that answers first could hand us a key it chose and then
    /// read the library we sync to it.
    pub fn accept(&self, response: &ClaimResponse) -> Result<PairOutcome, PairError> {
        let expected = keyed(
            &self.secret,
            b"server",
            &[&self.nonce_c, &response.nonce_s],
        );
        // `blake3::Hash`'s `PartialEq` is constant-time; comparing the raw
        // arrays instead would leak the position of the first wrong byte.
        if expected != blake3::Hash::from(response.proof_s) {
            return Err(PairError::BadProof);
        }

        let pad = seal_pad(&self.secret, &self.nonce_c, &response.nonce_s);
        let mut key = response.sealed_key;
        for (byte, pad_byte) in key.iter_mut().zip(pad) {
            *byte ^= pad_byte;
        }

        Ok(PairOutcome {
            device_id: response.device_id.clone(),
            name: response.name.clone(),
            key: PeerKey::from_bytes(key),
        })
    }
}

// ── Server side ─────────────────────────────────────────────────

/// The side that answers `/pair/claim`, and the state that bounds it.
///
/// Holds the attempt counter and the deadline, because those are what turn a
/// 20-bit guess into something a human-scale attacker cannot win.
pub struct PairingWindow {
    device_id: String,
    name: String,
    own_code: PairCode,
    /// Set when this machine's user types the peer's code. Until then no
    /// secret exists here, which is precisely the point of the mutual step.
    peer_code: Option<PairCode>,
    opened_at_ms: i64,
    attempts: u32,
    aborted: bool,
}

impl PairingWindow {
    pub fn open(
        device_id: impl Into<String>,
        name: impl Into<String>,
        own_code: PairCode,
        now_ms: i64,
    ) -> Self {
        Self {
            device_id: device_id.into(),
            name: name.into(),
            own_code,
            peer_code: None,
            opened_at_ms: now_ms,
            attempts: 0,
            aborted: false,
        }
    }

    /// The code to show on screen.
    pub fn own_code(&self) -> &PairCode {
        &self.own_code
    }

    /// The user typed the other device's code.
    pub fn enter_peer_code(&mut self, code: PairCode) {
        self.peer_code = Some(code);
    }

    /// Milliseconds until the window closes, clamped at zero — the UI's
    /// countdown reads this rather than keeping its own deadline, so the two
    /// cannot disagree about when pairing stopped working.
    pub fn remaining_ms(&self, now_ms: i64) -> i64 {
        (self.opened_at_ms + WINDOW_MS - now_ms).max(0)
    }

    pub fn is_open(&self, now_ms: i64) -> bool {
        !self.aborted && self.remaining_ms(now_ms) > 0
    }

    /// Wrong attempts so far. Shown in the modal so a user who fat-fingered
    /// a digit knows how much rope is left.
    pub fn attempts(&self) -> u32 {
        self.attempts
    }

    /// Verify a claim and, on success, mint the long-term key.
    ///
    /// Returns the response to send back plus this side's own record of the
    /// pairing; the caller persists the latter to the trust file.
    ///
    /// A repeat claim while the window is open is answered normally with a
    /// *fresh* key rather than rejected: if the response is lost in flight
    /// the client retries, and refusing would strand a pairing the user
    /// already completed. Both sides simply end up storing the newer key.
    pub fn handle_claim(
        &mut self,
        request: &ClaimRequest,
        now_ms: i64,
        rng: &impl RandomSource,
    ) -> anyhow::Result<Result<(ClaimResponse, PairOutcome), PairError>> {
        if self.aborted {
            return Ok(Err(PairError::Aborted));
        }
        if self.remaining_ms(now_ms) == 0 {
            return Ok(Err(PairError::Expired));
        }
        let Some(peer_code) = self.peer_code.as_ref() else {
            return Ok(Err(PairError::TooEarly));
        };

        let secret = pair_secret(&self.own_code, peer_code);
        let expected = keyed(&secret, b"client", &[&request.nonce_c]);
        if expected != blake3::Hash::from(request.proof_c) {
            self.attempts += 1;
            if self.attempts >= MAX_ATTEMPTS {
                // Burn the window rather than the codes alone: leaving it
                // open with a spent counter would let the next guess through.
                self.aborted = true;
                return Ok(Err(PairError::Aborted));
            }
            return Ok(Err(PairError::BadProof));
        }

        let key: [u8; 32] = rng.array()?;
        let nonce_s: [u8; 16] = rng.array()?;
        let pad = seal_pad(&secret, &request.nonce_c, &nonce_s);
        let mut sealed_key = key;
        for (byte, pad_byte) in sealed_key.iter_mut().zip(pad) {
            *byte ^= pad_byte;
        }

        let response = ClaimResponse {
            device_id: self.device_id.clone(),
            name: self.name.clone(),
            nonce_s,
            proof_s: *keyed(&secret, b"server", &[&request.nonce_c, &nonce_s]).as_bytes(),
            sealed_key,
        };
        let outcome = PairOutcome {
            device_id: request.device_id.clone(),
            name: request.name.clone(),
            key: PeerKey::from_bytes(key),
        };
        Ok(Ok((response, outcome)))
    }
}

// ── Primitives ──────────────────────────────────────────────────

/// `keyed_hash(secret, label || parts...)`.
///
/// Every `parts` element is a fixed-width nonce and every label is a distinct
/// constant, so concatenation cannot be made ambiguous by an attacker-chosen
/// value — there are none in the input.
fn keyed(secret: &[u8; 32], label: &[u8], parts: &[&[u8]]) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new_keyed(secret);
    hasher.update(label);
    for part in parts {
        hasher.update(part);
    }
    hasher.finalize()
}

/// The one-time pad the long-term key is sealed with.
fn seal_pad(secret: &[u8; 32], nonce_c: &[u8; 16], nonce_s: &[u8; 16]) -> [u8; 32] {
    let mut material = [0u8; 64];
    material[..32].copy_from_slice(secret);
    material[32..48].copy_from_slice(nonce_c);
    material[48..].copy_from_slice(nonce_s);
    blake3::derive_key(SEAL_CONTEXT, &material)
}

/// Fixed-size byte arrays as base64 in JSON.
mod b64_array {
    use base64::Engine as _;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer, const N: usize>(
        value: &[u8; N],
        s: S,
    ) -> Result<S::Ok, S::Error> {
        s.serialize_str(&base64::engine::general_purpose::STANDARD.encode(value))
    }

    pub fn deserialize<'de, D: Deserializer<'de>, const N: usize>(
        d: D,
    ) -> Result<[u8; N], D::Error> {
        use serde::de::Error as _;
        let text = String::deserialize(d)?;
        let bytes = base64::engine::general_purpose::STANDARD
            .decode(&text)
            .map_err(D::Error::custom)?;
        bytes
            .try_into()
            .map_err(|_| D::Error::custom(format!("expected {N} bytes")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::SeededRandom;

    const T0: i64 = 1_700_000_000_000;

    fn code(text: &str) -> PairCode {
        PairCode::parse(text).expect("test codes are well-formed")
    }

    /// Runs the whole handshake with both users cooperating, and returns what
    /// each side ends up holding.
    fn full_handshake(
        client_code: &PairCode,
        server_code: &PairCode,
        client_believes_server_code_is: &PairCode,
    ) -> Result<(PairOutcome, PairOutcome), PairError> {
        let client_rng = SeededRandom::new(1);
        let server_rng = SeededRandom::new(2);

        let mut window = PairingWindow::open("dev-server", "Workstation", server_code.clone(), T0);
        window.enter_peer_code(client_code.clone());

        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            client_code,
            client_believes_server_code_is,
            &client_rng,
        )
        .unwrap();

        let (response, server_side) = window
            .handle_claim(&initiator.claim(), T0, &server_rng)
            .unwrap()?;
        let client_side = initiator.accept(&response)?;
        Ok((client_side, server_side))
    }

    // ── Codes and the secret ────────────────────────────────────

    #[test]
    fn the_secret_does_not_depend_on_which_code_is_first() {
        let a = code("482107");
        let b = code("903115");
        assert_eq!(pair_secret(&a, &b), pair_secret(&b, &a));
    }

    #[test]
    fn different_code_pairs_derive_different_secrets() {
        assert_ne!(
            pair_secret(&code("482107"), &code("903115")),
            pair_secret(&code("482107"), &code("903116"))
        );
    }

    #[test]
    fn a_leading_zero_is_part_of_the_code() {
        // The codes are hashed as digits, so `042107` must not collide with
        // whatever `42107` would parse to if we accepted short input.
        assert!(PairCode::parse("42107").is_err());
        assert_ne!(
            pair_secret(&code("042107"), &code("111111")),
            pair_secret(&code("420107"), &code("111111"))
        );
    }

    #[test]
    fn codes_are_parsed_as_the_user_sees_them() {
        assert_eq!(code("482 107").as_str(), "482107");
        assert_eq!(code("482-107").as_str(), "482107");
        for bad in ["", "12345", "1234567", "4821o7", "abcdef"] {
            assert_eq!(PairCode::parse(bad), Err(PairError::MalformedCode), "{bad:?}");
        }
    }

    #[test]
    fn generated_codes_are_six_digits_and_stream_dependent() {
        let rng = SeededRandom::new(9);
        let first = PairCode::generate(&rng).unwrap();
        let second = PairCode::generate(&rng).unwrap();
        for produced in [&first, &second] {
            assert_eq!(produced.as_str().len(), 6);
            assert!(produced.as_str().bytes().all(|b| b.is_ascii_digit()));
        }
        assert_ne!(first, second);
        // Same seed, same code — this is what makes the tests below
        // reproducible rather than merely usually-passing.
        assert_eq!(PairCode::generate(&SeededRandom::new(9)).unwrap(), first);
    }

    #[test]
    fn a_broken_random_source_fails_rather_than_producing_a_guessable_code() {
        // A source stuck at 0xFF only ever yields draws above the rejection
        // limit; the loop must give up instead of falling back to a constant.
        let stuck = crate::random::FnRandom(|buf: &mut [u8]| {
            buf.fill(0xFF);
            Ok(())
        });
        assert!(PairCode::generate(&stuck).is_err());
    }

    // ── The happy path ──────────────────────────────────────────

    #[test]
    fn both_sides_end_up_with_the_same_key() {
        let client_code = code("482107");
        let server_code = code("903115");
        let (client_side, server_side) =
            full_handshake(&client_code, &server_code, &server_code).unwrap();

        assert_eq!(client_side.key, server_side.key, "the sealed key round-trips");
        assert_eq!(client_side.device_id, "dev-server");
        assert_eq!(client_side.name, "Workstation");
        assert_eq!(server_side.device_id, "dev-client");
        assert_eq!(server_side.name, "Laptop");
    }

    #[test]
    fn the_key_is_not_sent_in_the_clear() {
        let client_code = code("482107");
        let server_code = code("903115");
        let client_rng = SeededRandom::new(1);
        let server_rng = SeededRandom::new(2);

        let mut window = PairingWindow::open("dev-server", "Workstation", server_code.clone(), T0);
        window.enter_peer_code(client_code.clone());
        let initiator =
            Initiator::new("dev-client", "Laptop", &client_code, &server_code, &client_rng).unwrap();
        let (response, server_side) = window
            .handle_claim(&initiator.claim(), T0, &server_rng)
            .unwrap()
            .unwrap();

        assert_ne!(
            &response.sealed_key,
            server_side.key.as_bytes(),
            "the key must be sealed on the wire, not merely relabelled"
        );
    }

    // ── The mutual half ─────────────────────────────────────────

    #[test]
    fn a_server_that_cannot_produce_proof_s_is_rejected() {
        // A rogue master on the LAN: it sees the claim, knows the device ids
        // and the client nonce, but not the codes. The best it can do is
        // invent a proof and a key of its choosing.
        let client_code = code("482107");
        let server_code = code("903115");
        let client_rng = SeededRandom::new(1);
        let initiator =
            Initiator::new("dev-client", "Laptop", &client_code, &server_code, &client_rng).unwrap();
        let claim = initiator.claim();

        let forged = ClaimResponse {
            device_id: "dev-server".into(),
            name: "Workstation".into(),
            nonce_s: [0x5A; 16],
            proof_s: [0xAB; 32],
            sealed_key: [0xCD; 32],
        };
        assert_eq!(initiator.accept(&forged).unwrap_err(), PairError::BadProof);
        assert_eq!(claim.device_id, "dev-client");
    }

    #[test]
    fn a_response_bound_to_a_different_client_nonce_is_rejected() {
        // Replaying a proof captured from another pairing attempt: it is a
        // real proof, just not for this nonce_c.
        let client_code = code("482107");
        let server_code = code("903115");
        let secret = pair_secret(&client_code, &server_code);
        let nonce_s = [7u8; 16];

        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            &client_code,
            &server_code,
            &SeededRandom::new(1),
        )
        .unwrap();

        let stale = ClaimResponse {
            device_id: "dev-server".into(),
            name: "Workstation".into(),
            nonce_s,
            proof_s: *keyed(&secret, b"server", &[&[3u8; 16][..], &nonce_s]).as_bytes(),
            sealed_key: [0; 32],
        };
        assert_eq!(initiator.accept(&stale).unwrap_err(), PairError::BadProof);
    }

    #[test]
    fn a_wrong_secret_does_not_unseal_the_key() {
        let right = pair_secret(&code("482107"), &code("903115"));
        let wrong = pair_secret(&code("482107"), &code("903116"));
        let (nonce_c, nonce_s) = ([1u8; 16], [2u8; 16]);
        let key = [0x11u8; 32];

        let mut sealed = key;
        for (byte, pad) in sealed.iter_mut().zip(seal_pad(&right, &nonce_c, &nonce_s)) {
            *byte ^= pad;
        }

        let mut recovered = sealed;
        for (byte, pad) in recovered
            .iter_mut()
            .zip(seal_pad(&wrong, &nonce_c, &nonce_s))
        {
            *byte ^= pad;
        }
        assert_ne!(recovered, key);

        let mut correct = sealed;
        for (byte, pad) in correct.iter_mut().zip(seal_pad(&right, &nonce_c, &nonce_s)) {
            *byte ^= pad;
        }
        assert_eq!(correct, key);
    }

    #[test]
    fn the_seal_pad_is_bound_to_both_nonces() {
        // Reusing a pad across handshakes would leak the XOR of two keys.
        let secret = [4u8; 32];
        assert_ne!(
            seal_pad(&secret, &[1u8; 16], &[2u8; 16]),
            seal_pad(&secret, &[1u8; 16], &[3u8; 16])
        );
        assert_ne!(
            seal_pad(&secret, &[1u8; 16], &[2u8; 16]),
            seal_pad(&secret, &[9u8; 16], &[2u8; 16])
        );
    }

    #[test]
    fn the_pairing_secret_and_the_seal_pad_are_domain_separated() {
        let secret = pair_secret(&code("482107"), &code("903115"));
        assert_ne!(
            secret,
            seal_pad(&secret, &[0u8; 16], &[0u8; 16]),
            "one context string must not produce both values"
        );
    }

    // ── Window enforcement ──────────────────────────────────────

    #[test]
    fn a_claim_before_the_user_types_the_code_is_too_early() {
        let server_rng = SeededRandom::new(2);
        let mut window = PairingWindow::open("dev-server", "Workstation", code("903115"), T0);
        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            &code("482107"),
            &code("903115"),
            &SeededRandom::new(1),
        )
        .unwrap();

        assert_eq!(
            window
                .handle_claim(&initiator.claim(), T0, &server_rng)
                .unwrap()
                .unwrap_err(),
            PairError::TooEarly
        );
        assert_eq!(
            window.attempts(),
            0,
            "waiting for the other user is not a failed attempt"
        );

        // Once the user types it, the same claim succeeds — the client polls
        // with a stable nonce, so this is the real retry path.
        window.enter_peer_code(code("482107"));
        assert!(window
            .handle_claim(&initiator.claim(), T0, &server_rng)
            .unwrap()
            .is_ok());
    }

    #[test]
    fn a_wrong_code_fails_the_client_proof() {
        let err = full_handshake(&code("482107"), &code("903115"), &code("903116")).unwrap_err();
        assert_eq!(err, PairError::BadProof);
    }

    #[test]
    fn five_wrong_attempts_abort_the_window() {
        let server_rng = SeededRandom::new(2);
        let mut window = PairingWindow::open("dev-server", "Workstation", code("903115"), T0);
        window.enter_peer_code(code("482107"));

        // A guesser who has seen the master's code but not the servant's.
        let guesses = ["482106", "482108", "482117", "482207", "483107"];
        for (index, guess) in guesses.iter().enumerate() {
            let attacker = Initiator::new(
                "dev-attacker",
                "Someone",
                &code(guess),
                &code("903115"),
                &SeededRandom::new(index as u64),
            )
            .unwrap();
            let outcome = window
                .handle_claim(&attacker.claim(), T0, &server_rng)
                .unwrap()
                .unwrap_err();
            let expected = if index + 1 < MAX_ATTEMPTS as usize {
                PairError::BadProof
            } else {
                PairError::Aborted
            };
            assert_eq!(outcome, expected, "attempt {}", index + 1);
        }

        assert!(!window.is_open(T0), "the window must be dead, not merely out of tries");

        // And the *correct* code no longer helps — otherwise the limit would
        // only slow an attacker down rather than stopping one.
        let honest = Initiator::new(
            "dev-client",
            "Laptop",
            &code("482107"),
            &code("903115"),
            &SeededRandom::new(1),
        )
        .unwrap();
        assert_eq!(
            window
                .handle_claim(&honest.claim(), T0, &server_rng)
                .unwrap()
                .unwrap_err(),
            PairError::Aborted
        );
    }

    #[test]
    fn the_window_expires_after_three_minutes() {
        let server_rng = SeededRandom::new(2);
        let mut window = PairingWindow::open("dev-server", "Workstation", code("903115"), T0);
        window.enter_peer_code(code("482107"));
        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            &code("482107"),
            &code("903115"),
            &SeededRandom::new(1),
        )
        .unwrap();

        let last_moment = T0 + WINDOW_MS - 1;
        assert!(window.is_open(last_moment));
        assert_eq!(window.remaining_ms(last_moment), 1);
        assert!(window
            .handle_claim(&initiator.claim(), last_moment, &server_rng)
            .unwrap()
            .is_ok());

        assert!(!window.is_open(T0 + WINDOW_MS));
        assert_eq!(window.remaining_ms(T0 + WINDOW_MS + 5_000), 0, "clamped, not negative");
        assert_eq!(
            window
                .handle_claim(&initiator.claim(), T0 + WINDOW_MS, &server_rng)
                .unwrap()
                .unwrap_err(),
            PairError::Expired
        );
    }

    #[test]
    fn a_retried_claim_is_answered_with_a_fresh_key() {
        // The response can be lost in flight; refusing the retry would strand
        // a pairing the user already completed.
        let server_rng = SeededRandom::new(2);
        let mut window = PairingWindow::open("dev-server", "Workstation", code("903115"), T0);
        window.enter_peer_code(code("482107"));
        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            &code("482107"),
            &code("903115"),
            &SeededRandom::new(1),
        )
        .unwrap();

        let (first, first_local) = window
            .handle_claim(&initiator.claim(), T0, &server_rng)
            .unwrap()
            .unwrap();
        let (second, second_local) = window
            .handle_claim(&initiator.claim(), T0 + 1_000, &server_rng)
            .unwrap()
            .unwrap();

        assert_ne!(first_local.key, second_local.key);
        assert_eq!(
            initiator.accept(&second).unwrap().key,
            second_local.key,
            "the client and the server agree on the newer key"
        );
        assert_eq!(initiator.accept(&first).unwrap().key, first_local.key);
    }

    // ── Wire format ─────────────────────────────────────────────

    #[test]
    fn claim_messages_round_trip_through_json() {
        let client_code = code("482107");
        let server_code = code("903115");
        let mut window = PairingWindow::open("dev-server", "Workstation", server_code.clone(), T0);
        window.enter_peer_code(client_code.clone());
        let initiator = Initiator::new(
            "dev-client",
            "Laptop",
            &client_code,
            &server_code,
            &SeededRandom::new(1),
        )
        .unwrap();

        let claim = initiator.claim();
        let claim: ClaimRequest =
            serde_json::from_str(&serde_json::to_string(&claim).unwrap()).unwrap();

        let (response, server_side) = window
            .handle_claim(&claim, T0, &SeededRandom::new(2))
            .unwrap()
            .unwrap();
        let response: ClaimResponse =
            serde_json::from_str(&serde_json::to_string(&response).unwrap()).unwrap();

        assert_eq!(initiator.accept(&response).unwrap().key, server_side.key);
    }

    #[test]
    fn codes_stay_out_of_debug_output() {
        assert_eq!(format!("{:?}", code("482107")), "PairCode(<hidden>)");
    }
}
