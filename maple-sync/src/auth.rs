//! Request signing: proving which paired device sent a request.
//!
//! After pairing, the long-term key **never crosses the wire again**. Every
//! request carries a header instead:
//!
//! ```text
//! Authorization: Maple <device_id>:<unix_ms>:<nonce>:<mac>
//! mac = keyed_hash(key, method ‖ path ‖ unix_ms ‖ nonce ‖ blake3(body))
//! ```
//!
//! A request is accepted only if the MAC verifies under the key stored for
//! `device_id`, the timestamp is within [`MAX_SKEW_MS`] of now, and the nonce
//! has not been seen recently.
//!
//! # What this does and does not buy
//!
//! Covering the method and path stops a captured `GET /sync/pull` from being
//! replayed as `POST /sync/apply`; covering a hash of the body stops the rows
//! inside it being rewritten in flight. It does **not** hide any of it — the
//! transport is plain HTTP, so an eavesdropper reads every request in full.
//! See the crate docs.
//!
//! # Why the timestamp *and* the nonce
//!
//! Either alone is insufficient. A timestamp bounds how long a captured
//! request stays useful but permits replay inside that window; a nonce set
//! catches replays but would have to grow forever. Together the set only has
//! to remember one skew window's worth, which is what makes [`NonceRing`]
//! boundable.

use std::collections::{HashSet, VecDeque};

use crate::random::RandomSource;
use crate::trust::PeerKey;

/// How far a request's timestamp may be from local time, either direction.
///
/// Both directions, because the two machines' clocks are independent and the
/// sender's may be ahead of the receiver's; rejecting "from the future"
/// strictly would break pairs whose clocks differ by a second.
pub const MAX_SKEW_MS: i64 = 5 * 60 * 1000;

/// Scheme token in the `Authorization` header.
pub const SCHEME: &str = "Maple";

/// Default number of nonces remembered per process.
///
/// Entries older than a skew window are dropped anyway, so this cap only
/// binds under a flood — and a flood of requests that reach the nonce check
/// is a flood of requests whose MAC already verified, meaning the sender
/// holds the key. Dropping the oldest in that case is acceptable; refusing
/// service would not be.
pub const DEFAULT_NONCE_CAPACITY: usize = 4096;

/// Wall-clock milliseconds since the epoch, for production callers.
///
/// Everything in this module takes `now_ms` as an argument instead of calling
/// this, so tests can pin time and assert on the exact boundary rather than
/// racing it.
pub fn now_ms() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as i64)
        .unwrap_or(0)
}

// ── Errors ──────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AuthError {
    /// Header absent, wrong scheme, or not four colon-separated fields.
    Malformed,
    /// Timestamp outside ±[`MAX_SKEW_MS`].
    StaleTimestamp,
    /// MAC did not verify under this peer's key.
    BadSignature,
    /// This nonce was already used within the window.
    Replay,
}

impl std::fmt::Display for AuthError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let text = match self {
            Self::Malformed => "malformed Authorization header",
            Self::StaleTimestamp => "request timestamp is outside the accepted window",
            Self::BadSignature => "request signature did not verify",
            Self::Replay => "request nonce was already used",
        };
        f.write_str(text)
    }
}

impl std::error::Error for AuthError {}

// ── The signature ───────────────────────────────────────────────

/// A parsed (or freshly built) `Authorization: Maple …` credential.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedRequest {
    /// Who claims to have sent this. Untrusted until [`Self::verify`] runs —
    /// it is only the hint that says which key to check against.
    pub device_id: String,
    pub timestamp_ms: i64,
    pub nonce: [u8; 16],
    pub mac: [u8; 32],
}

impl SignedRequest {
    /// Sign a request with an explicit nonce.
    pub fn sign(
        key: &PeerKey,
        device_id: impl Into<String>,
        method: &str,
        path: &str,
        body: &[u8],
        now_ms: i64,
        nonce: [u8; 16],
    ) -> Self {
        let device_id = device_id.into();
        let mac = mac(key, method, path, now_ms, &nonce, body);
        Self {
            device_id,
            timestamp_ms: now_ms,
            nonce,
            mac: *mac.as_bytes(),
        }
    }

    /// Sign with a nonce drawn from `rng`.
    pub fn sign_with(
        key: &PeerKey,
        device_id: impl Into<String>,
        method: &str,
        path: &str,
        body: &[u8],
        now_ms: i64,
        rng: &impl RandomSource,
    ) -> anyhow::Result<Self> {
        let nonce = rng.array()?;
        Ok(Self::sign(key, device_id, method, path, body, now_ms, nonce))
    }

    /// Render the header value, including the scheme token.
    pub fn header(&self) -> String {
        format!(
            "{SCHEME} {}:{}:{}:{}",
            self.device_id,
            self.timestamp_ms,
            b64(&self.nonce),
            b64(&self.mac)
        )
    }

    /// Parse a header value. Does no cryptography — the caller looks up the
    /// key for [`Self::device_id`] and then calls [`Self::verify`].
    pub fn parse(header: &str) -> Result<Self, AuthError> {
        let rest = header
            .strip_prefix(SCHEME)
            .and_then(|r| r.strip_prefix(' '))
            .ok_or(AuthError::Malformed)?;

        // Exactly four fields: a device id containing a colon would otherwise
        // let a caller shift the parse and pick which field is the MAC.
        let mut fields = rest.split(':');
        let (Some(device_id), Some(ts), Some(nonce), Some(mac), None) = (
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
            fields.next(),
        ) else {
            return Err(AuthError::Malformed);
        };
        if device_id.is_empty() {
            return Err(AuthError::Malformed);
        }

        Ok(Self {
            device_id: device_id.to_owned(),
            timestamp_ms: ts.parse().map_err(|_| AuthError::Malformed)?,
            nonce: un_b64(nonce)?,
            mac: un_b64(mac)?,
        })
    }

    /// Check the credential against the request it claims to cover.
    ///
    /// Order matters: the freshness and MAC checks run *before* the nonce is
    /// recorded, so unauthenticated traffic cannot fill the ring and evict
    /// the entries that stop a real replay.
    pub fn verify(
        &self,
        key: &PeerKey,
        method: &str,
        path: &str,
        body: &[u8],
        now_ms: i64,
        ring: &mut NonceRing,
    ) -> Result<(), AuthError> {
        if (now_ms - self.timestamp_ms).abs() > MAX_SKEW_MS {
            return Err(AuthError::StaleTimestamp);
        }

        let expected = mac(key, method, path, self.timestamp_ms, &self.nonce, body);
        // `blake3::Hash`'s `PartialEq` is constant-time — comparing the byte
        // arrays directly would short-circuit on the first mismatch and leak
        // how much of a forged MAC was right.
        if expected != blake3::Hash::from(self.mac) {
            return Err(AuthError::BadSignature);
        }

        if !ring.remember(&self.device_id, &self.nonce, now_ms) {
            return Err(AuthError::Replay);
        }
        Ok(())
    }
}

/// `keyed_hash(key, method ‖ path ‖ unix_ms ‖ nonce ‖ blake3(body))`.
///
/// Each variable-length field is length-prefixed. Without that, `GET` +
/// `/a/b` and `GE` + `T/a/b` hash identically, and an attacker who can pick
/// part of a path gets to move the boundary.
///
/// The body is hashed rather than fed in whole so signing a multi-megabyte
/// batch does not mean holding it twice.
fn mac(
    key: &PeerKey,
    method: &str,
    path: &str,
    timestamp_ms: i64,
    nonce: &[u8; 16],
    body: &[u8],
) -> blake3::Hash {
    let mut hasher = blake3::Hasher::new_keyed(key.as_bytes());
    for field in [method.as_bytes(), path.as_bytes()] {
        hasher.update(&(field.len() as u64).to_le_bytes());
        hasher.update(field);
    }
    hasher.update(&timestamp_ms.to_le_bytes());
    hasher.update(nonce);
    hasher.update(blake3::hash(body).as_bytes());
    hasher.finalize()
}

fn b64(bytes: &[u8]) -> String {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

fn un_b64<const N: usize>(text: &str) -> Result<[u8; N], AuthError> {
    use base64::Engine as _;
    base64::engine::general_purpose::STANDARD
        .decode(text)
        .map_err(|_| AuthError::Malformed)?
        .try_into()
        .map_err(|_| AuthError::Malformed)
}

// ── Replay protection ───────────────────────────────────────────

/// Nonces seen recently, scoped per device.
///
/// Scoped per device because two peers drawing the same 16 random bytes is
/// not something either can be blamed for, and one peer must not be able to
/// invalidate another's request by using its nonce first.
pub struct NonceRing {
    capacity: usize,
    /// Oldest first, so expiry is a pop from the front.
    order: VecDeque<(i64, String)>,
    seen: HashSet<String>,
}

impl Default for NonceRing {
    fn default() -> Self {
        Self::with_capacity(DEFAULT_NONCE_CAPACITY)
    }
}

impl NonceRing {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            order: VecDeque::new(),
            seen: HashSet::new(),
        }
    }

    /// Number of nonces currently remembered. Exposed for tests and for a
    /// future status readout, not used in the decision.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Record a nonce. Returns `false` if it was already there — i.e. this is
    /// a replay.
    ///
    /// Entries older than a full skew window are forgotten first: a request
    /// carrying such a timestamp is rejected before it ever reaches here, so
    /// remembering it any longer buys nothing.
    fn remember(&mut self, device_id: &str, nonce: &[u8; 16], now_ms: i64) -> bool {
        let cutoff = now_ms - MAX_SKEW_MS;
        while let Some((seen_at, _)) = self.order.front() {
            if *seen_at >= cutoff {
                break;
            }
            let (_, key) = self.order.pop_front().expect("front was just observed");
            self.seen.remove(&key);
        }

        let key = format!("{device_id}:{}", b64(nonce));
        if !self.seen.insert(key.clone()) {
            return false;
        }
        self.order.push_back((now_ms, key));

        while self.order.len() > self.capacity {
            let (_, evicted) = self.order.pop_front().expect("len exceeds capacity >= 1");
            self.seen.remove(&evicted);
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::random::SeededRandom;

    const T0: i64 = 1_700_000_000_000;

    fn key() -> PeerKey {
        PeerKey::from_bytes([3u8; 32])
    }

    fn signed(now_ms: i64) -> SignedRequest {
        SignedRequest::sign(&key(), "dev-a", "POST", "/sync/apply", b"{}", now_ms, [1u8; 16])
    }

    #[test]
    fn a_valid_request_passes() {
        let mut ring = NonceRing::new();
        let request = signed(T0);
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0, &mut ring),
            Ok(())
        );
    }

    #[test]
    fn the_header_round_trips() {
        let request = signed(T0);
        let parsed = SignedRequest::parse(&request.header()).expect("parse");
        assert_eq!(parsed, request);
    }

    #[test]
    fn a_replayed_request_is_rejected() {
        let mut ring = NonceRing::new();
        let request = signed(T0);
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0, &mut ring),
            Ok(())
        );
        // Byte-identical capture, resent one second later — still inside the
        // freshness window, so only the nonce ring can catch it.
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0 + 1_000, &mut ring),
            Err(AuthError::Replay)
        );
    }

    #[test]
    fn a_six_minute_old_request_is_rejected() {
        let mut ring = NonceRing::new();
        let request = signed(T0);
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0 + 6 * 60 * 1000, &mut ring),
            Err(AuthError::StaleTimestamp)
        );
    }

    #[test]
    fn a_request_from_the_future_is_rejected_too() {
        let mut ring = NonceRing::new();
        let request = signed(T0);
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0 - 6 * 60 * 1000, &mut ring),
            Err(AuthError::StaleTimestamp)
        );
    }

    #[test]
    fn the_skew_boundary_is_inclusive() {
        let mut ring = NonceRing::new();
        let request = signed(T0);
        assert_eq!(
            request.verify(&key(), "POST", "/sync/apply", b"{}", T0 + MAX_SKEW_MS, &mut ring),
            Ok(()),
            "exactly at the limit must still pass"
        );
        let other = SignedRequest::sign(
            &key(),
            "dev-a",
            "POST",
            "/sync/apply",
            b"{}",
            T0,
            [2u8; 16],
        );
        assert_eq!(
            other.verify(&key(), "POST", "/sync/apply", b"{}", T0 + MAX_SKEW_MS + 1, &mut ring),
            Err(AuthError::StaleTimestamp)
        );
    }

    #[test]
    fn tampering_with_the_method_invalidates_the_mac() {
        let mut ring = NonceRing::new();
        assert_eq!(
            signed(T0).verify(&key(), "DELETE", "/sync/apply", b"{}", T0, &mut ring),
            Err(AuthError::BadSignature)
        );
    }

    #[test]
    fn tampering_with_the_path_invalidates_the_mac() {
        let mut ring = NonceRing::new();
        assert_eq!(
            signed(T0).verify(&key(), "POST", "/sync/pull", b"{}", T0, &mut ring),
            Err(AuthError::BadSignature)
        );
    }

    #[test]
    fn tampering_with_the_body_invalidates_the_mac() {
        let mut ring = NonceRing::new();
        assert_eq!(
            signed(T0).verify(&key(), "POST", "/sync/apply", b"{\"evil\":1}", T0, &mut ring),
            Err(AuthError::BadSignature)
        );
    }

    #[test]
    fn another_peers_key_does_not_verify() {
        let mut ring = NonceRing::new();
        let stranger = PeerKey::from_bytes([4u8; 32]);
        assert_eq!(
            signed(T0).verify(&stranger, "POST", "/sync/apply", b"{}", T0, &mut ring),
            Err(AuthError::BadSignature)
        );
    }

    #[test]
    fn moving_the_boundary_between_method_and_path_changes_the_mac() {
        // The length prefixes exist for exactly this: "GET" + "/a/b" and
        // "GE" + "T/a/b" concatenate to the same bytes.
        let a = SignedRequest::sign(&key(), "dev-a", "GET", "/a/b", b"", T0, [1u8; 16]);
        let b = SignedRequest::sign(&key(), "dev-a", "GE", "T/a/b", b"", T0, [1u8; 16]);
        assert_ne!(a.mac, b.mac);
    }

    #[test]
    fn a_replayed_timestamp_is_not_enough_to_reuse_a_nonce() {
        // Re-signing with a fresh timestamp but the same nonce is what an
        // attacker who cannot forge a MAC would try if the ring keyed on the
        // pair rather than the nonce.
        let mut ring = NonceRing::new();
        assert_eq!(
            signed(T0).verify(&key(), "POST", "/sync/apply", b"{}", T0, &mut ring),
            Ok(())
        );
        let later = SignedRequest::sign(
            &key(),
            "dev-a",
            "POST",
            "/sync/apply",
            b"{}",
            T0 + 1_000,
            [1u8; 16],
        );
        assert_eq!(
            later.verify(&key(), "POST", "/sync/apply", b"{}", T0 + 1_000, &mut ring),
            Err(AuthError::Replay)
        );
    }

    #[test]
    fn two_devices_may_use_the_same_nonce() {
        let mut ring = NonceRing::new();
        for device in ["dev-a", "dev-b"] {
            let request =
                SignedRequest::sign(&key(), device, "POST", "/sync/apply", b"{}", T0, [1u8; 16]);
            assert_eq!(
                request.verify(&key(), "POST", "/sync/apply", b"{}", T0, &mut ring),
                Ok(()),
                "{device} should not be blocked by another device's nonce"
            );
        }
    }

    #[test]
    fn the_ring_forgets_nonces_that_can_no_longer_be_replayed() {
        let mut ring = NonceRing::with_capacity(64);
        assert_eq!(
            signed(T0).verify(&key(), "POST", "/sync/apply", b"{}", T0, &mut ring),
            Ok(())
        );
        assert_eq!(ring.len(), 1);

        // A later request prunes anything older than a full skew window.
        let fresh = SignedRequest::sign(
            &key(),
            "dev-a",
            "POST",
            "/sync/apply",
            b"{}",
            T0 + 2 * MAX_SKEW_MS,
            [9u8; 16],
        );
        assert_eq!(
            fresh.verify(
                &key(),
                "POST",
                "/sync/apply",
                b"{}",
                T0 + 2 * MAX_SKEW_MS,
                &mut ring
            ),
            Ok(())
        );
        assert_eq!(ring.len(), 1, "the stale entry should have been dropped");
    }

    #[test]
    fn the_ring_stays_bounded() {
        let mut ring = NonceRing::with_capacity(8);
        let rng = SeededRandom::new(42);
        for _ in 0..100 {
            let request =
                SignedRequest::sign_with(&key(), "dev-a", "GET", "/x", b"", T0, &rng).unwrap();
            assert_eq!(request.verify(&key(), "GET", "/x", b"", T0, &mut ring), Ok(()));
        }
        assert_eq!(ring.len(), 8);
    }

    #[test]
    fn malformed_headers_are_rejected() {
        for header in [
            "",
            "Basic dXNlcjpwYXNz",
            "Maple",
            "Maple dev-a:1700:abc",
            "Maple dev-a:1700:AAAAAAAAAAAAAAAAAAAAAA==:AAAA:extra",
            "Maple :1700:AAAAAAAAAAAAAAAAAAAAAA==:AAAA",
            "Maple dev-a:not-a-number:AAAAAAAAAAAAAAAAAAAAAA==:AAAA",
            "Maple dev-a:1700:not base64!:AAAA",
        ] {
            assert_eq!(
                SignedRequest::parse(header),
                Err(AuthError::Malformed),
                "accepted {header:?}"
            );
        }
    }

    #[test]
    fn a_short_mac_is_malformed_rather_than_a_bad_signature() {
        // Caught at parse time, so verification never runs on a truncated
        // credential.
        let request = signed(T0);
        let truncated = format!(
            "{SCHEME} {}:{}:{}:{}",
            request.device_id,
            request.timestamp_ms,
            b64(&request.nonce),
            b64(&request.mac[..16])
        );
        assert_eq!(
            SignedRequest::parse(&truncated),
            Err(AuthError::Malformed)
        );
    }
}
