//! The wire contract between a master and its servants: routes, envelopes,
//! and the error vocabulary both sides agree on.
//!
//! Row-level types are **not** here — `SyncBatch` and its members live in
//! `maple_db::sync::wire`, because they are projections of that crate's
//! schema. This module holds only what wraps them: which URL carries what,
//! and how a failure is named.
//!
//! # Why errors are a closed vocabulary rather than a status code
//!
//! The servant has to sort failures into two piles that behave completely
//! differently: retry with backoff, or stop and tell the user to re-pair
//! (§1.4). An HTTP status alone cannot make that call — a 403 from a
//! reverse proxy in the way and a 403 from a rejected MAC deserve opposite
//! reactions. So every failure this server generates carries a machine-
//! readable [`ErrorCode`] in the body, and [`ErrorCode::is_fatal`] is the
//! single place that decides which pile it lands in.
//!
//! # Protocol version
//!
//! [`PROTOCOL_VERSION`] is bumped when the shape of these envelopes changes.
//! It is deliberately separate from the database schema version: two
//! installations can differ in schema (one has run a migration the other has
//! not) and still speak. `Hello` reports both so the servant can refuse a
//! link it cannot merge instead of half-applying rows it does not understand.

use serde::{Deserialize, Serialize};

/// Bumped on any incompatible change to the envelopes below.
pub const PROTOCOL_VERSION: u32 = 1;

/// Largest request body the server will read, in bytes.
///
/// The body has to be buffered in full before the MAC can be checked — the
/// signature covers a hash of it — so an unauthenticated caller gets to
/// choose how much memory the server allocates. A metadata batch at the
/// default 500-stamp boundary runs to a few megabytes even with 2 KB face
/// embeddings in it, so 64 MiB is generous by two orders of magnitude while
/// still bounding the damage.
pub const MAX_BODY_BYTES: u64 = 64 * 1024 * 1024;

/// The four routes P5 serves. Blob routes arrive in P7.
pub mod route {
    pub const HELLO: &str = "/sync/hello";
    pub const PAIR_CLAIM: &str = "/pair/claim";
    pub const PULL: &str = "/sync/pull";
    pub const PUSH: &str = "/sync/push";
}

// ── Envelopes ───────────────────────────────────────────────────

/// `GET /sync/hello` — who is answering, and can we talk to them.
///
/// Deliberately **unsigned**. It is the reachability probe, and it has to
/// work whether or not the caller is paired: §1.3 needs "the master is
/// there but rejected my credential" (red, re-pair) to look different from
/// "nothing answered" (red, retry), and a hello that required a valid
/// signature would collapse the two into one indistinguishable timeout.
///
/// The cost is that anyone on the LAN can learn a device's name and role.
/// On a network where that matters, so does the fact that the whole
/// protocol is plain HTTP — see the crate docs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hello {
    pub device_id: String,
    pub name: String,
    /// `"off"` | `"master"` | `"servant"`, matching `sync_identity.role`.
    pub role: String,
    pub protocol: u32,
    pub schema_version: i64,
}

/// `POST /sync/pull` — "send me everything you have stamped above `since`".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PullRequest {
    /// The caller's watermark. The response covers `rev > since`.
    pub since: i64,
    /// How many distinct stamps to include. The server clamps this; see
    /// `maple_db::sync::collect`, where the batch boundary is a stamp value
    /// rather than a row count so a stamp group is never split.
    pub max_revs: usize,
}

/// `POST /sync/push` — the result of merging the caller's batch.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PushResponse {
    /// Rows and tombstones that changed something here.
    pub applied: usize,
    /// Rows held back because a mandatory parent has not arrived. They are
    /// re-sent next pass: the caller must **not** advance its watermark past
    /// them, which is why this is reported rather than silently dropped.
    pub deferred: usize,
    /// The watermark the caller may now record for the push direction —
    /// echoed from the batch rather than recomputed, so a partially applied
    /// batch cannot advance it.
    pub acked_rev: i64,
}

/// The body of every non-2xx response this server produces.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorBody {
    pub code: ErrorCode,
    pub message: String,
}

impl ErrorBody {
    pub fn new(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }
}

// ── Error vocabulary ────────────────────────────────────────────

/// Every failure the sync protocol names.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorCode {
    /// The `Authorization` header was absent or unparseable.
    Malformed,
    /// The signature did not verify, or names a device we have no key for.
    /// Fatal: a key that is wrong now will be wrong in an hour.
    Unauthorized,
    /// Timestamp outside the skew window. **Not** fatal — this is what a
    /// badly-set clock looks like, and clocks get corrected.
    StaleTimestamp,
    /// Nonce already used. Not fatal: a retry of a request whose response
    /// was lost looks exactly like this, and the next attempt draws a fresh
    /// nonce.
    Replay,
    /// The master's user has not typed this device's pairing code yet. The
    /// client polls; §2.1's 425.
    TooEarly,
    /// The pairing code did not match.
    BadCode,
    /// The pairing window expired or was aborted; both codes are burnt.
    PairingClosed,
    /// No pairing window is open on the master.
    NoPairingWindow,
    /// Protocol or schema versions cannot interoperate. Not fatal: the other
    /// machine may be updated while this one keeps running, and then the
    /// link heals itself without a re-pair.
    Incompatible,
    /// The request body did not parse, or asked for something nonsensical.
    BadRequest,
    /// Something failed on the server. Retryable by definition — the client
    /// cannot tell a transient lock contention from a real bug, and treating
    /// it as fatal would need a human to clear a hiccup.
    Internal,
}

impl ErrorCode {
    /// HTTP status to send alongside.
    pub fn http_status(self) -> u16 {
        match self {
            Self::Malformed | Self::BadRequest => 400,
            Self::Unauthorized | Self::StaleTimestamp | Self::Replay => 401,
            Self::BadCode => 403,
            Self::PairingClosed | Self::NoPairingWindow => 410,
            Self::Incompatible => 409,
            // 425 Too Early, per §2.1 — the client polls rather than failing.
            Self::TooEarly => 425,
            Self::Internal => 500,
        }
    }

    /// Whether this failure will still be a failure after a retry.
    ///
    /// This is the §1.4 decision, in one place. Only a rejected credential
    /// qualifies: everything else either heals on its own (a corrected
    /// clock, a fresh nonce, an updated peer) or is a bug worth retrying
    /// past. Getting this wrong in the permissive direction costs battery;
    /// getting it wrong in the strict direction strands a working pairing
    /// behind a `Re-pair required` the user cannot clear by re-pairing,
    /// because nothing was broken.
    pub fn is_fatal(self) -> bool {
        matches!(self, Self::Unauthorized)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Malformed => "malformed",
            Self::Unauthorized => "unauthorized",
            Self::StaleTimestamp => "stale_timestamp",
            Self::Replay => "replay",
            Self::TooEarly => "too_early",
            Self::BadCode => "bad_code",
            Self::PairingClosed => "pairing_closed",
            Self::NoPairingWindow => "no_pairing_window",
            Self::Incompatible => "incompatible",
            Self::BadRequest => "bad_request",
            Self::Internal => "internal",
        }
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl From<crate::auth::AuthError> for ErrorCode {
    fn from(error: crate::auth::AuthError) -> Self {
        use crate::auth::AuthError;
        match error {
            AuthError::Malformed => Self::Malformed,
            AuthError::StaleTimestamp => Self::StaleTimestamp,
            AuthError::BadSignature => Self::Unauthorized,
            AuthError::Replay => Self::Replay,
        }
    }
}

impl From<crate::pairing::PairError> for ErrorCode {
    fn from(error: crate::pairing::PairError) -> Self {
        use crate::pairing::PairError;
        match error {
            PairError::MalformedCode => Self::BadRequest,
            PairError::TooEarly => Self::TooEarly,
            PairError::BadProof => Self::BadCode,
            PairError::Aborted | PairError::Expired => Self::PairingClosed,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::auth::AuthError;
    use crate::pairing::PairError;

    #[test]
    fn only_a_rejected_credential_is_fatal() {
        // The list is short on purpose: every other code names something
        // that can stop being true without anyone re-pairing.
        let all = [
            ErrorCode::Malformed,
            ErrorCode::Unauthorized,
            ErrorCode::StaleTimestamp,
            ErrorCode::Replay,
            ErrorCode::TooEarly,
            ErrorCode::BadCode,
            ErrorCode::PairingClosed,
            ErrorCode::NoPairingWindow,
            ErrorCode::Incompatible,
            ErrorCode::BadRequest,
            ErrorCode::Internal,
        ];
        let fatal: Vec<ErrorCode> = all.into_iter().filter(|c| c.is_fatal()).collect();
        assert_eq!(fatal, vec![ErrorCode::Unauthorized]);
    }

    #[test]
    fn a_bad_signature_is_fatal_but_a_bad_clock_is_not() {
        // Both arrive as a 401. The status alone cannot separate them, which
        // is exactly why the body carries a code.
        assert!(ErrorCode::from(AuthError::BadSignature).is_fatal());
        assert!(!ErrorCode::from(AuthError::StaleTimestamp).is_fatal());
        assert!(!ErrorCode::from(AuthError::Replay).is_fatal());
        assert_eq!(ErrorCode::from(AuthError::BadSignature).http_status(), 401);
        assert_eq!(ErrorCode::from(AuthError::StaleTimestamp).http_status(), 401);
    }

    #[test]
    fn too_early_is_the_425_that_makes_polling_work() {
        assert_eq!(ErrorCode::from(PairError::TooEarly).http_status(), 425);
        assert!(!ErrorCode::from(PairError::TooEarly).is_fatal());
    }

    #[test]
    fn a_burnt_pairing_window_is_gone_not_forbidden() {
        // 410 rather than 403: the codes existed and stopped existing, and
        // the client's correct response is to close the modal, not retry.
        for error in [PairError::Aborted, PairError::Expired] {
            assert_eq!(ErrorCode::from(error), ErrorCode::PairingClosed);
            assert_eq!(ErrorCode::from(error).http_status(), 410);
        }
    }

    #[test]
    fn codes_round_trip_through_json() {
        for code in [
            ErrorCode::Unauthorized,
            ErrorCode::TooEarly,
            ErrorCode::Incompatible,
            ErrorCode::NoPairingWindow,
        ] {
            let body = ErrorBody::new(code, "context");
            let json = serde_json::to_string(&body).expect("serialise");
            assert!(json.contains(code.as_str()), "{json} should name {code}");
            let back: ErrorBody = serde_json::from_str(&json).expect("parse");
            assert_eq!(back.code, code);
            assert_eq!(back.message, "context");
        }
    }
}
