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

/// Largest thumbnail a client will read from `/blob/thumb/`.
///
/// A thumbnail is ~10 KB. This is not a tuning knob; it is the point past
/// which whatever is answering is not a Maple master, and the client should
/// stop reading rather than fill memory on its word.
pub const MAX_THUMB_BYTES: u64 = 4 * 1024 * 1024;

/// Largest original a client will read from `/blob/orig/`.
///
/// Full-res loading is memory-only by the relay contract (§3.6), so this
/// number is an allocation the servant actually makes. Generous enough for a
/// large raw file, bounded so a hostile or broken master cannot stream until
/// the app dies.
pub const MAX_ORIG_BYTES: u64 = 256 * 1024 * 1024;

/// Every route the master serves.
///
/// The two blob routes are **prefixes**: the content hash is the rest of the
/// path, hex-encoded. Content addressing rather than a row id is what makes a
/// transfer dedupable — two libraries that both hold a photo ask for the same
/// URL — and it means a servant can request exactly the bytes it lacks
/// without either side agreeing on rowids.
pub mod route {
    pub const HELLO: &str = "/sync/hello";
    pub const PAIR_CLAIM: &str = "/pair/claim";
    pub const PULL: &str = "/sync/pull";
    pub const PUSH: &str = "/sync/push";

    /// `GET /blob/thumb/{hex_hash}` → WebP, rendered on the master if it has
    /// no cached copy.
    pub const BLOB_THUMB: &str = "/blob/thumb/";
    /// `GET /blob/orig/{hex_hash}[?raw=1]` → the original file's bytes,
    /// streamed. `?raw=1` asks for the companion raw file instead.
    ///
    /// `POST` to the same URL **sends** those bytes, which is how a photo
    /// gets from a servant to its master (§3.8). One URL for both directions
    /// is not a shortcut: the hash in it is what the receiver checks the
    /// bytes against, so read and write name a blob the same way or neither
    /// can be verified.
    pub const BLOB_ORIG: &str = "/blob/orig/";

    /// `POST /sync/wanted` → the hashes the answering device is missing.
    ///
    /// A master never dials a servant — it does not know how to reach one —
    /// so "the master fetches the servant's originals" happens as the servant
    /// asking what is wanted and uploading it.
    pub const WANTED: &str = "/sync/wanted";

    /// Build a blob path. The MAC covers the path exactly as written here,
    /// query string included, so client and server must never build it two
    /// different ways.
    pub fn blob(prefix: &str, hash: &[u8; 32], raw: bool) -> String {
        let mut out = String::with_capacity(prefix.len() + 64 + 6);
        out.push_str(prefix);
        out.push_str(&hex(hash));
        if raw {
            out.push_str("?raw=1");
        }
        out
    }

    /// A hash as it appears in a URL or a `wanted` list.
    pub fn hex(hash: &[u8; 32]) -> String {
        let mut out = String::with_capacity(64);
        for byte in hash {
            out.push_str(&format!("{byte:02x}"));
        }
        out
    }

    /// Inverse of [`hex`]. `None` for anything that is not 64 hex characters,
    /// on the same reasoning as [`blob_hash`]: a hash is fixed-width, so a
    /// short one is a malformed request rather than a miss.
    pub fn unhex(text: &str) -> Option<[u8; 32]> {
        if text.len() != 64 {
            return None;
        }
        let mut out = [0u8; 32];
        for (i, byte) in out.iter_mut().enumerate() {
            *byte = u8::from_str_radix(text.get(i * 2..i * 2 + 2)?, 16).ok()?;
        }
        Some(out)
    }

    /// Parse the hex hash out of a blob path. `None` if it is not 64 hex
    /// characters — a hash is fixed-width, so anything else is a malformed
    /// request rather than a miss.
    pub fn blob_hash(path: &str, prefix: &str) -> Option<[u8; 32]> {
        unhex(path.strip_prefix(prefix)?)
    }
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
    /// The caller's file mode for this link, as `PeerMode::as_str`.
    ///
    /// The mode is the *servant's* setting — it is the servant's disk that
    /// fills or does not — and it is chosen in the servant's settings card.
    /// Reporting it here is what keeps the master's own peer list honest
    /// instead of showing the `relay` that pairing defaulted to forever.
    /// Optional so a master can read a P6 servant's request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode: Option<String>,
}

/// `POST /sync/wanted` — "which hashes are you missing?".
///
/// Asked by a servant in **full** or **partial** mode, which is why the
/// answer needs no mode of its own: a device that is not going to upload
/// anything never asks.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WantedRequest {
    /// How many hashes to name. The server clamps it.
    pub limit: usize,
}

/// The answer: hashes of photos the master lists but does not hold.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct WantedResponse {
    /// Hex, lower case, 64 characters each — [`route::hex`].
    pub hashes: Vec<String>,
}

impl WantedResponse {
    /// Decode, dropping anything malformed rather than failing the pass.
    ///
    /// A hash that does not parse is one blob not transferred; refusing the
    /// whole list would stop a working link over a single bad entry.
    pub fn decoded(&self) -> Vec<[u8; 32]> {
        self.hashes.iter().filter_map(|h| route::unhex(h)).collect()
    }
}

/// `POST /blob/orig/{hash}` — what the receiver did with the bytes.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct UploadResponse {
    /// Whether the file was filed into the library. `false` for a companion
    /// raw, which waits, staged, for the display file it belongs to.
    pub stored: bool,
    /// Where it landed, for the log on the sending side. Never used to
    /// derive anything — it names a path on the *other* machine.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub path: Option<String>,
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
    /// No blob with that hash here. Emphatically **not** fatal: a hash
    /// *mutates* when a photo is losslessly rotated, so a servant can hold a
    /// row whose hash the master has already replaced. The next metadata sync
    /// carries the new hash and the fetch succeeds — retrying is exactly
    /// right, and stopping the link over one missing thumbnail would not be.
    NotFound,
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
            Self::NotFound => 404,
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
            Self::NotFound => "not_found",
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
            ErrorCode::NotFound,
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
    fn a_blob_path_round_trips_through_its_own_parser() {
        // Client and server must agree byte for byte: the MAC covers this
        // string, so a mismatch presents as an authentication failure.
        let hash = [0xABu8; 32];
        let path = route::blob(route::BLOB_THUMB, &hash, false);
        assert_eq!(path, format!("/blob/thumb/{}", "ab".repeat(32)));
        assert_eq!(route::blob_hash(&path, route::BLOB_THUMB), Some(hash));

        let raw = route::blob(route::BLOB_ORIG, &hash, true);
        assert!(raw.ends_with("?raw=1"), "{raw}");
        // The parser is handed the path with the query already split off, as
        // the dispatcher does.
        let bare = raw.split('?').next().unwrap();
        assert_eq!(route::blob_hash(bare, route::BLOB_ORIG), Some(hash));
    }

    #[test]
    fn a_hash_that_is_not_64_hex_characters_is_refused() {
        // Truncating is the difference between "no such blob" and "serve
        // whichever photo happens to share this prefix".
        assert_eq!(route::blob_hash("/blob/thumb/abcd", route::BLOB_THUMB), None);
        assert_eq!(
            route::blob_hash(&format!("/blob/thumb/{}", "zz".repeat(32)), route::BLOB_THUMB),
            None
        );
        assert_eq!(route::blob_hash("/sync/pull", route::BLOB_THUMB), None);
    }

    #[test]
    fn a_missing_blob_never_stops_the_link() {
        // A rotated photo changes hash, so the servant asks for one the
        // master no longer has. That must cost one thumbnail, not the pairing.
        assert!(!ErrorCode::NotFound.is_fatal());
        assert_eq!(ErrorCode::NotFound.http_status(), 404);
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
