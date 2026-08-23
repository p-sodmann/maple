//! The servant's HTTP client.
//!
//! A blocking `ureq` agent, mirroring [`crate::server`] route for route. It
//! signs everything the master requires a signature for and turns every
//! failure into a [`SyncFailure`] the worker can act on without knowing
//! anything about HTTP.
//!
//! # Why failures are classified here and not in the worker
//!
//! §1.4 hinges on one distinction: a rejected credential must never be
//! retried, and everything else must be. Making that call needs the HTTP
//! status, the error body's [`ErrorCode`], *and* the knowledge that a
//! transport-level error (connection refused, timeout) is by definition not
//! an authentication problem. All three live here; the worker sees only
//! [`SyncFailure::kind`] and asks the backoff what to do.
//!
//! # Determinism
//!
//! The clock and the random source are injected, as everywhere else in this
//! crate. A signature is `keyed_hash(key, method ‖ path ‖ unix_ms ‖ nonce ‖
//! blake3(body))`, so pinning both makes a request reproducible byte for
//! byte — which is what lets a test assert that the *server* rejects a replay
//! rather than that two random requests happened to differ.

use std::time::Duration;

use maple_db::SyncBatch;

use crate::auth::SignedRequest;
use crate::backoff::FailureKind;
use crate::pairing::{ClaimRequest, ClaimResponse};
use crate::protocol::{
    route, ErrorBody, ErrorCode, Hello, PullRequest, PushResponse, UploadResponse, WantedRequest,
    WantedResponse, MAX_ORIG_BYTES, MAX_THUMB_BYTES, PROTOCOL_VERSION,
};
use crate::random::SharedRandom;
use crate::server::Clock;
use crate::trust::PeerKey;
use maple_state::PeerMode;

/// How long a single request may take.
///
/// Generous because a first pull on a large library serialises a lot of rows
/// on the master before a byte comes back, and tight enough that a servant on
/// a dead network reaches its retry rather than hanging until the user
/// notices. The `image_loader` watchdog uses 30 s for the same reason; sync
/// batches are bigger, so this is longer.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(120);

/// Ask the master to close the connection once it has answered.
///
/// `tiny_http` reads each accepted connection on a thread from a pool, and
/// the task holding a keep-alive connection stays parked in `read()` waiting
/// for a *second* request that a one-shot client never sends. The pool grows
/// when every thread is busy, but a burst of connections arriving while one
/// thread is still idle all queue behind that one thread — and it pins itself
/// on the first of them. The rest are accepted and never parsed: the server's
/// loop looks idle, the request bytes sit unread in the socket, and the
/// caller waits out its whole timeout for a reply nobody is writing.
///
/// It is a race, so it is intermittent, and nothing hit it before P6 — one
/// worker, one connection, every five minutes. A relay servant's grid opens
/// one per tile and leaves them pooled afterwards, which is the shape that
/// triggers it; it was reproducible against two running instances, showing up
/// as three of six tiles blank until `ureq` gave up two minutes later.
/// Closing after each response costs one handshake per request on a LAN and
/// means no task ever parks on an idle socket.
const CLOSE: (&str, &str) = ("connection", "close");

/// A failed request, already sorted into "retry" or "stop".
#[derive(Debug, Clone)]
pub struct SyncFailure {
    pub kind: FailureKind,
    /// The master's code, when it sent one. Absent for transport failures.
    pub code: Option<ErrorCode>,
    pub message: String,
}

impl SyncFailure {
    fn transport(message: impl Into<String>) -> Self {
        Self {
            kind: FailureKind::Unreachable,
            code: None,
            message: message.into(),
        }
    }

    fn from_code(code: ErrorCode, message: impl Into<String>) -> Self {
        Self {
            kind: if code.is_fatal() {
                FailureKind::Auth
            } else {
                FailureKind::Unreachable
            },
            code: Some(code),
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SyncFailure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.code {
            Some(code) => write!(f, "{code}: {}", self.message),
            None => f.write_str(&self.message),
        }
    }
}

impl std::error::Error for SyncFailure {}

/// Talks to one master.
pub struct SyncClient {
    /// `http://host:port`, no trailing slash.
    base: String,
    device_id: String,
    agent: ureq::Agent,
    clock: Clock,
    rng: SharedRandom,
}

impl SyncClient {
    /// The URL this client dials, for a message that has to name it. Not the
    /// `host:port` it was built from — that is the caller's to remember.
    pub fn address(&self) -> &str {
        &self.base
    }

    /// `address` is `host:port`; the scheme is added here because there is
    /// only one — the transport is plain HTTP by design (§2.2), and letting a
    /// caller pass `https://` would promise a confidentiality this protocol
    /// does not provide.
    pub fn new(
        address: &str,
        device_id: impl Into<String>,
        clock: Clock,
        rng: SharedRandom,
    ) -> Self {
        Self::with_timeout(address, device_id, clock, rng, REQUEST_TIMEOUT)
    }

    /// Same, with an explicit per-request timeout.
    ///
    /// Pairing wants a much shorter one than a sync pass: the modal polls
    /// `/pair/claim` once a second while waiting for the other user to type,
    /// and a two-minute timeout there would mean a mistyped address hangs the
    /// whole window instead of failing while the codes are still valid.
    pub fn with_timeout(
        address: &str,
        device_id: impl Into<String>,
        clock: Clock,
        rng: SharedRandom,
        timeout: Duration,
    ) -> Self {
        let config = ureq::Agent::config_builder()
            .timeout_global(Some(timeout))
            // A non-2xx must come back as a *response*, not an error: the
            // master's `ErrorBody` is the only thing that says whether this
            // failure is worth retrying, and ureq's default turns the status
            // into an error that has thrown the body away.
            .http_status_as_error(false)
            .build();
        Self {
            base: format!("http://{}", address.trim().trim_end_matches('/')),
            device_id: device_id.into(),
            agent: config.into(),
            clock,
            rng,
        }
    }

    /// `GET /sync/hello` — unsigned, so it answers whether or not we are
    /// paired. That is the point: it separates "nothing there" from "there
    /// but rejecting me".
    pub fn hello(&self) -> Result<Hello, SyncFailure> {
        let response = self
            .agent
            .get(&self.url(route::HELLO))
            .header(CLOSE.0, CLOSE.1)
            .call()
            .map_err(to_failure)?;
        decode(response)
    }

    /// Check that a master we can reach is one we can actually merge with.
    ///
    /// Called before the first pass rather than trusted implicitly: applying
    /// a batch from a peer whose schema differs would write rows this build
    /// cannot represent, and unlike a network failure that is not something
    /// a retry recovers from — it needs the other machine updated.
    pub fn check_compatible(hello: &Hello) -> Result<(), SyncFailure> {
        if hello.protocol != PROTOCOL_VERSION {
            return Err(SyncFailure::from_code(
                ErrorCode::Incompatible,
                format!(
                    "peer speaks sync protocol {} but this build speaks {PROTOCOL_VERSION}",
                    hello.protocol
                ),
            ));
        }
        Ok(())
    }

    /// `POST /pair/claim` — unsigned; there is no key yet, which is the whole
    /// reason this route exists.
    pub fn pair_claim(&self, request: &ClaimRequest) -> Result<ClaimResponse, SyncFailure> {
        let body = serde_json::to_vec(request)
            .map_err(|e| SyncFailure::transport(format!("could not encode claim: {e}")))?;
        let response = self
            .agent
            .post(&self.url(route::PAIR_CLAIM))
            .header(CLOSE.0, CLOSE.1)
            .content_type("application/json")
            .send(&body[..])
            .map_err(to_failure)?;
        decode(response)
    }

    /// `POST /sync/pull` — everything the master has stamped above `since`.
    ///
    /// `mode` rides along so the master's settings card can show what this
    /// link actually does with files, rather than the `relay` that pairing
    /// defaulted to. It is told, not asked: the servant's disk is the one at
    /// stake, so the servant's setting is the one that counts.
    pub fn pull(
        &self,
        key: &PeerKey,
        since: i64,
        max_revs: usize,
        mode: PeerMode,
    ) -> Result<SyncBatch, SyncFailure> {
        self.signed_post(
            key,
            route::PULL,
            &PullRequest {
                since,
                max_revs,
                mode: Some(mode.as_str().to_owned()),
            },
        )
    }

    /// `POST /sync/wanted` — hashes the master lists but does not hold.
    pub fn wanted(&self, key: &PeerKey, limit: usize) -> Result<Vec<[u8; 32]>, SyncFailure> {
        let response: WantedResponse =
            self.signed_post(key, route::WANTED, &WantedRequest { limit })?;
        Ok(response.decoded())
    }

    /// `POST /sync/push` — merge our batch into the master.
    pub fn push(&self, key: &PeerKey, batch: &SyncBatch) -> Result<PushResponse, SyncFailure> {
        self.signed_post(key, route::PUSH, batch)
    }

    /// `GET /blob/thumb/{hash}` — a WebP thumbnail for one photo.
    ///
    /// Sized and encoded by the *master*, not by this device's thumbnail
    /// settings: it is the master that holds the file, and asking it to
    /// re-render per servant preference would trade a visible improvement
    /// nobody asked for against a cache hit on every request.
    pub fn blob_thumb(&self, key: &PeerKey, hash: &[u8; 32]) -> Result<Vec<u8>, SyncFailure> {
        let path = route::blob(route::BLOB_THUMB, hash, false);
        self.signed_get(key, &path, MAX_THUMB_BYTES)
    }

    /// `GET /blob/orig/{hash}[?raw=1]` — the original file's bytes.
    pub fn blob_orig(
        &self,
        key: &PeerKey,
        hash: &[u8; 32],
        raw: bool,
    ) -> Result<Vec<u8>, SyncFailure> {
        let path = route::blob(route::BLOB_ORIG, hash, raw);
        self.signed_get(key, &path, MAX_ORIG_BYTES)
    }

    /// `POST /blob/orig/{hash}[?raw=1]` — send an original to the master.
    ///
    /// # Why this one signs an empty body
    ///
    /// Every other signed request MACs a hash of its body, which is what
    /// forces the server to buffer the whole thing before it can decide
    /// whether the caller is anyone. Doing that here would mean holding a
    /// 100 MB raw file in the master's memory to check a signature — for the
    /// one route whose bodies are photographs.
    ///
    /// It is unnecessary, and only because the blob is *content-addressed*:
    /// the hash is in the path, the path is signed, and the master hashes
    /// what arrives and rejects it unless the two match. A tampered body
    /// fails that check exactly as it would fail a MAC, so the master can
    /// stream the upload to disk and verify as it goes. What the signature
    /// still buys is that an unpaired machine cannot make it write anything
    /// at all.
    ///
    /// The exception is `raw = true`: the schema hashes the display file, not
    /// its companion, so there is nothing to check a raw upload against. See
    /// [`crate::transfer`].
    pub fn upload_orig(
        &self,
        key: &PeerKey,
        hash: &[u8; 32],
        raw: bool,
        body: &mut dyn std::io::Read,
    ) -> Result<UploadResponse, SyncFailure> {
        let path = route::blob(route::BLOB_ORIG, hash, raw);
        let credential = SignedRequest::sign_with(
            key,
            self.device_id.clone(),
            "POST",
            &path,
            &[],
            (self.clock)(),
            &self.rng,
        )
        .map_err(|e| SyncFailure::transport(format!("could not sign request: {e}")))?;

        let response = self
            .agent
            .post(&self.url(&path))
            .header("Authorization", &credential.header())
            .header(CLOSE.0, CLOSE.1)
            .content_type("application/octet-stream")
            // Streamed, not buffered: `send` with a reader means a 100 MB raw
            // file crosses the wire in chunks rather than sitting in this
            // servant's memory beside the copy already on its disk.
            .send(ureq::SendBody::from_reader(body))
            .map_err(to_failure)?;
        decode(response)
    }

    /// A signed `GET` whose response is bytes rather than JSON.
    ///
    /// A `GET` carries no body, so the MAC covers method and path only —
    /// which is exactly why `path` here must be the string that also goes
    /// into the URL, query string and all. `route::blob` builds both from one
    /// place for that reason.
    fn signed_get(
        &self,
        key: &PeerKey,
        path: &str,
        limit: u64,
    ) -> Result<Vec<u8>, SyncFailure> {
        let credential = SignedRequest::sign_with(
            key,
            self.device_id.clone(),
            "GET",
            path,
            &[],
            (self.clock)(),
            &self.rng,
        )
        .map_err(|e| SyncFailure::transport(format!("could not sign request: {e}")))?;

        let mut response = self
            .agent
            .get(&self.url(path))
            .header("Authorization", &credential.header())
            .header(CLOSE.0, CLOSE.1)
            .call()
            .map_err(to_failure)?;

        let status = response.status().as_u16();
        if !(200..300).contains(&status) {
            // The error body is JSON and small; read it as text so `classify`
            // can find the code that says whether this is worth retrying.
            let text = response
                .body_mut()
                .read_to_string()
                .unwrap_or_else(|e| format!("<unreadable error body: {e}>"));
            return Err(classify(status, &text));
        }

        response
            .body_mut()
            .with_config()
            .limit(limit)
            .read_to_vec()
            .map_err(|e| SyncFailure::transport(format!("could not read blob: {e}")))
    }

    fn signed_post<B: serde::Serialize, R: serde::de::DeserializeOwned>(
        &self,
        key: &PeerKey,
        path: &str,
        body: &B,
    ) -> Result<R, SyncFailure> {
        let body = serde_json::to_vec(body)
            .map_err(|e| SyncFailure::transport(format!("could not encode request: {e}")))?;

        // The MAC covers the path as written in the URL, so the two must be
        // built from the same string — a normalisation applied to one and not
        // the other fails every request with a signature error that looks
        // like a wrong key.
        let credential = SignedRequest::sign_with(
            key,
            self.device_id.clone(),
            "POST",
            path,
            &body,
            (self.clock)(),
            &self.rng,
        )
        .map_err(|e| SyncFailure::transport(format!("could not sign request: {e}")))?;

        let response = self
            .agent
            .post(&self.url(path))
            .header("Authorization", &credential.header())
            .header(CLOSE.0, CLOSE.1)
            .content_type("application/json")
            .send(&body[..])
            .map_err(to_failure)?;
        decode(response)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }
}

// ── Response handling ───────────────────────────────────────────

/// Read a response, mapping a non-2xx status to a classified failure.
fn decode<T: serde::de::DeserializeOwned>(
    mut response: ureq::http::Response<ureq::Body>,
) -> Result<T, SyncFailure> {
    let status = response.status().as_u16();
    let text = response
        .body_mut()
        .read_to_string()
        .map_err(|e| SyncFailure::transport(format!("could not read response: {e}")))?;

    if !(200..300).contains(&status) {
        return Err(classify(status, &text));
    }
    serde_json::from_str(&text)
        .map_err(|e| SyncFailure::transport(format!("could not parse response: {e}")))
}

/// Decide whether a failed response is worth retrying.
///
/// The master's own [`ErrorBody`] decides it. When the body is missing or
/// unparseable — a proxy in the way, some other service on the port — the
/// failure is treated as **retryable**, because we then have no evidence that
/// our credential is bad. Declaring `Re-pair required` on a stray 403 from
/// something that is not the master would send the user to fix a pairing that
/// was never broken, and re-pairing would not clear it.
fn classify(status: u16, body: &str) -> SyncFailure {
    match serde_json::from_str::<ErrorBody>(body) {
        Ok(error) => SyncFailure::from_code(error.code, error.message),
        Err(_) => SyncFailure::transport(format!(
            "master answered HTTP {status} with no usable error body"
        )),
    }
}

/// Transport-level failure: connection refused, DNS, timeout, TLS. Never an
/// authentication problem — nothing got far enough to check a signature.
fn to_failure(error: ureq::Error) -> SyncFailure {
    SyncFailure::transport(error.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn a_rejected_credential_is_the_one_thing_that_stops_retrying() {
        let body = serde_json::to_string(&ErrorBody::new(
            ErrorCode::Unauthorized,
            "no key for this device",
        ))
        .unwrap();
        let failure = classify(401, &body);
        assert_eq!(failure.kind, FailureKind::Auth);
        assert_eq!(failure.code, Some(ErrorCode::Unauthorized));
    }

    #[test]
    fn a_skewed_clock_shares_the_status_but_not_the_verdict() {
        // Both are 401. Only the body separates "your key is dead" from
        // "your clock is wrong", and the second heals itself.
        let body =
            serde_json::to_string(&ErrorBody::new(ErrorCode::StaleTimestamp, "too old")).unwrap();
        let failure = classify(401, &body);
        assert_eq!(failure.kind, FailureKind::Unreachable);
    }

    #[test]
    fn an_unparseable_error_body_is_retryable() {
        // A 403 from a captive portal or a proxy is not evidence that our
        // pairing is broken, and treating it as fatal would strand a link
        // that re-pairing cannot fix because nothing was wrong with it.
        let failure = classify(403, "<html>Forbidden</html>");
        assert_eq!(failure.kind, FailureKind::Unreachable);
        assert_eq!(failure.code, None);
        assert!(failure.message.contains("403"), "{}", failure.message);
    }

    #[test]
    fn a_protocol_mismatch_is_reported_but_still_retryable() {
        // The other machine may be updated while this one keeps running.
        let hello = Hello {
            device_id: "dev-m".into(),
            name: "Workstation".into(),
            role: "master".into(),
            protocol: PROTOCOL_VERSION + 1,
            schema_version: 19,
        };
        let failure = SyncClient::check_compatible(&hello).expect_err("must refuse");
        assert_eq!(failure.code, Some(ErrorCode::Incompatible));
        assert_eq!(failure.kind, FailureKind::Unreachable);
    }

    #[test]
    fn a_matching_protocol_is_accepted() {
        let hello = Hello {
            device_id: "dev-m".into(),
            name: "Workstation".into(),
            role: "master".into(),
            protocol: PROTOCOL_VERSION,
            schema_version: 19,
        };
        assert!(SyncClient::check_compatible(&hello).is_ok());
    }

    #[test]
    fn the_base_url_is_plain_http_and_has_no_trailing_slash() {
        // https:// would promise a confidentiality this protocol does not
        // provide; a trailing slash would put `//sync/pull` in the MAC.
        let client = SyncClient::new(
            " 192.168.1.20:7645/ ",
            "dev-s",
            Arc::new(|| 0),
            Arc::new(crate::FnRandom(|buf: &mut [u8]| {
                buf.fill(0);
                Ok(())
            })),
        );
        assert_eq!(client.url(route::PULL), "http://192.168.1.20:7645/sync/pull");
    }
}
