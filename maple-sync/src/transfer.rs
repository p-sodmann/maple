//! Moving the photographs themselves.
//!
//! P5 and P6 moved knowledge: a servant learned every row the master had, and
//! P6 let it *display* a photo it does not hold by fetching pixels on demand.
//! This module is the remaining half of §3.8 — putting originals on disk —
//! and it is what makes **full** and **partial** mean something:
//!
//! | mode | originals go |
//! |---|---|
//! | full | both ways: the servant downloads what the master has, and uploads what the master lacks |
//! | partial | one way: the servant uploads its own; master-only photos stay relayed |
//! | relay | nowhere. P6 already did everything relay needs |
//!
//! # Both directions are driven by the servant
//!
//! §3.8 says "the master fetches the servant's originals", and the master
//! cannot: it has no client and no idea how to reach a servant, which may be
//! on a different network behind a NAT that only lets connections out. So the
//! *shape* is inverted while the outcome is not — the servant asks
//! `POST /sync/wanted` what the master is missing and uploads it. The master
//! stays a pure server, which is also what keeps a laptop that is asleep from
//! being something the master has to handle.
//!
//! # What is verified, and what cannot be
//!
//! A display file is content-addressed: the hash is in the URL, and both
//! sides check the bytes against it before writing them into a library. A
//! **companion raw is not** — the schema hashes the display file and has no
//! column for a second digest, so `?raw=1` carries bytes nothing can check.
//! They are accepted on the strength of the pairing alone, which is the same
//! trust that already lets a peer push arbitrary metadata rows; the blast
//! radius is one photo's companion. Giving `images` a `raw_hash` column would
//! close it, and that is a schema change, not a transport one.
//!
//! # Nothing is placed until everything has arrived
//!
//! A photo with a companion is two transfers, and a library row can only name
//! one pair of paths. Both are staged in a hidden `.incoming` directory first
//! and moved into place together, so a failure halfway leaves the row
//! untouched and the next pass simply tries again. The alternative — adopting
//! the display file and hoping the raw follows — produces a row that says it
//! has no companion when the sender knows it has one, and nothing would ever
//! revisit it.
//!
//! # The move and the row change under one lock
//!
//! Staging is network I/O and happens with no lock held; the *last* two steps
//! — rename into the library, then point the row at it — are taken together
//! under the database mutex. The library scanner runs every 60 seconds and
//! inserts any file it finds that no row claims, and `images.path` is UNIQUE,
//! so a scan landing between those two steps would insert a row for the file
//! and the adoption would then fail on the constraint — leaving the photo
//! relayed forever with a duplicate row beside it. Holding the lock puts the
//! collision on the scanner's insert instead, where it is a logged no-op:
//! the row it wanted to create already exists. Both steps are cheap (a rename
//! within one directory tree), so the lock is held for microseconds.

use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use maple_db::{Database, MissingOriginal};
use maple_state::PeerMode;

use crate::client::{SyncClient, SyncFailure};
use crate::protocol::{ErrorCode, MAX_ORIG_BYTES};
use crate::trust::PeerKey;

/// Directory inside the library where a transfer stages bytes before filing
/// them.
///
/// Hidden, which is load-bearing rather than tidy: `maple_import::scan` skips
/// dot-directories, so a half-downloaded file is never picked up by the
/// 60-second scanner and inserted as a photo of its own.
pub const STAGING_DIR: &str = ".incoming";

/// Most photos one pass will move in one direction.
///
/// A cap rather than "until the queue is empty" for the same reason
/// `worker::MAX_ROUNDS` is one: a first full sync of a real library is tens
/// of thousands of files, and a pass that ran until it finished would ignore
/// a quit request for hours. [`TransferOutcome::more_pending`] tells the
/// worker to come straight back instead of sleeping out its interval, so the
/// cap costs a round trip, not a day.
pub const MAX_TRANSFERS_PER_PASS: usize = 256;

/// How this device files photos it receives.
///
/// The same three values a card import uses, so a photo that arrives over the
/// wire lands where the same photo would have landed had it been imported
/// here — which is the whole of what §3.8 asks of "full" mode. Passed in
/// rather than read from `settings.toml` here, because a transport crate
/// reading the user's settings file would be a second source of truth for
/// something the UI already owns.
#[derive(Debug, Clone)]
pub struct LibraryLayout {
    pub library_dir: PathBuf,
    pub folder_template: String,
    pub filename_template: String,
}

impl LibraryLayout {
    /// Where a blob waits between arriving and being filed.
    ///
    /// Inside `library_dir` rather than the system temp directory, so the
    /// final move is a rename on the same filesystem instead of a copy of
    /// every byte a second time.
    pub fn staging_dir(&self) -> PathBuf {
        self.library_dir.join(STAGING_DIR)
    }

    /// The staging path for one blob. Named by hash — and by `raw` — so a
    /// retry overwrites its own half-finished attempt rather than
    /// accumulating one file per try.
    pub fn staged_path(&self, hash: &[u8; 32], raw: bool) -> PathBuf {
        let suffix = if raw { "raw" } else { "orig" };
        self.staging_dir()
            .join(format!("{}.{suffix}", crate::protocol::route::hex(hash)))
    }

    /// Move a staged photo — and its companion raw, when one is staged — into
    /// the library under this device's templates.
    ///
    /// One call for the pair, never one per file. `maple_import::place_pair`
    /// carries the reason: the library scanner regroups from disk by
    /// directory and stem, so a companion filed anywhere other than beside
    /// its display file is not a companion at all but a second photograph,
    /// and the scanner mints an `images` row for it that then replicates.
    /// Deriving the companion's own destination could not guarantee
    /// otherwise — the two blobs are staged under synthetic `<hash>.orig` /
    /// `<hash>.raw` names, so the raw's own EXIF is unreadable and its date
    /// came out as the moment it arrived.
    pub fn place(
        &self,
        staged: &Path,
        original_name: &str,
        companion: Option<(&Path, &str)>,
    ) -> anyhow::Result<(PathBuf, Option<PathBuf>)> {
        maple_import::place_pair(
            staged,
            &self.library_dir,
            &self.folder_template,
            &self.filename_template,
            original_name,
            companion,
        )
    }

    /// Drop whatever is staged for `hash`, in either form.
    ///
    /// Called after a successful adoption and after a failed one: a staged
    /// file that is never claimed is a photo-sized leak in a directory nobody
    /// looks at.
    pub fn discard(&self, hash: &[u8; 32]) {
        for raw in [false, true] {
            let path = self.staged_path(hash, raw);
            if path.exists() {
                if let Err(e) = std::fs::remove_file(&path) {
                    tracing::warn!("sync: could not clear {}: {e}", path.display());
                }
            }
        }
    }
}

/// What one pass moved.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct TransferOutcome {
    pub downloaded: usize,
    pub uploaded: usize,
    /// Photos that could not be moved this time — a master that no longer has
    /// the file, or bytes that failed their hash check. Not an error: each
    /// costs one photo, and the next pass tries again.
    pub skipped: usize,
    /// Whether the cap was hit with work still queued, so the worker should
    /// come back at once rather than sleeping out its interval.
    pub more_pending: bool,
}

impl TransferOutcome {
    pub fn moved(&self) -> usize {
        self.downloaded + self.uploaded
    }
}

/// Reports `(files done, files in this batch)` as a transfer runs.
///
/// The pill would otherwise say "Synced" through the whole of a first full
/// sync — metadata finishes in seconds and the photographs take hours, and
/// the honest reading of that is "still syncing".
pub type Progress<'a> = &'a dyn Fn(usize, usize);

/// Run the file half of a sync pass.
///
/// `should_stop` is checked between photos — a transfer is the longest thing
/// a pass does, and a quit request that waited for it would hold the app open
/// for the length of a download.
pub fn transfer(
    db: &Arc<Mutex<Database>>,
    client: &SyncClient,
    key: &PeerKey,
    mode: PeerMode,
    layout: &LibraryLayout,
    should_stop: &dyn Fn() -> bool,
    progress: Progress<'_>,
) -> Result<TransferOutcome, SyncFailure> {
    let mut outcome = TransferOutcome::default();
    if mode == PeerMode::Relay {
        // Nothing to do, and no request made: a relay servant that asked what
        // the master wanted would be offering files it is never going to send.
        return Ok(outcome);
    }

    if mode == PeerMode::Full {
        download(db, client, key, layout, should_stop, progress, &mut outcome)?;
    }
    if should_stop() {
        return Ok(outcome);
    }
    upload(db, client, key, should_stop, progress, &mut outcome)?;
    Ok(outcome)
}

// ── Down: the master's photos onto this disk ────────────────────

fn download(
    db: &Arc<Mutex<Database>>,
    client: &SyncClient,
    key: &PeerKey,
    layout: &LibraryLayout,
    should_stop: &dyn Fn() -> bool,
    progress: Progress<'_>,
    outcome: &mut TransferOutcome,
) -> Result<(), SyncFailure> {
    // Read the whole queue for this pass in one go, then let the lock go: the
    // rest of this function is network I/O, and holding the database mutex
    // across a download would stall the UI thread for the length of it.
    let queue = {
        let guard = lock(db);
        guard
            .originals_to_fetch(MAX_TRANSFERS_PER_PASS + 1)
            .map_err(internal)?
    };
    outcome.more_pending |= queue.len() > MAX_TRANSFERS_PER_PASS;
    let total = queue.len().min(MAX_TRANSFERS_PER_PASS);

    for (done, row) in queue.into_iter().take(MAX_TRANSFERS_PER_PASS).enumerate() {
        progress(done, total);
        if should_stop() {
            outcome.more_pending = true;
            return Ok(());
        }
        match fetch_one(client, key, layout, &row) {
            Ok(true) => match commit(db, layout, &row) {
                Ok(()) => outcome.downloaded += 1,
                Err(e) => {
                    tracing::error!("sync: could not file {}: {e}", row.filename);
                    layout.discard(&row.hash);
                    outcome.skipped += 1;
                }
            },
            Ok(false) => outcome.skipped += 1,
            Err(failure) => {
                layout.discard(&row.hash);
                return Err(failure);
            }
        }
    }
    Ok(())
}

/// Fetch one photo, and its companion when it has one, into staging.
///
/// `Ok(false)` is a photo this pass could not have: the master no longer
/// holds it, or what it sent did not hash to what was asked for. Both are
/// per-photo and neither should stop the pass — a hash *mutates* on lossless
/// rotation, so a servant legitimately asks for one the master has already
/// replaced.
fn fetch_one(
    client: &SyncClient,
    key: &PeerKey,
    layout: &LibraryLayout,
    row: &MissingOriginal,
) -> Result<bool, SyncFailure> {
    let display = match client.blob_orig(key, &row.hash, false) {
        Ok(bytes) => bytes,
        Err(failure) if failure.code == Some(ErrorCode::NotFound) => {
            tracing::info!("sync: master no longer holds {}", row.filename);
            return Ok(false);
        }
        Err(failure) => return Err(failure),
    };

    if maple_import::hash_bytes(&display) != row.hash {
        // Not "corrupt in transit" specifically — a truncated read looks the
        // same as a tampered body — but either way these bytes are not the
        // photograph this row names, and writing them into a library would
        // make the lie permanent.
        tracing::warn!("sync: {} did not match its hash, discarding", row.filename);
        return Ok(false);
    }

    if row.raw_filename.is_some() {
        match client.blob_orig(key, &row.hash, true) {
            Ok(bytes) => {
                stage(layout, &row.hash, true, &bytes).map_err(internal)?;
            }
            Err(failure) if failure.code == Some(ErrorCode::NotFound) => {
                // The row says there is a companion and the master says there
                // is not. Its own scanner will drop the column soon enough;
                // take the display file now rather than blocking on it.
                tracing::info!("sync: no companion for {} after all", row.filename);
            }
            Err(failure) => return Err(failure),
        }
    }

    stage(layout, &row.hash, false, &display).map_err(internal)?;
    Ok(true)
}

/// Move what was staged into the library and point the row at it, with the
/// database lock held across both. See the module docs for why that pairing
/// is not optional.
fn commit(
    db: &Arc<Mutex<Database>>,
    layout: &LibraryLayout,
    row: &MissingOriginal,
) -> anyhow::Result<()> {
    let guard = lock(db);
    let staged_raw = layout.staged_path(&row.hash, true);
    let companion = match (&row.raw_filename, staged_raw.exists()) {
        (Some(name), true) => Some((staged_raw.as_path(), name.as_str())),
        _ => None,
    };
    let (placed, placed_raw) =
        layout.place(&layout.staged_path(&row.hash, false), &row.filename, companion)?;
    guard.adopt_original(row.id, &placed, placed_raw.as_deref())?;
    drop(guard);
    layout.discard(&row.hash);
    Ok(())
}

fn stage(
    layout: &LibraryLayout,
    hash: &[u8; 32],
    raw: bool,
    bytes: &[u8],
) -> anyhow::Result<()> {
    std::fs::create_dir_all(layout.staging_dir())?;
    let mut file = std::fs::File::create(layout.staged_path(hash, raw))?;
    file.write_all(bytes)?;
    Ok(())
}

// ── Up: this disk's photos onto the master ──────────────────────

fn upload(
    db: &Arc<Mutex<Database>>,
    client: &SyncClient,
    key: &PeerKey,
    should_stop: &dyn Fn() -> bool,
    progress: Progress<'_>,
    outcome: &mut TransferOutcome,
) -> Result<(), SyncFailure> {
    let wanted = client.wanted(key, MAX_TRANSFERS_PER_PASS + 1)?;
    outcome.more_pending |= wanted.len() > MAX_TRANSFERS_PER_PASS;
    let total = wanted.len().min(MAX_TRANSFERS_PER_PASS);

    for (done, hash) in wanted.into_iter().take(MAX_TRANSFERS_PER_PASS).enumerate() {
        progress(done, total);
        if should_stop() {
            outcome.more_pending = true;
            return Ok(());
        }
        // The master wants a hash; whether *this* device is the one that can
        // supply it is a local question. In a star topology with two or more
        // servants, most of what a master lacks belongs to somebody else.
        let paths = {
            let guard = lock(db);
            let original = guard.blob_path(&hash, false).map_err(internal)?;
            let raw = guard.blob_path(&hash, true).map_err(internal)?;
            original.map(|path| (path, raw))
        };
        let Some((original, raw)) = paths else {
            continue;
        };

        // The companion goes first and is *staged* on the master, which then
        // files both the moment the display file verifies. Sending it second
        // would be too late: adopting the display flips the row to local, and
        // the master would have nowhere to put a companion any more.
        if let Some(raw) = raw {
            if let Err(e) = send_file(client, key, &hash, true, &raw)? {
                tracing::warn!("sync: master refused the companion for {}: {e}", raw.display());
            }
        }
        match send_file(client, key, &hash, false, &original)? {
            Ok(response) => {
                if response.stored {
                    outcome.uploaded += 1;
                } else {
                    outcome.skipped += 1;
                }
            }
            Err(e) => {
                tracing::warn!("sync: master refused {}: {e}", original.display());
                outcome.skipped += 1;
            }
        }
    }
    Ok(())
}

/// Send one file, streamed from disk.
///
/// The nested `Result` separates the two failures that must not be confused:
/// the outer one is the link (unreachable, unauthorised) and ends the pass;
/// the inner one is this file (the master stopped wanting it between asking
/// and receiving) and costs only this file.
#[allow(clippy::type_complexity)]
fn send_file(
    client: &SyncClient,
    key: &PeerKey,
    hash: &[u8; 32],
    raw: bool,
    path: &Path,
) -> Result<Result<crate::protocol::UploadResponse, String>, SyncFailure> {
    let mut file = match std::fs::File::open(path) {
        Ok(file) => file,
        Err(e) => {
            // The row says the file is here and it is not. The scanner marks
            // it missing within the minute; until then this is one file, not
            // a broken link.
            return Ok(Err(format!("could not open {}: {e}", path.display())));
        }
    };
    // The companion's extension travels with it: the master files a raw
    // beside its display file under that file's stem, so the extension is the
    // only part of this name it needs — and the only way it can learn one for
    // a photo whose row predates `origin_raw_path`.
    let ext = raw
        .then(|| path.extension().and_then(|e| e.to_str()))
        .flatten();
    match client.upload_orig(key, hash, raw, ext, &mut file) {
        Ok(response) => Ok(Ok(response)),
        Err(failure)
            if matches!(
                failure.code,
                Some(ErrorCode::NotFound) | Some(ErrorCode::BadRequest)
            ) =>
        {
            Ok(Err(failure.to_string()))
        }
        Err(failure) => Err(failure),
    }
}

// ── Receiving, on either side ───────────────────────────────────

/// Stream `reader` into `dest`, returning the BLAKE3 of what was written.
///
/// Streamed rather than buffered because this is the one route whose bodies
/// are photographs: a 100 MB raw file would otherwise be held in memory
/// whole, on a machine whose job is to be a passive server. `limit` bounds
/// what a caller can make this write to disk.
pub fn receive_to_file(
    reader: &mut dyn Read,
    dest: &Path,
    limit: u64,
) -> anyhow::Result<[u8; 32]> {
    if let Some(parent) = dest.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut file = std::fs::File::create(dest)?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = vec![0u8; 64 * 1024];
    let mut written = 0u64;
    loop {
        let read = reader.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        written += read as u64;
        if written > limit {
            drop(file);
            let _ = std::fs::remove_file(dest);
            anyhow::bail!("upload exceeds the {limit}-byte limit");
        }
        hasher.update(&buffer[..read]);
        file.write_all(&buffer[..read])?;
    }
    file.sync_all()?;
    Ok(*hasher.finalize().as_bytes())
}

/// The cap on a single uploaded original, matching what a client will read in
/// the other direction.
pub const MAX_UPLOAD_BYTES: u64 = MAX_ORIG_BYTES;

fn internal(error: impl std::fmt::Display) -> SyncFailure {
    SyncFailure {
        kind: crate::backoff::FailureKind::Unreachable,
        code: None,
        message: error.to_string(),
    }
}

fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}
