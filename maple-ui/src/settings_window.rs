//! Settings window controller (Slint port of views/settings_window.rs).
//!
//! A second top-level Window held as a `thread_local!` singleton that shows
//! read-only configuration info, offers destructive clear actions (AI
//! descriptions, thumbnail cache), and — since P4 — owns the Sync card.
//!
//! # The first write-back in this window
//!
//! Everything else here is read-only display. The Sync card is the first
//! control that persists anything, and almost all of it persists to the
//! *database* rather than to `settings.toml`: role and device name live in
//! `sync_identity`, per-peer mode in `sync_peers`. See
//! `maple_state::sync` for why, and `maple_db::sync_peers` for the accessors.
//! `Settings::save` is deliberately never called from here — it rewrites the
//! file wholesale and discards the user's comments.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{ComponentHandle, Model, SharedString, Timer, TimerMode};

use maple_state::SyncRole;
use maple_sync::TrustedPeer;

use crate::services::settings as settings_service;
use crate::sync_pairing::{PairingController, PairingDeps, PairingSide};
use crate::sync_supervisor::SyncSupervisor;
use crate::{PairingDeviceItem, SettingsWindow, SyncPeerItem};

/// How often the pairing modal's countdown refreshes.
const PAIRING_TICK: Duration = Duration::from_secs(1);

/// The window plus the countdown timer that must outlive the callback that
/// started it — a `slint::Timer` that is dropped stops immediately.
///
/// The [`PairingController`] itself needs no slot here: the modal's callbacks
/// each hold an `Rc` clone of it, so it lives exactly as long as they do.
struct SettingsHandle {
    window: SettingsWindow,
    _pairing_timer: Rc<RefCell<Option<Timer>>>,
}

thread_local! {
    static SETTINGS: RefCell<Option<SettingsHandle>> = const { RefCell::new(None) };
}

/// Open (or reuse) the settings window, syncing the current dark-mode state.
///
/// Takes the sync supervisor alongside `db`: the card does not merely display
/// the role, it changes it, and a role change has to start or stop a real
/// listener or worker rather than only repaint a radio button.
pub fn open(db: Arc<Mutex<maple_db::Database>>, sync: Rc<SyncSupervisor>, is_dark: bool) {
    if SETTINGS.with(|s| s.borrow().is_none()) {
        match build(db.clone(), sync.clone()) {
            Ok(handle) => SETTINGS.with(|cell| *cell.borrow_mut() = Some(handle)),
            Err(e) => {
                tracing::error!("Failed to build settings window: {e}");
                return;
            }
        }
    }
    SETTINGS.with(|cell| {
        let guard = cell.borrow();
        if let Some(handle) = guard.as_ref() {
            let win = &handle.window;
            win.set_dark(is_dark);
            populate(win);
            populate_sync(win, &db, &sync);
            if let Err(e) = win.show() {
                tracing::error!("Failed to show settings window: {e}");
            }
        }
    });
}

/// Refresh the destination-path summary rows, if the settings window is
/// currently open. Called by `path_template_window` after a save so the
/// two windows stay in sync without a full re-`open()`.
pub fn refresh_path_template_display() {
    SETTINGS.with(|s| {
        let guard = s.borrow();
        if let Some(win) = guard.as_ref().map(|h| &h.window) {
            let settings = maple_state::Settings::load();
            win.set_path_template_folder(if settings.path_template.folder.is_empty() {
                "(flat)".into()
            } else {
                settings.path_template.folder.into()
            });
            win.set_path_template_filename(SharedString::from(settings.path_template.filename));
        }
    });
}

/// Propagate a theme change to the settings window while it is open.
pub fn set_dark(dark: bool) {
    SETTINGS.with(|s| {
        let guard = s.borrow();
        if let Some(handle) = guard.as_ref() {
            handle.window.set_dark(dark);
        }
    });
}

fn build(
    db: Arc<Mutex<maple_db::Database>>,
    sync: Rc<SyncSupervisor>,
) -> Result<SettingsHandle, slint::PlatformError> {
    let window = SettingsWindow::new()?;

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_clear_ai_descriptions({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let result = settings_service::clear_ai_descriptions(&db);
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Some(n) => format!("Cleared {n} AI description{}.", if n == 1 { "" } else { "s" }),
                    None => "Failed to clear AI descriptions.".to_owned(),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    window.on_delete_all_face_data({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let result = settings_service::clear_face_data(&db);
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Some((faces, persons)) => format!(
                        "Deleted {faces} face{} and {persons} person{}. \
                         Re-detection will run in the background.",
                        if faces == 1 { "" } else { "s" },
                        if persons == 1 { "" } else { "s" },
                    ),
                    None => "Failed to delete face data.".to_owned(),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    window.on_open_debug_compare({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            crate::debug_compare::open(db.clone(), is_dark);
        }
    });

    window.on_open_path_template({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            crate::path_template_window::open(db.clone(), is_dark);
        }
    });

    window.on_clear_thumbnail_cache({
        let w = window.as_weak();
        move || {
            // The thumbnail cache path comes from settings; clearing means
            // removing all files in the .thumbcache directory.
            let settings = maple_state::Settings::load();
            let cache_dir = settings.library_dir.join(".thumbcache");
            let result = std::fs::remove_dir_all(&cache_dir)
                .and_then(|_| std::fs::create_dir_all(&cache_dir));
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Ok(_) => "Thumbnail cache cleared.".to_owned(),
                    Err(e) => format!("Failed to clear cache: {e}"),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    let handle = wire_sync(&window, db, sync);

    Ok(handle)
}

/// Populate the window with current settings values.
fn populate(window: &SettingsWindow) {
    let s = maple_state::Settings::load();
    window.set_library_dir(s.library_dir.to_string_lossy().into_owned().into());
    window.set_database_path(s.database_path.to_string_lossy().into_owned().into());
    let cache = s.library_dir.join(".thumbcache");
    window.set_cache_path(cache.to_string_lossy().into_owned().into());
    window.set_thumbnail_quality(
        SharedString::from(format!("{}%", s.thumbnails.quality))
    );
    window.set_path_template_folder(
        if s.path_template.folder.is_empty() {
            "(flat)".into()
        } else {
            s.path_template.folder.clone().into()
        },
    );
    window.set_path_template_filename(SharedString::from(s.path_template.filename.clone()));
    window.set_ai_endpoint(
        if s.ai.enabled {
            s.ai.server_url.clone().into()
        } else {
            SharedString::new()
        },
    );
    window.set_face_threshold(
        if s.face.enabled {
            format!("{:.0}%", s.face.similarity_threshold * 100.0).into()
        } else {
            SharedString::new()
        },
    );
    window.set_status_text(SharedString::new());
}

// ── Sync card ───────────────────────────────────────────────────

/// Wire every Sync-card and pairing-modal callback, and return the handle
/// that keeps the countdown timer alive.
fn wire_sync(
    window: &SettingsWindow,
    db: Arc<Mutex<maple_db::Database>>,
    sync: Rc<SyncSupervisor>,
) -> SettingsHandle {
    // The same slot the master's listener answers claims from — see
    // `maple_sync::pairing::PairingSlot`. Sharing it is what lets a code
    // typed here verify a proof arriving on another thread.
    let pairing = Rc::new(RefCell::new(PairingController::new(sync.pairing_slot())));
    let pairing_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));

    window.on_sync_role_selected({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        move |role_text| {
            let role = SyncRole::parse(&role_text);
            let Some(w) = w.upgrade() else { return };
            {
                let guard = maple_db::lock_db(&db);
                if let Err(e) = guard.set_sync_role(role) {
                    tracing::error!("Failed to set sync role: {e}");
                    w.set_status_text(SharedString::from(format!("Could not change role: {e}")));
                    return;
                }
            }
            // Not just a repaint: this stops whichever of the listener and
            // the worker was running and starts the other, or neither.
            sync.restart();
            populate_sync(&w, &db, &sync);
        }
    });

    window.on_sync_rename_device({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        move |name| {
            let Some(w) = w.upgrade() else { return };
            let mut renamed = false;
            {
                let guard = maple_db::lock_db(&db);
                match guard.set_device_name(&name) {
                    Ok(()) => {
                        w.set_status_text(SharedString::from("Device renamed."));
                        renamed = true;
                    }
                    Err(e) => {
                        tracing::error!("Failed to rename device: {e}");
                        w.set_status_text(SharedString::from(format!("Could not rename: {e}")));
                    }
                }
            }
            if renamed {
                // The name is baked into the mDNS record, so a master that
                // does not re-publish keeps introducing itself by its old
                // name in every pick-list on the network. Only on success:
                // a rename that failed left the stored name alone, and the
                // record already carries it.
                sync.renamed();
            }
            populate_sync(&w, &db, &sync);
        }
    });

    window.on_sync_peer_mode_cycled({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        move |device_id| {
            let Some(w) = w.upgrade() else { return };
            {
                let guard = maple_db::lock_db(&db);
                // Read-then-write rather than trusting the displayed chip:
                // the row the user clicked was rendered from a list that may
                // be a repaint behind the database.
                let current = match guard.sync_peer(&device_id) {
                    Ok(Some(peer)) => peer.mode,
                    Ok(None) => return,
                    Err(e) => {
                        tracing::error!("Failed to read peer {device_id}: {e}");
                        return;
                    }
                };
                if guard.sync_role().unwrap_or_default() != SyncRole::Servant {
                    // The chip is not clickable on a master, but the callback
                    // is reachable from the markup and the write would look
                    // like it took until the servant's next pull put it back.
                    tracing::debug!("Ignoring a mode change on a master: the servant owns it");
                    return;
                }
                if let Err(e) = guard.set_sync_peer_mode(&device_id, current.next()) {
                    tracing::error!("Failed to set peer mode: {e}");
                }
            }
            populate_sync(&w, &db, &sync);
        }
    });

    window.on_sync_unpair({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        move |device_id| {
            let Some(w) = w.upgrade() else { return };
            let message = unpair(&db, &sync, &device_id);
            w.set_status_text(SharedString::from(message));
            // A running worker is holding a key that no longer exists.
            sync.restart();
            populate_sync(&w, &db, &sync);
        }
    });

    // ── Pairing modal ───────────────────────────────────────────

    window.on_pairing_begin({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        let pairing = pairing.clone();
        let pairing_timer = pairing_timer.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let deps = pairing_deps(&db, &sync);

            if let Err(e) = pairing.borrow_mut().open(&deps, maple_sync::now_ms()) {
                tracing::error!("Failed to open pairing window: {e}");
                w.set_status_text(SharedString::from(format!("Could not start pairing: {e}")));
                return;
            }

            w.set_pairing_entered(SharedString::new());
            w.set_pairing_address(SharedString::from(suggested_address(&sync)));
            pairing
                .borrow_mut()
                .set_address(w.get_pairing_address().as_str());
            w.set_pairing_open(true);
            refresh_pairing(&w, &pairing);
            start_pairing_timer(&w, &db, &sync, &pairing, &pairing_timer);
        }
    });

    window.on_pairing_code_edited({
        let w = window.as_weak();
        let pairing = pairing.clone();
        move |text| {
            let Some(w) = w.upgrade() else { return };
            let normalised = pairing.borrow_mut().set_entered(&text);
            w.set_pairing_entered(SharedString::from(normalised));
            refresh_pairing(&w, &pairing);
        }
    });

    window.on_pairing_address_edited({
        let w = window.as_weak();
        let pairing = pairing.clone();
        move |text| {
            let Some(w) = w.upgrade() else { return };
            pairing.borrow_mut().set_address(&text);
            refresh_pairing(&w, &pairing);
        }
    });

    window.on_pairing_device_chosen({
        let w = window.as_weak();
        let pairing = pairing.clone();
        move |address| {
            let Some(w) = w.upgrade() else { return };
            pairing.borrow_mut().choose_device(&address);
            // Written back into the field as well, because the field is the
            // single source of truth for what gets dialled — see
            // `PairingController::choose_device`.
            w.set_pairing_address(address);
            refresh_pairing(&w, &pairing);
        }
    });

    window.on_pairing_submit({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        let pairing = pairing.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let deps = pairing_deps(&db, &sync);
            if let Err(e) = pairing.borrow_mut().submit(&deps, maple_sync::now_ms()) {
                w.set_status_text(SharedString::from(e.to_string()));
            }
            refresh_pairing(&w, &pairing);
        }
    });

    window.on_pairing_cancel({
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        let pairing = pairing.clone();
        let pairing_timer = pairing_timer.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            pairing.borrow_mut().cancel();
            close_pairing(&w, &db, &sync, &pairing, &pairing_timer);
        }
    });

    SettingsHandle {
        window: window.clone_strong(),
        _pairing_timer: pairing_timer,
    }
}

/// Which half of the handshake this device plays, and what it signs with.
fn pairing_deps(db: &Arc<Mutex<maple_db::Database>>, sync: &Rc<SyncSupervisor>) -> PairingDeps {
    let guard = maple_db::lock_db(db);
    let device_id = guard.device_id().to_owned();
    let role = guard.sync_role().unwrap_or(SyncRole::Off);
    let name = guard.device_name().unwrap_or_default();
    drop(guard);

    let name = if name.trim().is_empty() {
        // A peer's settings card shows this, and a blank row is worse than a
        // dull one. Short enough to stay a label, unique enough to tell two
        // unnamed devices apart.
        format!("maple-{}", &device_id[..device_id.len().min(6)])
    } else {
        name
    };

    PairingDeps {
        device_id,
        device_name: name,
        // A master listens and is claimed; anything else dials. An `Off`
        // device pairing is treated as the dialling side, since it has no
        // listener running for a claim to reach.
        side: if role == SyncRole::Master {
            PairingSide::Master
        } else {
            PairingSide::Servant
        },
        clock: sync.clock(),
        rng: sync.rng(),
        // Starts a browse the first time a servant-side modal opens, and
        // hands the master side `None` — a master is claimed, it dials
        // nobody, and a listener browsing for itself is noise.
        discovery: if role == SyncRole::Master {
            None
        } else {
            sync.discovery()
        },
    }
}

/// Pre-fill the address field with the master we already know about, if any,
/// so re-pairing after a revoked key does not mean typing it again.
fn suggested_address(sync: &Rc<SyncSupervisor>) -> String {
    let trust = match sync.trust().lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    };
    trust
        .peers()
        .iter()
        .find_map(|peer| peer.address.clone())
        .unwrap_or_default()
}

/// Store a pairing this device completed as the initiator.
///
/// Only the servant reaches this: on a master the listener has already
/// written both stores by the time it answered, because a response lost in
/// flight must not leave the master without the key it just handed out.
fn persist_pairing(
    db: &Arc<Mutex<maple_db::Database>>,
    sync: &Rc<SyncSupervisor>,
    pairing: &Rc<RefCell<PairingController>>,
) {
    let address = pairing.borrow().address().to_owned();
    let Some(outcome) = pairing.borrow_mut().take_outcome() else {
        return;
    };

    {
        let mut trust = match sync.trust().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = trust.upsert_peer(TrustedPeer {
            device_id: outcome.device_id.clone(),
            key: outcome.key,
            // Recorded here and nowhere else: it is how the worker finds the
            // master on every later launch, with no code prompt.
            address: Some(address),
        }) {
            tracing::error!("Failed to store the pairing key: {e}");
            return;
        }
    }

    let guard = maple_db::lock_db(db);
    if let Err(e) = guard.upsert_sync_peer(
        &outcome.device_id,
        Some(outcome.name.as_str()),
        // Relay stores nothing, so a pairing completed before the user has
        // chosen a mode cannot start filling a disk.
        maple_state::PeerMode::Relay,
    ) {
        tracing::error!("Failed to record the paired device: {e}");
    }
}

/// Forget a peer in both stores.
///
/// The bookkeeping row and the key live in different files on purpose —
/// secrets stay out of the database that gets copied between machines — so
/// unpairing is two writes, and neither alone leaves a consistent state. The
/// key goes first: a leftover row without a key syncs nothing, while a
/// leftover key without a row still authenticates.
fn unpair(
    db: &Arc<Mutex<maple_db::Database>>,
    sync: &Rc<SyncSupervisor>,
    device_id: &str,
) -> String {
    {
        let mut trust = match sync.trust().lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        };
        if let Err(e) = trust.remove_peer(device_id) {
            tracing::error!("Failed to rewrite sync_trust.json: {e}");
            return format!("Could not remove the stored key: {e}");
        }
    }

    let guard = maple_db::lock_db(db);
    match guard.remove_sync_peer(device_id) {
        Ok(true) => "Device unpaired.".to_owned(),
        Ok(false) => "That device was already unpaired.".to_owned(),
        Err(e) => {
            tracing::error!("Failed to remove peer {device_id}: {e}");
            format!("Could not unpair: {e}")
        }
    }
}

/// Fill the Sync card from the database and settings.
fn populate_sync(
    window: &SettingsWindow,
    db: &Arc<Mutex<maple_db::Database>>,
    sync: &Rc<SyncSupervisor>,
) {
    let guard = maple_db::lock_db(db);
    let settings = maple_state::Settings::load();

    let role = guard.sync_role().unwrap_or_default();
    window.set_sync_role(SharedString::from(role.as_str()));
    window.set_sync_device_name(SharedString::from(guard.device_name().unwrap_or_default()));
    window.set_sync_role_detail(SharedString::from(match role {
        SyncRole::Master => format!("listening {}", settings.sync.listen_addr),
        SyncRole::Servant => format!("every {}", every(settings.sync.interval_secs)),
        SyncRole::Off => String::new(),
    }));

    let peers = guard.list_sync_peers().unwrap_or_else(|e| {
        tracing::error!("Failed to list sync peers: {e}");
        Vec::new()
    });
    let now = maple_sync::now_ms();
    // How many photos this library lists but cannot open, per peer, and how
    // many it holds itself. Both are needed to say what a mode is costing:
    // on a master the first number is the servant's photos it will never
    // receive, and on a servant the second is its own photos that will never
    // leave.
    let remote_counts: std::collections::HashMap<String, i64> = guard
        .remote_original_counts()
        .unwrap_or_default()
        .into_iter()
        .filter_map(|(device, n)| device.map(|d| (d, n)))
        .collect();
    let local_count = guard.local_original_count().unwrap_or(0);
    let items: Vec<SyncPeerItem> = peers
        .into_iter()
        .map(|peer| {
            let held = remote_counts.get(&peer.device_id).copied().unwrap_or(0);
            let (pending, stuck) =
                pending_line(role, peer.mode, &peer.display_name(), held, local_count);
            SyncPeerItem {
                device_id: SharedString::from(peer.device_id.clone()),
                name: SharedString::from(peer.display_name()),
                mode: SharedString::from(peer.mode.label()),
                mode_hint: SharedString::from(peer.mode.explanation()),
                pending: SharedString::from(pending),
                pending_stuck: stuck,
                // See `SyncPeerItem::mode-editable`: on a master the column is
                // a record of what the servant reported on its last pull, and
                // `server::pull` overwrites it on the next one.
                mode_editable: role == SyncRole::Servant,
                last_seen: SharedString::from(match peer.last_seen_at {
                    Some(seen) => maple_sync::relative_time(now - seen),
                    None => "never".to_owned(),
                }),
                online: peer
                    .last_seen_at
                    .is_some_and(|seen| now - seen < ONLINE_WINDOW_MS),
            }
        })
        .collect();
    window.set_sync_peers(slint::ModelRc::new(slint::VecModel::from(items)));
    drop(guard);

    let _ = sync;
}

/// The line under a peer that says what its mode is costing *here*, and
/// whether that cost is a queue or a standstill.
///
/// The mode chip alone cannot say this, because the same three words mean
/// different things on the two ends of one link:
///
/// - On a **servant**, the mode is the user's own choice and governs both
///   directions. Relay means this device's photos never leave it — not a
///   backlog that drains when the master wakes up, but a permanent state, so
///   it is flagged rather than counted down.
/// - On a **master**, the chip is a *record of what the servant chose* (the
///   servant sends its mode on every pull and the master stores it), and no
///   setting on this machine can change what arrives: a master runs no worker
///   and has no route to a servant. So the master's line names the number of
///   unloadable tiles and points at where the fix is.
///
/// Returns `(text, stuck)`; `stuck` is what draws it as a warning.
fn pending_line(
    role: SyncRole,
    mode: maple_state::PeerMode,
    peer_name: &str,
    held_on_peer: i64,
    local_photos: i64,
) -> (String, bool) {
    match role {
        SyncRole::Master => {
            if held_on_peer == 0 {
                return (String::new(), false);
            }
            let plural = if held_on_peer == 1 { "photo" } else { "photos" };
            if mode.moves_originals() {
                (
                    format!("{held_on_peer} {plural} still on {peer_name}, waiting to be sent."),
                    false,
                )
            } else {
                (
                    format!(
                        "{held_on_peer} {plural} listed here but held on {peer_name}, and none                          will arrive while it is in Relay. Change the mode on {peer_name}."
                    ),
                    true,
                )
            }
        }
        SyncRole::Servant => {
            if mode.moves_originals() {
                if held_on_peer == 0 {
                    return (String::new(), false);
                }
                let plural = if held_on_peer == 1 { "photo" } else { "photos" };
                (format!("{held_on_peer} {plural} still to come from {peer_name}."), false)
            } else if local_photos > 0 {
                let plural = if local_photos == 1 { "photo" } else { "photos" };
                (
                    format!(
                        "None of this device's {local_photos} {plural} are copied to                          {peer_name}; it lists them as tiles it cannot open."
                    ),
                    true,
                )
            } else {
                (String::new(), false)
            }
        }
        // The card still lists peers with sync switched off; nothing is
        // moving in either direction and saying so per-peer would be noise.
        SyncRole::Off => (String::new(), false),
    }
}

/// How recently a peer must have been seen to count as online. Two sync
/// intervals' grace at the default cadence, so a single missed pass does not
/// flicker the dot.
const ONLINE_WINDOW_MS: i64 = 10 * 60 * 1000;

/// "5 min" / "30 s" / "2 h" — the servant's idle cadence.
fn every(secs: u64) -> String {
    match secs {
        s if s < 60 => format!("{s} s"),
        s if s < 3600 => format!("{} min", s / 60),
        s => format!("{} h", s / 3600),
    }
}

/// Push the controller's current view into the modal's properties.
fn refresh_pairing(window: &SettingsWindow, pairing: &Rc<RefCell<PairingController>>) {
    let view = pairing.borrow_mut().tick(maple_sync::now_ms());
    window.set_pairing_own_code(SharedString::from(view.own_code));
    window.set_pairing_countdown(SharedString::from(view.countdown));
    window.set_pairing_message(SharedString::from(view.message));
    window.set_pairing_can_submit(view.can_submit);
    window.set_pairing_needs_address(view.needs_address);
    let devices: Vec<PairingDeviceItem> = view
        .devices
        .into_iter()
        .map(|device| PairingDeviceItem {
            label: SharedString::from(device.label),
            address: SharedString::from(device.address),
            chosen: device.chosen,
        })
        .collect();
    // Only when it actually changed. This runs once a second for as long as
    // the modal is open, and replacing the model rebuilds the repeater's
    // items — which resets the `has-hover` a row is drawing itself from, so
    // an unconditional set makes the highlight blink under the pointer once
    // a second. Same rule as the library grid's refresh.
    if !same_devices(&window.get_pairing_devices(), &devices) {
        window.set_pairing_devices(slint::ModelRc::new(slint::VecModel::from(devices)));
    }
    if window.get_pairing_entered().as_str() != view.entered {
        window.set_pairing_entered(SharedString::from(view.entered));
    }
}

/// Whether the pick-list on screen already says exactly this.
fn same_devices(shown: &slint::ModelRc<PairingDeviceItem>, fresh: &[PairingDeviceItem]) -> bool {
    shown.row_count() == fresh.len()
        && shown.iter().zip(fresh).all(|(a, b)| {
            a.address == b.address && a.label == b.label && a.chosen == b.chosen
        })
}

/// Start the one-second countdown. Replaces any timer already running, so
/// reopening the modal cannot leave two of them ticking.
fn start_pairing_timer(
    window: &SettingsWindow,
    db: &Arc<Mutex<maple_db::Database>>,
    sync: &Rc<SyncSupervisor>,
    pairing: &Rc<RefCell<PairingController>>,
    slot: &Rc<RefCell<Option<Timer>>>,
) {
    let timer = Timer::default();
    timer.start(TimerMode::Repeated, PAIRING_TICK, {
        let w = window.as_weak();
        let db = db.clone();
        let sync = sync.clone();
        let pairing = pairing.clone();
        let slot = slot.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            refresh_pairing(&w, &pairing);
            if !pairing.borrow().is_open() {
                // Expired, aborted, failed, or paired — `tick` has already
                // discarded both codes either way, so all that is left is to
                // store anything that succeeded and take the modal down.
                close_pairing(&w, &db, &sync, &pairing, &slot);
            }
        }
    });
    *slot.borrow_mut() = Some(timer);
}

/// Take the modal down, stop the countdown, and report how it ended.
fn close_pairing(
    window: &SettingsWindow,
    db: &Arc<Mutex<maple_db::Database>>,
    sync: &Rc<SyncSupervisor>,
    pairing: &Rc<RefCell<PairingController>>,
    slot: &Rc<RefCell<Option<Timer>>>,
) {
    // Dropping the timer stops it; leaving it running would keep waking the
    // event loop for a modal that is no longer on screen.
    *slot.borrow_mut() = None;

    // Before anything else: a pairing this device completed as the initiator
    // is only in memory until this runs.
    persist_pairing(db, sync, pairing);

    window.set_pairing_open(false);
    window.set_pairing_entered(SharedString::new());
    window.set_pairing_own_code(SharedString::new());
    if let Some(end) = pairing.borrow().ended() {
        window.set_status_text(SharedString::from(end.message()));
    }
    // A fresh pairing gives a servant the address and key its worker was
    // missing, so this is where a link actually comes up.
    sync.restart();
    populate_sync(window, db, sync);
}
