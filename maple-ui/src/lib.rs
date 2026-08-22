//! maple-ui — Slint UI for Maple.
//!
//! Navigation: persistent sidebar shell with Library and Import views embedded
//! in AppWindow.  Secondary windows (detail, settings, collections, face-tag,
//! import browser) are opened as separate OS windows from callbacks.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use maple_db::SearchQuery;

slint::include_modules!();

mod collections_page;
mod date;
mod debug_compare;
mod detail;
mod face_crop;
mod face_overlay;
mod face_tag;
mod grid;
mod image_loader;
mod import;
mod import_page;
mod library_page;
mod paging;
mod path_template_window;
mod people_page;
mod remote;
mod rep_crop;
mod services;
mod settings_window;
mod sync_pairing;
mod sync_status;
mod sync_supervisor;
pub mod thumbnail;
mod transforms;

use grid::LibraryGrid;

/// Shared handles built once at the top of [`run`] and passed to each
/// `wire_*` function instead of threading a dozen parameters through.
///
/// Every field is a cheap clone target (`Rc`/`Arc`/`Copy`), and the window is
/// held only as a [`slint::Weak`] — a callback that clones fields out of an
/// `AppCtx` therefore can't capture a strong `AppWindow` and leak the window
/// through a reference cycle.
pub(crate) struct AppCtx {
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<maple_db::ThumbnailCache>,
    /// The shell window — weak, see the struct docs.
    window: slint::Weak<AppWindow>,
    grid: LibraryGrid,
    /// Tracks the query last passed to `grid.load()` (search text / person
    /// filter / …) so the date-view toggle can reload with it unchanged.
    current_query: Rc<RefCell<SearchQuery>>,
    /// The collection select-mode is currently "editing" — set by following a
    /// gallery card into a filtered Library view, or by clicking a sidebar
    /// dot while already in select-mode. Drives both the checkbox sync in
    /// `apply_marquee`/`apply_membership` and the sidebar/gallery highlight
    /// (`library-active-collection-id`). `None` outside select-mode or before
    /// a target has been picked.
    select_target: Rc<Cell<Option<i64>>>,
    /// See the closure built in [`run`].
    resync_selection: Rc<dyn Fn()>,
    /// In-memory memoization of gallery cover crops, mirroring the People
    /// page's own `CropCache` (see `rep_crop`) — separate instance since
    /// collections and persons are keyed independently.
    coll_crop_cache: rep_crop::CropCache,
    thumb_px: u32,
    thumb_quality: u8,
    /// Owns the sync listener or worker, whichever this device's role calls
    /// for, and restarts them when that changes.
    sync: Rc<sync_supervisor::SyncSupervisor>,
}

/// Boot the UI. Blocks until the window closes.
pub fn run() -> anyhow::Result<()> {
    let settings = maple_state::Settings::load();

    let db = match maple_db::Database::open(&settings.database_path) {
        Ok(db) => Arc::new(Mutex::new(db)),
        Err(e) => {
            tracing::error!(
                "Failed to open library database at {}: {e}",
                settings.database_path.display()
            );
            let fallback = std::env::temp_dir().join("maple_library_fallback.db");
            Arc::new(Mutex::new(
                maple_db::Database::open(&fallback).expect("could not open fallback database"),
            ))
        }
    };

    let cache_dir = settings.library_dir.join(".thumbcache");
    let cache = Arc::new(maple_db::ThumbnailCache::open(&cache_dir).unwrap_or_else(|e| {
        tracing::warn!("Thumbnail cache unavailable: {e}");
        let fallback = std::env::temp_dir().join("maple_thumbcache_fallback");
        maple_db::ThumbnailCache::open(&fallback).expect("could not open fallback thumbnail cache")
    }));

    maple_db::LibraryScanner::new(db.clone(), settings.library_dir.clone(), Some(cache.clone()))
        .spawn();

    // Backfill EXIF metadata (curated fields + comprehensive tags) for any
    // records inserted since the last run. Safe to call repeatedly — it only
    // touches records where `exif_extracted = 0`.
    maple_db::spawn_metadata_filler(db.clone());

    if settings.stacks.enabled {
        maple_db::spawn_hasher(db.clone(), settings.stacks.clone(), Some(cache.clone()));
    }

    let window = AppWindow::new()?;

    // ── Library grid ───────────────────────────────────────────────
    let settings = maple_state::Settings::load();
    let grid = library_page::create_grid(&window, &settings, db.clone(), cache.clone());

    let current_query: Rc<RefCell<SearchQuery>> = Rc::new(RefCell::new(SearchQuery::default()));
    let select_target: Rc<Cell<Option<i64>>> = Rc::new(Cell::new(None));

    // Re-applies the active select-target's membership after a "same
    // context" grid reload (date-view toggle, `grid::request_reload` from
    // rotation/restructure) — those calls reload with the *same* query, not
    // a context switch, so unlike navigation they must not silently drop an
    // in-progress selection. No-op when select-mode is off or has no target.
    let resync_selection: Rc<dyn Fn()> = {
        let db = db.clone();
        let grid = grid.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        Rc::new(move || {
            let Some(id) = select_target.get() else { return };
            let Some(win) = w.upgrade() else { return };
            if !win.get_library_select_mode() {
                return;
            }
            let member_ids = services::collections::member_ids(&db, id);
            let n = grid.apply_membership(&member_ids);
            win.set_library_selected_count(n);
        })
    };

    // ── Sync ───────────────────────────────────────────────────────
    // Seeded from the stored role, so an installation that has never been set
    // up starts grey rather than claiming to be connecting to nothing.
    let (device_id, device_name, sync_role) = {
        let guard = maple_db::lock_db(&db);
        (
            guard.device_id().to_owned(),
            guard.device_name().unwrap_or_default(),
            guard.sync_role().unwrap_or_default(),
        )
    };
    let sync_status = maple_sync::SyncStatus::cell(sync_role);
    let trust = match maple_sync::TrustStore::open_default(&device_id, &device_name) {
        Ok(store) => Arc::new(Mutex::new(store)),
        Err(e) => {
            // A corrupt trust file is surfaced, not overwritten: it holds
            // every paired device's key, and silently starting fresh would
            // present as "everything needs re-pairing" while hiding why.
            tracing::error!("Sync disabled — could not read the key store: {e}");
            sync_status
                .lock()
                .map(|mut s| s.last_error = Some(e.to_string()))
                .unwrap_or_default();
            Arc::new(Mutex::new(
                maple_sync::TrustStore::open(
                    std::env::temp_dir().join("maple_sync_trust_fallback.json"),
                    &device_id,
                    &device_name,
                )
                .expect("a fresh fallback store cannot be corrupt"),
            ))
        }
    };
    let sync = sync_supervisor::SyncSupervisor::new(
        db.clone(),
        trust,
        maple_sync::PairingSlot::new(),
        sync_status.clone(),
        sync_pairing::db_random(db.clone()),
        cache.clone(),
        settings.thumbnails.size,
        settings.thumbnails.quality,
    );

    let ctx = AppCtx {
        db: db.clone(),
        cache: cache.clone(),
        window: window.as_weak(),
        grid: grid.clone(),
        current_query: current_query.clone(),
        select_target: select_target.clone(),
        resync_selection: resync_selection.clone(),
        coll_crop_cache: Arc::new(Mutex::new(HashMap::new())),
        thumb_px: settings.thumbnails.size,
        thumb_quality: settings.thumbnails.quality,
        sync: sync.clone(),
    };

    // Let other windows (rotation, library restructure, …) request a grid
    // reload without needing `grid`/`current_query` threaded through their
    // own constructors — see `grid::request_reload`.
    grid::register(grid, current_query, resync_selection);

    // Installs the paging feed and loads the first page — everything the
    // initial `load` can reach must already be wired at this point.
    library_page::wire_grid(&window, &ctx);

    // ── Collections page ───────────────────────────────────────────
    collections_page::wire(&window, &ctx);

    // ── People page ────────────────────────────────────────────────
    people_page::wire(&window, &ctx);

    // ── Library navigation (sidebar, filter chip, search, tiles) ────
    library_page::wire_navigation(&window, &ctx);

    // ── Library mass-select ────────────────────────────────────────
    library_page::wire_mass_select(&window, &ctx);

    // ── Import page ────────────────────────────────────────────────
    import_page::wire(&window, &ctx);

    // ── AI / Faces toggles ─────────────────────────────────────────
    wire_toggles(&window, &ctx);

    // ── Theme propagation ──────────────────────────────────────────
    wire_theme(&window);

    // ── Other secondary windows ────────────────────────────────────
    wire_secondary_windows(&window, &ctx);

    // ── Sync status pill ───────────────────────────────────────────
    // Held for the life of `run`: a `slint::Timer` that is dropped stops, so
    // letting this handle fall out of scope would freeze the pill on whatever
    // it showed at startup.
    let _sync_pill = sync_status::wire(&window, sync_status, maple_sync::now_ms);

    // Starts the listener or the worker, per the stored role. After
    // `grid::register`, because a first pass can finish before `run()` does
    // and calls `request_reload`.
    sync.restart();

    window.run()?;

    // The listener and the worker both hold the database mutex during a
    // merge; stopping them before `run` returns keeps one from being
    // mid-transaction while the connection is torn down.
    sync.stop();
    Ok(())
}

/// Wire the header toggles that drive background work and the grid layout.
fn wire_toggles(window: &AppWindow, ctx: &AppCtx) {
    let face_tagger: Arc<Mutex<Option<maple_db::FaceTagger>>> = Arc::new(Mutex::new(None));

    window.on_toggle_ai(|| {
        tracing::info!("AI tagging toggle — wiring for phase 2");
    });

    window.on_toggle_date_view({
        let w = ctx.window.clone();
        let grid = ctx.grid.clone();
        let current_query = ctx.current_query.clone();
        let resync_selection = ctx.resync_selection.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let on = !w.get_date_view_on();
            w.set_date_view_on(on);
            grid.set_date_view(on);
            grid.load(current_query.borrow().clone());
            resync_selection();
        }
    });

    window.on_toggle_faces({
        let w = ctx.window.clone();
        let db = ctx.db.clone();
        let face_tagger = face_tagger.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            if w.get_faces_on() {
                // Stop (signal is implicit — tagger checks stop flag).
                w.set_faces_on(false);
                return;
            }
            w.set_faces_on(true);
            let db = db.clone();
            let face_tagger = face_tagger.clone();
            std::thread::spawn(move || {
                let settings = maple_state::Settings::load();
                if !settings.face.models_available() {
                    tracing::warn!("Detect Faces: no face model configured");
                    return;
                }
                let device: maple_db::models::ModelDevice =
                    settings.face.device.parse().unwrap_or_default();
                match maple_db::FaceDetector::with_device(
                    &settings.face.detector_model,
                    settings.face.embedder_path(),
                    &device,
                    settings.face.detector_type,
                    None,
                ) {
                    Ok(detector) => {
                        let tagger = maple_db::spawn_face_tagger(db, detector);
                        if let Ok(mut g) = face_tagger.lock() {
                            *g = Some(tagger);
                        }
                    }
                    Err(e) => tracing::error!("Detect Faces: failed to load model: {e}"),
                }
            });
        }
    });
}

/// Fan a theme change out to every secondary window that may be open.
fn wire_theme(window: &AppWindow) {
    window.on_theme_toggled(|is_dark| {
        settings_window::set_dark(is_dark);
        detail::set_dark(is_dark);
        debug_compare::set_dark(is_dark);
        import::set_dark(is_dark);
        path_template_window::set_dark(is_dark);
    });
}

/// Wire the buttons that open a separate top-level window.
fn wire_secondary_windows(window: &AppWindow, ctx: &AppCtx) {
    window.on_settings_clicked({
        let db = ctx.db.clone();
        let sync = ctx.sync.clone();
        let w = ctx.window.clone();
        move || {
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            settings_window::open(db.clone(), sync.clone(), is_dark);
        }
    });
    window.on_tag_faces_clicked({
        let db = ctx.db.clone();
        move || face_tag::open(db.clone())
    });
    window.on_debug_clicked(|| tracing::info!("debug — placeholder"));
}
