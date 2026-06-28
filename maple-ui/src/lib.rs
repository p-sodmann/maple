//! maple-ui — Slint UI for Maple.
//!
//! GTK4/libadwaita has been fully replaced with Slint (sprint P5).  The
//! background-work architecture is unchanged — all heavy work runs on
//! `std::thread` + `std::sync::mpsc`; UI-side delivery uses `slint::Timer`
//! and `Weak::upgrade_in_event_loop`.
//!
//! The Slint markup in `ui/` is compiled by `build.rs` and pulled in below.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{Timer, TimerMode};

use maple_db::SearchQuery;


slint::include_modules!();

mod collections_window;
mod detail;
mod face_overlay;
mod face_tag;
mod grid;
mod image_loader;
mod import;
mod settings_window;
pub mod thumbnail;

use grid::LibraryGrid;

/// Search-input debounce, matching the old 200 ms GTK SearchEntry debounce.
const SEARCH_DEBOUNCE_MS: u64 = 200;

/// Boot the UI and run the main window. Blocks until the window closes.
///
/// Opens the library database and thumbnail cache (each with a temp-dir
/// fallback so a bad configured path never crashes startup), starts the
/// background library scanner and the optional stack hasher, then builds and
/// runs the Slint application window.
pub fn run() -> anyhow::Result<()> {
    let settings = maple_state::Settings::load();

    let db = match maple_db::Database::open(&settings.database_path) {
        Ok(db) => Arc::new(Mutex::new(db)),
        Err(e) => {
            tracing::error!(
                "Failed to open library database at {}: {e}",
                settings.database_path.display()
            );
            // Proceed on a temp fallback rather than crashing on startup.
            let fallback = std::env::temp_dir().join("maple_library_fallback.db");
            Arc::new(Mutex::new(
                maple_db::Database::open(&fallback).expect("Could not open fallback database"),
            ))
        }
    };

    // Open the thumbnail cache alongside the database.
    let cache_dir = settings.library_dir.join(".thumbcache");
    let cache = Arc::new(maple_db::ThumbnailCache::open(&cache_dir).unwrap_or_else(|e| {
        tracing::warn!("Thumbnail cache unavailable at {}: {e}", cache_dir.display());
        let fallback = std::env::temp_dir().join("maple_thumbcache_fallback");
        maple_db::ThumbnailCache::open(&fallback).expect("could not open fallback thumbnail cache")
    }));

    // Start the background library scanner immediately so the DB stays in sync
    // with the library directory from the moment the app launches.
    maple_db::LibraryScanner::new(db.clone(), settings.library_dir.clone(), Some(cache.clone()))
        .spawn();

    // Start the background hasher if stack detection is enabled.
    if settings.stacks.enabled {
        maple_db::spawn_hasher(db.clone(), settings.stacks.clone(), Some(cache.clone()));
    }

    let window = AppWindow::new()?;

    // ── Library grid ──────────────────────────────────────────────
    let settings = maple_state::Settings::load();
    let grid = LibraryGrid::new(
        db.clone(),
        cache.clone(),
        settings.thumbnails.quality,
        settings.thumbnails.size,
    );
    window.set_library_items(grid.model());
    window.set_library_cell_size(settings.thumbnails.size as f32);

    // Initial grid load when the library page is first shown.
    window.on_library_shown({
        let grid = grid.clone();
        move || grid.load(SearchQuery::default())
    });

    // Debounced search: each keystroke restarts a single-shot timer; only the
    // final query (after the user pauses) hits the DB.
    let search_debounce: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    window.on_library_search_changed({
        let grid = grid.clone();
        let search_debounce = search_debounce.clone();
        move |text| {
            let grid = grid.clone();
            let text = text.to_string();
            let timer = Timer::default();
            timer.start(
                TimerMode::SingleShot,
                Duration::from_millis(SEARCH_DEBOUNCE_MS),
                move || {
                    let mut q = SearchQuery::default();
                    if !text.trim().is_empty() {
                        q = q.with_text(&text);
                    }
                    grid.load(q);
                },
            );
            *search_debounce.borrow_mut() = Some(timer);
        }
    });

    // Cell click → open the detail/lightbox window with a snapshot of the
    // records the grid is currently showing (for prev/next navigation).
    window.on_library_activated({
        let records = grid.records();
        let db = db.clone();
        move |idx| {
            let snapshot = records.borrow().clone();
            if (idx as usize) < snapshot.len() {
                detail::open(snapshot, idx as usize, db.clone());
            }
        }
    });

    // Phase 7 windows.
    window.on_settings_clicked({
        let db = db.clone();
        move || settings_window::open(db.clone())
    });
    window.on_collections_clicked({
        let db = db.clone();
        move || collections_window::open(db.clone())
    });
    window.on_debug_clicked(|| tracing::info!("debug window — phase 7 placeholder"));
    window.on_import_requested({
        let db = db.clone();
        move || import::open(db.clone())
    });

    // "Detect Faces" — load the ONNX face detector on a background thread to
    // avoid blocking the UI, then hand the FaceTagger back via an Arc so it
    // stays alive on the main thread for the window's lifetime.
    let face_tagger: Arc<Mutex<Option<maple_db::FaceTagger>>> = Arc::new(Mutex::new(None));
    window.on_detect_faces_clicked({
        let db = db.clone();
        let face_tagger = face_tagger.clone();
        move || {
            let db = db.clone();
            let face_tagger = face_tagger.clone();
            std::thread::spawn(move || {
                let settings = maple_state::Settings::load();
                if !settings.face.models_available() {
                    tracing::warn!(
                        "Detect Faces: no face model configured — set [face] detector_model in settings.toml"
                    );
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
                        if let Ok(mut guard) = face_tagger.lock() {
                            *guard = Some(tagger);
                        }
                        tracing::info!("Face tagger started");
                    }
                    Err(e) => tracing::error!("Detect Faces: failed to load model: {e}"),
                }
            });
        }
    });

    // "Tag Faces" — open the dedicated face-tagging window.
    window.on_tag_faces_clicked({
        let db = db.clone();
        move || face_tag::open(db.clone())
    });

    // Keep handles alive for the window's lifetime.
    let _db = db;
    let _cache = cache;
    let _face_tagger = face_tagger;

    window.run()?;
    Ok(())
}
