//! maple-ui — Slint UI for Maple.
//!
//! Navigation: persistent sidebar shell with Library and Import views embedded
//! in AppWindow.  Secondary windows (detail, settings, collections, face-tag,
//! import browser) are opened as separate OS windows from callbacks.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{ModelRc, SharedString, Timer, TimerMode, VecModel};

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

/// Search-input debounce (ms).
const SEARCH_DEBOUNCE_MS: u64 = 200;

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

    if settings.stacks.enabled {
        maple_db::spawn_hasher(db.clone(), settings.stacks.clone(), Some(cache.clone()));
    }

    let window = AppWindow::new()?;

    // ── Library grid ───────────────────────────────────────────────
    let settings = maple_state::Settings::load();
    let grid = LibraryGrid::new(
        db.clone(),
        cache.clone(),
        settings.thumbnails.quality,
        settings.thumbnails.size,
    );
    window.set_library_items(grid.model());
    window.set_library_cell_size(settings.thumbnails.size as f32);

    // Library is the default page — load immediately.
    grid.load(SearchQuery::default());

    window.on_library_shown({
        let grid = grid.clone();
        move || grid.load(SearchQuery::default())
    });

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

    // ── Import page ────────────────────────────────────────────────
    // Shared source-path state for the embedded picker.
    let import_source: Rc<RefCell<std::path::PathBuf>> =
        Rc::new(RefCell::new(std::path::PathBuf::new()));

    // Build the initial location lists.
    window.on_import_page_shown({
        let w = window.as_weak();
        let settings = maple_state::Settings::load();
        let home = home_dir();
        move || {
            let Some(w) = w.upgrade() else { return };
            let locs = build_import_locations(&settings, &home);
            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_browse({
        let w = window.as_weak();
        let source = import_source.clone();
        move || {
            let picked = rfd::FileDialog::new()
                .set_title("Choose source folder")
                .pick_folder();
            if let Some(path) = picked {
                let name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| path.to_string_lossy().into_owned());
                let path_str = path.to_string_lossy().into_owned();
                *source.borrow_mut() = path;
                if let Some(w) = w.upgrade() {
                    w.set_import_source_name(SharedString::from(name));
                    w.set_import_source_path(SharedString::from(path_str));
                    w.set_import_source_count(0);
                }
            }
        }
    });

    window.on_import_clear({
        let w = window.as_weak();
        let source = import_source.clone();
        move || {
            *source.borrow_mut() = std::path::PathBuf::new();
            if let Some(w) = w.upgrade() {
                w.set_import_source_path(SharedString::default());
                w.set_import_source_name(SharedString::default());
                w.set_import_source_count(0);
            }
        }
    });

    window.on_import_location_selected({
        let w = window.as_weak();
        let source = import_source.clone();
        let settings = maple_state::Settings::load();
        let home = home_dir();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            let mut locs = build_import_locations(&settings, &home);

            // Mark selected, deselect others.
            for loc in &mut locs {
                loc.is_selected = loc.id == id.as_str();
            }

            if let Some(loc) = locs.iter().find(|l| l.id == id.as_str()) {
                *source.borrow_mut() = std::path::PathBuf::from(loc.path.as_str());
                w.set_import_source_name(loc.name.clone().into());
                w.set_import_source_path(loc.path.clone().into());
                w.set_import_source_count(loc.count);
            }

            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_star_toggled({
        let w = window.as_weak();
        let settings = maple_state::Settings::load();
        let home = home_dir();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            // Toggle star in-place (no persistence for now).
            let mut locs = build_import_locations(&settings, &home);
            for loc in &mut locs {
                if loc.id == id.as_str() {
                    loc.is_starred = !loc.is_starred;
                }
            }
            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_start_scan({
        let db = db.clone();
        let source = import_source.clone();
        move || {
            let src = source.borrow().clone();
            if src.as_os_str().is_empty() {
                return;
            }
            import::open_with_source(db.clone(), src);
        }
    });

    // ── AI / Faces toggles ─────────────────────────────────────────
    let face_tagger: Arc<Mutex<Option<maple_db::FaceTagger>>> = Arc::new(Mutex::new(None));

    window.on_toggle_ai(|| {
        tracing::info!("AI tagging toggle — wiring for phase 2");
    });

    window.on_toggle_faces({
        let w = window.as_weak();
        let db = db.clone();
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

    // ── Other secondary windows ────────────────────────────────────
    window.on_settings_clicked({
        let db = db.clone();
        move || settings_window::open(db.clone())
    });
    window.on_collections_clicked({
        let db = db.clone();
        move || collections_window::open(db.clone())
    });
    window.on_tag_faces_clicked({
        let db = db.clone();
        move || face_tag::open(db.clone())
    });
    window.on_debug_clicked(|| tracing::info!("debug — placeholder"));

    let _db = db;
    let _cache = cache;
    let _face_tagger = face_tagger;

    window.run()?;
    Ok(())
}

fn home_dir() -> std::path::PathBuf {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(std::path::PathBuf::from)
        .unwrap_or_default()
}

// ── Import location helpers ────────────────────────────────────────

struct LocData {
    id:          String,
    name:        String,
    path:        String,
    count:       i32,
    is_starred:  bool,
    is_selected: bool,
}

fn build_import_locations(
    settings: &maple_state::Settings,
    home: &std::path::PathBuf,
) -> Vec<LocData> {
    let mut locs: Vec<LocData> = Vec::new();

    // Pictures directory
    let pictures = home.join("Pictures");
    if pictures.is_dir() {
        locs.push(LocData {
            id: "pictures".into(),
            name: "Pictures Library".into(),
            path: pictures.to_string_lossy().into_owned(),
            count: 0,
            is_starred: true,
            is_selected: false,
        });
    }

    // Desktop
    let desktop = home.join("Desktop");
    if desktop.is_dir() {
        locs.push(LocData {
            id: "desktop".into(),
            name: "Desktop".into(),
            path: desktop.to_string_lossy().into_owned(),
            count: 0,
            is_starred: false,
            is_selected: false,
        });
    }

    // Downloads
    let downloads = home.join("Downloads");
    if downloads.is_dir() {
        locs.push(LocData {
            id: "downloads".into(),
            name: "Downloads".into(),
            path: downloads.to_string_lossy().into_owned(),
            count: 0,
            is_starred: false,
            is_selected: false,
        });
    }

    // Library directory (from settings) if different from Pictures
    let lib = settings.library_dir.clone();
    if lib.is_dir() && lib != home.join("Pictures") {
        locs.push(LocData {
            id: "library".into(),
            name: "Library Folder".into(),
            path: lib.to_string_lossy().into_owned(),
            count: 0,
            is_starred: false,
            is_selected: false,
        });
    }

    locs
}

fn partition_locations(locs: Vec<LocData>) -> (ModelRc<ImportLocation>, ModelRc<ImportLocation>) {
    let mut favs: Vec<ImportLocation> = Vec::new();
    let mut recents: Vec<ImportLocation> = Vec::new();
    for l in locs {
        let item = ImportLocation {
            id:          l.id.into(),
            name:        l.name.into(),
            path:        l.path.into(),
            count:       l.count,
            is_starred:  l.is_starred,
            is_selected: l.is_selected,
        };
        if item.is_starred {
            favs.push(item);
        } else {
            recents.push(item);
        }
    }
    (
        ModelRc::from(Rc::new(VecModel::from(favs))),
        ModelRc::from(Rc::new(VecModel::from(recents))),
    )
}
