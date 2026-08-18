//! maple-ui — Slint UI for Maple.
//!
//! Navigation: persistent sidebar shell with Library and Import views embedded
//! in AppWindow.  Secondary windows (detail, settings, collections, face-tag,
//! import browser) are opened as separate OS windows from callbacks.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{ModelRc, SharedString, Timer, TimerMode, VecModel};

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
mod paging;
mod path_template_window;
mod people_page;
mod rep_crop;
mod services;
mod settings_window;
pub mod thumbnail;
mod transforms;

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
    let grid = LibraryGrid::new(
        db.clone(),
        cache.clone(),
        settings.thumbnails.quality,
        settings.thumbnails.size,
    );
    window.set_library_items(grid.model());
    window.set_library_date_groups(grid.date_groups_model());
    window.set_library_cell_size(settings.thumbnails.size as f32);

    // Tracks the query last passed to `grid.load()` (search text / person
    // filter / …) so the date-view toggle can reload with it unchanged.
    let current_query: Rc<RefCell<SearchQuery>> = Rc::new(RefCell::new(SearchQuery::default()));

    // The collection select-mode is currently "editing" — set by following a
    // gallery card into a filtered Library view, or by clicking a sidebar
    // dot while already in select-mode. Drives both the checkbox sync in
    // `apply_marquee`/`apply_membership` and the sidebar/gallery highlight
    // (`library-active-collection-id`). `None` outside select-mode or before
    // a target has been picked.
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

    // Let other windows (rotation, library restructure, …) request a grid
    // reload without needing `grid`/`current_query` threaded through their
    // own constructors — see `grid::request_reload`.
    grid::register(grid.clone(), current_query.clone(), resync_selection.clone());

    // The grid only holds the pages loaded so far, so the header's photo
    // count comes from a COUNT query rather than from the item model.
    grid.on_total_count({
        let w = window.as_weak();
        move |total| {
            if let Some(win) = w.upgrade() {
                win.set_library_total_count(total.map_or(-1, |n| n as i32));
            }
        }
    });

    // Endless scrolling: the grid view reports how many items the viewport
    // is approaching (plus a prefetch lead) and the grid appends pages until
    // it has them.
    window.on_library_request_more({
        let grid = grid.clone();
        move |rows| grid.request_more(rows)
    });

    // Library is the default page — load the first page immediately.
    grid.load(SearchQuery::default());

    // ── Collections page ───────────────────────────────────────────
    // Capture thumb settings for background thumbnail loads.
    let coll_thumb_px = settings.thumbnails.size;
    let coll_thumb_quality = settings.thumbnails.quality;

    // Shared record list — populated by load_thumbs, read by on_collections_open_image.
    let coll_records: Arc<Mutex<Vec<maple_db::LibraryImage>>> =
        Arc::new(Mutex::new(Vec::new()));

    // In-memory memoization of gallery cover crops, mirroring the People
    // page's own `CropCache` (see `rep_crop`) — separate instance since
    // collections and persons are keyed independently.
    let coll_crop_cache: rep_crop::CropCache = Arc::new(Mutex::new(HashMap::new()));

    // Populate sidebar + gallery immediately (so dots/cards show even before navigating).
    collections_page::reload(&window, &db);
    collections_page::load_gallery(&window, &db, &cache, coll_thumb_quality, &coll_crop_cache);

    window.on_collections_page_shown({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move || {
            if let Some(win) = w.upgrade() {
                collections_page::reload(&win, &db);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_collections_activated({
        let grid = grid.clone();
        let w = window.as_weak();
        let current_query = current_query.clone();
        let select_target = select_target.clone();
        move |collection_id, name| {
            let q = SearchQuery::default().with_collection(collection_id as i64);
            *current_query.borrow_mut() = q.clone();
            grid.load(q);
            select_target.set(Some(collection_id as i64));
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(name);
                win.set_library_active_collection_id(collection_id);
                win.set_page(crate::Page::Library);
            }
        }
    });

    window.on_collections_edit_save({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move |id, name, color| {
            let name = name.trim().to_string();
            if name.is_empty() {
                return;
            }
            let hex = format!("#{:02x}{:02x}{:02x}", color.red(), color.green(), color.blue());
            services::collections::rename_collection(&db, id as i64, &name);
            services::collections::set_collection_color(&db, id as i64, &hex);
            if let Some(win) = w.upgrade() {
                collections_page::reload_keep_sel(&win, &db);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_collections_edit_delete({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move |id| {
            services::collections::delete_collection(&db, &cache, id as i64);
            if let Some(win) = w.upgrade() {
                collections_page::reload(&win, &db);
                collections_page::clear_detail(&win);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_collections_select({
        let db = db.clone();
        let cache = cache.clone();
        let coll_records = coll_records.clone();
        let w = window.as_weak();
        move |id| {
            if let Some(win) = w.upgrade() {
                // Clear old thumbs immediately; push fresh detail.
                win.set_collections_thumbs(slint::ModelRc::default());
                collections_page::push_detail(&win, &db, id);
            }
            collections_page::load_thumbs(
                id,
                db.clone(),
                cache.clone(),
                coll_thumb_px,
                coll_thumb_quality,
                w.clone(),
                coll_records.clone(),
            );
        }
    });

    window.on_collections_open_image({
        let coll_records = coll_records.clone();
        let db = db.clone();
        let w = window.as_weak();
        move |image_id| {
            let records = coll_records.lock().ok().map(|g| g.clone()).unwrap_or_default();
            // Find the index of the clicked image within the loaded records.
            let idx = records.iter().position(|r| r.id == image_id as i64).unwrap_or(0);
            if !records.is_empty() {
                let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
                detail::open(records, idx, db.clone(), is_dark);
            }
        }
    });

    window.on_collections_create({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move |name, color, parent_id| {
            let name = name.trim().to_string();
            if name.is_empty() { return; }
            let hex = format!(
                "#{:02x}{:02x}{:02x}",
                color.red(),
                color.green(),
                color.blue(),
            );
            let pid = if parent_id >= 0 { Some(parent_id as i64) } else { None };
            services::collections::create_collection(&db, &name, &hex, pid);
            if let Some(win) = w.upgrade() {
                collections_page::reload(&win, &db);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_collections_delete({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move |id| {
            services::collections::delete_collection(&db, &cache, id as i64);
            if let Some(win) = w.upgrade() {
                collections_page::reload(&win, &db);
                collections_page::clear_detail(&win);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_collections_rename({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move |id, name| {
            let name = name.trim().to_string();
            if name.is_empty() { return; }
            services::collections::rename_collection(&db, id as i64, &name);
            if let Some(win) = w.upgrade() {
                collections_page::reload_keep_sel(&win, &db);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    // ── People page ────────────────────────────────────────────────
    let people = people_page::PeoplePage::new(db.clone(), cache.clone(), coll_thumb_quality);
    window.set_people_items(people.model());

    window.on_people_page_shown({
        let people = people.clone();
        let w = window.as_weak();
        move || {
            let w2 = w.clone();
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
            }
            people.load(w2);
        }
    });

    window.on_people_person_activated({
        let grid = grid.clone();
        let w = window.as_weak();
        let current_query = current_query.clone();
        let select_target = select_target.clone();
        move |person_id, name| {
            let q = SearchQuery::default().with_person(person_id as i64);
            *current_query.borrow_mut() = q.clone();
            grid.load(q);
            select_target.set(None);
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(name);
                win.set_library_active_collection_id(-1);
                win.set_page(crate::Page::Library);
            }
        }
    });

    window.on_people_edit_save({
        let db = db.clone();
        let people = people.clone();
        let w = window.as_weak();
        move |person_id, name| {
            let name = name.trim().to_string();
            if name.is_empty() {
                return;
            }
            services::faces::rename_person(&db, person_id as i64, &name);
            if let Some(win) = w.upgrade() {
                people.load(win.as_weak());
            }
        }
    });

    window.on_people_edit_delete({
        let db = db.clone();
        let cache = cache.clone();
        let people = people.clone();
        let w = window.as_weak();
        move |person_id| {
            services::faces::delete_person(&db, &cache, person_id as i64);
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
                people.load(win.as_weak());
            }
        }
    });

    window.on_people_tag_faces({
        let db = db.clone();
        let people = people.clone();
        let w = window.as_weak();
        move || {
            face_tag::open(db.clone());
            // Refresh untagged count after the wizard is closed (best-effort;
            // the window fires this callback synchronously, so the count
            // updates when the user returns to the People page).
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
            }
        }
    });

    window.on_library_shown({
        let grid = grid.clone();
        let current_query = current_query.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        move || {
            let q = SearchQuery::default();
            *current_query.borrow_mut() = q.clone();
            grid.load(q);
            select_target.set(None);
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(SharedString::new());
                win.set_library_active_collection_id(-1);
            }
        }
    });

    window.on_library_filter_cleared({
        let grid = grid.clone();
        let current_query = current_query.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        move || {
            let q = SearchQuery::default();
            *current_query.borrow_mut() = q.clone();
            grid.load(q);
            select_target.set(None);
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(SharedString::new());
                win.set_library_active_collection_id(-1);
            }
        }
    });

    let search_debounce: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));
    window.on_library_search_changed({
        let grid = grid.clone();
        let search_debounce = search_debounce.clone();
        let current_query = current_query.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        move |text| {
            let grid = grid.clone();
            let current_query = current_query.clone();
            let text = text.to_string();
            select_target.set(None);
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(SharedString::new());
                win.set_library_active_collection_id(-1);
            }
            let timer = Timer::default();
            timer.start(
                TimerMode::SingleShot,
                Duration::from_millis(SEARCH_DEBOUNCE_MS),
                move || {
                    let mut q = SearchQuery::default();
                    if !text.trim().is_empty() {
                        q = q.with_text(&text);
                    }
                    *current_query.borrow_mut() = q.clone();
                    grid.load(q);
                },
            );
            *search_debounce.borrow_mut() = Some(timer);
        }
    });

    window.on_library_activated({
        let records = grid.records();
        let db = db.clone();
        let w = window.as_weak();
        move |idx| {
            let snapshot = records.borrow().clone();
            if (idx as usize) < snapshot.len() {
                let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
                detail::open(snapshot, idx as usize, db.clone(), is_dark);
            }
        }
    });

    // ── Library mass-select (sidebar "Add Images" toggle → collection dots) ──
    //
    // Entering select-mode doesn't require a target collection — you can
    // just tick photos, then click a sidebar dot to add them (handled
    // below). But if a target is already set (from following a gallery card
    // into a filtered view, or a previous dot click this session), checkbox
    // state is immediately synced to that collection's real membership so
    // deselecting one is a removal, matching a freshly-clicked dot.
    window.on_toggle_select_mode({
        let db = db.clone();
        let grid = grid.clone();
        let select_target = select_target.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let w = window.as_weak();
        move || {
            let Some(win) = w.upgrade() else { return };
            let on = !win.get_library_select_mode();
            win.set_library_select_mode(on);
            if on {
                win.set_page(crate::Page::Library);
                let n = match select_target.get() {
                    Some(id) => {
                        let member_ids = services::collections::member_ids(&db, id);
                        grid.apply_membership(&member_ids)
                    }
                    None => grid.selected_ids().len() as i32,
                };
                win.set_library_selected_count(n);
            } else {
                grid.clear_selection();
                select_target.set(None);
                win.set_library_selected_count(0);
                win.set_library_active_collection_id(-1);
                // Batch-refresh sidebar counts + gallery covers now that the
                // session's incremental add/removes (each already committed
                // live, see apply_marquee) are done.
                collections_page::reload(&win, &db);
                collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_library_marquee_select({
        let db = db.clone();
        let cache = cache.clone();
        let coll_crop_cache = coll_crop_cache.clone();
        let grid = grid.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        move |base, count, x0, y0, x1, y1, columns| {
            let target = select_target.get();
            let n = grid.apply_marquee(base, count, (x0, y0, x1, y1), columns, target);
            if let Some(win) = w.upgrade() {
                win.set_library_selected_count(n);
                // Live-refresh the sidebar dot's count + gallery card cover
                // right after each add/remove, not just on toggle-off.
                if target.is_some() {
                    collections_page::reload(&win, &db);
                    collections_page::load_gallery(&win, &db, &cache, coll_thumb_quality, &coll_crop_cache);
                }
            }
        }
    });

    // Sidebar dot clicked while in select-mode: make that collection the
    // editing target — sync checkboxes to its real membership (most tiles
    // in a general/unfiltered view won't be members) and highlight the dot.
    // From here every tap/drag live-adds/removes via `apply_marquee`.
    window.on_collections_set_select_target({
        let db = db.clone();
        let grid = grid.clone();
        let select_target = select_target.clone();
        let w = window.as_weak();
        move |collection_id| {
            let collection_id = collection_id as i64;
            let member_ids = services::collections::member_ids(&db, collection_id);
            let n = grid.apply_membership(&member_ids);
            select_target.set(Some(collection_id));
            if let Some(win) = w.upgrade() {
                win.set_library_selected_count(n);
                win.set_library_active_collection_id(collection_id as i32);
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
        let db = db.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move || {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
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
        let db = db.clone();
        let source = import_source.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let mut locs = build_import_locations(&settings, &dirs, &starred);

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
        let db = db.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
            if let Some(loc) = locs.iter().find(|l| l.id == id.as_str()) {
                let path = loc.path.as_str();
                if starred.contains(path) {
                    if let Ok(g) = db.lock() { let _ = g.remove_starred_path(path); }
                } else if let Ok(g) = db.lock() { let _ = g.add_starred_path(path); }
            }
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_start_scan({
        let db = db.clone();
        let source = import_source.clone();
        let w = window.as_weak();
        move || {
            let src = source.borrow().clone();
            if src.as_os_str().is_empty() {
                return;
            }
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            import::open_with_source(db.clone(), src, is_dark);
        }
    });

    // ── AI / Faces toggles ─────────────────────────────────────────
    let face_tagger: Arc<Mutex<Option<maple_db::FaceTagger>>> = Arc::new(Mutex::new(None));

    window.on_toggle_ai(|| {
        tracing::info!("AI tagging toggle — wiring for phase 2");
    });

    window.on_toggle_date_view({
        let w = window.as_weak();
        let grid = grid.clone();
        let current_query = current_query.clone();
        let resync_selection = resync_selection.clone();
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

    // ── Theme propagation ──────────────────────────────────────────
    window.on_theme_toggled(|is_dark| {
        settings_window::set_dark(is_dark);
        detail::set_dark(is_dark);
        debug_compare::set_dark(is_dark);
        import::set_dark(is_dark);
        path_template_window::set_dark(is_dark);
    });

    // ── Other secondary windows ────────────────────────────────────
    window.on_settings_clicked({
        let db = db.clone();
        let w = window.as_weak();
        move || {
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            settings_window::open(db.clone(), is_dark);
        }
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

// ── Import location helpers ────────────────────────────────────────

/// Well-known user folders, resolved via OS "known folder" APIs
/// (`directories::UserDirs`) rather than manual home-dir joins, so
/// relocated folders (e.g. OneDrive-redirected Pictures on Windows)
/// are still found.
struct KnownDirs {
    pictures: Option<std::path::PathBuf>,
    desktop: Option<std::path::PathBuf>,
    downloads: Option<std::path::PathBuf>,
}

impl KnownDirs {
    fn from_os() -> Self {
        let user_dirs = directories::UserDirs::new();
        Self {
            pictures: user_dirs.as_ref().and_then(|u| u.picture_dir()).map(Into::into),
            desktop: user_dirs.as_ref().and_then(|u| u.desktop_dir()).map(Into::into),
            downloads: user_dirs.as_ref().and_then(|u| u.download_dir()).map(Into::into),
        }
    }
}

struct LocData {
    id:          String,
    name:        String,
    path:        String,
    count:       i32,
    is_starred:  bool,
    is_selected: bool,
}

fn starred_paths(db: &std::sync::Arc<std::sync::Mutex<maple_db::Database>>) -> std::collections::HashSet<String> {
    db.lock()
        .ok()
        .and_then(|g| g.get_starred_paths().ok())
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn build_import_locations(
    settings: &maple_state::Settings,
    dirs: &KnownDirs,
    starred: &std::collections::HashSet<String>,
) -> Vec<LocData> {
    let mut locs: Vec<LocData> = Vec::new();

    // Pictures directory
    if let Some(pictures) = &dirs.pictures {
        if pictures.is_dir() {
            let path = pictures.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "pictures".into(),
                name: "Pictures Library".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Desktop
    if let Some(desktop) = &dirs.desktop {
        if desktop.is_dir() {
            let path = desktop.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "desktop".into(),
                name: "Desktop".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Downloads
    if let Some(downloads) = &dirs.downloads {
        if downloads.is_dir() {
            let path = downloads.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "downloads".into(),
                name: "Downloads".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Library directory (from settings) if different from Pictures
    let lib = settings.library_dir.clone();
    if lib.is_dir() && dirs.pictures.as_deref() != Some(lib.as_path()) {
        let path = lib.to_string_lossy().into_owned();
        locs.push(LocData {
            id: "library".into(),
            name: "Library Folder".into(),
            is_starred: starred.contains(&path),
            path,
            count: 0,
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn build_import_locations_finds_pictures_desktop_and_dedupes_library() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir(home.join("Pictures")).unwrap();
        std::fs::create_dir(home.join("Desktop")).unwrap();
        // No Downloads dir created — should be skipped.

        let mut settings = maple_state::Settings::default();
        settings.library_dir = home.join("Pictures"); // same as Pictures -> should not duplicate

        let dirs = KnownDirs {
            pictures: Some(home.join("Pictures")),
            desktop: Some(home.join("Desktop")),
            downloads: Some(home.join("Downloads")),
        };

        let starred = HashSet::new();
        let locs = build_import_locations(&settings, &dirs, &starred);

        let ids: Vec<&str> = locs.iter().map(|l| l.id.as_str()).collect();
        assert!(ids.contains(&"pictures"));
        assert!(ids.contains(&"desktop"));
        assert!(!ids.contains(&"downloads"));
        assert!(!ids.contains(&"library"));
    }
}
