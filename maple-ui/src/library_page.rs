//! Library-page controller.
//!
//! Owns the wiring for the Library view embedded in AppWindow: the grid and
//! its endless-scroll feed, the navigation callbacks that swap the active
//! query (sidebar entry, filter chip, search box, tile activation), and the
//! collection mass-select mode driven from the sidebar dots.
//!
//! Called from `lib.rs` during startup — see [`crate::AppCtx`] for the shared
//! handles these functions clone out of.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{SharedString, Timer, TimerMode};

use maple_db::SearchQuery;

use crate::grid::LibraryGrid;
use crate::{collections_page, detail, services, AppCtx, AppWindow};

/// Search-input debounce (ms).
const SEARCH_DEBOUNCE_MS: u64 = 200;

/// Build the grid and bind its models to the shell window.
pub fn create_grid(
    window: &AppWindow,
    settings: &maple_state::Settings,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<maple_db::ThumbnailCache>,
) -> LibraryGrid {
    let grid = LibraryGrid::new(
        db,
        cache,
        settings.thumbnails.quality,
        settings.thumbnails.size,
    );
    window.set_library_items(grid.model());
    window.set_library_date_groups(grid.date_groups_model());
    window.set_library_cell_size(settings.thumbnails.size as f32);
    grid
}

/// Wire the grid's count/paging feed and load the first page.
///
/// Must run before anything can scroll the grid: `request-more` fires from a
/// `changed want-count` binding in `library.slint`, so the handler has to be
/// installed before the initial `load` puts items on screen.
pub fn wire_grid(window: &AppWindow, ctx: &AppCtx) {
    // The grid only holds the pages loaded so far, so the header's photo
    // count comes from a COUNT query rather than from the item model.
    ctx.grid.on_total_count({
        let w = ctx.window.clone();
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
        let grid = ctx.grid.clone();
        move |rows| grid.request_more(rows)
    });

    // Library is the default page — load the first page immediately.
    ctx.grid.load(SearchQuery::default());
}

/// Wire the callbacks that swap which records the grid is showing.
pub fn wire_navigation(window: &AppWindow, ctx: &AppCtx) {
    window.on_library_shown({
        let grid = ctx.grid.clone();
        let current_query = ctx.current_query.clone();
        let select_target = ctx.select_target.clone();
        let w = ctx.window.clone();
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
        let grid = ctx.grid.clone();
        let current_query = ctx.current_query.clone();
        let select_target = ctx.select_target.clone();
        let w = ctx.window.clone();
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
        let grid = ctx.grid.clone();
        let search_debounce = search_debounce.clone();
        let current_query = ctx.current_query.clone();
        let select_target = ctx.select_target.clone();
        let w = ctx.window.clone();
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
        let records = ctx.grid.records();
        let db = ctx.db.clone();
        let w = ctx.window.clone();
        move |idx| {
            let snapshot = records.borrow().clone();
            if (idx as usize) < snapshot.len() {
                let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
                detail::open(snapshot, idx as usize, db.clone(), is_dark);
            }
        }
    });
}

/// Wire the library mass-select (sidebar "Add Images" toggle → collection dots).
///
/// Entering select-mode doesn't require a target collection — you can
/// just tick photos, then click a sidebar dot to add them (handled
/// below). But if a target is already set (from following a gallery card
/// into a filtered view, or a previous dot click this session), checkbox
/// state is immediately synced to that collection's real membership so
/// deselecting one is a removal, matching a freshly-clicked dot.
pub fn wire_mass_select(window: &AppWindow, ctx: &AppCtx) {
    window.on_toggle_select_mode({
        let db = ctx.db.clone();
        let grid = ctx.grid.clone();
        let select_target = ctx.select_target.clone();
        let cache = ctx.cache.clone();
        let coll_crop_cache = ctx.coll_crop_cache.clone();
        let thumb_quality = ctx.thumb_quality;
        let w = ctx.window.clone();
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
                collections_page::load_gallery(&win, &db, &cache, thumb_quality, &coll_crop_cache);
            }
        }
    });

    window.on_library_marquee_select({
        let db = ctx.db.clone();
        let cache = ctx.cache.clone();
        let coll_crop_cache = ctx.coll_crop_cache.clone();
        let grid = ctx.grid.clone();
        let select_target = ctx.select_target.clone();
        let thumb_quality = ctx.thumb_quality;
        let w = ctx.window.clone();
        move |base, count, x0, y0, x1, y1, columns| {
            let target = select_target.get();
            let n = grid.apply_marquee(base, count, (x0, y0, x1, y1), columns, target);
            if let Some(win) = w.upgrade() {
                win.set_library_selected_count(n);
                // Live-refresh the sidebar dot's count + gallery card cover
                // right after each add/remove, not just on toggle-off.
                if target.is_some() {
                    collections_page::reload(&win, &db);
                    collections_page::load_gallery(&win, &db, &cache, thumb_quality, &coll_crop_cache);
                }
            }
        }
    });

    // Sidebar dot clicked while in select-mode: make that collection the
    // editing target — sync checkboxes to its real membership (most tiles
    // in a general/unfiltered view won't be members) and highlight the dot.
    // From here every tap/drag live-adds/removes via `apply_marquee`.
    window.on_collections_set_select_target({
        let db = ctx.db.clone();
        let grid = ctx.grid.clone();
        let select_target = ctx.select_target.clone();
        let w = ctx.window.clone();
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
}
