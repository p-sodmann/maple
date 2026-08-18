//! Collections inline-page controller.
//!
//! Owns all DB interaction for the CollectionsPage embedded in AppWindow.
//! Called from lib.rs to wire up the collections-* callbacks.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, VecModel};

use maple_db::{SearchQuery, ThumbnailCache};

use crate::rep_crop::{self, CropCache};
use crate::services::collections::{self, load_entries};
use crate::thumbnail;
use crate::transforms::{color_to_hex, hex_to_color, optional_id, record_index, trimmed_name};
use crate::{detail, AppCtx, AppWindow, CollectionEntry, CollectionGalleryItem, ThumbItem};

/// `(rgb_bytes_w_h, filename, image_id)` — Send-safe thumbnail payload
/// produced off the main thread, consumed by `upgrade_in_event_loop`.
type RawThumb = (Option<(Vec<u8>, u32, u32)>, String, i32);

/// Cover-crop render size — matches the People page's `CROP_PX` so both
/// galleries share one visual scale.
const COVER_PX: u32 = 120;

// ── Wiring ────────────────────────────────────────────────────────

/// Populate the page and wire its callbacks.
///
/// Called from `lib.rs` during startup — see [`AppCtx`] for the shared
/// handles the closures clone out of.
pub fn wire(window: &AppWindow, ctx: &AppCtx) {
    let db = &ctx.db;
    let cache = &ctx.cache;
    let thumb_px = ctx.thumb_px;
    let thumb_quality = ctx.thumb_quality;
    let crop_cache = &ctx.coll_crop_cache;

    // Shared record list — populated by load_thumbs, read by on_collections_open_image.
    let records: Arc<Mutex<Vec<maple_db::LibraryImage>>> = Arc::new(Mutex::new(Vec::new()));

    // Populate sidebar + gallery immediately (so dots/cards show even before navigating).
    reload(window, db);
    load_gallery(window, db, cache, thumb_quality, crop_cache);

    window.on_collections_page_shown({
        let db = db.clone();
        let cache = cache.clone();
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move || {
            if let Some(win) = w.upgrade() {
                reload(&win, &db);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });

    window.on_collections_activated({
        let grid = ctx.grid.clone();
        let w = ctx.window.clone();
        let current_query = ctx.current_query.clone();
        let select_target = ctx.select_target.clone();
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
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move |id, name, color| {
            let Some(name) = trimmed_name(&name) else { return };
            let hex = color_to_hex(color);
            collections::rename_collection(&db, id as i64, &name);
            collections::set_collection_color(&db, id as i64, &hex);
            if let Some(win) = w.upgrade() {
                reload_keep_sel(&win, &db);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });

    window.on_collections_edit_delete({
        let db = db.clone();
        let cache = cache.clone();
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move |id| {
            collections::delete_collection(&db, &cache, id as i64);
            if let Some(win) = w.upgrade() {
                reload(&win, &db);
                clear_detail(&win);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });

    window.on_collections_select({
        let db = db.clone();
        let cache = cache.clone();
        let records = records.clone();
        let w = ctx.window.clone();
        move |id| {
            if let Some(win) = w.upgrade() {
                // Clear old thumbs immediately; push fresh detail.
                win.set_collections_thumbs(ModelRc::default());
                push_detail(&win, &db, id);
            }
            load_thumbs(
                id,
                db.clone(),
                cache.clone(),
                thumb_px,
                thumb_quality,
                w.clone(),
                records.clone(),
            );
        }
    });

    window.on_collections_open_image({
        let records = records.clone();
        let db = db.clone();
        let w = ctx.window.clone();
        move |image_id| {
            let records = records.lock().ok().map(|g| g.clone()).unwrap_or_default();
            // Find the index of the clicked image within the loaded records.
            let idx = record_index(&records, image_id as i64);
            if !records.is_empty() {
                let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
                detail::open(records, idx, db.clone(), is_dark);
            }
        }
    });

    window.on_collections_create({
        let db = db.clone();
        let cache = cache.clone();
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move |name, color, parent_id| {
            let Some(name) = trimmed_name(&name) else { return };
            let hex = color_to_hex(color);
            collections::create_collection(&db, &name, &hex, optional_id(parent_id));
            if let Some(win) = w.upgrade() {
                reload(&win, &db);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });

    window.on_collections_delete({
        let db = db.clone();
        let cache = cache.clone();
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move |id| {
            collections::delete_collection(&db, &cache, id as i64);
            if let Some(win) = w.upgrade() {
                reload(&win, &db);
                clear_detail(&win);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });

    window.on_collections_rename({
        let db = db.clone();
        let cache = cache.clone();
        let crop_cache = crop_cache.clone();
        let w = ctx.window.clone();
        move |id, name| {
            let Some(name) = trimmed_name(&name) else { return };
            collections::rename_collection(&db, id as i64, &name);
            if let Some(win) = w.upgrade() {
                reload_keep_sel(&win, &db);
                load_gallery(&win, &db, &cache, thumb_quality, &crop_cache);
            }
        }
    });
}

// ── Public API ────────────────────────────────────────────────────

/// Reload all collections from DB, push to `collections-list` and
/// `sidebar-collections`, clear the detail panel.
pub fn reload(window: &AppWindow, db: &Arc<Mutex<maple_db::Database>>) {
    let entries = load_entries(db);
    window.set_collections_list(make_model(entries.clone()));
    window.set_sidebar_collections(make_model(entries));
}

/// Reload collections and keep the detail panel for the currently-selected id.
pub fn reload_keep_sel(window: &AppWindow, db: &Arc<Mutex<maple_db::Database>>) {
    let entries = load_entries(db);
    window.set_collections_list(make_model(entries.clone()));
    window.set_sidebar_collections(make_model(entries));
    // Re-push sel-* for the selected id so the count stays fresh.
    let sel_id = window.get_collections_selected_id();
    if sel_id >= 0 {
        push_detail(window, db, sel_id);
    }
}

/// Reload the gallery grid (People-page-style cards with a computed cover
/// image) and kick off async cover renders for any collection whose cover
/// isn't already in `crop_cache`.
///
/// Must be called from the Slint event-loop thread. Mirrors
/// `PeoplePage::load` (`people_page.rs`) — see [`rep_crop`] for the shared
/// decode/cache logic.
pub fn load_gallery(
    window: &AppWindow,
    db: &Arc<Mutex<maple_db::Database>>,
    cache: &Arc<ThumbnailCache>,
    quality: u8,
    crop_cache: &CropCache,
) {
    let colls = collections::load_all_with_representatives(db);

    let items: Vec<CollectionGalleryItem> = colls
        .iter()
        .map(|c| {
            let cached = crop_cache.lock().ok().and_then(|cc| {
                cc.get(&c.id)
                    .and_then(|(rep_id, pixels)| (Some(*rep_id) == c.image_id).then(|| pixels.clone()))
            });
            let (cover_image, cover_loaded) = match cached {
                Some(pixels) => (rep_crop::image_from_rgb(&pixels, COVER_PX), true),
                None => (Image::default(), false),
            };
            CollectionGalleryItem {
                id: c.id as i32,
                name: SharedString::from(c.name.as_str()),
                color: hex_to_color(&c.color),
                image_count: c.image_count as i32,
                cover_image,
                cover_loaded,
            }
        })
        .collect();

    let model = ModelRc::from(Rc::new(VecModel::from(items)));
    window.set_collections_gallery_items(model);

    let w = window.as_weak();
    for (idx, coll) in colls.into_iter().enumerate() {
        let (Some(path), Some(image_id)) = (coll.image_path, coll.image_id) else {
            continue;
        };
        let already_cached = crop_cache.lock().ok().is_some_and(|cc| {
            cc.get(&coll.id).is_some_and(|(cached_id, _)| *cached_id == image_id)
        });
        if already_cached {
            continue;
        }

        let w2 = w.clone();
        let expected_id = coll.id as i32;
        let coll_id = coll.id;
        let crop_cache = crop_cache.clone();
        let cache = cache.clone();
        std::thread::spawn(move || {
            let redb_cached = cache.get_cover(coll_id);
            let cache_for_store = cache.clone();
            let Ok(pixels) = rep_crop::extract_and_cache(
                &path,
                None,
                COVER_PX,
                quality,
                redb_cached,
                |webp| {
                    if let Err(e) = cache_for_store.insert_cover(coll_id, webp) {
                        tracing::warn!("insert_cover {coll_id}: {e}");
                    }
                },
            ) else {
                return;
            };
            let _ = w2.upgrade_in_event_loop(move |win| {
                let pixels = Arc::new(pixels);
                if let Ok(mut c) = crop_cache.lock() {
                    c.insert(coll_id, (image_id, pixels.clone()));
                }
                let model = win.get_collections_gallery_items();
                if let Some(mut item) = model.row_data(idx) {
                    if item.id == expected_id {
                        item.cover_image = rep_crop::image_from_rgb(&pixels, COVER_PX);
                        item.cover_loaded = true;
                        model.set_row_data(idx, item);
                    }
                }
            });
        });
    }
}

/// Push detail properties for one collection (no thumbnail load).
pub fn push_detail(window: &AppWindow, db: &Arc<Mutex<maple_db::Database>>, id: i32) {
    if let Some(d) = collections::load_collection_detail(db, id) {
        window.set_collections_sel_name(SharedString::from(d.name));
        window.set_collections_sel_color(d.color);
        window.set_collections_sel_count(d.image_count);
        window.set_collections_sel_parent_name(SharedString::from(d.parent_name));
    }
}

/// Clear detail panel (after delete or deselect).
pub fn clear_detail(window: &AppWindow) {
    window.set_collections_selected_id(-1);
    window.set_collections_sel_name(SharedString::default());
    window.set_collections_sel_count(0);
    window.set_collections_thumbs(ModelRc::default());
    window.set_collections_sel_parent_name(SharedString::default());
}

/// Spawn a background thread that loads up to 12 thumbnails for `id`.
///
/// Raw RGB bytes are decoded off the main thread; the final conversion to
/// `slint::Image` (which is !Send) happens on the event loop via
/// `upgrade_in_event_loop`. The loaded `LibraryImage` records are stored in
/// `records_out` (main-thread access via Arc<Mutex>) so `open_image` can open
/// the detail viewer with the full record list.
pub fn load_thumbs(
    id: i32,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<ThumbnailCache>,
    thumb_px: u32,
    thumb_quality: u8,
    weak: slint::Weak<AppWindow>,
    records_out: Arc<Mutex<Vec<maple_db::LibraryImage>>>,
) {
    std::thread::spawn(move || {
        let images = maple_db::lock_db(&db)
            .search_images(&SearchQuery::default().with_collection(id as i64).with_limit(12))
            .unwrap_or_default();

        // Produce Send-safe raw RGB tuples. slint::Image is built on the
        // main thread inside upgrade_in_event_loop below.
        let raws: Vec<RawThumb> = images
            .iter()
            .map(|img| {
                let rgb = load_thumb_rgb(img, thumb_px, thumb_quality, &cache).ok();
                let name = img
                    .path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_default();
                (rgb, name, img.id as i32)
            })
            .collect();

        let _ = weak.upgrade_in_event_loop(move |win| {
            // Store records so the open-image callback can pass them to the detail viewer.
            if let Ok(mut guard) = records_out.lock() {
                *guard = images;
            }
            let items: Vec<ThumbItem> = raws
                .into_iter()
                .map(|(raw_rgb, name, img_id)| {
                    let (img, loaded) = match raw_rgb {
                        Some((rgb, w, h)) => (rgb_to_image(&rgb, w, h), true),
                        None => (Image::default(), false),
                    };
                    ThumbItem {
                        id: img_id,
                        image: img,
                        name: SharedString::from(name.as_str()),
                        loaded,
                        unsupported: !loaded,
                        stack_size: 0,
                        score: SharedString::default(),
                        selected: false,
                    }
                })
                .collect();
            win.set_collections_thumbs(ModelRc::from(Rc::new(VecModel::from(items))));
        });
    });
}

// ── Internals ─────────────────────────────────────────────────────

fn make_model(entries: Vec<CollectionEntry>) -> ModelRc<CollectionEntry> {
    ModelRc::from(Rc::new(VecModel::from(entries)))
}

fn load_thumb_rgb(
    rec: &maple_db::LibraryImage,
    max_size: u32,
    quality: u8,
    cache: &ThumbnailCache,
) -> anyhow::Result<(Vec<u8>, u32, u32)> {
    if let Some(hash) = rec.hash {
        if let Some(webp) = cache.get(&hash) {
            return thumbnail::decode_webp_rgb(&webp);
        }
    }
    let (rgb, w, h) = thumbnail::render_to_rgb(&rec.path, max_size)?;
    if let Some(hash) = rec.hash {
        let webp = thumbnail::encode_webp_rgb(&rgb, w, h, quality);
        if let Err(e) = cache.insert(&hash, &webp) {
            tracing::warn!("Thumb cache write failed for {}: {e}", rec.path.display());
        }
    }
    Ok((rgb, w, h))
}

fn rgb_to_image(rgb: &[u8], width: u32, height: u32) -> Image {
    if rgb.len() != (width as usize * height as usize * 3) {
        return Image::default();
    }
    let mut buf = SharedPixelBuffer::<Rgb8Pixel>::new(width, height);
    buf.make_mut_bytes().copy_from_slice(rgb);
    Image::from_rgb8(buf)
}
