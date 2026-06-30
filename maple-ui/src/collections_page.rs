//! Collections inline-page controller.
//!
//! Owns all DB interaction for the CollectionsPage embedded in AppWindow.
//! Called from lib.rs to wire up the collections-* callbacks.

use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{Image, ModelRc, Rgb8Pixel, SharedPixelBuffer, SharedString, VecModel};

use maple_db::{SearchQuery, ThumbnailCache};

use crate::thumbnail;
use crate::{AppWindow, CollectionEntry, ThumbItem};

/// `(rgb_bytes_w_h, filename, image_id)` — Send-safe thumbnail payload
/// produced off the main thread, consumed by `upgrade_in_event_loop`.
type RawThumb = (Option<(Vec<u8>, u32, u32)>, String, i32);

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

/// Push detail properties for one collection (no thumbnail load).
pub fn push_detail(window: &AppWindow, db: &Arc<Mutex<maple_db::Database>>, id: i32) {
    if let Ok(g) = db.lock() {
        if let Some(c) = g.collection_by_id(id as i64).ok().flatten() {
            window.set_collections_sel_name(SharedString::from(c.name));
            window.set_collections_sel_color(hex_to_color(&c.color));
            window.set_collections_sel_count(c.image_count as i32);
            let parent_name = c.parent_id
                .and_then(|pid| g.collection_by_id(pid).ok().flatten())
                .map(|p| p.name)
                .unwrap_or_default();
            window.set_collections_sel_parent_name(SharedString::from(parent_name));
        }
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
        let images = db
            .lock()
            .ok()
            .and_then(|g| {
                g.search_images(&SearchQuery::default().with_collection(id as i64).with_limit(12))
                    .ok()
            })
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
                    }
                })
                .collect();
            win.set_collections_thumbs(ModelRc::from(Rc::new(VecModel::from(items))));
        });
    });
}

// ── Internals ─────────────────────────────────────────────────────

fn load_entries(db: &Arc<Mutex<maple_db::Database>>) -> Vec<CollectionEntry> {
    let colls = db
        .lock()
        .ok()
        .and_then(|g| g.all_collections().ok())
        .unwrap_or_default();
    flatten_tree(&colls)
}

/// DFS-flatten the collection tree into a display list with depth info.
///
/// `all_collections` returns rows sorted by name, so siblings within each
/// parent level already arrive in alphabetical order.
fn flatten_tree(colls: &[maple_db::Collection]) -> Vec<CollectionEntry> {
    use std::collections::HashMap;

    // Map parent_id → indices of children (in the `colls` slice)
    let mut children: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut root_indices: Vec<usize> = Vec::new();

    for (idx, c) in colls.iter().enumerate() {
        match c.parent_id {
            Some(pid) => children.entry(pid).or_default().push(idx),
            None => root_indices.push(idx),
        }
    }

    // Iterative DFS using a stack of (index, depth).
    // Roots are pushed in reverse order so the first root pops first.
    let mut result: Vec<CollectionEntry> = Vec::with_capacity(colls.len());
    let mut stack: Vec<(usize, i32)> = root_indices.into_iter().rev().map(|i| (i, 0)).collect();

    while let Some((idx, depth)) = stack.pop() {
        let c = &colls[idx];
        result.push(CollectionEntry {
            id: c.id as i32,
            name: SharedString::from(c.name.as_str()),
            color: hex_to_color(&c.color),
            image_count: c.image_count as i32,
            parent_id: c.parent_id.map(|id| id as i32).unwrap_or(-1),
            depth,
        });
        // Push children in reverse sorted order so they pop in sorted order.
        if let Some(kids) = children.get(&c.id) {
            for &kid_idx in kids.iter().rev() {
                stack.push((kid_idx, depth + 1));
            }
        }
    }

    result
}

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

pub fn hex_to_color(hex: &str) -> slint::Color {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return slint::Color::from_rgb_u8(r, g, b);
        }
    }
    slint::Color::from_rgb_u8(0xa0, 0x9a, 0x8e)
}
