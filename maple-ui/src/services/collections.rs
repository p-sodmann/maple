//! Collection queries — DB access + data-shape composition for the
//! Collections page and the detail-window add-to-collection picker.

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

use slint::SharedString;

use maple_db::SearchQuery;

use crate::transforms::hex_to_color;
use crate::{CollectionChip, CollectionEntry};

/// Load all collection chips for the add-to-collection picker in the detail
/// window. Returns an empty Vec on DB error.
pub fn load_all_collections(db: &Arc<Mutex<maple_db::Database>>) -> Vec<CollectionChip> {
    db.lock()
        .ok()
        .and_then(|g| g.all_collections().ok())
        .unwrap_or_default()
        .iter()
        .map(|c| CollectionChip {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: hex_to_color(&c.color),
        })
        .collect()
}

/// Flattened collection tree for the Collections page list + sidebar.
pub fn load_entries(db: &Arc<Mutex<maple_db::Database>>) -> Vec<CollectionEntry> {
    let colls = db
        .lock()
        .ok()
        .and_then(|g| g.all_collections().ok())
        .unwrap_or_default();
    flatten_tree(&colls)
}

/// Detail-panel fields for one collection: name, color, image count, and
/// its parent's name (empty string if top-level or parent lookup fails).
pub struct CollectionDetail {
    pub name: String,
    pub color: slint::Color,
    pub image_count: i32,
    pub parent_name: String,
}

pub fn load_collection_detail(
    db: &Arc<Mutex<maple_db::Database>>,
    id: i32,
) -> Option<CollectionDetail> {
    let g = db.lock().ok()?;
    let c = g.collection_by_id(id as i64).ok().flatten()?;
    let parent_name = c
        .parent_id
        .and_then(|pid| g.collection_by_id(pid).ok().flatten())
        .map(|p| p.name)
        .unwrap_or_default();
    Some(CollectionDetail {
        name: c.name,
        color: hex_to_color(&c.color),
        image_count: c.image_count as i32,
        parent_name,
    })
}

/// Create a collection; `parent_id: None` for top-level. Returns `false` on
/// DB error (best-effort, matches existing callback behavior).
pub fn create_collection(
    db: &Arc<Mutex<maple_db::Database>>,
    name: &str,
    hex_color: &str,
    parent_id: Option<i64>,
) -> bool {
    db.lock()
        .ok()
        .and_then(|g| g.create_collection(name, hex_color, parent_id).ok())
        .is_some()
}

pub fn rename_collection(db: &Arc<Mutex<maple_db::Database>>, id: i64, name: &str) -> bool {
    db.lock()
        .ok()
        .and_then(|g| g.rename_collection(id, name).ok())
        .is_some()
}

pub fn set_collection_color(db: &Arc<Mutex<maple_db::Database>>, id: i64, hex_color: &str) -> bool {
    db.lock()
        .ok()
        .and_then(|g| g.set_collection_color(id, hex_color).ok())
        .is_some()
}

/// Delete a collection and best-effort evict its cached cover crop.
pub fn delete_collection(
    db: &Arc<Mutex<maple_db::Database>>,
    cache: &maple_db::ThumbnailCache,
    id: i64,
) -> bool {
    let ok = db
        .lock()
        .ok()
        .and_then(|g| g.delete_collection(id).ok())
        .is_some();
    if ok {
        if let Err(e) = cache.remove_cover(id) {
            tracing::warn!("remove_cover {id}: {e}");
        }
    }
    ok
}

/// Add one image to a collection and refresh its representative cover.
/// Returns `false` on DB error (best-effort).
pub fn add_image_to_collection(
    db: &Arc<Mutex<maple_db::Database>>,
    collection_id: i64,
    image_id: i64,
) -> bool {
    let Ok(g) = db.lock() else { return false };
    if let Err(e) = g.add_image_to_collection(collection_id, image_id) {
        tracing::warn!("add_image_to_collection {image_id} -> {collection_id}: {e}");
        return false;
    }
    if let Err(e) = g.update_collection_representative(collection_id) {
        tracing::warn!("update_collection_representative {collection_id}: {e}");
    }
    true
}

/// Remove one image from a collection and refresh its representative cover.
/// Returns `false` on DB error (best-effort).
pub fn remove_image_from_collection(
    db: &Arc<Mutex<maple_db::Database>>,
    collection_id: i64,
    image_id: i64,
) -> bool {
    let Ok(g) = db.lock() else { return false };
    if let Err(e) = g.remove_image_from_collection(collection_id, image_id) {
        tracing::warn!("remove_image_from_collection {image_id} <- {collection_id}: {e}");
        return false;
    }
    if let Err(e) = g.update_collection_representative(collection_id) {
        tracing::warn!("update_collection_representative {collection_id}: {e}");
    }
    true
}

/// Every image id currently in `collection_id`. Used to sync select-mode
/// checkboxes to real membership when a sidebar dot becomes the "editing
/// target" (see `grid::apply_membership`).
pub fn member_ids(db: &Arc<Mutex<maple_db::Database>>, collection_id: i64) -> HashSet<i64> {
    db.lock()
        .ok()
        .and_then(|g| g.search_images(&SearchQuery::default().with_collection(collection_id)).ok())
        .map(|imgs| imgs.into_iter().map(|i| i.id).collect())
        .unwrap_or_default()
}

/// Load every collection with its representative cover-image path (People-
/// page equivalent for the Collections gallery).
pub fn load_all_with_representatives(
    db: &Arc<Mutex<maple_db::Database>>,
) -> Vec<maple_db::CollectionWithRep> {
    db.lock()
        .ok()
        .and_then(|g| g.all_collections_with_representatives().ok())
        .unwrap_or_default()
}

/// DFS-flatten the collection tree into a display list with depth info.
///
/// `all_collections` returns rows sorted by name, so siblings within each
/// parent level already arrive in alphabetical order.
pub fn flatten_tree(colls: &[maple_db::Collection]) -> Vec<CollectionEntry> {
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

#[cfg(test)]
mod tests {
    use super::*;

    fn coll(id: i64, name: &str, parent_id: Option<i64>) -> maple_db::Collection {
        maple_db::Collection {
            id,
            name: name.into(),
            color: "#336699".into(),
            created_at: 0,
            image_count: 0,
            parent_id,
        }
    }

    #[test]
    fn flatten_tree_orders_roots_before_children_alphabetically() {
        // As returned by `all_collections` (sorted by name): Nested is a
        // child of Trip, both are top-level-adjacent in the input slice.
        let colls = vec![coll(1, "Nested", Some(2)), coll(2, "Trip", None)];
        let entries = flatten_tree(&colls);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["Trip", "Nested"]);
        assert_eq!(entries[0].depth, 0);
        assert_eq!(entries[1].depth, 1);
        assert_eq!(entries[1].parent_id, 2);
    }

    #[test]
    fn flatten_tree_marks_top_level_parent_id_as_negative_one() {
        let colls = vec![coll(1, "Solo", None)];
        let entries = flatten_tree(&colls);
        assert_eq!(entries[0].parent_id, -1);
    }
}
