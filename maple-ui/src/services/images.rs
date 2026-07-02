//! Image/detail queries — DB access for the library grid and the detail
//! window, no Slint types (callers build `ModelRc`s from the plain values
//! returned here).

use std::sync::{Arc, Mutex};

use maple_db::{FaceDetection, LibraryImage, SearchQuery};

use crate::services::faces::load_embedding_matrix;
use crate::transforms::{hex_to_color, EmbeddingMatrix};
use crate::CollectionChip;

/// Run a library search. Empty Vec on DB error (matches existing callback
/// behavior — the grid just renders nothing rather than surfacing a dialog).
pub fn search_library(db: &Arc<Mutex<maple_db::Database>>, query: &SearchQuery) -> Vec<LibraryImage> {
    db.lock()
        .ok()
        .and_then(|d| d.search_images(query).ok())
        .unwrap_or_default()
}

/// Bundled fetch for opening the detail view on one image: its face
/// detections, the known-person embedding matrix (for suggestion matching),
/// and its collection chip memberships.
pub struct ImageDetail {
    pub faces: Vec<FaceDetection>,
    pub embeddings: EmbeddingMatrix,
    pub collection_chips: Vec<CollectionChip>,
}

pub fn load_image_detail(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> ImageDetail {
    let faces = db
        .lock()
        .ok()
        .and_then(|g| g.faces_for_image(image_id).ok())
        .unwrap_or_default();
    let embeddings = load_embedding_matrix(db);
    let collection_chips = load_collection_chips(db, image_id);
    ImageDetail { faces, embeddings, collection_chips }
}

/// Collection chips for one image (used both on initial load and after
/// add/remove-from-collection actions).
pub fn load_collection_chips(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> Vec<CollectionChip> {
    db.lock()
        .ok()
        .and_then(|d| d.collections_for_image(image_id).ok())
        .unwrap_or_default()
        .iter()
        .map(|c| CollectionChip {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: hex_to_color(&c.color),
        })
        .collect()
}

/// AI descriptions and full EXIF tag list for the info popup. Each is a
/// `(label, value)` pair in DB-returned order.
pub struct ImageInfoData {
    pub ai_descriptions: Vec<(String, String)>,
    pub exif_tags: Vec<(String, String)>,
}

pub fn load_image_info_data(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> ImageInfoData {
    let ai_descriptions = db
        .lock()
        .ok()
        .and_then(|g| g.ai_descriptions_for_image(image_id).ok())
        .unwrap_or_default();
    let exif_tags = db
        .lock()
        .ok()
        .and_then(|g| g.exif_tags_for_image(image_id).ok())
        .unwrap_or_default();
    ImageInfoData { ai_descriptions, exif_tags }
}
