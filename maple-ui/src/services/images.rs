//! Image/detail queries — DB access for the library grid and the detail
//! window, no Slint types (callers build `ModelRc`s from the plain values
//! returned here).

use std::sync::{Arc, Mutex};

use maple_db::{lock_db, FaceDetection, LibraryImage, SearchQuery};

use crate::services::faces::load_embedding_matrix;
use crate::transforms::{hex_to_color, EmbeddingMatrix};
use crate::CollectionChip;

/// Run a library search. Empty Vec on DB error (matches existing callback
/// behavior — the grid just renders nothing rather than surfacing a dialog).
pub fn search_library(db: &Arc<Mutex<maple_db::Database>>, query: &SearchQuery) -> Vec<LibraryImage> {
    lock_db(db).search_images(query).unwrap_or_default()
}

/// Row count the search's filters match, ignoring its `limit`/`offset` —
/// the total the library grid pages through. `None` on a DB error or for a
/// query with no countable total (see `Database::count_images`).
pub fn count_library(db: &Arc<Mutex<maple_db::Database>>, query: &SearchQuery) -> Option<usize> {
    lock_db(db).count_images(query).ok().flatten()
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
    let faces = lock_db(db).faces_for_image(image_id).unwrap_or_default();
    let embeddings = load_embedding_matrix(db);
    let collection_chips = load_collection_chips(db, image_id);
    ImageDetail { faces, embeddings, collection_chips }
}

/// Collection chips for one image (used both on initial load and after
/// add/remove-from-collection actions).
pub fn load_collection_chips(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> Vec<CollectionChip> {
    lock_db(db)
        .collections_for_image(image_id)
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
    // Two cheap indexed reads about the same image — one guard, so the popup
    // can't show descriptions and tags from either side of a concurrent write.
    let guard = lock_db(db);
    let ai_descriptions = guard.ai_descriptions_for_image(image_id).unwrap_or_default();
    let exif_tags = guard.exif_tags_for_image(image_id).unwrap_or_default();
    drop(guard);
    ImageInfoData { ai_descriptions, exif_tags }
}

/// Debug helper: cosine similarity between two images' stored stack-detection
/// embeddings, keyed by the current `StackSettings::algorithm_key()`. Returns
/// `(filename_a, filename_b, similarity)` on success, or a user-facing error
/// message (unknown id / no stored embedding / DB error).
pub fn compare_embeddings(
    db: &Arc<Mutex<maple_db::Database>>,
    id_a: i64,
    id_b: i64,
) -> Result<(String, String, f32), String> {
    let algorithm = maple_state::Settings::load().stacks.algorithm_key();
    let guard = lock_db(db);

    let display_name = |id: i64| -> Result<String, String> {
        let img = guard
            .image_by_id(id)
            .map_err(|e| e.to_string())?
            .ok_or_else(|| format!("No image with id {id}."))?;
        Ok(img.meta.filename.unwrap_or_else(|| img.path.to_string_lossy().into_owned()))
    };
    let name_a = display_name(id_a)?;
    let name_b = display_name(id_b)?;

    let similarity = guard
        .similarity_for_images(id_a, id_b, &algorithm)
        .map_err(|e| e.to_string())?
        .ok_or_else(|| {
            format!(
                "No stored embedding for one or both images under algorithm \"{algorithm}\". \
                 Enable stack detection in Settings and let the background hasher run."
            )
        })?;

    Ok((name_a, name_b, similarity))
}
