//! Person/face-queue queries for the People page and the face-tagging window.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use maple_db::{lock_db, FaceDetection, PersonWithRep};

use crate::transforms::is_real_detection;

/// All named persons with their representative-face crop info, for the
/// People page grid.
pub fn load_all_persons(db: &Arc<Mutex<maple_db::Database>>) -> Vec<PersonWithRep> {
    lock_db(db).all_persons_with_representatives().unwrap_or_default()
}

/// Count of detected faces not yet assigned to a person or skipped.
pub fn load_untagged_face_count(db: &Arc<Mutex<maple_db::Database>>) -> usize {
    lock_db(db).untagged_face_count().unwrap_or(0)
}

/// One face awaiting review in the face-tagging queue, with the source
/// image path needed to render its crop.
pub struct UntaggedFace {
    pub path: PathBuf,
    pub face: FaceDetection,
}

/// Collect every real untagged non-skipped face across all present images.
pub fn collect_untagged_faces(db: &Arc<Mutex<maple_db::Database>>) -> Vec<UntaggedFace> {
    let guard = lock_db(db);
    let image_ids = guard.images_with_untagged_faces().unwrap_or_default();
    let mut out = Vec::new();
    for image_id in image_ids {
        let Some(img) = guard.image_by_id(image_id).ok().flatten() else { continue };
        for face in guard.faces_for_image(image_id).unwrap_or_default() {
            if face.person_id.is_none() && !face.skipped && is_real_detection(&face) {
                out.push(UntaggedFace { path: img.path.clone(), face });
            }
        }
    }
    out
}
