//! Trivial DB-wide clear operations for the Settings window's debug actions.

use std::sync::{Arc, Mutex};

use maple_db::lock_db;

/// Delete all AI descriptions. Returns the number of rows deleted, or
/// `None` on DB error.
pub fn clear_ai_descriptions(db: &Arc<Mutex<maple_db::Database>>) -> Option<usize> {
    lock_db(db).clear_all_ai_descriptions().ok()
}

/// Delete all face detections and persons. Returns `(faces, persons)`
/// deleted, or `None` on DB error.
pub fn clear_face_data(db: &Arc<Mutex<maple_db::Database>>) -> Option<(usize, usize)> {
    lock_db(db).clear_all_face_data().ok()
}
