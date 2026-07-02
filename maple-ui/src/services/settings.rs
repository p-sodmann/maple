//! Trivial DB-wide clear operations for the Settings window's debug actions.

use std::sync::{Arc, Mutex};

/// Delete all AI descriptions. Returns the number of rows deleted, or
/// `None` on DB error.
pub fn clear_ai_descriptions(db: &Arc<Mutex<maple_db::Database>>) -> Option<usize> {
    db.lock().ok().and_then(|g| g.clear_all_ai_descriptions().ok())
}

/// Delete all face detections and persons. Returns `(faces, persons)`
/// deleted, or `None` on DB error.
pub fn clear_face_data(db: &Arc<Mutex<maple_db::Database>>) -> Option<(usize, usize)> {
    db.lock().ok().and_then(|g| g.clear_all_face_data().ok())
}
