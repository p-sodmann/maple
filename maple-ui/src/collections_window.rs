//! Collections manager window — provides `all_chips` for the detail-window
//! add-to-collection picker.  The old pop-up manager is no longer opened from
//! the main window (the inline CollectionsPage replaced it).

use std::sync::{Arc, Mutex};

/// Load all collection chips for the add-to-collection picker in the detail
/// window.  Returns an empty Vec on DB error.
pub fn all_chips(db: &Arc<Mutex<maple_db::Database>>) -> Vec<crate::CollectionChip> {
    db.lock()
        .ok()
        .and_then(|g| g.all_collections().ok())
        .unwrap_or_default()
        .iter()
        .map(|c| crate::CollectionChip {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: crate::transforms::hex_to_color(&c.color),
        })
        .collect()
}
