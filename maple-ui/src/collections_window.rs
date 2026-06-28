//! Collections manager window controller (Slint port of
//! views/library/collection_manager.rs).
//!
//! A second top-level Window (thread_local! singleton) for creating and
//! deleting collections. The collection-add action from the detail window goes
//! through the inline picker panel in detail.slint instead.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use crate::{CollectionEntry, CollectionsWindow};

struct CollWin {
    window: CollectionsWindow,
    db: Arc<Mutex<maple_db::Database>>,
}

thread_local! {
    static COLLECTIONS: RefCell<Option<CollWin>> = const { RefCell::new(None) };
}

/// Open (or reuse) the collections manager window.
pub fn open(db: Arc<Mutex<maple_db::Database>>) {
    if COLLECTIONS.with(|c| c.borrow().is_none()) {
        match build(db.clone()) {
            Ok(cw) => COLLECTIONS.with(|cell| *cell.borrow_mut() = Some(cw)),
            Err(e) => {
                tracing::error!("Failed to build collections window: {e}");
                return;
            }
        }
    }
    COLLECTIONS.with(|cell| {
        let guard = cell.borrow();
        if let Some(cw) = guard.as_ref() {
            reload_model(&cw.window, &cw.db);
            if let Err(e) = cw.window.show() {
                tracing::error!("Failed to show collections window: {e}");
            }
        }
    });
}

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
            color: parse_hex_color(&c.color),
        })
        .collect()
}

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<CollWin, slint::PlatformError> {
    let window = CollectionsWindow::new()?;

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_create_collection({
        let w = window.as_weak();
        let db = db.clone();
        move |name, color| {
            let name = name.trim().to_owned();
            if name.is_empty() {
                return;
            }
            let hex = format!(
                "#{:02x}{:02x}{:02x}",
                color.red(),
                color.green(),
                color.blue(),
            );
            let result = db.lock().ok().and_then(|g| g.create_collection(&name, &hex).ok());
            if let Some(w) = w.upgrade() {
                if result.is_some() {
                    reload_model(&w, &db);
                    w.set_status_text(SharedString::new());
                } else {
                    w.set_status_text("Failed to create collection.".into());
                }
            }
        }
    });

    window.on_delete_collection({
        let w = window.as_weak();
        let db = db.clone();
        move |id| {
            let result = db.lock().ok().and_then(|g| g.delete_collection(id as i64).ok());
            if let Some(w) = w.upgrade() {
                if result.is_some() {
                    reload_model(&w, &db);
                    w.set_status_text(SharedString::new());
                } else {
                    w.set_status_text("Failed to delete collection.".into());
                }
            }
        }
    });

    Ok(CollWin { window, db })
}

fn reload_model(window: &CollectionsWindow, db: &Arc<Mutex<maple_db::Database>>) {
    let colls = db
        .lock()
        .ok()
        .and_then(|g| g.all_collections().ok())
        .unwrap_or_default();
    let entries: Vec<CollectionEntry> = colls
        .iter()
        .map(|c| CollectionEntry {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: parse_hex_color(&c.color),
            image_count: 0,
        })
        .collect();
    window.set_collections(ModelRc::from(Rc::new(VecModel::from(entries))));
    window.set_new_name(SharedString::new());
    window.set_status_text(SharedString::new());
}

fn parse_hex_color(hex: &str) -> slint::Color {
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
    slint::Color::from_rgb_u8(0x9a, 0x9a, 0x9a)
}
