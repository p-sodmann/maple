//! Settings window controller (Slint port of views/settings_window.rs).
//!
//! A second top-level Window held as a `thread_local!` singleton that shows
//! read-only configuration info and offers destructive clear actions (AI
//! descriptions, thumbnail cache).

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, SharedString};

use crate::SettingsWindow;

thread_local! {
    static SETTINGS: RefCell<Option<SettingsWindow>> = const { RefCell::new(None) };
}

/// Open (or reuse) the settings window, syncing the current dark-mode state.
pub fn open(db: Arc<Mutex<maple_db::Database>>, is_dark: bool) {
    if SETTINGS.with(|s| s.borrow().is_none()) {
        match build(db) {
            Ok(win) => SETTINGS.with(|cell| *cell.borrow_mut() = Some(win)),
            Err(e) => {
                tracing::error!("Failed to build settings window: {e}");
                return;
            }
        }
    }
    SETTINGS.with(|cell| {
        let guard = cell.borrow();
        if let Some(win) = guard.as_ref() {
            win.set_dark(is_dark);
            populate(win);
            if let Err(e) = win.show() {
                tracing::error!("Failed to show settings window: {e}");
            }
        }
    });
}

/// Propagate a theme change to the settings window while it is open.
pub fn set_dark(dark: bool) {
    SETTINGS.with(|s| {
        let guard = s.borrow();
        if let Some(win) = guard.as_ref() {
            win.set_dark(dark);
        }
    });
}

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<SettingsWindow, slint::PlatformError> {
    let window = SettingsWindow::new()?;

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_clear_ai_descriptions({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let result = db.lock().ok().and_then(|g| g.clear_all_ai_descriptions().ok());
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Some(n) => format!("Cleared {n} AI description{}.", if n == 1 { "" } else { "s" }),
                    None => "Failed to clear AI descriptions.".to_owned(),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    window.on_delete_all_face_data({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let result = db.lock().ok().and_then(|g| g.clear_all_face_data().ok());
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Some((faces, persons)) => format!(
                        "Deleted {faces} face{} and {persons} person{}. \
                         Re-detection will run in the background.",
                        if faces == 1 { "" } else { "s" },
                        if persons == 1 { "" } else { "s" },
                    ),
                    None => "Failed to delete face data.".to_owned(),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    window.on_clear_thumbnail_cache({
        let w = window.as_weak();
        move || {
            // The thumbnail cache path comes from settings; clearing means
            // removing all files in the .thumbcache directory.
            let settings = maple_state::Settings::load();
            let cache_dir = settings.library_dir.join(".thumbcache");
            let result = std::fs::remove_dir_all(&cache_dir)
                .and_then(|_| std::fs::create_dir_all(&cache_dir));
            if let Some(w) = w.upgrade() {
                let msg = match result {
                    Ok(_) => "Thumbnail cache cleared.".to_owned(),
                    Err(e) => format!("Failed to clear cache: {e}"),
                };
                w.set_status_text(SharedString::from(msg));
            }
        }
    });

    Ok(window)
}

/// Populate the window with current settings values.
fn populate(window: &SettingsWindow) {
    let s = maple_state::Settings::load();
    window.set_library_dir(s.library_dir.to_string_lossy().into_owned().into());
    window.set_database_path(s.database_path.to_string_lossy().into_owned().into());
    let cache = s.library_dir.join(".thumbcache");
    window.set_cache_path(cache.to_string_lossy().into_owned().into());
    window.set_thumbnail_quality(
        SharedString::from(format!("{}%", s.thumbnails.quality))
    );
    window.set_ai_endpoint(
        if s.ai.enabled {
            s.ai.server_url.clone().into()
        } else {
            SharedString::new()
        },
    );
    window.set_face_threshold(
        if s.face.enabled {
            format!("{:.0}%", s.face.similarity_threshold * 100.0).into()
        } else {
            SharedString::new()
        },
    );
    window.set_status_text(SharedString::new());
}
