use std::sync::{Arc, Mutex};

use libadwaita as adw;

use crate::views::home;

/// Build the main application window.
pub fn build_window(app: &adw::Application) -> adw::ApplicationWindow {
    let settings = maple_state::Settings::load();

    let db = match maple_db::Database::open(&settings.database_path) {
        Ok(db) => Arc::new(Mutex::new(db)),
        Err(e) => {
            tracing::error!(
                "Failed to open library database at {}: {e}",
                settings.database_path.display()
            );
            // Proceed without a database rather than crashing on startup.
            // A fresh in-memory-equivalent DB (temp path) keeps the rest of
            // the UI functional even if the configured path is inaccessible.
            let fallback = std::env::temp_dir().join("maple_library_fallback.db");
            Arc::new(Mutex::new(
                maple_db::Database::open(&fallback)
                    .expect("Could not open fallback database"),
            ))
        }
    };

    // Start the background library scanner immediately so the DB stays
    // in sync with the library directory from the moment the app launches.
    maple_db::LibraryScanner::new(db.clone(), settings.library_dir).spawn();

    let toast_overlay = adw::ToastOverlay::new();
    let nav_view = adw::NavigationView::new();

    let home_page = home::build_home_page(&nav_view, &toast_overlay, db);
    nav_view.push(&home_page);

    toast_overlay.set_child(Some(&nav_view));

    adw::ApplicationWindow::builder()
        .application(app)
        .title("Maple")
        .default_width(900)
        .default_height(600)
        .content(&toast_overlay)
        .build()
}
