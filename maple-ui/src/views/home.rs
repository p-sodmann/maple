//! Home page — the root navigation page shown at application start.
//!
//! Presents two entry points:
//!   • Import Photos → pushes the source/destination picker flow
//!   • Library       → pushes the searchable library browser

use std::sync::{Arc, Mutex};

use maple_db::ThumbnailCache;

use gtk4::prelude::*;
use libadwaita as adw;

use super::{library, settings_window, source_picker};
use crate::widgets;

/// Build the home page and wire navigation into `nav_view`.
pub fn build_home_page(
    nav_view: &adw::NavigationView,
    toast_overlay: &adw::ToastOverlay,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<ThumbnailCache>,
) -> adw::NavigationPage {
    let import_btn = gtk4::Button::builder()
        .child(&adw::ButtonContent::builder()
            .icon_name("folder-download-symbolic")
            .label("Import Photos")
            .build())
        .css_classes(["suggested-action", "pill"])
        .build();

    let library_btn = gtk4::Button::builder()
        .child(&adw::ButtonContent::builder()
            .icon_name("view-grid-symbolic")
            .label("Browse Library")
            .build())
        .css_classes(["pill"])
        .build();

    // Stacked full-width buttons inside the clamp so both share a width.
    let buttons = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .build();
    buttons.append(&import_btn);
    buttons.append(&library_btn);

    let clamp = adw::Clamp::builder()
        .maximum_size(280)
        .child(&buttons)
        .build();

    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(24)
        .build();
    content.append(&widgets::logo_picture(200));
    content.append(&clamp);

    // The logo already carries the wordmark, so the status page only
    // adds the tagline beneath it.
    let status_page = adw::StatusPage::builder()
        .description("Import and browse your photo library.")
        .child(&content)
        .build();

    let settings_btn = gtk4::Button::builder()
        .icon_name("preferences-system-symbolic")
        .tooltip_text("Settings")
        .css_classes(["flat"])
        .build();

    let header = adw::HeaderBar::new();
    header.pack_end(&settings_btn);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&status_page));
    toolbar_view.add_css_class("maple-hero");

    let page = adw::NavigationPage::builder()
        .title("Maple")
        .child(&toolbar_view)
        .build();

    // ── Settings button ──────────────────────────────────────────
    settings_btn.connect_clicked({
        let db = db.clone();
        let cache = cache.clone();
        move |btn| {
            if let Some(win) = btn.root().and_downcast::<gtk4::Window>() {
                settings_window::open_settings(&win, &db, cache.clone(), || {});
            }
        }
    });

    // ── Import button ────────────────────────────────────────────
    import_btn.connect_clicked({
        let nav_view = nav_view.clone();
        let toast_overlay = toast_overlay.clone();
        let db = db.clone();
        move |_| {
            let picker = source_picker::build_picker_page(&nav_view, &toast_overlay, db.clone());
            nav_view.push(&picker);
        }
    });

    // ── Library button ───────────────────────────────────────────
    library_btn.connect_clicked({
        let nav_view = nav_view.clone();
        let db = db.clone();
        let cache = cache.clone();
        move |_| {
            let lib = library::build_library_page(&nav_view, db.clone(), cache.clone());
            nav_view.push(&lib);
        }
    });

    page
}
