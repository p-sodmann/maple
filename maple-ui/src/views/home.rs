//! Home page — the root navigation page shown at application start.
//!
//! Presents two entry points:
//!   • Import Photos → pushes the source/destination picker flow
//!   • Library       → pushes the searchable library browser

use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;

use super::{library, source_picker};

/// Build the home page and wire navigation into `nav_view`.
pub fn build_home_page(
    nav_view: &adw::NavigationView,
    toast_overlay: &adw::ToastOverlay,
    db: Arc<Mutex<maple_db::Database>>,
) -> adw::NavigationPage {
    let import_btn = gtk4::Button::builder()
        .label("Import Photos")
        .css_classes(["suggested-action", "pill"])
        .halign(gtk4::Align::Center)
        .build();

    let library_btn = gtk4::Button::builder()
        .label("Browse Library")
        .css_classes(["pill"])
        .halign(gtk4::Align::Center)
        .build();

    let buttons = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .halign(gtk4::Align::Center)
        .build();
    buttons.append(&import_btn);
    buttons.append(&library_btn);

    let clamp = adw::Clamp::builder()
        .maximum_size(360)
        .child(&buttons)
        .build();

    let status_page = adw::StatusPage::builder()
        .icon_name("camera-photo-symbolic")
        .title("Maple")
        .description("Import and browse your photo library.")
        .child(&clamp)
        .build();

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&adw::HeaderBar::new());
    toolbar_view.set_content(Some(&status_page));

    let page = adw::NavigationPage::builder()
        .title("Maple")
        .child(&toolbar_view)
        .build();

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
        move |_| {
            let lib = library::build_library_page(&nav_view, db.clone());
            nav_view.push(&lib);
        }
    });

    page
}
