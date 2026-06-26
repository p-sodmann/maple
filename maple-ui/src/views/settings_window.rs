//! Application settings window.
//!
//! Two-column layout: left = "Database", right = "Advanced".
//! Each column is an `adw::PreferencesGroup` so rows can be appended freely.
//!
//! To add a new setting:
//!   1. Build an `adw::ActionRow` (or any GTK widget) for the relevant column.
//!   2. Call `db_group.add(&row)` or `advanced_group.add(&row)`.
//!   3. For destructive actions, call `confirm_action(parent, …)` to show
//!      the standard "Are you sure?" alert before executing.

use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::{Database, ThumbnailCache};

/// Open the settings window as a modal transient of `parent`.
///
/// `on_changed` is called after any action that requires a grid reload
/// (e.g. clearing the cache after a size change).
pub fn open_settings(
    parent: &impl IsA<gtk4::Window>,
    db: &Arc<Mutex<Database>>,
    cache: Arc<ThumbnailCache>,
    on_changed: impl Fn() + 'static,
) {
    let db = db.clone();
    let on_changed = std::rc::Rc::new(on_changed);

    let settings = maple_state::Settings::load();

    // ── Database column ───────────────────────────────────────────
    let db_group = adw::PreferencesGroup::builder()
        .title("Database")
        .build();

    // Database path row.
    let db_path = settings.database_path.to_string_lossy().into_owned();
    let db_path_row = adw::ActionRow::builder()
        .title("Database path")
        .subtitle(&db_path)
        .build();
    db_group.add(&db_path_row);

    // Thumbnail size row.
    let size_adj = gtk4::Adjustment::new(
        settings.thumbnails.size as f64,
        100.0,
        500.0,
        10.0,  // step
        50.0,  // page
        0.0,
    );
    let size_row = adw::SpinRow::builder()
        .title("Thumbnail size")
        .subtitle("Pixel size of grid thumbnails (clears cache when changed)")
        .adjustment(&size_adj)
        .climb_rate(1.0)
        .digits(0)
        .build();
    db_group.add(&size_row);

    // ── Advanced column ────────────────────────────────────────────
    let advanced_group = adw::PreferencesGroup::builder()
        .title("Advanced")
        .build();

    let clear_ai_row = adw::ActionRow::builder()
        .title("Empty AI descriptions")
        .subtitle("Delete all AI-generated image descriptions")
        .build();

    let clear_ai_btn = gtk4::Button::builder()
        .label("Empty…")
        .css_classes(["destructive-action"])
        .valign(gtk4::Align::Center)
        .build();
    clear_ai_row.add_suffix(&clear_ai_btn);
    clear_ai_row.set_activatable_widget(Some(&clear_ai_btn));
    advanced_group.add(&clear_ai_row);

    let clear_cache_row = adw::ActionRow::builder()
        .title("Delete thumbnail cache")
        .subtitle("Remove all cached WebP previews; they will be regenerated on next view")
        .build();

    let clear_cache_btn = gtk4::Button::builder()
        .label("Delete…")
        .css_classes(["destructive-action"])
        .valign(gtk4::Align::Center)
        .build();
    clear_cache_row.add_suffix(&clear_cache_btn);
    clear_cache_row.set_activatable_widget(Some(&clear_cache_btn));
    advanced_group.add(&clear_cache_row);

    // ── Two-column layout ─────────────────────────────────────────
    let left_col = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .hexpand(true)
        .build();
    left_col.append(&db_group);

    let right_col = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(0)
        .hexpand(true)
        .build();
    right_col.append(&advanced_group);

    let columns = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(24)
        .margin_start(24)
        .margin_end(24)
        .margin_top(24)
        .margin_bottom(24)
        .build();
    columns.append(&left_col);
    columns.append(&right_col);

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .child(&columns)
        .build();

    let header = adw::HeaderBar::new();

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&scroll));

    let win = adw::Window::builder()
        .title("Settings")
        .default_width(680)
        .default_height(420)
        .transient_for(parent)
        .modal(true)
        .build();
    win.set_content(Some(&toolbar_view));

    // ── Thumbnail size change ────────────────────────────────────
    size_row.connect_value_notify({
        let cache = cache.clone();
        let on_changed = on_changed.clone();
        move |row| {
            let new_size = row.value() as u32;
            let mut s = maple_state::Settings::load();
            s.thumbnails.size = new_size;
            if let Err(e) = s.save() {
                tracing::warn!("Failed to save settings: {e}");
            }
            if let Err(e) = cache.clear() {
                tracing::warn!("Failed to clear thumbnail cache: {e}");
            } else {
                tracing::info!("Thumbnail cache cleared after size change to {new_size}px");
            }
            on_changed();
        }
    });

    // ── "Empty AI descriptions" confirmation ──────────────────────
    clear_ai_btn.connect_clicked({
        let win = win.clone();
        let db = db.clone();
        let on_changed = on_changed.clone();
        move |_| {
            confirm_action(
                &win,
                "Empty AI Descriptions?",
                "All AI-generated descriptions will be deleted and will need to be regenerated.",
                "Empty All",
                {
                    let db = db.clone();
                    let on_changed = on_changed.clone();
                    move || {
                        if let Ok(d) = db.lock() {
                            match d.clear_all_ai_descriptions() {
                                Ok(n) => tracing::info!("Cleared {n} AI descriptions"),
                                Err(e) => tracing::error!("Failed to clear AI descriptions: {e}"),
                            }
                        }
                        on_changed();
                    }
                },
            );
        }
    });

    // ── "Delete thumbnail cache" confirmation ─────────────────────
    clear_cache_btn.connect_clicked({
        let win = win.clone();
        move |_| {
            confirm_action(
                &win,
                "Delete Thumbnail Cache?",
                "All cached preview images will be deleted and regenerated when you next browse the library.",
                "Delete Cache",
                {
                    let cache = cache.clone();
                    move || {
                        if let Err(e) = cache.clear() {
                            tracing::error!("Failed to clear thumbnail cache: {e}");
                        } else {
                            tracing::info!("Thumbnail cache cleared");
                        }
                    }
                },
            );
        }
    });

    win.present();
}

// ── Shared helper ─────────────────────────────────────────────────

/// Show a destructive-action confirmation alert.
///
/// `on_confirm` is called only if the user clicks the destructive button.
/// Extend settings with new dangerous actions by calling this function —
/// no boilerplate needed at each call site.
fn confirm_action(
    parent: &adw::Window,
    heading: &str,
    body: &str,
    confirm_label: &str,
    on_confirm: impl Fn() + 'static,
) {
    let alert = adw::AlertDialog::builder()
        .heading(heading)
        .body(body)
        .build();

    alert.add_response("cancel", "Cancel");
    alert.add_response("confirm", confirm_label);
    alert.set_response_appearance("confirm", adw::ResponseAppearance::Destructive);
    alert.set_default_response(Some("cancel"));
    alert.set_close_response("cancel");

    alert.connect_response(None, move |_, response| {
        if response == "confirm" {
            on_confirm();
        }
    });

    alert.present(Some(parent));
}
