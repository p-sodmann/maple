//! Collection manager window — create, rename, recolour and delete collections.

use std::sync::{Arc, Mutex};

use gtk4::gdk;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::Database;

/// Preset palette colours offered when creating / recolouring a collection.
const PALETTE: &[(&str, &str)] = &[
    ("Red",    "#e01b24"),
    ("Orange", "#ff7800"),
    ("Yellow", "#f5c211"),
    ("Green",  "#33d17a"),
    ("Teal",   "#26a269"),
    ("Blue",   "#3584e4"),
    ("Purple", "#9141ac"),
    ("Pink",   "#e66100"),
];

/// Open the collection manager as a transient window.
///
/// `on_changed` is called whenever collections are mutated so the caller
/// can refresh any dependent UI (e.g. the library filter popover).
pub fn open_manager(
    parent: &impl IsA<gtk4::Window>,
    db: &Arc<Mutex<Database>>,
    on_changed: impl Fn() + 'static,
) {
    let db = db.clone();
    let on_changed = std::rc::Rc::new(on_changed);

    let list_box = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::None)
        .css_classes(["boxed-list"])
        .build();

    populate_list(&list_box, &db, &on_changed);

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .margin_start(16)
        .margin_end(16)
        .margin_top(12)
        .margin_bottom(16)
        .build();
    scroll.set_child(Some(&list_box));

    // ── Header with "+" button ──────────────────────────────────
    let add_btn = gtk4::Button::builder()
        .icon_name("list-add-symbolic")
        .tooltip_text("New collection")
        .css_classes(["flat"])
        .build();

    let header = adw::HeaderBar::new();
    header.pack_start(&add_btn);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&scroll));

    let win = adw::Window::builder()
        .title("Manage Collections")
        .default_width(400)
        .default_height(480)
        .transient_for(parent)
        .build();
    win.set_content(Some(&toolbar_view));

    add_btn.connect_clicked({
        let db = db.clone();
        let list_box = list_box.clone();
        let on_changed = on_changed.clone();
        let win = win.clone();
        move |_| {
            open_new_collection_dialog(&win, &db, &list_box, &on_changed);
        }
    });

    win.present();
}

// ── Populate / rebuild the list ─────────────────────────────────

fn populate_list(
    list_box: &gtk4::ListBox,
    db: &Arc<Mutex<Database>>,
    on_changed: &std::rc::Rc<impl Fn() + 'static>,
) {
    // Remove all children.
    while let Some(child) = list_box.first_child() {
        list_box.remove(&child);
    }

    let collections = db
        .lock()
        .ok()
        .and_then(|d| d.all_collections().ok())
        .unwrap_or_default();

    if collections.is_empty() {
        let label = gtk4::Label::builder()
            .label("No collections yet")
            .css_classes(["dim-label"])
            .margin_top(24)
            .margin_bottom(24)
            .build();
        list_box.append(&label);
        return;
    }

    for coll in &collections {
        let row = build_collection_row(coll, list_box, db, on_changed);
        list_box.append(&row);
    }
}

fn build_collection_row(
    coll: &maple_db::Collection,
    list_box: &gtk4::ListBox,
    db: &Arc<Mutex<Database>>,
    on_changed: &std::rc::Rc<impl Fn() + 'static>,
) -> adw::ActionRow {
    let color_dot = gtk4::DrawingArea::builder()
        .content_width(16)
        .content_height(16)
        .valign(gtk4::Align::Center)
        .build();

    let hex = coll.color.clone();
    color_dot.set_draw_func(move |_, cr, w, h| {
        if let Some(rgba) = parse_hex(&hex) {
            cr.set_source_rgba(
                rgba.red() as f64,
                rgba.green() as f64,
                rgba.blue() as f64,
                1.0,
            );
            let radius = w.min(h) as f64 / 2.0;
            cr.arc(w as f64 / 2.0, h as f64 / 2.0, radius, 0.0, 2.0 * std::f64::consts::PI);
            let _ = cr.fill();
        }
    });

    let count_label = format!("{} image{}", coll.image_count, if coll.image_count == 1 { "" } else { "s" });

    let delete_btn = gtk4::Button::builder()
        .icon_name("user-trash-symbolic")
        .tooltip_text("Delete collection")
        .css_classes(["flat", "circular"])
        .valign(gtk4::Align::Center)
        .build();

    let row = adw::ActionRow::builder()
        .title(&coll.name)
        .subtitle(&count_label)
        .build();
    row.add_prefix(&color_dot);
    row.add_suffix(&delete_btn);

    let coll_id = coll.id;
    delete_btn.connect_clicked({
        let db = db.clone();
        let list_box = list_box.clone();
        let on_changed = on_changed.clone();
        move |_| {
            if let Ok(d) = db.lock() {
                let _ = d.delete_collection(coll_id);
            }
            populate_list(&list_box, &db, &on_changed);
            on_changed();
        }
    });

    row
}

// ── New collection dialog ───────────────────────────────────────

fn open_new_collection_dialog(
    parent: &adw::Window,
    db: &Arc<Mutex<Database>>,
    list_box: &gtk4::ListBox,
    on_changed: &std::rc::Rc<impl Fn() + 'static>,
) {
    let name_entry = gtk4::Entry::builder()
        .placeholder_text("Collection name")
        .hexpand(true)
        .build();

    let selected_color = std::rc::Rc::new(std::cell::RefCell::new(PALETTE[5].1.to_string()));

    // Colour buttons row.
    let color_row = gtk4::FlowBox::builder()
        .selection_mode(gtk4::SelectionMode::None)
        .max_children_per_line(8)
        .min_children_per_line(8)
        .column_spacing(6)
        .row_spacing(6)
        .homogeneous(true)
        .build();

    for &(_name, hex) in PALETTE {
        let btn = gtk4::Button::builder()
            .css_classes(["flat", "circular"])
            .width_request(32)
            .height_request(32)
            .tooltip_text(_name)
            .build();

        let da = gtk4::DrawingArea::builder()
            .content_width(24)
            .content_height(24)
            .build();

        let hex_c = hex.to_string();
        da.set_draw_func(move |_, cr, w, h| {
            if let Some(rgba) = parse_hex(&hex_c) {
                cr.set_source_rgba(
                    rgba.red() as f64,
                    rgba.green() as f64,
                    rgba.blue() as f64,
                    1.0,
                );
                let radius = w.min(h) as f64 / 2.0;
                cr.arc(w as f64 / 2.0, h as f64 / 2.0, radius, 0.0, 2.0 * std::f64::consts::PI);
                let _ = cr.fill();
            }
        });
        btn.set_child(Some(&da));

        let color_ref = selected_color.clone();
        let hex_owned = hex.to_string();
        btn.connect_clicked(move |_| {
            *color_ref.borrow_mut() = hex_owned.clone();
        });

        color_row.insert(&btn, -1);
    }

    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_start(16)
        .margin_end(16)
        .margin_top(16)
        .margin_bottom(16)
        .build();
    content.append(&name_entry);
    content.append(&color_row);

    let create_btn = gtk4::Button::builder()
        .label("Create")
        .css_classes(["suggested-action"])
        .build();

    let cancel_btn = gtk4::Button::builder()
        .label("Cancel")
        .build();

    let btn_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .halign(gtk4::Align::End)
        .build();
    btn_box.append(&cancel_btn);
    btn_box.append(&create_btn);
    content.append(&btn_box);

    let dialog_header = adw::HeaderBar::new();
    let dialog_toolbar = adw::ToolbarView::new();
    dialog_toolbar.add_top_bar(&dialog_header);
    dialog_toolbar.set_content(Some(&content));

    let dialog = adw::Window::builder()
        .title("New Collection")
        .default_width(340)
        .default_height(220)
        .transient_for(parent)
        .modal(true)
        .build();
    dialog.set_content(Some(&dialog_toolbar));

    cancel_btn.connect_clicked({
        let dialog = dialog.clone();
        move |_| dialog.close()
    });

    create_btn.connect_clicked({
        let db = db.clone();
        let list_box = list_box.clone();
        let on_changed = on_changed.clone();
        let dialog = dialog.clone();
        let name_entry = name_entry.clone();
        let selected_color = selected_color.clone();
        move |_| {
            let name = name_entry.text().trim().to_string();
            if name.is_empty() {
                return;
            }
            let color = selected_color.borrow().clone();
            if let Ok(d) = db.lock() {
                let _ = d.create_collection(&name, &color);
            }
            populate_list(&list_box, &db, &on_changed);
            on_changed();
            dialog.close();
        }
    });

    // Enter key in name entry also creates.
    name_entry.connect_activate({
        let create_btn = create_btn.clone();
        move |_| create_btn.emit_clicked()
    });

    dialog.present();
}

// ── Helpers ─────────────────────────────────────────────────────

fn parse_hex(hex: &str) -> Option<gdk::RGBA> {
    let rgba = gdk::RGBA::parse(hex).ok()?;
    Some(rgba)
}
