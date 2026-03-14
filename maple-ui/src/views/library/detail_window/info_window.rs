//! Info popup window — shows all database fields for the current image.

use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::{ImageStatus, LibraryImage};

/// Open a non-modal popup window showing all database fields for `image`.
pub(super) fn open_info_window(
    image: &LibraryImage,
    db: &Arc<Mutex<maple_db::Database>>,
    parent: &adw::Window,
) {
    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .build();

    let ai_descriptions = db
        .lock()
        .ok()
        .and_then(|d| d.ai_descriptions_for_image(image.id).ok())
        .unwrap_or_default();

    populate_info_panel(&panel, image, &ai_descriptions);

    let scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .margin_start(16)
        .margin_end(16)
        .margin_top(12)
        .margin_bottom(16)
        .build();
    scroll.set_child(Some(&panel));

    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&scroll));

    let title = image.meta.filename.as_deref().unwrap_or("Image Info");
    let win = adw::Window::builder()
        .title(title)
        .default_width(420)
        .default_height(520)
        .transient_for(parent)
        .build();
    win.set_content(Some(&toolbar_view));
    win.present();
}

fn populate_info_panel(
    panel: &gtk4::Box,
    image: &LibraryImage,
    ai_descriptions: &[(String, String)],
) {
    let m = &image.meta;

    // ── File group ────────────────────────────────────────────────
    let file_group = adw::PreferencesGroup::builder().title("File").build();

    if let Some(ref name) = m.filename {
        file_group.add(&make_row("Name", name));
    }
    if let Some(s) = image.path.to_str() {
        file_group.add(
            &adw::ActionRow::builder()
                .title("Path")
                .subtitle(s)
                .subtitle_lines(3)
                .build(),
        );
    }
    file_group.add(&make_row("Added", &format_timestamp(image.added_at)));
    file_group.add(&make_row(
        "Status",
        match image.status {
            ImageStatus::Present => "Present",
            ImageStatus::Missing => "Missing",
        },
    ));
    file_group.add(&make_row("ID", &image.id.to_string()));
    panel.append(&file_group);

    // ── Camera group ──────────────────────────────────────────────
    let cam_group = adw::PreferencesGroup::builder().title("Camera").build();
    let mut has_cam = false;

    if let Some(ref v) = m.make {
        cam_group.add(&make_row("Make", v));
        has_cam = true;
    }
    if let Some(ref v) = m.model {
        cam_group.add(&make_row("Model", v));
        has_cam = true;
    }
    if let Some(ref v) = m.lens {
        cam_group.add(&make_row("Lens", v));
        has_cam = true;
    }
    if let Some(fl) = m.focal_length {
        cam_group.add(&make_row("Focal Length", &format!("{fl:.0} mm")));
        has_cam = true;
    }
    if let Some(ap) = m.aperture {
        cam_group.add(&make_row("Aperture", &format!("f/{ap:.1}")));
        has_cam = true;
    }
    if let Some(iso) = m.iso {
        cam_group.add(&make_row("ISO", &iso.to_string()));
        has_cam = true;
    }
    if has_cam {
        panel.append(&cam_group);
    }

    // ── Image group ───────────────────────────────────────────────
    let img_group = adw::PreferencesGroup::builder().title("Image").build();
    let mut has_img = false;

    if let Some(ts) = m.taken_at {
        img_group.add(&make_row("Taken", &format_timestamp(ts)));
        has_img = true;
    }
    if let (Some(w), Some(h)) = (m.width, m.height) {
        img_group.add(&make_row("Dimensions", &format!("{w} × {h}")));
        has_img = true;
    }
    if let Some(orient) = m.orientation {
        let label = match orient {
            1 => "Normal",
            2 => "Mirrored horizontally",
            3 => "Rotated 180°",
            4 => "Mirrored vertically",
            5 => "Mirrored + rotated 90° CW",
            6 => "Rotated 90° CW",
            7 => "Mirrored + rotated 90° CCW",
            8 => "Rotated 90° CCW",
            _ => "Unknown",
        };
        img_group.add(&make_row("Orientation", label));
        has_img = true;
    }
    if has_img {
        panel.append(&img_group);
    }

    // ── AI descriptions group ─────────────────────────────────────
    if !ai_descriptions.is_empty() {
        let ai_group = adw::PreferencesGroup::builder()
            .title("AI Description")
            .build();

        for (model_id, description) in ai_descriptions {
            // Use an ExpanderRow so long descriptions don't dominate the panel.
            let row = adw::ExpanderRow::builder()
                .title(model_id)
                .build();

            let label = gtk4::Label::builder()
                .label(description)
                .wrap(true)
                .wrap_mode(gtk4::pango::WrapMode::Word)
                .xalign(0.0)
                .selectable(true)
                .margin_start(12)
                .margin_end(12)
                .margin_top(8)
                .margin_bottom(8)
                .build();
            label.add_css_class("body");

            let row_content = gtk4::ListBoxRow::builder()
                .activatable(false)
                .selectable(false)
                .build();
            row_content.set_child(Some(&label));
            row.add_row(&row_content);

            ai_group.add(&row);
        }

        panel.append(&ai_group);
    }
}

fn make_row(title: &str, subtitle: &str) -> adw::ActionRow {
    adw::ActionRow::builder()
        .title(title)
        .subtitle(subtitle)
        .build()
}

/// Format a Unix timestamp (seconds since epoch) as `YYYY-MM-DD HH:MM:SS`.
fn format_timestamp(ts: i64) -> String {
    // civil_from_days — https://howardhinnant.github.io/date_algorithms.html
    let z = ts.div_euclid(86400) + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };

    let rem = ts.rem_euclid(86400);
    let h = rem / 3600;
    let min = (rem % 3600) / 60;
    let s = rem % 60;

    format!("{y:04}-{m:02}-{d:02}  {h:02}:{min:02}:{s:02}")
}
