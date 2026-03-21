//! Thin metadata strip shown below the image.

use gtk4::prelude::*;

use maple_db::LibraryImage;

pub(super) fn build_empty_info_bar() -> gtk4::Box {
    gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(16)
        .margin_start(12)
        .margin_end(12)
        .margin_top(6)
        .margin_bottom(6)
        .build()
}

/// Clear and repopulate `bar` with a one-line summary from `image`.
pub(super) fn fill_info_bar(bar: &gtk4::Box, image: &LibraryImage) {
    while let Some(child) = bar.last_child() {
        bar.remove(&child);
    }

    let m = &image.meta;
    let mut fields: Vec<String> = Vec::new();

    match (&m.make, &m.model) {
        (Some(make), Some(model)) => fields.push(format!("{make} {model}")),
        (Some(make), None) => fields.push(make.clone()),
        _ => {}
    }
    if let Some(ref lens) = m.lens {
        fields.push(lens.clone());
    }
    if let (Some(fl), Some(ap)) = (m.focal_length, m.aperture) {
        fields.push(format!("{fl:.0} mm  f/{ap:.1}"));
    }
    if let Some(iso) = m.iso {
        fields.push(format!("ISO {iso}"));
    }
    if let (Some(w), Some(h)) = (m.width, m.height) {
        fields.push(format!("{w} × {h}"));
    }

    let n = fields.len();
    for (i, text) in fields.iter().enumerate() {
        let label = gtk4::Label::new(Some(text));
        label.add_css_class("caption");
        label.add_css_class("dim-label");
        label.set_ellipsize(gtk4::pango::EllipsizeMode::End);
        bar.append(&label);
        if i + 1 < n {
            let sep = gtk4::Label::new(Some("·"));
            sep.add_css_class("caption");
            sep.add_css_class("dim-label");
            bar.append(&sep);
        }
    }
}
