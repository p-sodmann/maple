//! Small shared UI building blocks used across views.

use gtk4::gdk;
use gtk4::glib;
use gtk4::prelude::*;

/// The application logo as a `Picture`, scaled to `size` × `size`.
///
/// The PNG has a baked-in white background, so it is clipped to a
/// rounded card (`.maple-logo`) instead of sitting as a bare square.
pub(crate) fn logo_picture(size: i32) -> gtk4::Picture {
    let bytes = glib::Bytes::from_static(include_bytes!("../../assets/logo.png"));
    let texture = gdk::Texture::from_bytes(&bytes).expect("failed to load logo");
    let picture = gtk4::Picture::for_paintable(&texture);
    picture.set_content_fit(gtk4::ContentFit::Cover);
    picture.set_size_request(size, size);
    picture.set_halign(gtk4::Align::Center);
    picture.set_overflow(gtk4::Overflow::Hidden);
    picture.add_css_class("maple-logo");
    picture
}

/// A round colour swatch of `size` px, filled with the CSS colour `hex`.
///
/// Used for collection colour dots in chips, list rows, and pickers.
pub(crate) fn color_dot(hex: &str, size: i32) -> gtk4::DrawingArea {
    let dot = gtk4::DrawingArea::builder()
        .content_width(size)
        .content_height(size)
        .valign(gtk4::Align::Center)
        .build();
    let hex = hex.to_owned();
    dot.set_draw_func(move |_, cr, w, h| {
        if let Ok(rgba) = gdk::RGBA::parse(&hex) {
            cr.set_source_rgba(
                rgba.red() as f64,
                rgba.green() as f64,
                rgba.blue() as f64,
                1.0,
            );
            let r = w.min(h) as f64 / 2.0;
            cr.arc(
                w as f64 / 2.0,
                h as f64 / 2.0,
                r,
                0.0,
                2.0 * std::f64::consts::PI,
            );
            let _ = cr.fill();
        }
    });
    dot
}
