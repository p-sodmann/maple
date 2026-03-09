//! Detail window — singleton full-size viewer with zoom + pan.
//!
//! Only one detail window exists at a time.  Activating a new image
//! updates the existing window instead of opening a second one.
//!
//! Controls:
//!   • Scroll wheel      — zoom in / out
//!   • Left-button drag  — pan when zoomed in
//!   • Open button       — launch in default application
//!   • Copy path button  — write absolute path to clipboard

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::mpsc;
use std::time::Duration;

use gtk4::gdk;
use gtk4::gio;
use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::LibraryImage;

// ── Constants ────────────────────────────────────────────────────

const ZOOM_STEP: f64 = 1.15;
const ZOOM_MAX: f64  = 20.0;

const MAX_WIN_W: i32 = 1400;
const MAX_WIN_H: i32 = 860;
/// Approximate pixel height of header bar + info strip combined.
const CHROME_H: i32 = 82;

// ── Singleton state ──────────────────────────────────────────────

#[derive(Clone)]
struct DetailContext {
    window: adw::Window,
    picture: gtk4::Picture,
    info_bar: gtk4::Box,
    scrolled: gtk4::ScrolledWindow,
    zoom: Rc<Cell<f64>>,
    img_dims: Rc<Cell<(i32, i32)>>,
    /// Shared with action-button closures so they always use the current path.
    current_path: Rc<RefCell<PathBuf>>,
}

thread_local! {
    static DETAIL_CTX: RefCell<Option<DetailContext>> = const { RefCell::new(None) };
}

// ── Public API ───────────────────────────────────────────────────

/// Open (or update) the singleton detail window for `image`.
pub fn open(image: &LibraryImage, parent: &gtk4::Window) {
    // Reuse an existing visible window.
    let ctx = DETAIL_CTX.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|c| c.window.is_visible())
            .cloned()
    });

    if let Some(ctx) = ctx {
        update_context(&ctx, image);
        ctx.window.present();
        return;
    }

    let ctx = build_window(image, parent);

    // Clear the singleton when this window closes — but guard against rapid
    // open/close where a newer window may already be stored.
    let window_ref = ctx.window.clone();
    ctx.window.connect_destroy(move |_| {
        DETAIL_CTX.with(|cell| {
            let is_current = cell
                .borrow()
                .as_ref()
                .map_or(false, |c| c.window == window_ref);
            if is_current {
                *cell.borrow_mut() = None;
            }
        });
    });

    DETAIL_CTX.with(|cell| *cell.borrow_mut() = Some(ctx.clone()));
    ctx.window.present();
}

// ── Window builder ───────────────────────────────────────────────

fn build_window(image: &LibraryImage, parent: &gtk4::Window) -> DetailContext {
    let current_path = Rc::new(RefCell::new(image.path.clone()));
    let zoom: Rc<Cell<f64>> = Rc::new(Cell::new(1.0));
    let img_dims: Rc<Cell<(i32, i32)>> = Rc::new(Cell::new((0, 0)));

    // ── Picture widget inside a scrolled container ────────────────
    let picture = gtk4::Picture::builder()
        .content_fit(gtk4::ContentFit::Contain)
        .hexpand(true)
        .vexpand(true)
        .build();

    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Never)
        .hexpand(true)
        .vexpand(true)
        .build();
    scrolled.set_child(Some(&picture));

    wire_zoom_and_pan(&scrolled, &picture, &zoom, &img_dims);

    // ── Action buttons ────────────────────────────────────────────
    let open_btn = gtk4::Button::builder()
        .icon_name("external-link-symbolic")
        .tooltip_text("Open in default application")
        .css_classes(["flat"])
        .build();

    let copy_btn = gtk4::Button::builder()
        .icon_name("edit-copy-symbolic")
        .tooltip_text("Copy file path to clipboard")
        .css_classes(["flat"])
        .build();

    open_btn.connect_clicked({
        let path = current_path.clone();
        move |_| launch_default_app(&path.borrow())
    });

    copy_btn.connect_clicked({
        let path = current_path.clone();
        move |btn| {
            if let Some(s) = path.borrow().to_str() {
                btn.clipboard().set_text(s);
            }
        }
    });

    let header = adw::HeaderBar::new();
    header.pack_end(&open_btn);
    header.pack_end(&copy_btn);

    // ── Metadata info strip ───────────────────────────────────────
    let info_bar = build_empty_info_bar();
    fill_info_bar(&info_bar, image);

    // ── Layout ────────────────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    content.append(&info_bar);
    content.append(&scrolled);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let filename = image.meta.filename.as_deref().unwrap_or("Image");
    let window = adw::Window::builder()
        .title(filename)
        .default_width(960)
        .default_height(720)
        .transient_for(parent)
        .build();
    window.set_content(Some(&toolbar_view));

    load_image(image.path.clone(), &picture, &scrolled, &zoom, &img_dims, &window);

    DetailContext { window, picture, info_bar, scrolled, zoom, img_dims, current_path }
}

// ── Context update ───────────────────────────────────────────────

fn update_context(ctx: &DetailContext, image: &LibraryImage) {
    let filename = image.meta.filename.as_deref().unwrap_or("Image");
    ctx.window.set_title(Some(filename));
    *ctx.current_path.borrow_mut() = image.path.clone();
    fill_info_bar(&ctx.info_bar, image);
    reset_zoom(&ctx.picture, &ctx.scrolled, &ctx.zoom);
    load_image(
        image.path.clone(),
        &ctx.picture,
        &ctx.scrolled,
        &ctx.zoom,
        &ctx.img_dims,
        &ctx.window,
    );
}

// ── Info bar ─────────────────────────────────────────────────────

fn build_empty_info_bar() -> gtk4::Box {
    gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(16)
        .margin_start(12)
        .margin_end(12)
        .margin_top(6)
        .margin_bottom(6)
        .build()
}

/// Clear and repopulate `bar` with metadata from `image`.
fn fill_info_bar(bar: &gtk4::Box, image: &LibraryImage) {
    while let Some(child) = bar.last_child() {
        bar.remove(&child);
    }

    let m = &image.meta;
    let mut fields: Vec<String> = Vec::new();

    if let Some(ref name) = m.filename {
        fields.push(name.clone());
    }
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

// ── Async image loader ───────────────────────────────────────────

fn load_image(
    path: PathBuf,
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    zoom: &Rc<Cell<f64>>,
    img_dims: &Rc<Cell<(i32, i32)>>,
    window: &adw::Window,
) {
    let (tx, rx) = mpsc::channel::<Vec<u8>>();
    std::thread::spawn(move || {
        if let Ok(bytes) = std::fs::read(&path) {
            let _ = tx.send(bytes);
        }
    });

    let picture = picture.clone();
    let scrolled = scrolled.clone();
    let zoom = zoom.clone();
    let img_dims = img_dims.clone();
    let window = window.clone();

    glib::timeout_add_local(Duration::from_millis(32), move || {
        match rx.try_recv() {
            Ok(bytes) => {
                let gb = glib::Bytes::from(&bytes);
                if let Ok(texture) = gdk::Texture::from_bytes(&gb) {
                    let iw = texture.width();
                    let ih = texture.height();
                    img_dims.set((iw, ih));
                    let (dw, dh) = display_size(iw, ih);
                    window.set_default_size(dw, dh);
                    reset_zoom(&picture, &scrolled, &zoom);
                    picture.set_paintable(Some(&texture));
                }
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => glib::ControlFlow::Break,
        }
    });
}

// ── Zoom / pan ───────────────────────────────────────────────────

fn wire_zoom_and_pan(
    scrolled: &gtk4::ScrolledWindow,
    picture: &gtk4::Picture,
    zoom: &Rc<Cell<f64>>,
    img_dims: &Rc<Cell<(i32, i32)>>,
) {
    // Track pointer position (widget-local coords) for pointer-anchored zoom.
    let pointer: Rc<Cell<(f64, f64)>> = Rc::new(Cell::new((0.0, 0.0)));
    let motion = gtk4::EventControllerMotion::new();
    motion.connect_motion({
        let pointer = pointer.clone();
        move |_, x, y| pointer.set((x, y))
    });
    scrolled.add_controller(motion);

    // Scroll wheel → zoom
    let scroll_ctrl =
        gtk4::EventControllerScroll::new(gtk4::EventControllerScrollFlags::VERTICAL);
    scroll_ctrl.connect_scroll({
        let picture = picture.clone();
        let scrolled = scrolled.clone();
        let zoom = zoom.clone();
        let img_dims = img_dims.clone();
        let pointer = pointer.clone();
        move |_, _dx, dy| {
            let old = zoom.get();
            let next = if dy > 0.0 {
                old / ZOOM_STEP  // scroll down → zoom out
            } else {
                old * ZOOM_STEP  // scroll up   → zoom in
            }
            .clamp(1.0, ZOOM_MAX);
            zoom.set(next);
            apply_zoom(&picture, &scrolled, old, next, img_dims.get(), pointer.get());
            glib::Propagation::Stop
        }
    });
    scrolled.add_controller(scroll_ctrl);

    // Left-button drag → pan
    let drag_start_h: Rc<Cell<f64>> = Rc::new(Cell::new(0.0));
    let drag_start_v: Rc<Cell<f64>> = Rc::new(Cell::new(0.0));
    let drag = gtk4::GestureDrag::new();

    drag.connect_drag_begin({
        let scrolled = scrolled.clone();
        let drag_start_h = drag_start_h.clone();
        let drag_start_v = drag_start_v.clone();
        move |_, _, _| {
            drag_start_h.set(scrolled.hadjustment().value());
            drag_start_v.set(scrolled.vadjustment().value());
        }
    });

    drag.connect_drag_update({
        let scrolled = scrolled.clone();
        let drag_start_h = drag_start_h.clone();
        let drag_start_v = drag_start_v.clone();
        move |_, offset_x, offset_y| {
            scrolled.hadjustment().set_value(drag_start_h.get() - offset_x);
            scrolled.vadjustment().set_value(drag_start_v.get() - offset_y);
        }
    });

    scrolled.add_controller(drag);
}

/// Apply `new_zoom` (1.0 = fit-to-window, >1.0 = zoomed in).
/// `(px, py)` is the pointer position in scrolled-widget coordinates; the
/// image pixel under that point is kept stationary through the zoom.
fn apply_zoom(
    picture: &gtk4::Picture,
    scrolled: &gtk4::ScrolledWindow,
    old_zoom: f64,
    new_zoom: f64,
    (img_w, img_h): (i32, i32),
    (px, py): (f64, f64),
) {
    if img_w == 0 || img_h == 0 {
        return;
    }

    let vw = scrolled.width() as f64;
    let vh = scrolled.height() as f64;
    // Screen pixels per image pixel at zoom == 1.0 (the "fit" scale).
    let fit = if vw > 0.0 && vh > 0.0 {
        f64::min(vw / img_w as f64, vh / img_h as f64)
    } else {
        1.0
    };

    // Image-space coordinates of the pixel currently under the pointer.
    let (cx, cy) = if old_zoom <= 1.0 {
        // Fit mode: ContentFit::Contain centers the image in the viewport.
        let img_left = (vw - img_w as f64 * fit) / 2.0;
        let img_top  = (vh - img_h as f64 * fit) / 2.0;
        (
            ((px - img_left) / fit).clamp(0.0, img_w as f64),
            ((py - img_top)  / fit).clamp(0.0, img_h as f64),
        )
    } else {
        let ppx = fit * old_zoom;
        (
            (scrolled.hadjustment().value() + px) / ppx,
            (scrolled.vadjustment().value() + py) / ppx,
        )
    };

    if new_zoom <= 1.0 {
        picture.set_content_fit(gtk4::ContentFit::Contain);
        picture.set_size_request(-1, -1);
        scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Never);
    } else {
        picture.set_content_fit(gtk4::ContentFit::Fill);
        let ppx = fit * new_zoom;
        let w = (img_w as f64 * ppx).round() as i32;
        let h = (img_h as f64 * ppx).round() as i32;
        picture.set_size_request(w, h);
        scrolled.set_policy(gtk4::PolicyType::Automatic, gtk4::PolicyType::Automatic);

        // Scroll so the image pixel under the pointer stays under the pointer.
        // Pre-configure adjustment bounds before set_value to avoid clamping.
        let th = (cx * ppx - px).max(0.0);
        let tv = (cy * ppx - py).max(0.0);
        let hadj = scrolled.hadjustment();
        hadj.set_upper(w as f64);
        hadj.set_page_size(vw);
        hadj.set_value(th);
        let vadj = scrolled.vadjustment();
        vadj.set_upper(h as f64);
        vadj.set_page_size(vh);
        vadj.set_value(tv);
    }
}

fn reset_zoom(picture: &gtk4::Picture, scrolled: &gtk4::ScrolledWindow, zoom: &Cell<f64>) {
    zoom.set(1.0);
    picture.set_content_fit(gtk4::ContentFit::Contain);
    picture.set_size_request(-1, -1);
    scrolled.set_policy(gtk4::PolicyType::Never, gtk4::PolicyType::Never);
}

// ── Window sizing ─────────────────────────────────────────────────

/// Compute a window size that fits `img_w × img_h` within the screen budget.
fn display_size(img_w: i32, img_h: i32) -> (i32, i32) {
    if img_w <= 0 || img_h <= 0 {
        return (960, 720);
    }
    let usable_h = MAX_WIN_H - CHROME_H;
    let scale = f64::min(
        MAX_WIN_W as f64 / img_w as f64,
        usable_h as f64 / img_h as f64,
    );
    let w = (img_w as f64 * scale).round() as i32;
    let h = (img_h as f64 * scale).round() as i32 + CHROME_H;
    (w.max(400), h.max(300))
}

// ── System launcher ───────────────────────────────────────────────

fn launch_default_app(path: &PathBuf) {
    let file = gio::File::for_path(path);
    if let Err(e) =
        gio::AppInfo::launch_default_for_uri(&file.uri(), None::<&gio::AppLaunchContext>)
    {
        tracing::warn!("Failed to open {} in default app: {e}", path.display());
    }
}
