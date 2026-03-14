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
//!   • Persons button    — toggle face-detection overlay (green = assigned,
//!                         blue = unknown); click a box to assign a person

mod face_overlay;
mod image_load;
mod info_bar;
mod info_window;
mod zoom_pan;

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::gio;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::LibraryImage;

use face_overlay::FaceOverlay;

// ── Singleton state ──────────────────────────────────────────────

#[derive(Clone)]
struct DetailContext {
    window: adw::Window,
    picture: gtk4::Picture,
    info_bar: gtk4::Box,
    /// Full record for the currently displayed image; used by the info popup.
    current_image: Rc<RefCell<LibraryImage>>,
    scrolled: gtk4::ScrolledWindow,
    zoom: Rc<Cell<f64>>,
    img_dims: Rc<Cell<(i32, i32)>>,
    /// Shared with action-button closures so they always use the current path.
    current_path: Rc<RefCell<PathBuf>>,
    /// Face overlay — loads detections for each image and draws boxes.
    face_overlay: FaceOverlay,
}

thread_local! {
    static DETAIL_CTX: RefCell<Option<DetailContext>> = const { RefCell::new(None) };
}

// ── Public API ───────────────────────────────────────────────────

/// Open (or update) the singleton detail window for `image`.
pub fn open(image: &LibraryImage, parent: &gtk4::Window, db: &Arc<Mutex<maple_db::Database>>) {
    // Reuse an existing visible window.
    let ctx = DETAIL_CTX.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|c| c.window.is_visible())
            .cloned()
    });

    if let Some(ctx) = ctx {
        update_context(&ctx, image, db);
        ctx.window.present();
        return;
    }

    let ctx = build_window(image, parent, db);

    // Clear the singleton when this window closes.
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

fn build_window(
    image: &LibraryImage,
    parent: &gtk4::Window,
    db: &Arc<Mutex<maple_db::Database>>,
) -> DetailContext {
    let current_path = Rc::new(RefCell::new(image.path.clone()));
    let current_image = Rc::new(RefCell::new(image.clone()));
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

    zoom_pan::wire_zoom_and_pan(&scrolled, &picture, &zoom, &img_dims);

    // ── Face overlay (wraps scrolled) ─────────────────────────────
    let settings = maple_state::Settings::load();
    let face_overlay = FaceOverlay::new(
        &scrolled,
        zoom.clone(),
        img_dims.clone(),
        db.clone(),
        settings.face.similarity_threshold,
    );

    // Load detections for the first image immediately.
    face_overlay.load_for_image(image.id, db);

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

    let info_btn = gtk4::Button::builder()
        .icon_name("dialog-information-symbolic")
        .tooltip_text("Show image information")
        .css_classes(["flat"])
        .build();

    let persons_btn = gtk4::ToggleButton::builder()
        .icon_name("system-users-symbolic")
        .tooltip_text("Show detected faces / assign persons")
        .css_classes(["flat"])
        .build();

    // Disable the persons button if no ONNX models are configured.
    if !settings.face.models_available() {
        persons_btn.set_tooltip_text(Some(
            "Face detection unavailable — set face.detector_model to the \
             atksh ONNX model path in settings.toml",
        ));
    }

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

    persons_btn.connect_toggled({
        let face_overlay = face_overlay.clone();
        move |btn| {
            face_overlay.set_visible(btn.is_active());
        }
    });

    let header = adw::HeaderBar::new();
    header.pack_end(&open_btn);
    header.pack_end(&copy_btn);
    header.pack_end(&info_btn);
    header.pack_end(&persons_btn);

    // ── Metadata info strip ───────────────────────────────────────
    let info_bar = info_bar::build_empty_info_bar();
    info_bar::fill_info_bar(&info_bar, image);

    // ── Layout ────────────────────────────────────────────────────
    // face_overlay.container is the gtk4::Overlay that wraps the scrolled window.
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    content.append(&info_bar);
    content.append(&face_overlay.container);

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

    info_btn.connect_clicked({
        let current_image = current_image.clone();
        let window = window.clone();
        let db = db.clone();
        move |_| info_window::open_info_window(&current_image.borrow(), &db, &window)
    });

    image_load::load_image(
        image.path.clone(),
        &picture,
        &scrolled,
        &zoom,
        &img_dims,
        &window,
    );

    DetailContext {
        window,
        picture,
        info_bar,
        current_image,
        scrolled,
        zoom,
        img_dims,
        current_path,
        face_overlay,
    }
}

// ── Context update ───────────────────────────────────────────────

fn update_context(
    ctx: &DetailContext,
    image: &LibraryImage,
    db: &Arc<Mutex<maple_db::Database>>,
) {
    let filename = image.meta.filename.as_deref().unwrap_or("Image");
    ctx.window.set_title(Some(filename));
    *ctx.current_path.borrow_mut() = image.path.clone();
    info_bar::fill_info_bar(&ctx.info_bar, image);
    *ctx.current_image.borrow_mut() = image.clone();
    zoom_pan::reset_zoom(&ctx.picture, &ctx.scrolled, &ctx.zoom);
    image_load::load_image(
        image.path.clone(),
        &ctx.picture,
        &ctx.scrolled,
        &ctx.zoom,
        &ctx.img_dims,
        &ctx.window,
    );
    // Reload face detections for the new image.
    ctx.face_overlay.load_for_image(image.id, db);
}

// ── System launcher ──────────────────────────────────────────────

fn launch_default_app(path: &PathBuf) {
    let file = gio::File::for_path(path);
    if let Err(e) =
        gio::AppInfo::launch_default_for_uri(&file.uri(), None::<&gio::AppLaunchContext>)
    {
        tracing::warn!("Failed to open {} in default app: {}", path.display(), e);
    }
}
