//! Detail window — singleton full-size viewer with zoom + pan.
//!
//! Only one detail window exists at a time.  Activating a new image
//! updates the existing window instead of opening a second one.
//!
//! Controls:
//!   • Scroll wheel      — zoom in / out
//!   • Arrow Left / Up   — previous image
//!   • Arrow Right / Down — next image
//!   • Left-button drag         — pan when zoomed in
//!   • Open button              — launch in default application
//!   • Copy path button         — write absolute path to clipboard
//!   • Persons button           — toggle face-detection overlay (green = assigned,
//!                                blue = unknown); click a box to assign a person
//!   • Collections button       — open window to manage image collections
//!   • Fullscreen button        — toggle borderless fullscreen
//!   • Configurable hotkey (default "a") — add image to last-used collection

mod collections;
mod face_overlay;
mod image_load;
mod info_bar;
mod info_window;
mod zoom_pan;

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use gtk4::gdk;
use gtk4::glib;
use gtk4::gio;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::LibraryImage;

use collections::CollectionBar;
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
    /// Toast overlay for showing notifications inside the detail window.
    #[allow(dead_code)]
    toast_overlay: adw::ToastOverlay,
    /// Collection bar — chips + add-to-collection logic.
    collection_bar: CollectionBar,
    /// Filename label in the bottom bar.
    filename_label: gtk4::Label,
    /// Database handle.
    db: Arc<Mutex<maple_db::Database>>,
    /// Header bar reference — kept alive for fullscreen toggling.
    #[allow(dead_code)]
    header: adw::HeaderBar,
    /// Whether the window is currently in borderless fullscreen mode.
    #[allow(dead_code)]
    is_fullscreen: Rc<Cell<bool>>,
    /// Ordered list of images visible in the grid at the time this image was opened.
    images: Rc<RefCell<Vec<LibraryImage>>>,
    /// Position of the currently displayed image within `images`.
    current_index: Rc<Cell<usize>>,
    /// True while an image load is in flight; navigation is blocked until cleared.
    is_loading: Rc<Cell<bool>>,
}

thread_local! {
    static DETAIL_CTX: RefCell<Option<DetailContext>> = const { RefCell::new(None) };
}

// ── Public API ───────────────────────────────────────────────────

/// Open (or update) the singleton detail window for `image`.
///
/// `index` is the position of `image` within `images`; the full `images` list
/// enables previous/next navigation from inside the detail window.
pub fn open(
    image: &LibraryImage,
    index: usize,
    images: Vec<LibraryImage>,
    parent: &gtk4::Window,
    db: &Arc<Mutex<maple_db::Database>>,
) {
    // Reuse an existing visible window.
    let ctx = DETAIL_CTX.with(|cell| {
        cell.borrow()
            .as_ref()
            .filter(|c| c.window.is_visible())
            .cloned()
    });

    if let Some(ctx) = ctx {
        ctx.current_index.set(index);
        *ctx.images.borrow_mut() = images;
        update_context(&ctx, image, db);
        ctx.window.present();
        return;
    }

    let ctx = build_window(image, index, images, parent, db);

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
    index: usize,
    images: Vec<LibraryImage>,
    parent: &gtk4::Window,
    db: &Arc<Mutex<maple_db::Database>>,
) -> DetailContext {
    let current_path = Rc::new(RefCell::new(image.path.clone()));
    let current_image = Rc::new(RefCell::new(image.clone()));
    let zoom: Rc<Cell<f64>> = Rc::new(Cell::new(1.0));
    let img_dims: Rc<Cell<(i32, i32)>> = Rc::new(Cell::new((0, 0)));
    let is_fullscreen: Rc<Cell<bool>> = Rc::new(Cell::new(false));
    let images: Rc<RefCell<Vec<LibraryImage>>> = Rc::new(RefCell::new(images));
    let current_index: Rc<Cell<usize>> = Rc::new(Cell::new(index));
    let is_loading: Rc<Cell<bool>> = Rc::new(Cell::new(true));

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
        .css_classes(["maple-photo-surface"])
        .build();
    scrolled.set_child(Some(&picture));

    zoom_pan::wire_zoom_and_pan(&scrolled, &picture, &zoom, &img_dims);

    // ── Face overlay (wraps scrolled) ─────────────────────────────
    let settings = maple_state::Settings::load();
    let face_overlay = FaceOverlay::new(
        &scrolled,
        &picture,
        zoom.clone(),
        img_dims.clone(),
        db.clone(),
    );

    // Load detections for the first image immediately.
    face_overlay.load_for_image(image.id, db);

    // ── Toast overlay ─────────────────────────────────────────────
    let toast_overlay = adw::ToastOverlay::new();
    // Drop target: files dropped on the detail window open the import browser
    // in the main window via the app-wide ImportCtx registered in window.rs.
    toast_overlay.add_controller(crate::views::drop_import::make_drop_target());

    // ── Collection bar ────────────────────────────────────────────
    let collection_bar = CollectionBar::new(
        current_image.clone(),
        db.clone(),
        toast_overlay.clone(),
    );

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

    let add_face_btn = gtk4::ToggleButton::builder()
        .icon_name("list-add-symbolic")
        .tooltip_text("Draw a bounding box to tag a new face")
        .css_classes(["flat"])
        .sensitive(false)
        .build();

    let collections_btn = gtk4::Button::builder()
        .icon_name("folder-symbolic")
        .tooltip_text("Manage collections for this image")
        .css_classes(["flat"])
        .build();

    let fullscreen_btn = gtk4::Button::builder()
        .icon_name("view-fullscreen-symbolic")
        .tooltip_text("Toggle fullscreen")
        .css_classes(["flat"])
        .build();

    let rotate_ccw_btn = gtk4::Button::builder()
        .icon_name("object-rotate-left-symbolic")
        .tooltip_text("Rotate 90° counter-clockwise")
        .css_classes(["flat"])
        .build();

    let rotate_cw_btn = gtk4::Button::builder()
        .icon_name("object-rotate-right-symbolic")
        .tooltip_text("Rotate 90° clockwise")
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
        let add_face_btn = add_face_btn.clone();
        move |btn| {
            let active = btn.is_active();
            face_overlay.set_visible(active);
            add_face_btn.set_sensitive(active);
            if !active {
                add_face_btn.set_active(false);
                face_overlay.set_draw_mode(false);
            }
        }
    });

    add_face_btn.connect_toggled({
        let face_overlay = face_overlay.clone();
        move |btn| {
            face_overlay.set_draw_mode(btn.is_active());
        }
    });

    collections_btn.connect_clicked({
        let collection_bar = collection_bar.clone();
        move |btn| {
            let parent_window = btn
                .root()
                .and_then(|r| r.downcast::<gtk4::Window>().ok());
            if let Some(win) = parent_window {
                collection_bar.reload();
                // Open the collection picker window.
                collections::open_picker(&collection_bar, &win);
            }
        }
    });

    let header = adw::HeaderBar::new();
    header.pack_end(&open_btn);
    header.pack_end(&copy_btn);
    header.pack_end(&info_btn);
    header.pack_end(&fullscreen_btn);
    header.pack_start(&persons_btn);
    header.pack_start(&add_face_btn);
    header.pack_start(&collections_btn);
    header.pack_start(&rotate_ccw_btn);
    header.pack_start(&rotate_cw_btn);

    // ── Metadata info strip ───────────────────────────────────────
    let info_bar = info_bar::build_empty_info_bar();
    info_bar::fill_info_bar(&info_bar, image);

    // ── Bottom bar: collection chips (left) + filename (right) ───
    let filename_label = gtk4::Label::builder()
        .label(image.meta.filename.as_deref().unwrap_or(""))
        .halign(gtk4::Align::End)
        .hexpand(true)
        .margin_end(8)
        .css_classes(["dim-label", "caption"])
        .build();

    let bottom_bar = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .margin_top(4)
        .margin_bottom(4)
        .margin_start(8)
        .margin_end(4)
        .build();
    bottom_bar.append(&collection_bar.chips);
    bottom_bar.append(&filename_label);

    // ── Layout ────────────────────────────────────────────────────
    // Info strip and collection bar are real toolbars so they share the
    // header/footer surface instead of floating over the photo.
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.add_top_bar(&info_bar);
    toolbar_view.add_bottom_bar(&bottom_bar);
    toolbar_view.set_content(Some(&face_overlay.container));

    toast_overlay.set_child(Some(&toolbar_view));

    let filename = image.meta.filename.as_deref().unwrap_or("Image");
    let window = adw::Window::builder()
        .title(filename)
        .default_width(960)
        .default_height(720)
        .transient_for(parent)
        .build();
    window.set_content(Some(&toast_overlay));

    // ── Fullscreen toggle ─────────────────────────────────────────
    fullscreen_btn.connect_clicked({
        let window = window.clone();
        let header = header.clone();
        let info_bar = info_bar.clone();
        let is_fullscreen = is_fullscreen.clone();
        move |btn| {
            let going_fullscreen = !is_fullscreen.get();
            is_fullscreen.set(going_fullscreen);

            if going_fullscreen {
                window.fullscreen();
                header.set_show_title(false);
                header.set_decoration_layout(Some(""));
                info_bar.set_visible(false);
                btn.set_icon_name("view-restore-symbolic");
                btn.set_tooltip_text(Some("Exit fullscreen"));
            } else {
                window.unfullscreen();
                header.set_show_title(true);
                header.set_decoration_layout(None);
                info_bar.set_visible(true);
                btn.set_icon_name("view-fullscreen-symbolic");
                btn.set_tooltip_text(Some("Toggle fullscreen"));
            }
        }
    });

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
        {
            let is_loading = is_loading.clone();
            move || is_loading.set(false)
        },
    );

    let ctx = DetailContext {
        window,
        picture,
        info_bar,
        current_image,
        scrolled,
        zoom,
        img_dims,
        current_path,
        face_overlay,
        toast_overlay,
        collection_bar,
        filename_label,
        db: db.clone(),
        header,
        is_fullscreen,
        images,
        current_index,
        is_loading,
    };

    // Load initial collection chips.
    ctx.collection_bar.reload();

    // ── Hotkey controller ────────────────────────────────────────
    wire_hotkey(&ctx, &settings);

    // ── Navigation (arrow keys) ───────────────────────────────────
    wire_navigation(&ctx);

    // ── Rotation buttons ─────────────────────────────────────────
    wire_rotation(&ctx, &rotate_cw_btn, &rotate_ccw_btn);

    ctx
}

// ── Context update ───────────────────────────────────────────────

fn update_context(
    ctx: &DetailContext,
    image: &LibraryImage,
    db: &Arc<Mutex<maple_db::Database>>,
) {
    let filename = image.meta.filename.as_deref().unwrap_or("Image");
    ctx.window.set_title(Some(filename));
    ctx.filename_label.set_label(filename);
    *ctx.current_path.borrow_mut() = image.path.clone();
    info_bar::fill_info_bar(&ctx.info_bar, image);
    *ctx.current_image.borrow_mut() = image.clone();
    zoom_pan::reset_zoom(&ctx.picture, &ctx.scrolled, &ctx.zoom);
    ctx.is_loading.set(true);
    image_load::load_image(
        image.path.clone(),
        &ctx.picture,
        &ctx.scrolled,
        &ctx.zoom,
        &ctx.img_dims,
        &ctx.window,
        {
            let is_loading = ctx.is_loading.clone();
            move || is_loading.set(false)
        },
    );
    // Reload face detections for the new image.
    ctx.face_overlay.load_for_image(image.id, db);
    // Reload collection chips.
    ctx.collection_bar.reload();
}

// ── Navigation ───────────────────────────────────────────────────

/// Move `delta` steps in the image list (+1 = next, -1 = prev).
///
/// No-ops while a load is in flight so rapid key/scroll presses do not
/// queue up multiple image transitions.
fn navigate_relative(ctx: &DetailContext, delta: i32) {
    if ctx.is_loading.get() {
        return;
    }
    let cur = ctx.current_index.get();
    let new_image = {
        let images = ctx.images.borrow();
        let len = images.len();
        if len == 0 {
            return;
        }
        let new_idx = (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize;
        if new_idx == cur {
            return;
        }
        ctx.current_index.set(new_idx);
        images[new_idx].clone()
    };
    update_context(ctx, &new_image, &ctx.db.clone());
}

// ── Rotation ─────────────────────────────────────────────────────

fn wire_rotation(ctx: &DetailContext, cw_btn: &gtk4::Button, ccw_btn: &gtk4::Button) {
    for (btn, clockwise) in [(&cw_btn.clone(), true), (&ccw_btn.clone(), false)] {
        btn.connect_clicked({
            let ctx = ctx.clone();
            let cw_btn = cw_btn.clone();
            let ccw_btn = ccw_btn.clone();
            move |_| rotate_image(&ctx, clockwise, &cw_btn, &ccw_btn)
        });
    }
}

fn rotate_image(ctx: &DetailContext, clockwise: bool, cw_btn: &gtk4::Button, ccw_btn: &gtk4::Button) {
    let image_id = ctx.current_image.borrow().id;
    let path = ctx.current_path.borrow().clone();

    cw_btn.set_sensitive(false);
    ccw_btn.set_sensitive(false);

    let (tx, rx) = mpsc::channel::<Result<(u16, [u8; 32]), String>>();
    std::thread::spawn(move || {
        let _ = tx.send(
            maple_db::rotate_image_file(&path, clockwise).map_err(|e| e.to_string()),
        );
    });

    let ctx = ctx.clone();
    let cw_btn = cw_btn.clone();
    let ccw_btn = ccw_btn.clone();

    glib::timeout_add_local(Duration::from_millis(32), move || {
        match rx.try_recv() {
            Ok(Ok((new_orientation, new_hash))) => {
                if let Err(e) = maple_db::lock_db(&ctx.db)
                    .update_image_hash_and_orientation(image_id, &new_hash, new_orientation as i64)
                {
                    tracing::warn!("Failed to update DB after rotation: {e}");
                }
                {
                    let mut img = ctx.current_image.borrow_mut();
                    img.meta.orientation = Some(new_orientation as i64);
                    img.hash = Some(new_hash);
                }
                {
                    let idx = ctx.current_index.get();
                    let mut images = ctx.images.borrow_mut();
                    if let Some(img) = images.get_mut(idx) {
                        img.meta.orientation = Some(new_orientation as i64);
                        img.hash = Some(new_hash);
                    }
                }
                ctx.is_loading.set(true);
                let is_loading = ctx.is_loading.clone();
                image_load::load_image(
                    ctx.current_path.borrow().clone(),
                    &ctx.picture,
                    &ctx.scrolled,
                    &ctx.zoom,
                    &ctx.img_dims,
                    &ctx.window,
                    move || is_loading.set(false),
                );
                cw_btn.set_sensitive(true);
                ccw_btn.set_sensitive(true);
                glib::ControlFlow::Break
            }
            Ok(Err(msg)) => {
                let toast = adw::Toast::new(&format!("Rotation failed: {msg}"));
                ctx.toast_overlay.add_toast(toast);
                cw_btn.set_sensitive(true);
                ccw_btn.set_sensitive(true);
                glib::ControlFlow::Break
            }
            Err(mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
            Err(mpsc::TryRecvError::Disconnected) => {
                cw_btn.set_sensitive(true);
                ccw_btn.set_sensitive(true);
                glib::ControlFlow::Break
            }
        }
    });
}

fn wire_navigation(ctx: &DetailContext) {
    let key_ctrl = gtk4::EventControllerKey::new();
    key_ctrl.connect_key_pressed({
        let ctx = ctx.clone();
        move |_, key, _, _| match key {
            gdk::Key::Left | gdk::Key::Up => {
                navigate_relative(&ctx, -1);
                glib::Propagation::Stop
            }
            gdk::Key::Right | gdk::Key::Down => {
                navigate_relative(&ctx, 1);
                glib::Propagation::Stop
            }
            _ => glib::Propagation::Proceed,
        }
    });
    ctx.window.add_controller(key_ctrl);
}

// ── Hotkey ───────────────────────────────────────────────────────

fn wire_hotkey(ctx: &DetailContext, settings: &maple_state::Settings) {
    let hotkey_name = settings.collections.add_hotkey.clone();
    let target_key = gdk::Key::from_name(&hotkey_name).unwrap_or(gdk::Key::a);

    let hotkeys = crate::HotkeyManager::new();
    let collection_bar = ctx.collection_bar.clone();
    hotkeys.register("add-to-collection", target_key, move || {
        if let Some(coll_id) = collection_bar.last_collection_id.get() {
            collection_bar.add_to_collection(coll_id);
            return gtk4::glib::Propagation::Stop;
        }
        gtk4::glib::Propagation::Proceed
    });
    hotkeys.attach(&ctx.window);
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
