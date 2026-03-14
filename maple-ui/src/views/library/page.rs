//! Library browser page — wires the search bar, grid, and detail window.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;

use maple_db::SearchQuery;

use super::{detail_window, grid::LibraryGrid, search_bar};

/// Build the library browser page.
///
/// Opens a background metadata-filler thread so that newly imported images
/// get their EXIF fields populated while the user browses.
///
/// If `settings.ai.enabled` is true, the AI tagger starts automatically.
/// A header button lets the user start or stop it at any time.
pub fn build_library_page(
    _nav_view: &adw::NavigationView,
    db: Arc<Mutex<maple_db::Database>>,
) -> adw::NavigationPage {
    let settings = maple_state::Settings::load();

    // ── Grid ──────────────────────────────────────────────────────
    let library_grid = LibraryGrid::new(db.clone(), {
        let db = db.clone();
        move |image, window| {
            detail_window::open(&image, &window, &db);
        }
    });

    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .build();
    scrolled.set_child(Some(library_grid.widget()));

    // ── Search bar ────────────────────────────────────────────────
    let search_entry = search_bar::build_search_entry({
        let grid = library_grid.clone();
        move |text| {
            grid.load(SearchQuery::default().with_text(&text));
        }
    });

    let search_clamp = adw::Clamp::builder()
        .maximum_size(600)
        .child(&search_entry)
        .margin_top(8)
        .margin_bottom(8)
        .build();

    // ── Layout ────────────────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    content.append(&search_clamp);
    content.append(&scrolled);

    // ── Header with AI tagger button ──────────────────────────────
    let header = adw::HeaderBar::new();

    let ai_button = gtk4::Button::with_label("Tag with AI");
    ai_button.set_tooltip_text(Some("Describe images with AI to improve search"));
    header.pack_end(&ai_button);

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let page = adw::NavigationPage::builder()
        .title("Library")
        .child(&toolbar_view)
        .build();

    // ── Face tagger button ────────────────────────────────────────
    let face_btn = gtk4::Button::with_label("Detect Faces");
    face_btn.set_tooltip_text(Some("Detect and tag faces in all library images"));
    if !settings.face.models_available() {
        face_btn.set_sensitive(false);
        face_btn.set_tooltip_text(Some(
            "Face detection unavailable — set face.detector_model to the \
             atksh ONNX model path in settings.toml",
        ));
    }
    header.pack_end(&face_btn);

    let face_tagger: Rc<RefCell<Option<maple_db::FaceTagger>>> = Rc::new(RefCell::new(None));

    // Helper: load the face detector in a background thread, then spawn the
    // tagger on the main thread once loading completes.  This keeps the UI
    // responsive during the potentially slow ORT model-load + graph-build step.
    let start_face_detector: Rc<dyn Fn()> = Rc::new({
        let db = db.clone();
        let face_tagger = face_tagger.clone();
        let face_btn = face_btn.clone();
        let settings = settings.clone();
        move || {
            let detector_path = settings.face.detector_model.clone();
            let embedder_path = settings.face.embedder_path().map(|p| p.to_owned());
            let device: maple_db::models::ModelDevice =
                settings.face.device.parse().unwrap_or_default();

            face_btn.set_sensitive(false);
            face_btn.set_label("Loading model…");

            let (tx, rx) =
                std::sync::mpsc::sync_channel::<Result<maple_db::FaceDetector, String>>(1);

            std::thread::Builder::new()
                .name("face-model-loader".to_owned())
                .spawn(move || {
                    let result = maple_db::FaceDetector::with_device(
                        &detector_path,
                        embedder_path.as_deref(),
                        &device,
                    )
                    .map_err(|e| format!("{e:#}"));
                    let _ = tx.send(result);
                })
                .expect("failed to spawn face model loader thread");

            glib::timeout_add_local(Duration::from_millis(200), {
                let db = db.clone();
                let face_tagger = face_tagger.clone();
                let face_btn = face_btn.clone();
                move || match rx.try_recv() {
                    Ok(Ok(detector)) => {
                        *face_tagger.borrow_mut() =
                            Some(maple_db::spawn_face_tagger(db.clone(), detector));
                        face_btn.set_label("Stop Face Detection");
                        face_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("Failed to load face detector: {e}");
                        face_btn.set_label("Detect Faces");
                        face_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        face_btn.set_label("Detect Faces");
                        face_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                }
            });
        }
    });

    face_btn.connect_clicked({
        let face_tagger = face_tagger.clone();
        let face_btn = face_btn.clone();
        let start_face_detector = start_face_detector.clone();
        move |_| {
            let mut guard = face_tagger.borrow_mut();
            if let Some(t) = guard.take() {
                t.stop();
                face_btn.set_label("Detect Faces");
            } else {
                drop(guard); // release borrow before start_face_detector borrows it
                start_face_detector();
            }
        }
    });

    // ── AI tagger state ───────────────────────────────────────────
    // Holds the running tagger handle; None means stopped.
    let tagger: Rc<RefCell<Option<maple_db::AiTagger>>> = Rc::new(RefCell::new(None));

    let update_button_label = {
        let ai_button = ai_button.clone();
        move |running: bool| {
            if running {
                ai_button.set_label("Stop AI Tagging");
            } else {
                ai_button.set_label("Tag with AI");
            }
        }
    };

    // Helper: build a fresh describer from current settings.
    let make_describer = {
        let settings = settings.clone();
        move || {
            maple_db::LmStudioDescriber::new(
                &settings.ai.server_url,
                &settings.ai.model,
                &settings.ai.prompt,
            )
        }
    };

    // Toggle button: start if stopped, stop if running.
    ai_button.connect_clicked({
        let db = db.clone();
        let tagger = tagger.clone();
        let update_label = update_button_label.clone();
        let make_describer = make_describer.clone();
        move |_| {
            let mut guard = tagger.borrow_mut();
            if let Some(t) = guard.take() {
                t.stop();
                update_label(false);
            } else {
                *guard = Some(maple_db::spawn_ai_tagger(db.clone(), make_describer()));
                update_label(true);
            }
        }
    });

    // ── Initial load + background workers ─────────────────────────
    let loaded = Rc::new(Cell::new(false));
    page.connect_map({
        let grid = library_grid.clone();
        let db = db.clone();
        let tagger = tagger.clone();
        move |_| {
            if !loaded.get() {
                loaded.set(true);
                grid.load(SearchQuery::default());
                maple_db::spawn_metadata_filler(db.clone());

                if settings.ai.enabled {
                    *tagger.borrow_mut() =
                        Some(maple_db::spawn_ai_tagger(db.clone(), make_describer()));
                    update_button_label(true);
                }

                if settings.face.enabled && settings.face.models_available() {
                    start_face_detector();
                }
            }
        }
    });


    page
}
