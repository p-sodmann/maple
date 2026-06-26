//! Library browser page — wires the search bar, grid, and detail window.
//!
//! Reached from `home::build_home_page` via the "Browse Library" button.
//!
//! Background workers started here (if enabled in Settings):
//!   • **Metadata filler** — one-shot thread that populates EXIF fields for
//!     images missing metadata (runs once on page creation).
//!   • **AI tagger** — loops until stopped; describes images via LM Studio.
//!   • **Face tagger** — loops until stopped; detects faces + computes
//!     ArcFace embeddings via ONNX.
//!
//! All workers receive `Arc<Mutex<Database>>` and are stopped when the page
//! is destroyed (face/AI taggers) or when their one-shot work completes
//! (metadata filler).

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use gtk4::glib;
use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::{SearchQuery, ThumbnailCache};

use super::{build_face_tagging_page, collection_manager, detail_window, grid::LibraryGrid, search_bar};
use crate::views::settings_window;

/// Shared slot holding the "load the semantic model" action, so `reload_grid`
/// (defined first) can lazily trigger it on the first search without a
/// definition-order cycle.
type SemanticStartSlot = Rc<RefCell<Option<Rc<dyn Fn()>>>>;

/// Build the library browser page.
///
/// Opens a background metadata-filler thread so that newly imported images
/// get their EXIF fields populated while the user browses.
///
/// If `settings.ai.enabled` is true, the AI tagger starts automatically.
/// A header button lets the user start or stop it at any time.
pub fn build_library_page(
    nav_view: &adw::NavigationView,
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<ThumbnailCache>,
) -> adw::NavigationPage {
    let settings = maple_state::Settings::load();

    // ── Grid ──────────────────────────────────────────────────────
    let library_grid = LibraryGrid::new(
        db.clone(),
        cache.clone(),
        settings.thumbnails.quality,
        settings.thumbnails.size,
        {
            let db = db.clone();
            move |index, images, window| {
                let image = images[index].clone();
                detail_window::open(&image, index, images, &window, &db);
            }
        },
    );

    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .build();
    scrolled.set_child(Some(library_grid.widget()));

    // ── Collection filter state ──────────────────────────────────
    let active_collection: Rc<Cell<Option<i64>>> = Rc::new(Cell::new(None));
    let search_text: Rc<RefCell<String>> = Rc::new(RefCell::new(String::new()));

    // ── Semantic search state ─────────────────────────────────────
    // The resident sentence encoder embeds search queries; the worker embeds
    // descriptions in the background.  Both are populated once the semantic
    // index is built — which happens automatically on the first search (or via
    // the header button).  Until the model is ready, search is keyword-only.
    let semantic_encoder: Rc<RefCell<Option<maple_db::SemanticEncoder>>> =
        Rc::new(RefCell::new(None));
    let semantic_worker: Rc<RefCell<Option<maple_db::SentenceEmbedder>>> =
        Rc::new(RefCell::new(None));
    let semantic_k = settings.semantic.knn_k;
    // True while the model download/compile is in flight.
    let semantic_loading = Rc::new(Cell::new(false));
    // Set once an automatic (typing-triggered) load has been attempted, so a
    // failed load doesn't retry on every keystroke (the button can still retry).
    let semantic_auto_tried = Rc::new(Cell::new(false));
    // Holds the "load the semantic model" action; assigned once it is defined
    // below, so `reload_grid` (defined first) can trigger it without a cycle.
    let start_semantic_holder: SemanticStartSlot = Rc::new(RefCell::new(None));

    // Helper closure to reload the grid with both text + collection filter.
    let reload_grid = {
        let grid = library_grid.clone();
        let active_collection = active_collection.clone();
        let search_text = search_text.clone();
        let semantic_encoder = semantic_encoder.clone();
        let semantic_auto_tried = semantic_auto_tried.clone();
        let start_semantic_holder = start_semantic_holder.clone();
        Rc::new(move || {
            let mut q = SearchQuery::default();
            let text = search_text.borrow().clone();
            if !text.trim().is_empty() {
                q = q.with_text(&text);
                // Hybrid search: if a sentence encoder is loaded, embed the
                // query so keyword + semantic results are merged.
                let embedding = semantic_encoder
                    .borrow()
                    .as_ref()
                    .and_then(|enc| enc.embed_query(&text));
                match embedding {
                    Some(embedding) => q = q.with_semantic(embedding, semantic_k),
                    None => {
                        // Model not ready: lazily kick off a load the first time
                        // the user searches.  When it finishes it re-runs the
                        // search (see the loader completion handler below).
                        if !semantic_auto_tried.get() {
                            semantic_auto_tried.set(true);
                            let start = start_semantic_holder.borrow().clone();
                            if let Some(start) = start {
                                start();
                            }
                        }
                    }
                }
            }
            if let Some(cid) = active_collection.get() {
                q = q.with_collection(cid);
            }
            grid.load(q);
        })
    };

    // ── Search bar ────────────────────────────────────────────────
    let search_entry = search_bar::build_search_entry({
        let search_text = search_text.clone();
        let reload_grid = reload_grid.clone();
        move |text| {
            *search_text.borrow_mut() = text;
            reload_grid();
        }
    });

    // The search bar lives in a second top bar so it shares the header
    // surface instead of floating over the grid.
    let search_clamp = adw::Clamp::builder()
        .maximum_size(600)
        .child(&search_entry)
        .margin_top(6)
        .margin_bottom(8)
        .margin_start(12)
        .margin_end(12)
        .build();

    // ── Header ────────────────────────────────────────────────────
    let header = adw::HeaderBar::new();

    let ai_button = gtk4::Button::with_label("Tag with AI");
    ai_button.set_tooltip_text(Some("Describe images with AI to improve search"));
    header.pack_end(&ai_button);

    // ── Semantic index button ─────────────────────────────────────
    let semantic_button = gtk4::Button::with_label("Build Semantic Index");
    semantic_button.set_tooltip_text(Some(
        "Embed AI descriptions for semantic search \
         (downloads the sentence-transformer model on first use)",
    ));
    header.pack_end(&semantic_button);

    // ── Face tagger button ────────────────────────────────────────
    let face_detect_btn = gtk4::Button::with_label("Detect Faces");
    face_detect_btn.set_tooltip_text(Some("Detect and tag faces in all library images"));
    if !settings.face.models_available() {
        face_detect_btn.set_sensitive(false);
        face_detect_btn.set_tooltip_text(Some(
            "Face detection unavailable — set face.detector_model and \
             face.detector_type in settings.toml",
        ));
    }
    header.pack_end(&face_detect_btn);

    // ── Tag Faces button (opens the tagging wizard) ───────────────
    let tag_faces_btn = gtk4::Button::with_label("Tag Faces");
    tag_faces_btn.set_tooltip_text(Some("Step through untagged faces and assign names"));
    if !settings.face.models_available() {
        tag_faces_btn.set_sensitive(false);
    }
    header.pack_end(&tag_faces_btn);

    tag_faces_btn.connect_clicked({
        let nav_view = nav_view.clone();
        let db = db.clone();
        let settings = settings.clone();
        move |_| {
            let page = build_face_tagging_page(
                db.clone(),
                settings.face.tagging_top_k,
                nav_view.clone(),
                false,
            );
            nav_view.push(&page);
        }
    });


    // ── Settings button ───────────────────────────────────────────
    let settings_btn = gtk4::Button::builder()
        .icon_name("preferences-system-symbolic")
        .tooltip_text("Settings")
        .css_classes(["flat"])
        .build();
    header.pack_start(&settings_btn);

    settings_btn.connect_clicked({
        let db = db.clone();
        let cache = cache.clone();
        let library_grid = library_grid.clone();
        let reload_grid = reload_grid.clone();
        move |btn| {
            if let Some(win) = btn.root().and_downcast::<gtk4::Window>() {
                let grid = library_grid.clone();
                let reload = reload_grid.clone();
                settings_window::open_settings(&win, &db, cache.clone(), move || {
                    let new_size = maple_state::Settings::load().thumbnails.size;
                    grid.set_thumb_size(new_size);
                    reload();
                });
            }
        }
    });

    // ── Collection filter button ─────────────────────────────────
    let coll_menu_btn = gtk4::MenuButton::builder()
        .icon_name("folder-symbolic")
        .tooltip_text("Filter by collection")
        .css_classes(["flat"])
        .build();
    header.pack_start(&coll_menu_btn);

    // Build (and rebuild) the collection popover content.
    let build_coll_popover = {
        let db = db.clone();
        let active_collection = active_collection.clone();
        let reload_grid = reload_grid.clone();
        let coll_menu_btn = coll_menu_btn.clone();
        Rc::new(move || {
            let list = gtk4::ListBox::builder()
                .selection_mode(gtk4::SelectionMode::None)
                .css_classes(["boxed-list"])
                .build();

            // "All" row.
            let all_row = adw::ActionRow::builder()
                .title("All Images")
                .activatable(true)
                .build();
            if active_collection.get().is_none() {
                all_row.add_prefix(&gtk4::Image::from_icon_name("object-select-symbolic"));
            }
            list.append(&all_row);

            let collections = db
                .lock()
                .ok()
                .and_then(|d| d.all_collections().ok())
                .unwrap_or_default();

            for coll in &collections {
                let dot = crate::widgets::color_dot(&coll.color, 12);

                let subtitle = format!("{} image{}", coll.image_count, if coll.image_count == 1 { "" } else { "s" });
                let row = adw::ActionRow::builder()
                    .title(&coll.name)
                    .subtitle(&subtitle)
                    .activatable(true)
                    .build();
                row.add_prefix(&dot);

                if active_collection.get() == Some(coll.id) {
                    row.add_suffix(&gtk4::Image::from_icon_name("object-select-symbolic"));
                }

                list.append(&row);
            }

            // Separator + Manage row.
            let sep = gtk4::Separator::new(gtk4::Orientation::Horizontal);
            list.append(&sep);
            let manage_row = adw::ActionRow::builder()
                .title("Manage Collections…")
                .activatable(true)
                .build();
            list.append(&manage_row);

            // Click handling.
            let active_collection = active_collection.clone();
            let reload_grid = reload_grid.clone();
            let db2 = db.clone();
            let coll_menu_btn = coll_menu_btn.clone();
            let collections_clone = collections.clone();
            list.connect_row_activated(move |_, row| {
                let idx = row.index();
                if idx == 0 {
                    // "All"
                    active_collection.set(None);
                    coll_menu_btn.set_icon_name("folder-symbolic");
                    coll_menu_btn.popdown();
                    reload_grid();
                } else if idx as usize <= collections_clone.len() {
                    let coll = &collections_clone[(idx - 1) as usize];
                    active_collection.set(Some(coll.id));
                    coll_menu_btn.set_label(&coll.name);
                    coll_menu_btn.popdown();
                    reload_grid();
                } else {
                    // "Manage Collections…"
                    coll_menu_btn.popdown();
                    if let Some(win) = coll_menu_btn.root().and_downcast::<gtk4::Window>() {
                        let reload_grid = reload_grid.clone();
                        collection_manager::open_manager(&win, &db2, move || {
                            reload_grid();
                        });
                    }
                }
            });

            let scroll = gtk4::ScrolledWindow::builder()
                .hscrollbar_policy(gtk4::PolicyType::Never)
                .vscrollbar_policy(gtk4::PolicyType::Automatic)
                .max_content_height(400)
                .propagate_natural_height(true)
                .build();
            scroll.set_child(Some(&list));

            let popover = gtk4::Popover::new();
            popover.set_child(Some(&scroll));
            popover
        })
    };

    // Set an initial popover and rebuild its content each time it is shown.
    {
        let popover = build_coll_popover();
        coll_menu_btn.set_popover(Some(&popover));
    }
    // Rebuild on each open so collection list is fresh.
    coll_menu_btn.connect_notify_local(Some("active"), {
        let build_coll_popover = build_coll_popover.clone();
        let coll_menu_btn = coll_menu_btn.clone();
        move |btn, _| {
            // MenuButton "active" is true when the popover is about to show.
            if btn.is_active() {
                let popover = build_coll_popover();
                coll_menu_btn.set_popover(Some(&popover));
                popover.popup();
            }
        }
    });

    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.add_top_bar(&search_clamp);
    toolbar_view.set_content(Some(&scrolled));

    let page = adw::NavigationPage::builder()
        .title("Library")
        .child(&toolbar_view)
        .build();

    // ── Face detector state ───────────────────────────────────────
    let face_tagger: Rc<RefCell<Option<maple_db::FaceTagger>>> = Rc::new(RefCell::new(None));

    let start_face_detector: Rc<dyn Fn()> = Rc::new({
        let db = db.clone();
        let face_tagger = face_tagger.clone();
        let face_detect_btn = face_detect_btn.clone();
        let settings = settings.clone();
        move || {
            let detector_path = settings.face.detector_model.clone();
            let embedder_path = settings.face.embedder_path().map(|p| p.to_owned());
            let device: maple_db::models::ModelDevice =
                settings.face.device.parse().unwrap_or_default();
            let detector_kind = settings.face.detector_type;
            let debug_dir = settings
                .debug
                .then(|| maple_state::config_dir().join("aligned_faces"));

            face_detect_btn.set_sensitive(false);
            face_detect_btn.set_label("Loading model…");

            let (tx, rx) =
                std::sync::mpsc::sync_channel::<Result<maple_db::FaceDetector, String>>(1);

            std::thread::Builder::new()
                .name("face-model-loader".to_owned())
                .spawn(move || {
                    let result = maple_db::FaceDetector::with_device(
                        &detector_path,
                        embedder_path.as_deref(),
                        &device,
                        detector_kind,
                        debug_dir,
                    )
                    .map_err(|e| format!("{e:#}"));
                    let _ = tx.send(result);
                })
                .expect("failed to spawn face model loader thread");

            glib::timeout_add_local(Duration::from_millis(200), {
                let db = db.clone();
                let face_tagger = face_tagger.clone();
                let face_detect_btn = face_detect_btn.clone();
                move || match rx.try_recv() {
                    Ok(Ok(detector)) => {
                        *face_tagger.borrow_mut() =
                            Some(maple_db::spawn_face_tagger(db.clone(), detector));
                        face_detect_btn.set_label("Stop Face Detection");
                        face_detect_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("Failed to load face detector: {e}");
                        face_detect_btn.set_label("Detect Faces");
                        face_detect_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        face_detect_btn.set_label("Detect Faces");
                        face_detect_btn.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                }
            });
        }
    });

    face_detect_btn.connect_clicked({
        let face_tagger = face_tagger.clone();
        let face_detect_btn = face_detect_btn.clone();
        let start_face_detector = start_face_detector.clone();
        move |_| {
            let mut guard = face_tagger.borrow_mut();
            if let Some(t) = guard.take() {
                t.stop();
                face_detect_btn.set_label("Detect Faces");
            } else {
                drop(guard);
                start_face_detector();
            }
        }
    });

    // ── AI tagger state ───────────────────────────────────────────
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

    // ── Semantic index control ────────────────────────────────────
    // Loads the sentence-transformer model (downloading on first use) and
    // starts the background embedder.  Idempotent: a no-op while a load is in
    // flight, and only restarts the worker if the model is already resident.
    let start_semantic: Rc<dyn Fn()> = Rc::new({
        let db = db.clone();
        let semantic_encoder = semantic_encoder.clone();
        let semantic_worker = semantic_worker.clone();
        let semantic_loading = semantic_loading.clone();
        let semantic_button = semantic_button.clone();
        let settings = settings.clone();
        let reload_grid = reload_grid.clone();
        move || {
            // Already resident → just ensure the background worker is running.
            let resident = semantic_encoder.borrow().clone();
            if let Some(enc) = resident {
                if semantic_worker.borrow().is_none() {
                    *semantic_worker.borrow_mut() =
                        Some(maple_db::spawn_sentence_embedder(db.clone(), enc));
                }
                semantic_button.set_label("Stop Semantic Index");
                return;
            }
            // A load is already in flight — don't start a second one.
            if semantic_loading.get() {
                return;
            }
            semantic_loading.set(true);

            let sem = settings.semantic.clone();
            let device: maple_db::models::ModelDevice = sem.device.parse().unwrap_or_default();

            semantic_button.set_sensitive(false);
            semantic_button.set_label("Loading model…");

            // Download + load the model off the main thread (network + ONNX
            // graph compile are slow).
            let (tx, rx) =
                std::sync::mpsc::sync_channel::<Result<maple_db::SemanticEncoder, String>>(1);

            std::thread::Builder::new()
                .name("semantic-model-loader".to_owned())
                .spawn(move || {
                    let result = (|| -> Result<maple_db::SemanticEncoder, String> {
                        let (onnx, tokenizer) = maple_db::models::fetch_sentence_model(
                            &sem.model,
                            &sem.onnx_file,
                            &sem.tokenizer_file,
                        )
                        .map_err(|e| format!("{e:#}"))?;
                        let model = maple_db::models::ModelFactory::new()
                            .with_device(device)
                            .build_text_embedder(&onnx, &tokenizer)
                            .map_err(|e| format!("{e:#}"))?;
                        Ok(maple_db::SemanticEncoder::new(model, sem.model.clone()))
                    })();
                    let _ = tx.send(result);
                })
                .expect("failed to spawn semantic model loader thread");

            glib::timeout_add_local(Duration::from_millis(200), {
                let db = db.clone();
                let semantic_encoder = semantic_encoder.clone();
                let semantic_worker = semantic_worker.clone();
                let semantic_loading = semantic_loading.clone();
                let semantic_button = semantic_button.clone();
                let reload_grid = reload_grid.clone();
                move || match rx.try_recv() {
                    Ok(Ok(encoder)) => {
                        // Match the vector table to this model (rebuilds the
                        // index if the model / dimension changed).
                        if let Err(e) = maple_db::lock_db(&db)
                            .ensure_vec_table(encoder.model_id(), encoder.dim())
                        {
                            tracing::warn!("semantic: ensure_vec_table failed: {e}");
                        }
                        *semantic_worker.borrow_mut() =
                            Some(maple_db::spawn_sentence_embedder(db.clone(), encoder.clone()));
                        *semantic_encoder.borrow_mut() = Some(encoder);
                        semantic_loading.set(false);
                        semantic_button.set_label("Stop Semantic Index");
                        semantic_button.set_sensitive(true);
                        // Re-run the current search now that the query can be
                        // embedded (semantic results appear without retyping).
                        reload_grid();
                        glib::ControlFlow::Break
                    }
                    Ok(Err(e)) => {
                        tracing::warn!("Failed to load semantic model: {e}");
                        semantic_loading.set(false);
                        semantic_button.set_label("Build Semantic Index");
                        semantic_button.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => glib::ControlFlow::Continue,
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        semantic_loading.set(false);
                        semantic_button.set_label("Build Semantic Index");
                        semantic_button.set_sensitive(true);
                        glib::ControlFlow::Break
                    }
                }
            });
        }
    });

    // Publish the start action so `reload_grid` can lazily trigger a load on
    // the first search (breaks the definition-order cycle).
    *start_semantic_holder.borrow_mut() = Some(start_semantic.clone());

    semantic_button.connect_clicked({
        let semantic_worker = semantic_worker.clone();
        let semantic_button = semantic_button.clone();
        let start_semantic = start_semantic.clone();
        move |_| {
            let mut guard = semantic_worker.borrow_mut();
            if let Some(w) = guard.take() {
                w.stop();
                semantic_button.set_label("Build Semantic Index");
            } else {
                drop(guard);
                start_semantic();
            }
        }
    });

    // ── Initial load + background workers ─────────────────────────
    let loaded = Rc::new(Cell::new(false));
    page.connect_map({
        let reload_grid = reload_grid.clone();
        let db = db.clone();
        let tagger = tagger.clone();
        let start_semantic = start_semantic.clone();
        move |_| {
            if !loaded.get() {
                loaded.set(true);
                reload_grid();
                maple_db::spawn_metadata_filler(db.clone());

                if settings.ai.enabled {
                    *tagger.borrow_mut() =
                        Some(maple_db::spawn_ai_tagger(db.clone(), make_describer()));
                    update_button_label(true);
                }

                if settings.face.enabled && settings.face.models_available() {
                    start_face_detector();
                }

                if settings.semantic.enabled {
                    start_semantic();
                }
            }
        }
    });

    page
}
