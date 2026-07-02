//! Detail / lightbox window controller (Slint).
//!
//! Slint port of `views/library/detail_window/`. Opens a second top-level
//! [`DetailWindow`] showing one full-resolution image with pointer-anchored
//! zoom + pan, prev/next navigation through the records the grid was
//! displaying, a toggleable one-line EXIF strip, a collection-chips bar, and
//! a face detection overlay with click-to-assign and draw-new-box support.
//!
//! Like the old GTK detail window — and the existing `DEBUG_WIN` pattern —
//! the window is a singleton held in a `thread_local!` so re-activating a
//! grid cell reuses the same window instead of stacking new ones.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use maple_db::{FaceDetection, LibraryImage};

use crate::face_overlay::{
    assign_to_name, assign_to_person, build_face_boxes, build_suggestions, delete_face,
    insert_new_face,
};
use crate::image_loader;
use crate::services::images as image_service;
use crate::transforms::{format_unix_ts, is_real_detection, truncate_value, EmbeddingMatrix};
use crate::{CollectionChip, DetailWindow, ImageInfoRow};

thread_local! {
    /// The single live detail window (mirrors the old `DETAIL_CTX` singleton).
    static DETAIL: RefCell<Option<Detail>> = const { RefCell::new(None) };
}

/// Record list + face overlay state shared between [`navigate`] and
/// [`show_record`]. Cloning is cheap — every field is an `Rc`/`Arc`.
#[derive(Clone)]
struct NavState {
    records: Rc<RefCell<Vec<LibraryImage>>>,
    index: Rc<Cell<usize>>,
    db: Arc<Mutex<maple_db::Database>>,
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    known_embeddings: Rc<RefCell<EmbeddingMatrix>>,
    current_image_id: Rc<Cell<i64>>,
    /// Whether the "ALL EXIF METADATA" section in the info panel is expanded.
    /// Reset to collapsed whenever a different image is shown.
    show_all_exif: Rc<Cell<bool>>,
}

/// The detail window and its shared state.
///
/// Not `Clone` — the generated `DetailWindow` handle isn't, and there should
/// only ever be one. The strong handle lives solely in the [`DETAIL`]
/// thread-local; callbacks capture a [`slint::Weak`] plus the shared `Rc`
/// fields, so no reference cycle forms through Slint.
struct Detail {
    window: DetailWindow,
    nav: NavState,
}

/// Open (or reuse) the detail window for `records[index]`, syncing dark-mode state.
pub fn open(
    records: Vec<LibraryImage>,
    index: usize,
    db: Arc<Mutex<maple_db::Database>>,
    is_dark: bool,
) {
    if DETAIL.with(|d| d.borrow().is_none()) {
        match build(db) {
            Ok(d) => DETAIL.with(|cell| *cell.borrow_mut() = Some(d)),
            Err(e) => {
                tracing::error!("Failed to build detail window: {e}");
                return;
            }
        }
    }

    DETAIL.with(|cell| {
        let guard = cell.borrow();
        let Some(detail) = guard.as_ref() else { return };
        detail.window.set_dark(is_dark);
        let len = records.len();
        *detail.nav.records.borrow_mut() = records;
        detail.nav.index.set(index.min(len.saturating_sub(1)));

        show_current(detail);

        if let Err(e) = detail.window.show() {
            tracing::error!("Failed to show detail window: {e}");
        }
    });
}

/// Propagate a theme change to the detail window while it is open.
pub fn set_dark(dark: bool) {
    DETAIL.with(|d| {
        let guard = d.borrow();
        if let Some(detail) = guard.as_ref() {
            detail.window.set_dark(dark);
        }
    });
}

/// Build a fresh detail window and wire its callbacks (once).
fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<Detail, slint::PlatformError> {
    let window = DetailWindow::new()?;
    let records: Rc<RefCell<Vec<LibraryImage>>> = Rc::new(RefCell::new(Vec::new()));
    let index = Rc::new(Cell::new(0usize));
    let faces: Rc<RefCell<Vec<FaceDetection>>> = Rc::new(RefCell::new(Vec::new()));
    let known_embeddings: Rc<RefCell<EmbeddingMatrix>> =
        Rc::new(RefCell::new(EmbeddingMatrix::empty()));
    let current_image_id: Rc<Cell<i64>> = Rc::new(Cell::new(0));
    let show_all_exif: Rc<Cell<bool>> = Rc::new(Cell::new(false));
    let nav = NavState {
        records: records.clone(),
        index: index.clone(),
        db: db.clone(),
        faces: faces.clone(),
        known_embeddings: known_embeddings.clone(),
        current_image_id: current_image_id.clone(),
        show_all_exif: show_all_exif.clone(),
    };

    // ── Navigation ────────────────────────────────────────────────
    window.on_prev({
        let w = window.as_weak();
        let nav = nav.clone();
        move || navigate(&w, &nav, -1)
    });
    window.on_next({
        let w = window.as_weak();
        let nav = nav.clone();
        move || navigate(&w, &nav, 1)
    });

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_toggle_fullscreen({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let fs = !w.window().is_fullscreen();
                w.window().set_fullscreen(fs);
                w.set_is_fullscreen(fs);
            }
        }
    });

    window.on_open_external({
        let records = records.clone();
        let index = index.clone();
        move || {
            let recs = records.borrow();
            if let Some(rec) = recs.get(index.get()) {
                if let Err(e) = open::that_detached(&rec.path) {
                    tracing::warn!("Failed to open {} externally: {e}", rec.path.display());
                }
            }
        }
    });

    window.on_remove_collection({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        move |coll_id| {
            let image_id = match records.borrow().get(index.get()) {
                Some(rec) => rec.id,
                None => return,
            };
            if let Ok(d) = db.lock() {
                if let Err(e) = d.remove_image_from_collection(coll_id as i64, image_id) {
                    tracing::warn!(
                        "Failed to remove image {image_id} from collection {coll_id}: {e}"
                    );
                }
            }
            if let Some(w) = w.upgrade() {
                w.set_collection_chips(load_chips(&db, image_id));
            }
        }
    });

    // Load all collections and open the inline picker panel.
    window.on_add_to_collection({
        let w = window.as_weak();
        let db = db.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let chips = crate::services::collections::load_all_collections(&db);
            w.set_available_collections(ModelRc::from(Rc::new(VecModel::from(chips))));
            w.set_add_coll_panel_open(true);
        }
    });

    window.on_add_to_collection_confirm({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        move |coll_id| {
            let image_id = match records.borrow().get(index.get()) {
                Some(rec) => rec.id,
                None => return,
            };
            if let Ok(guard) = db.lock() {
                if let Err(e) = guard.add_image_to_collection(coll_id as i64, image_id) {
                    tracing::warn!("add_image_to_collection {image_id} → {coll_id}: {e}");
                    return;
                }
            }
            if let Some(w) = w.upgrade() {
                w.set_collection_chips(load_chips(&db, image_id));
            }
        }
    });

    window.on_toggle_all_exif({
        let w = window.as_weak();
        let nav = nav.clone();
        move || {
            nav.show_all_exif.set(!nav.show_all_exif.get());
            if let Some(window) = w.upgrade() {
                refresh_info_rows(&window, &nav);
            }
        }
    });

    // ── Face overlay ──────────────────────────────────────────────

    window.on_toggle_faces({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let now = !w.get_show_faces();
                w.set_show_faces(now);
            }
        }
    });

    window.on_toggle_face_draw({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let now = !w.get_face_draw_mode();
                w.set_face_draw_mode(now);
                if !now {
                    // Exiting draw mode clears any in-progress drag visuals.
                    w.set_draw_dragging(false);
                }
            }
        }
    });

    // Click hit-testing done in Rust using the geo-* properties exposed from
    // the Slint viewport geometry (zoom-area delegates all clicks here).
    window.on_face_area_clicked({
        let w = window.as_weak();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        move |vp_x, vp_y| {
            let Some(w) = w.upgrade() else { return };
            if !w.get_show_faces() || w.get_face_draw_mode() {
                return;
            }
            let img_left = w.get_geo_img_left();
            let img_top = w.get_geo_img_top();
            let disp_w = w.get_geo_disp_w();
            let disp_h = w.get_geo_disp_h();
            if disp_w <= 0.0 || disp_h <= 0.0 {
                return;
            }
            let face_id = {
                let f = faces.borrow();
                f.iter().find_map(|face| {
                    if !is_real_detection(face) {
                        return None;
                    }
                    let [x1, y1, x2, y2] = face.bbox;
                    let bx = img_left + x1 * disp_w;
                    let by = img_top + y1 * disp_h;
                    let bw = (x2 - x1) * disp_w;
                    let bh = (y2 - y1) * disp_h;
                    if vp_x >= bx && vp_x <= bx + bw && vp_y >= by && vp_y <= by + bh {
                        Some(face.id)
                    } else {
                        None
                    }
                })
            };
            let Some(fid) = face_id else { return };
            let embedding = {
                let f = faces.borrow();
                f.iter()
                    .find(|f| f.id == fid)
                    .map(|f| f.embedding.clone())
                    .unwrap_or_default()
            };
            let sugs = build_suggestions(&embedding, &known_embeddings.borrow());
            w.set_face_suggestions(sugs);
            w.set_face_panel_id(fid as i32);
            w.set_face_name_entry(SharedString::new());
            w.set_face_panel_open(true);
        }
    });

    window.on_face_assign_person({
        let w = window.as_weak();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let db = db.clone();
        move |face_id, person_id| {
            let Some(w) = w.upgrade() else { return };
            if assign_to_person(face_id as i64, person_id as i64, &mut faces.borrow_mut(), &db) {
                let boxes = build_face_boxes(&faces.borrow(), &known_embeddings.borrow());
                w.set_face_boxes(boxes);
                w.set_face_panel_open(false);
            }
        }
    });

    window.on_face_assign_name({
        let w = window.as_weak();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let db = db.clone();
        move |face_id, name| {
            let name = name.trim().to_owned();
            if name.is_empty() {
                return;
            }
            let Some(w) = w.upgrade() else { return };
            if assign_to_name(
                face_id as i64,
                &name,
                &mut faces.borrow_mut(),
                &mut known_embeddings.borrow_mut(),
                &db,
            )
            .is_some()
            {
                let boxes = build_face_boxes(&faces.borrow(), &known_embeddings.borrow());
                w.set_face_boxes(boxes);
                w.set_face_panel_open(false);
                w.set_face_name_entry(SharedString::new());
            }
        }
    });

    window.on_face_delete({
        let w = window.as_weak();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let db = db.clone();
        move |face_id| {
            let Some(w) = w.upgrade() else { return };
            delete_face(face_id as i64, &mut faces.borrow_mut(), &db);
            let boxes = build_face_boxes(&faces.borrow(), &known_embeddings.borrow());
            w.set_face_boxes(boxes);
            w.set_face_panel_open(false);
        }
    });

    // face-draw-done passes raw viewport coords; normalise here using the
    // geo-* properties (img-left, disp-w, etc.) exposed from the Slint model.
    window.on_face_draw_done({
        let w = window.as_weak();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let current_image_id = current_image_id.clone();
        let db = db.clone();
        move |vx0, vy0, vx1, vy1| {
            let Some(w) = w.upgrade() else { return };
            let iid = current_image_id.get();
            if iid == 0 {
                return;
            }
            // Convert from viewport coords to normalised image coords [0,1].
            let img_left = w.get_geo_img_left();
            let img_top = w.get_geo_img_top();
            let disp_w = w.get_geo_disp_w();
            let disp_h = w.get_geo_disp_h();
            if disp_w <= 0.0 || disp_h <= 0.0 {
                return;
            }
            let nx0 = ((vx0 - img_left) / disp_w).clamp(0.0, 1.0);
            let ny0 = ((vy0 - img_top) / disp_h).clamp(0.0, 1.0);
            let nx1 = ((vx1 - img_left) / disp_w).clamp(0.0, 1.0);
            let ny1 = ((vy1 - img_top) / disp_h).clamp(0.0, 1.0);
            // Sort corners so bbox is always [min_x, min_y, max_x, max_y].
            let (bx1, bx2) = if nx0 <= nx1 { (nx0, nx1) } else { (nx1, nx0) };
            let (by1, by2) = if ny0 <= ny1 { (ny0, ny1) } else { (ny1, ny0) };
            // Ignore tiny boxes (accidental clicks, < 0.5% of image side).
            if (bx2 - bx1) < 0.005 || (by2 - by1) < 0.005 {
                return;
            }
            let bbox = [bx1, by1, bx2, by2];
            if let Some(face_id) = insert_new_face(iid, bbox, &mut faces.borrow_mut(), &db) {
                let boxes = build_face_boxes(&faces.borrow(), &known_embeddings.borrow());
                w.set_face_boxes(boxes);
                // Immediately open the assignment panel for the new box.
                let sugs = build_suggestions(&[], &known_embeddings.borrow());
                w.set_face_suggestions(sugs);
                w.set_face_panel_id(face_id as i32);
                w.set_face_name_entry(SharedString::new());
                w.set_face_panel_open(true);
            }
        }
    });

    window.on_face_panel_close({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                w.set_face_panel_open(false);
                w.set_face_name_entry(SharedString::new());
            }
        }
    });

    Ok(Detail { window, nav })
}

/// Move `delta` steps through the record list (+1 next, -1 prev).
fn navigate(w: &slint::Weak<DetailWindow>, nav: &NavState, delta: i32) {
    let Some(window) = w.upgrade() else { return };
    if window.get_loading() {
        return;
    }
    let len = nav.records.borrow().len();
    if len == 0 {
        return;
    }
    let cur = nav.index.get();
    let new = (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize;
    if new == cur {
        return;
    }
    nav.index.set(new);
    show_record(&window, nav);
}

fn show_current(detail: &Detail) {
    show_record(&detail.window, &detail.nav);
}

/// Build the flat list of key-value rows for the info popup.
///
/// `show_all_exif` controls whether the comprehensive "ALL EXIF METADATA"
/// section (every captured tag beyond the curated rows above) is expanded.
fn build_info_rows(
    rec: &LibraryImage,
    db: &Arc<Mutex<maple_db::Database>>,
    face_count: usize,
    show_all_exif: bool,
) -> ModelRc<ImageInfoRow> {
    let mut rows: Vec<ImageInfoRow> = Vec::new();

    let section = |label: &str| ImageInfoRow {
        label: label.into(),
        value: SharedString::new(),
        is_section: true,
        is_toggle: false,
    };
    let row = |label: &str, value: String| ImageInfoRow {
        label: label.into(),
        value: truncate_value(&value).into(),
        is_section: false,
        is_toggle: false,
    };

    // ── File ────────────────────────────────────────────────────────
    rows.push(section("FILE"));
    if let Some(ref name) = rec.meta.filename {
        rows.push(row("Filename", name.clone()));
    }
    rows.push(row("Path", rec.path.to_string_lossy().into_owned()));
    if let Some(ref raw) = rec.raw_path {
        rows.push(row("RAW companion", raw.to_string_lossy().into_owned()));
    }
    rows.push(row(
        "Status",
        match rec.status {
            maple_db::ImageStatus::Present => "Present".into(),
            maple_db::ImageStatus::Missing => "Missing".into(),
        },
    ));
    rows.push(row("Added", format_unix_ts(rec.added_at)));
    if let Some(hash) = rec.hash {
        let hex: String = hash.iter().map(|b| format!("{b:02x}")).collect::<String>()[..16].to_owned();
        rows.push(row("Hash (BLAKE3)", format!("{hex}…")));
    }
    if let Some(stack_id) = rec.stack_id {
        let size = rec.stack_size.unwrap_or(0);
        rows.push(row("Stack", format!("#{stack_id}  ({size} images)")));
    }

    // ── Camera ──────────────────────────────────────────────────────
    rows.push(section("CAMERA"));
    match (&rec.meta.make, &rec.meta.model) {
        (Some(make), Some(model)) => rows.push(row("Camera", format!("{make} {model}"))),
        (Some(make), None) => rows.push(row("Camera", make.clone())),
        (None, Some(model)) => rows.push(row("Camera", model.clone())),
        _ => {}
    }
    if let Some(ref lens) = rec.meta.lens {
        rows.push(row("Lens", lens.clone()));
    }
    if let Some(fl) = rec.meta.focal_length {
        rows.push(row("Focal length", format!("{fl:.0} mm")));
    }
    if let Some(ap) = rec.meta.aperture {
        rows.push(row("Aperture", format!("f/{ap:.1}")));
    }
    if let Some(iso) = rec.meta.iso {
        rows.push(row("ISO", format!("{iso}")));
    }

    // ── Image ───────────────────────────────────────────────────────
    rows.push(section("IMAGE"));
    if let (Some(w), Some(h)) = (rec.meta.width, rec.meta.height) {
        rows.push(row("Dimensions", format!("{w} × {h} px")));
    }
    if let Some(ts) = rec.meta.taken_at {
        rows.push(row("Taken", format_unix_ts(ts)));
    }
    if let Some(ori) = rec.meta.orientation {
        let label = match ori {
            1 => "1 (Normal)".into(),
            3 => "3 (180°)".into(),
            6 => "6 (90° CW)".into(),
            8 => "8 (90° CCW)".into(),
            n => format!("{n}"),
        };
        rows.push(row("Orientation", label));
    }

    // ── AI descriptions ─────────────────────────────────────────────
    let info = image_service::load_image_info_data(db, rec.id);
    if !info.ai_descriptions.is_empty() {
        rows.push(section("AI DESCRIPTION"));
        for (model_id, desc) in info.ai_descriptions {
            rows.push(row(&model_id, desc));
        }
    }

    // ── Faces ───────────────────────────────────────────────────────
    if face_count > 0 {
        rows.push(section("FACES"));
        rows.push(row(
            "Detected",
            format!("{face_count} face{}", if face_count == 1 { "" } else { "s" }),
        ));
    }

    // ── All EXIF (comprehensive, collapsible) ─────────────────────────
    let exif_tags = info.exif_tags;
    if !exif_tags.is_empty() {
        rows.push(ImageInfoRow {
            label: format!(
                "ALL EXIF METADATA ({}) — {}",
                exif_tags.len(),
                if show_all_exif { "HIDE" } else { "SHOW" }
            )
            .into(),
            value: SharedString::new(),
            is_section: true,
            is_toggle: true,
        });
        if show_all_exif {
            for (tag, value) in &exif_tags {
                rows.push(row(tag, value.clone()));
            }
        }
    }

    ModelRc::from(Rc::new(VecModel::from(rows)))
}

/// Update window chrome for the current record and kick off the async decode.
fn show_record(window: &DetailWindow, nav: &NavState) {
    let rec = match nav.records.borrow().get(nav.index.get()) {
        Some(r) => r.clone(),
        None => return,
    };
    let db = &nav.db;

    let filename = rec.meta.filename.clone().unwrap_or_else(|| "Image".to_owned());
    window.set_filename(filename.into());
    window.set_error_text(SharedString::new());
    window.set_loading(true);
    window.set_face_panel_open(false);
    window.invoke_reset_view();

    // Load face detections, embeddings, and collection chips for this image.
    nav.current_image_id.set(rec.id);
    let detail = image_service::load_image_detail(db, rec.id);
    window.set_collection_chips(chips_model(detail.collection_chips));
    *nav.faces.borrow_mut() = detail.faces;
    *nav.known_embeddings.borrow_mut() = detail.embeddings;
    let boxes = build_face_boxes(&nav.faces.borrow(), &nav.known_embeddings.borrow());
    window.set_face_boxes(boxes);

    // Populate the info popup rows (works whether the panel is open or closed).
    // The "all EXIF" section collapses again whenever a different image loads.
    nav.show_all_exif.set(false);
    let face_count = nav.faces.borrow().len();
    window.set_image_info_rows(build_info_rows(&rec, db, face_count, false));

    image_loader::load_full_image(rec.path.clone(), window.as_weak());
}

/// Rebuild just the info-panel rows for the currently shown record, e.g.
/// after toggling the "all EXIF" section — no image reload needed.
fn refresh_info_rows(window: &DetailWindow, nav: &NavState) {
    let Some(rec) = nav.records.borrow().get(nav.index.get()).cloned() else {
        return;
    };
    let face_count = nav.faces.borrow().len();
    window.set_image_info_rows(build_info_rows(&rec, &nav.db, face_count, nav.show_all_exif.get()));
}

/// Load the current image's collection memberships as chip data.
fn load_chips(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> ModelRc<CollectionChip> {
    chips_model(image_service::load_collection_chips(db, image_id))
}

fn chips_model(chips: Vec<CollectionChip>) -> ModelRc<CollectionChip> {
    ModelRc::from(Rc::new(VecModel::from(chips)))
}
