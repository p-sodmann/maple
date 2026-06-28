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
    self, assign_to_name, assign_to_person, build_face_boxes, build_suggestions, delete_face,
    insert_new_face, EmbeddingMatrix,
};
use crate::image_loader;
use crate::{CollectionChip, DetailWindow};

thread_local! {
    /// The single live detail window (mirrors the old `DETAIL_CTX` singleton).
    static DETAIL: RefCell<Option<Detail>> = const { RefCell::new(None) };
}

/// The detail window and its shared state.
///
/// Not `Clone` — the generated `DetailWindow` handle isn't, and there should
/// only ever be one. The strong handle lives solely in the [`DETAIL`]
/// thread-local; callbacks capture a [`slint::Weak`] plus the shared `Rc`
/// fields, so no reference cycle forms through Slint.
struct Detail {
    window: DetailWindow,
    records: Rc<RefCell<Vec<LibraryImage>>>,
    index: Rc<Cell<usize>>,
    db: Arc<Mutex<maple_db::Database>>,
    // Face overlay state
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    known_embeddings: Rc<RefCell<EmbeddingMatrix>>,
    current_image_id: Rc<Cell<i64>>,
}

/// Open (or reuse) the detail window for `records[index]`.
pub fn open(records: Vec<LibraryImage>, index: usize, db: Arc<Mutex<maple_db::Database>>) {
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
        let len = records.len();
        *detail.records.borrow_mut() = records;
        detail.index.set(index.min(len.saturating_sub(1)));

        show_current(detail);

        if let Err(e) = detail.window.show() {
            tracing::error!("Failed to show detail window: {e}");
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

    // ── Navigation ────────────────────────────────────────────────
    window.on_prev({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let cid = current_image_id.clone();
        move || navigate(&w, &records, &index, &db, &faces, &known_embeddings, &cid, -1)
    });
    window.on_next({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        let faces = faces.clone();
        let known_embeddings = known_embeddings.clone();
        let cid = current_image_id.clone();
        move || navigate(&w, &records, &index, &db, &faces, &known_embeddings, &cid, 1)
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
            let chips = crate::collections_window::all_chips(&db);
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
                    if !face_overlay::is_real_detection(face) {
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
                let boxes = build_face_boxes(&faces.borrow(), &db, &known_embeddings.borrow());
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
                let boxes = build_face_boxes(&faces.borrow(), &db, &known_embeddings.borrow());
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
            let boxes = build_face_boxes(&faces.borrow(), &db, &known_embeddings.borrow());
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
                let boxes = build_face_boxes(&faces.borrow(), &db, &known_embeddings.borrow());
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

    Ok(Detail {
        window,
        records,
        index,
        db,
        faces,
        known_embeddings,
        current_image_id,
    })
}

/// Move `delta` steps through the record list (+1 next, -1 prev).
fn navigate(
    w: &slint::Weak<DetailWindow>,
    records: &Rc<RefCell<Vec<LibraryImage>>>,
    index: &Rc<Cell<usize>>,
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    known_embeddings: &Rc<RefCell<EmbeddingMatrix>>,
    current_image_id: &Rc<Cell<i64>>,
    delta: i32,
) {
    let Some(window) = w.upgrade() else { return };
    if window.get_loading() {
        return;
    }
    let len = records.borrow().len();
    if len == 0 {
        return;
    }
    let cur = index.get();
    let new = (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize;
    if new == cur {
        return;
    }
    index.set(new);
    show_record(&window, records, index, db, faces, known_embeddings, current_image_id);
}

fn show_current(detail: &Detail) {
    show_record(
        &detail.window,
        &detail.records,
        &detail.index,
        &detail.db,
        &detail.faces,
        &detail.known_embeddings,
        &detail.current_image_id,
    );
}

/// Update window chrome for the current record and kick off the async decode.
fn show_record(
    window: &DetailWindow,
    records: &Rc<RefCell<Vec<LibraryImage>>>,
    index: &Rc<Cell<usize>>,
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    known_embeddings: &Rc<RefCell<EmbeddingMatrix>>,
    current_image_id: &Rc<Cell<i64>>,
) {
    let rec = match records.borrow().get(index.get()) {
        Some(r) => r.clone(),
        None => return,
    };

    let filename = rec.meta.filename.clone().unwrap_or_else(|| "Image".to_owned());
    window.set_filename(filename.into());
    window.set_info_text(info_text(&rec).into());
    window.set_collection_chips(load_chips(db, rec.id));
    window.set_error_text(SharedString::new());
    window.set_loading(true);
    window.set_face_panel_open(false);
    window.invoke_reset_view();

    // Load face detections for this image.
    current_image_id.set(rec.id);
    let new_faces = db
        .lock()
        .ok()
        .and_then(|g| g.faces_for_image(rec.id).ok())
        .unwrap_or_default();
    *faces.borrow_mut() = new_faces;
    *known_embeddings.borrow_mut() = EmbeddingMatrix::build(db);
    let boxes = build_face_boxes(&faces.borrow(), db, &known_embeddings.borrow());
    window.set_face_boxes(boxes);

    image_loader::load_full_image(rec.path.clone(), window.as_weak());
}

/// Build the one-line EXIF summary (port of `info_bar.rs::fill_info_bar`).
fn info_text(image: &LibraryImage) -> String {
    let m = &image.meta;
    let mut fields: Vec<String> = Vec::new();

    match (&m.make, &m.model) {
        (Some(make), Some(model)) => fields.push(format!("{make} {model}")),
        (Some(make), None) => fields.push(make.clone()),
        _ => {}
    }
    if let Some(lens) = &m.lens {
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

    fields.join("  ·  ")
}

/// Load the current image's collection memberships as chip data.
fn load_chips(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> ModelRc<CollectionChip> {
    let collections = db
        .lock()
        .ok()
        .and_then(|d| d.collections_for_image(image_id).ok())
        .unwrap_or_default();

    let chips: Vec<CollectionChip> = collections
        .iter()
        .map(|c| CollectionChip {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: parse_hex_color(&c.color),
        })
        .collect();

    ModelRc::from(Rc::new(VecModel::from(chips)))
}

/// Parse a `#rrggbb` hex string into a Slint colour. Falls back to neutral grey.
fn parse_hex_color(hex: &str) -> slint::Color {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return slint::Color::from_rgb_u8(r, g, b);
        }
    }
    slint::Color::from_rgb_u8(0x9a, 0x9a, 0x9a)
}
