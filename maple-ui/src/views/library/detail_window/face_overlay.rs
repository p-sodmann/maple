//! Face detection overlay for the detail window.
//!
//! # Layout
//!
//! A [`gtk4::Overlay`] wraps the existing scrolled image area.  On top of it
//! sits a transparent [`gtk4::DrawingArea`] (`can-target = false`) that draws
//! coloured bounding boxes.  A [`gtk4::GestureClick`] on the overlay itself
//! detects clicks in those boxes and opens a person-assignment popover.
//!
//! # Coordinate mapping
//!
//! At zoom = 1 (ContentFit::Contain) the image is letterboxed inside the
//! scrolled viewport.  At zoom > 1 (ContentFit::Fill, explicit size_request)
//! the image fills a larger virtual canvas that is scrolled.  Both cases are
//! handled in [`face_screen_rect`].

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::{best_person_match, FaceDetection};

// ── Public state ──────────────────────────────────────────────────

/// Shared state for the face overlay.  Cheap to clone (`Rc`-backed).
#[derive(Clone)]
pub struct FaceOverlay {
    /// The container to place in the layout (replaces the raw scrolled window).
    pub container: gtk4::Overlay,
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    visible: Rc<Cell<bool>>,
    drawing_area: gtk4::DrawingArea,
}

impl FaceOverlay {
    /// Create an overlay that wraps `scrolled`.
    ///
    /// `zoom` and `img_dims` are shared with the zoom/pan system so that
    /// box positions are correct at any zoom level.
    pub fn new(
        scrolled: &gtk4::ScrolledWindow,
        zoom: Rc<Cell<f64>>,
        img_dims: Rc<Cell<(i32, i32)>>,
        db: Arc<Mutex<maple_db::Database>>,
        similarity_threshold: f32,
    ) -> Self {
        let faces: Rc<RefCell<Vec<FaceDetection>>> = Rc::new(RefCell::new(vec![]));
        let visible: Rc<Cell<bool>> = Rc::new(Cell::new(false));

        // ── Drawing area ──────────────────────────────────────────
        let drawing_area = gtk4::DrawingArea::builder()
            .hexpand(true)
            .vexpand(true)
            .can_target(false) // pass all events through
            .build();

        drawing_area.set_draw_func({
            let faces = faces.clone();
            let visible = visible.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            move |_da, cx, _w, _h| {
                if !visible.get() {
                    return;
                }
                let f = faces.borrow();
                if f.is_empty() {
                    return;
                }
                let (vw, vh) = (scrolled.width() as f64, scrolled.height() as f64);
                let (img_w, img_h) = img_dims.get();
                if img_w == 0 || img_h == 0 {
                    return;
                }
                let z = zoom.get();
                let scroll_x = scrolled.hadjustment().value();
                let scroll_y = scrolled.vadjustment().value();

                for face in f.iter() {
                    // Skip sentinel rows (no-face detected).
                    if face.confidence < 0.0 || face.bbox == [0.0, 0.0, 0.0, 0.0] {
                        continue;
                    }
                    let Some((sx, sy, sw, sh)) = face_screen_rect(
                        face.bbox,
                        img_w,
                        img_h,
                        vw,
                        vh,
                        z,
                        scroll_x,
                        scroll_y,
                    ) else {
                        continue;
                    };

                    // Blue = no person assigned; green = person assigned.
                    if face.person_id.is_some() {
                        cx.set_source_rgba(0.2, 0.85, 0.4, 0.9);
                    } else {
                        cx.set_source_rgba(0.2, 0.55, 1.0, 0.9);
                    }
                    cx.set_line_width(2.5);
                    cx.rectangle(sx, sy, sw, sh);
                    let _ = cx.stroke();

                    // Semi-transparent fill for hit-area visibility.
                    if face.person_id.is_some() {
                        cx.set_source_rgba(0.2, 0.85, 0.4, 0.08);
                    } else {
                        cx.set_source_rgba(0.2, 0.55, 1.0, 0.08);
                    }
                    cx.rectangle(sx, sy, sw, sh);
                    let _ = cx.fill();
                }
            }
        });

        // ── Overlay container ─────────────────────────────────────
        let container = gtk4::Overlay::new();
        container.set_child(Some(scrolled));
        container.add_overlay(&drawing_area);

        // ── Click gesture on the overlay (events pass through drawing area) ─
        let gesture = gtk4::GestureClick::new();
        gesture.connect_pressed({
            let faces = faces.clone();
            let visible = visible.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            let db = db.clone();
            let drawing_area = drawing_area.clone();
            move |gesture, n_press, x, y| {
                if n_press != 1 || !visible.get() {
                    return;
                }
                let (vw, vh) = (scrolled.width() as f64, scrolled.height() as f64);
                let (img_w, img_h) = img_dims.get();
                let z = zoom.get();
                let scroll_x = scrolled.hadjustment().value();
                let scroll_y = scrolled.vadjustment().value();

                let hit = faces.borrow().iter().enumerate().find_map(|(i, face)| {
                    if face.confidence < 0.0 || face.bbox == [0.0, 0.0, 0.0, 0.0] {
                        return None;
                    }
                    let (sx, sy, sw, sh) = face_screen_rect(
                        face.bbox, img_w, img_h, vw, vh, z, scroll_x, scroll_y,
                    )?;
                    if x >= sx && x <= sx + sw && y >= sy && y <= sy + sh {
                        Some((i, face.id, face.person_id, face.embedding.clone()))
                    } else {
                        None
                    }
                });

                if let Some((idx, face_id, current_person, embedding)) = hit {
                    // Prevent the click from also panning.
                    gesture.set_state(gtk4::EventSequenceState::Claimed);
                    let widget = gesture
                        .widget()
                        .and_downcast::<gtk4::Overlay>()
                        .expect("gesture widget is overlay");
                    open_person_dialog(
                        &widget,
                        face_id,
                        idx,
                        current_person,
                        &embedding,
                        similarity_threshold,
                        db.clone(),
                        faces.clone(),
                        drawing_area.clone(),
                    );
                }
            }
        });
        container.add_controller(gesture);

        Self { container, faces, visible, drawing_area }
    }

    /// Load (or reload) face detections for `image_id` and repaint.
    pub fn load_for_image(&self, image_id: i64, db: &Arc<Mutex<maple_db::Database>>) {
        let new_faces = db
            .lock()
            .unwrap()
            .faces_for_image(image_id)
            .unwrap_or_default();
        *self.faces.borrow_mut() = new_faces;
        self.drawing_area.queue_draw();
    }

    /// Show or hide the overlay (does not affect the underlying image).
    pub fn set_visible(&self, v: bool) {
        self.visible.set(v);
        self.drawing_area.queue_draw();
    }
}

// ── Person assignment dialog ──────────────────────────────────────

fn open_person_dialog(
    parent: &gtk4::Overlay,
    face_id: i64,
    face_idx: usize,
    current_person: Option<i64>,
    embedding: &[f32],
    similarity_threshold: f32,
    db: Arc<Mutex<maple_db::Database>>,
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    drawing_area: gtk4::DrawingArea,
) {
    // Build suggestion from cosine similarity.
    let suggestion: Option<(i64, String, f32)> = {
        let guard = db.lock().unwrap();
        guard
            .all_assigned_face_embeddings()
            .ok()
            .and_then(|known| best_person_match(embedding, &known, similarity_threshold))
    };

    // Current person name (if any).
    let current_name: Option<String> = current_person.and_then(|pid| {
        db.lock().unwrap().person_name(pid).ok().flatten()
    });

    // ── Dialog layout ─────────────────────────────────────────────
    let window = parent
        .root()
        .and_downcast::<gtk4::Window>()
        .expect("overlay has window root");

    let dialog = adw::Window::builder()
        .title("Assign Person")
        .default_width(320)
        .transient_for(&window)
        .modal(true)
        .build();

    let vbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_top(16)
        .margin_bottom(16)
        .margin_start(16)
        .margin_end(16)
        .build();

    // Suggestion banner.
    if let Some((_, ref name, sim)) = suggestion {
        let pct = (sim * 100.0) as u32;
        let hint = gtk4::Label::new(Some(&format!(
            "Suggested: {name}  ({pct}% match)"
        )));
        hint.add_css_class("caption");
        hint.add_css_class("dim-label");
        hint.set_halign(gtk4::Align::Start);
        vbox.append(&hint);
    }

    // Name entry, pre-filled with suggestion or current assignment.
    let entry = gtk4::Entry::builder()
        .placeholder_text("Person name")
        .hexpand(true)
        .build();
    if let Some(ref name) = current_name {
        entry.set_text(name);
    } else if let Some((_, ref name, _)) = suggestion {
        entry.set_text(name);
    }
    vbox.append(&entry);

    // Buttons.
    let btn_row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .halign(gtk4::Align::End)
        .build();

    let cancel_btn = gtk4::Button::with_label("Cancel");
    cancel_btn.add_css_class("flat");
    let assign_btn = gtk4::Button::with_label("Assign");
    assign_btn.add_css_class("suggested-action");

    btn_row.append(&cancel_btn);
    btn_row.append(&assign_btn);
    vbox.append(&btn_row);

    let header = adw::HeaderBar::new();
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&vbox));
    dialog.set_content(Some(&toolbar));

    cancel_btn.connect_clicked({
        let dialog = dialog.clone();
        move |_| dialog.close()
    });

    assign_btn.connect_clicked({
        let dialog = dialog.clone();
        let entry = entry.clone();
        let db = db.clone();
        let faces = faces.clone();
        let drawing_area = drawing_area.clone();
        move |_| {
            let name = entry.text().trim().to_owned();
            if name.is_empty() {
                return;
            }
            let guard = db.lock().unwrap();
            match guard.upsert_person(&name) {
                Ok(person_id) => {
                    let _ = guard.assign_face_to_person(face_id, Some(person_id));
                    drop(guard);
                    // Update in-memory face record.
                    if let Some(face) = faces.borrow_mut().get_mut(face_idx) {
                        face.person_id = Some(person_id);
                    }
                    drawing_area.queue_draw();
                    dialog.close();
                }
                Err(e) => {
                    tracing::warn!("failed to upsert person '{name}': {e}");
                }
            }
        }
    });

    dialog.present();
}

// ── Coordinate helpers ────────────────────────────────────────────

/// Map a normalised face bbox to screen coordinates inside the overlay widget.
///
/// Returns `None` when image dimensions are unknown.
///
/// At zoom ≤ 1.0 the image is in ContentFit::Contain (letterboxed).
/// At zoom > 1.0 the image is ContentFit::Fill with a size_request and
/// the scrolled window's adjustments carry the scroll offset.
fn face_screen_rect(
    [x1, y1, x2, y2]: [f32; 4],
    img_w: i32,
    img_h: i32,
    vw: f64,
    vh: f64,
    zoom: f64,
    scroll_x: f64,
    scroll_y: f64,
) -> Option<(f64, f64, f64, f64)> {
    if img_w == 0 || img_h == 0 || vw == 0.0 || vh == 0.0 {
        return None;
    }
    let fit = f64::min(vw / img_w as f64, vh / img_h as f64);

    let (sx, sy, sw, sh) = if zoom <= 1.0 {
        // Letterboxed: image centred inside the viewport.
        let draw_w = img_w as f64 * fit;
        let draw_h = img_h as f64 * fit;
        let off_x = (vw - draw_w) / 2.0;
        let off_y = (vh - draw_h) / 2.0;
        (
            off_x + x1 as f64 * draw_w,
            off_y + y1 as f64 * draw_h,
            (x2 - x1) as f64 * draw_w,
            (y2 - y1) as f64 * draw_h,
        )
    } else {
        // Zoomed: pixel-per-image-pixel = fit * zoom; subtract scroll offset.
        let ppx = fit * zoom;
        (
            x1 as f64 * img_w as f64 * ppx - scroll_x,
            y1 as f64 * img_h as f64 * ppx - scroll_y,
            (x2 - x1) as f64 * img_w as f64 * ppx,
            (y2 - y1) as f64 * img_h as f64 * ppx,
        )
    };

    Some((sx, sy, sw, sh))
}
