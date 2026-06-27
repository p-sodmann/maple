//! Face detection overlay for the detail window.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use adw::prelude::*;
use gtk4::prelude::*;
use libadwaita as adw;

use maple_db::FaceDetection;

use super::super::face_shared::{
    assign_face_to_name, assign_face_to_person, face_screen_rect, is_real_detection,
    person_name, EmbeddingMatrix,
};

/// Look up the display label for a face detection.
///
/// - Assigned face → person name (from DB).
/// - Unassigned face with embedding → best match name if above threshold, else "?".
/// - Unassigned face without embedding → "?".
///
/// Returns `(label, is_assigned, is_suggestion)`.
fn face_label(
    face: &FaceDetection,
    db: &Arc<Mutex<maple_db::Database>>,
    known: &EmbeddingMatrix,
    threshold: f32,
) -> (String, bool, bool) {
    if let Some(pid) = face.person_id {
        let name = db
            .lock()
            .ok()
            .and_then(|g| g.person_name(pid).ok().flatten())
            .unwrap_or_else(|| "?".into());
        return (name, true, false);
    }
    if face.embedding.is_empty() {
        return ("?".into(), false, false);
    }
    let matches = known.top_k(&face.embedding, 1);
    if let Some((_pid, name, sim)) = matches.first() {
        if sim.is_finite() && *sim >= threshold {
            return (name.clone(), false, true);
        }
    }
    ("?".into(), false, false)
}

/// Convert a screen-space point to normalised image coordinates [0, 1].
///
/// Inverse of `face_screen_rect`.  Returns `None` when image dimensions are
/// unknown or when the point lies outside the drawable canvas.
fn screen_to_norm(
    sx: f64,
    sy: f64,
    img_w: i32,
    img_h: i32,
    vw: f64,
    vh: f64,
    zoom: f64,
    scroll_x: f64,
    scroll_y: f64,
) -> Option<(f32, f32)> {
    if img_w == 0 || img_h == 0 || vw == 0.0 || vh == 0.0 {
        return None;
    }
    let fit = f64::min(vw / img_w as f64, vh / img_h as f64);
    let ppx = fit * zoom;
    let draw_w = img_w as f64 * ppx;
    let draw_h = img_h as f64 * ppx;
    let off_x = ((vw - draw_w) / 2.0).max(0.0);
    let off_y = ((vh - draw_h) / 2.0).max(0.0);
    let nx = ((sx + scroll_x - off_x) / draw_w) as f32;
    let ny = ((sy + scroll_y - off_y) / draw_h) as f32;
    Some((nx.clamp(0.0, 1.0), ny.clamp(0.0, 1.0)))
}

#[derive(Clone)]
pub struct FaceOverlay {
    /// Image overlay used to draw bounding boxes.
    pub container: gtk4::Overlay,
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    visible: Rc<Cell<bool>>,
    drawing_area: gtk4::DrawingArea,
    /// Embedding matrix rebuilt on each image load; shared with the click gesture.
    known_embeddings: Rc<RefCell<EmbeddingMatrix>>,
    /// When true the user is drawing a new bounding box (crosshair cursor).
    draw_mode: Rc<Cell<bool>>,
    /// The image currently displayed — needed to insert manual face detections.
    image_id: Rc<Cell<i64>>,
    /// Screen-space rectangle being drawn (x1,y1,x2,y2) while dragging.
    drag_rect: Rc<Cell<Option<(f64, f64, f64, f64)>>>,
}

impl FaceOverlay {
    pub fn new(
        scrolled: &gtk4::ScrolledWindow,
        picture: &gtk4::Picture,
        zoom: Rc<Cell<f64>>,
        img_dims: Rc<Cell<(i32, i32)>>,
        db: Arc<Mutex<maple_db::Database>>,
    ) -> Self {
        let settings = maple_state::Settings::load();
        let threshold = settings.face.similarity_threshold;

        let faces: Rc<RefCell<Vec<FaceDetection>>> = Rc::new(RefCell::new(vec![]));
        let visible: Rc<Cell<bool>> = Rc::new(Cell::new(false));
        let known_embeddings: Rc<RefCell<EmbeddingMatrix>> =
            Rc::new(RefCell::new(EmbeddingMatrix::empty()));
        let draw_mode: Rc<Cell<bool>> = Rc::new(Cell::new(false));
        let image_id: Rc<Cell<i64>> = Rc::new(Cell::new(0));
        let drag_rect: Rc<Cell<Option<(f64, f64, f64, f64)>>> = Rc::new(Cell::new(None));
        let replace_face: Rc<Cell<Option<(i64, usize)>>> = Rc::new(Cell::new(None));

        let drawing_area = gtk4::DrawingArea::builder()
            .hexpand(true)
            .vexpand(true)
            .can_target(false)
            .build();

        drawing_area.set_draw_func({
            let faces = faces.clone();
            let visible = visible.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            let known_embeddings = known_embeddings.clone();
            let db = db.clone();
            let draw_mode = draw_mode.clone();
            let drag_rect = drag_rect.clone();
            move |_da, cx, _w, _h| {
                if !visible.get() {
                    return;
                }

                let f = faces.borrow();
                let (vw, vh) = (scrolled.width() as f64, scrolled.height() as f64);
                let (img_w, img_h) = img_dims.get();
                if img_w == 0 || img_h == 0 {
                    return;
                }
                let z = zoom.get();
                let scroll_x = scrolled.hadjustment().value();
                let scroll_y = scrolled.vadjustment().value();

                // Draw existing face bounding boxes.
                if !f.is_empty() {
                    let known = known_embeddings.borrow();
                    for face in f.iter() {
                        if !is_real_detection(face) {
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

                        let (label, is_assigned, is_suggestion) =
                            face_label(face, &db, &known, threshold);

                        let (r, g, b) = if is_assigned {
                            (0.2, 0.85, 0.4)
                        } else if is_suggestion {
                            (1.0, 0.65, 0.0)
                        } else {
                            (0.2, 0.55, 1.0)
                        };

                        cx.set_source_rgba(r, g, b, 0.9);
                        cx.set_line_width(2.5);
                        cx.rectangle(sx, sy, sw, sh);
                        let _ = cx.stroke();

                        cx.set_source_rgba(r, g, b, 0.08);
                        cx.rectangle(sx, sy, sw, sh);
                        let _ = cx.fill();

                        let font_size = (sh * 0.18).clamp(10.0, 24.0);
                        cx.set_font_size(font_size);
                        let text_y = sy + sh + font_size + 2.0;

                        cx.set_source_rgba(0.0, 0.0, 0.0, 0.6);
                        let _ = cx.move_to(sx + 1.0, text_y + 1.0);
                        let _ = cx.show_text(&label);

                        cx.set_source_rgba(r, g, b, 1.0);
                        let _ = cx.move_to(sx, text_y);
                        let _ = cx.show_text(&label);
                    }
                }

                // Draw the live drag rectangle while the user is drawing a new box.
                if draw_mode.get() {
                    if let Some((x1, y1, x2, y2)) = drag_rect.get() {
                        let (rx, ry) = (x1.min(x2), y1.min(y2));
                        let (rw, rh) = ((x2 - x1).abs(), (y2 - y1).abs());
                        if rw > 1.0 && rh > 1.0 {
                            cx.set_source_rgba(0.0, 0.0, 0.0, 0.5);
                            cx.set_line_width(2.5);
                            cx.set_dash(&[6.0, 3.0], 0.0);
                            cx.rectangle(rx, ry, rw, rh);
                            let _ = cx.stroke();
                            cx.set_source_rgba(1.0, 1.0, 1.0, 0.9);
                            cx.set_dash(&[6.0, 3.0], 4.5);
                            cx.rectangle(rx, ry, rw, rh);
                            let _ = cx.stroke();
                            cx.set_dash(&[], 0.0);
                        }
                    }
                }
            }
        });

        for adj in [scrolled.hadjustment(), scrolled.vadjustment()] {
            adj.connect_value_changed({
                let drawing_area = drawing_area.clone();
                move |_| drawing_area.queue_draw()
            });
            adj.connect_changed({
                let drawing_area = drawing_area.clone();
                move |_| drawing_area.queue_draw()
            });
        }

        picture.connect_notify_local(Some("paintable"), {
            let drawing_area = drawing_area.clone();
            move |_, _| drawing_area.queue_draw()
        });

        let container = gtk4::Overlay::new();
        container.set_child(Some(scrolled));
        container.add_overlay(&drawing_area);

        // ── Click gesture: assign an existing face ────────────────
        let click_gesture = gtk4::GestureClick::new();
        click_gesture.connect_pressed({
            let faces = faces.clone();
            let visible = visible.clone();
            let draw_mode = draw_mode.clone();
            let replace_face = replace_face.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            let db = db.clone();
            let drawing_area = drawing_area.clone();
            let known_embeddings = known_embeddings.clone();
            move |gesture, n_press, x, y| {
                if n_press != 1 || !visible.get() || draw_mode.get() {
                    return;
                }
                let (vw, vh) = (scrolled.width() as f64, scrolled.height() as f64);
                let (img_w, img_h) = img_dims.get();
                let z = zoom.get();
                let scroll_x = scrolled.hadjustment().value();
                let scroll_y = scrolled.vadjustment().value();

                let hit = faces.borrow().iter().enumerate().find_map(|(i, face)| {
                    if !is_real_detection(face) {
                        return None;
                    }
                    let (sx, sy, sw, sh) =
                        face_screen_rect(face.bbox, img_w, img_h, vw, vh, z, scroll_x, scroll_y)?;
                    if x >= sx && x <= sx + sw && y >= sy && y <= sy + sh {
                        Some((i, face.id, face.person_id, face.embedding.clone()))
                    } else {
                        None
                    }
                });

                if let Some((idx, face_id, current_person, embedding)) = hit {
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
                        &db,
                        &faces,
                        &drawing_area,
                        &known_embeddings.borrow(),
                        &draw_mode,
                        &replace_face,
                    );
                }
            }
        });
        container.add_controller(click_gesture);

        // ── Draw gesture: create a new face by dragging ───────────
        // Capture phase so it intercepts before the pan GestureDrag on the
        // child ScrolledWindow.  The gesture denies the sequence when draw
        // mode is inactive so panning works normally.
        let drag_start: Rc<Cell<(f64, f64)>> = Rc::new(Cell::new((0.0, 0.0)));
        let draw_drag = gtk4::GestureDrag::new();
        draw_drag.set_propagation_phase(gtk4::PropagationPhase::Capture);

        draw_drag.connect_drag_begin({
            let draw_mode = draw_mode.clone();
            let visible = visible.clone();
            let drag_start = drag_start.clone();
            let drag_rect = drag_rect.clone();
            move |gesture, start_x, start_y| {
                if !draw_mode.get() || !visible.get() {
                    gesture.set_state(gtk4::EventSequenceState::Denied);
                    return;
                }
                gesture.set_state(gtk4::EventSequenceState::Claimed);
                drag_start.set((start_x, start_y));
                drag_rect.set(Some((start_x, start_y, start_x, start_y)));
            }
        });

        draw_drag.connect_drag_update({
            let drag_start = drag_start.clone();
            let drag_rect = drag_rect.clone();
            let drawing_area = drawing_area.clone();
            move |_, offset_x, offset_y| {
                let (sx, sy) = drag_start.get();
                drag_rect.set(Some((sx, sy, sx + offset_x, sy + offset_y)));
                drawing_area.queue_draw();
            }
        });

        draw_drag.connect_drag_end({
            let drag_start = drag_start.clone();
            let drag_rect = drag_rect.clone();
            let image_id = image_id.clone();
            let db = db.clone();
            let faces = faces.clone();
            let drawing_area = drawing_area.clone();
            let known_embeddings = known_embeddings.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            let draw_mode = draw_mode.clone();
            let replace_face = replace_face.clone();
            move |gesture, offset_x, offset_y| {
                drag_rect.set(None);
                drawing_area.queue_draw();

                // Ignore tiny drags (accidental clicks).
                if offset_x.abs() < 8.0 && offset_y.abs() < 8.0 {
                    return;
                }

                let (sx, sy) = drag_start.get();
                let (ex, ey) = (sx + offset_x, sy + offset_y);

                let (vw, vh) = (scrolled.width() as f64, scrolled.height() as f64);
                let (img_w, img_h) = img_dims.get();
                let z = zoom.get();
                let scroll_x = scrolled.hadjustment().value();
                let scroll_y = scrolled.vadjustment().value();

                let Some((nx1, ny1)) =
                    screen_to_norm(sx, sy, img_w, img_h, vw, vh, z, scroll_x, scroll_y)
                else {
                    return;
                };
                let Some((nx2, ny2)) =
                    screen_to_norm(ex, ey, img_w, img_h, vw, vh, z, scroll_x, scroll_y)
                else {
                    return;
                };

                let (x1, y1, x2, y2) = (
                    nx1.min(nx2),
                    ny1.min(ny2),
                    nx1.max(nx2),
                    ny1.max(ny2),
                );
                // Require the box to cover at least 0.5 % of each axis.
                if (x2 - x1) < 0.005 || (y2 - y1) < 0.005 {
                    return;
                }
                let bbox = [x1, y1, x2, y2];

                let overlay = match gesture.widget().and_downcast::<gtk4::Overlay>() {
                    Some(w) => w,
                    None => return,
                };

                // Check whether this draw is replacing an existing face's bbox.
                let replace = replace_face.get();
                replace_face.set(None);

                if let Some((existing_face_id, existing_face_idx)) = replace {
                    // Update existing bbox in DB.
                    if db
                        .lock()
                        .ok()
                        .and_then(|g| g.update_face_bbox(existing_face_id, bbox).ok())
                        .is_none()
                    {
                        return;
                    }
                    // Update in memory.
                    if let Some(face) = faces.borrow_mut().get_mut(existing_face_idx) {
                        face.bbox = bbox;
                    }
                    drawing_area.queue_draw();
                    // Exit draw mode — it was auto-entered for this redraw.
                    draw_mode.set(false);
                    overlay.set_cursor(None::<&gtk4::gdk::Cursor>);
                    // Re-open the assignment dialog for the same face.
                    let (current_person, emb) = {
                        let f = faces.borrow();
                        let face = f.get(existing_face_idx);
                        (
                            face.and_then(|f| f.person_id),
                            face.map(|f| f.embedding.clone()).unwrap_or_default(),
                        )
                    };
                    open_person_dialog(
                        &overlay,
                        existing_face_id,
                        existing_face_idx,
                        current_person,
                        &emb,
                        &db,
                        &faces,
                        &drawing_area,
                        &known_embeddings.borrow(),
                        &draw_mode,
                        &replace_face,
                    );
                } else {
                    // Insert a new face detection.
                    let iid = image_id.get();
                    if iid == 0 {
                        return;
                    }
                    let face_id = match db
                        .lock()
                        .ok()
                        .and_then(|g| g.insert_face_detection(iid, bbox, &[], 1.0).ok())
                    {
                        Some(id) => id,
                        None => return,
                    };
                    let face_idx = {
                        let mut f = faces.borrow_mut();
                        let idx = f.len();
                        f.push(FaceDetection {
                            id: face_id,
                            image_id: iid,
                            bbox,
                            embedding: vec![],
                            person_id: None,
                            confidence: 1.0,
                            skipped: false,
                        });
                        idx
                    };
                    drawing_area.queue_draw();
                    open_person_dialog(
                        &overlay,
                        face_id,
                        face_idx,
                        None,
                        &[],
                        &db,
                        &faces,
                        &drawing_area,
                        &known_embeddings.borrow(),
                        &draw_mode,
                        &replace_face,
                    );
                }
            }
        });
        container.add_controller(draw_drag);

        Self {
            container,
            faces,
            visible,
            drawing_area,
            known_embeddings,
            draw_mode,
            image_id,
            drag_rect,
        }
    }

    pub fn load_for_image(&self, image_id: i64, db: &Arc<Mutex<maple_db::Database>>) {
        self.image_id.set(image_id);
        let new_faces = db.lock().unwrap().faces_for_image(image_id).unwrap_or_default();
        *self.faces.borrow_mut() = new_faces;
        *self.known_embeddings.borrow_mut() = EmbeddingMatrix::build(db);
        self.drawing_area.queue_draw();
    }

    pub fn set_visible(&self, v: bool) {
        self.visible.set(v);
        self.drawing_area.queue_draw();
    }

    /// Enable or disable "draw new face" mode.
    ///
    /// When active the cursor changes to a crosshair and dragging over the
    /// image creates a new face detection bounding box.
    pub fn set_draw_mode(&self, active: bool) {
        self.draw_mode.set(active);
        if active {
            self.container.set_cursor_from_name(Some("crosshair"));
        } else {
            self.container.set_cursor(None::<&gtk4::gdk::Cursor>);
            self.drag_rect.set(None);
        }
        self.drawing_area.queue_draw();
    }
}

fn open_person_dialog(
    parent: &gtk4::Overlay,
    face_id: i64,
    face_idx: usize,
    current_person: Option<i64>,
    embedding: &[f32],
    db: &Arc<Mutex<maple_db::Database>>,
    faces: &Rc<RefCell<Vec<FaceDetection>>>,
    drawing_area: &gtk4::DrawingArea,
    known: &EmbeddingMatrix,
    draw_mode: &Rc<Cell<bool>>,
    replace_face: &Rc<Cell<Option<(i64, usize)>>>,
) {
    // All known persons, ranked by similarity when an embedding is available.
    let all_persons = known.top_k(embedding, usize::MAX);
    let current_name = person_name(db, current_person);

    let window = parent
        .root()
        .and_downcast::<gtk4::Window>()
        .expect("overlay has window root");

    let dialog = adw::Window::builder()
        .title("Assign Person")
        .default_width(380)
        .transient_for(&window)
        .modal(true)
        .build();

    let vbox = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(10)
        .margin_top(16)
        .margin_bottom(16)
        .margin_start(16)
        .margin_end(16)
        .build();

    // ── Suggestion list ────────────────────────────────────────────
    // Holds (row_box, lowercase_name) for live filtering by the entry.
    let mut suggestion_rows: Vec<(gtk4::Box, String)> = Vec::new();

    let suggestions_inner = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(2)
        .build();

    let suggestions_section = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(4)
        .build();

    if !all_persons.is_empty() {
        let hint = gtk4::Label::new(Some("Known persons"));
        hint.add_css_class("caption");
        hint.add_css_class("dim-label");
        hint.set_halign(gtk4::Align::Start);
        suggestions_section.append(&hint);

        for (person_id, name, sim) in all_persons {
            let row = gtk4::Box::builder()
                .orientation(gtk4::Orientation::Horizontal)
                .spacing(8)
                .hexpand(true)
                .build();

            let person_btn = gtk4::Button::with_label(&name);
            person_btn.set_hexpand(true);
            person_btn.set_halign(gtk4::Align::Fill);

            person_btn.connect_clicked({
                let dialog = dialog.clone();
                let db = db.clone();
                let faces = faces.clone();
                let drawing_area = drawing_area.clone();
                move |_| {
                    assign_face_to_person(&db, &faces, &drawing_area, face_idx, face_id, person_id);
                    dialog.close();
                }
            });

            row.append(&person_btn);
            if sim.is_finite() {
                let sim_label = gtk4::Label::new(Some(&format!("{:.1}%", sim * 100.0)));
                sim_label.add_css_class("dim-label");
                sim_label.set_width_chars(6);
                sim_label.set_xalign(1.0);
                row.append(&sim_label);
            }
            suggestions_inner.append(&row);
            suggestion_rows.push((row, name.to_lowercase()));
        }

        // Scrollable so a large person list does not blow up the dialog height.
        let scroll = gtk4::ScrolledWindow::builder()
            .hscrollbar_policy(gtk4::PolicyType::Never)
            .vscrollbar_policy(gtk4::PolicyType::Automatic)
            .propagate_natural_height(true)
            .child(&suggestions_inner)
            .build();
        scroll.set_max_content_height(220);
        suggestions_section.append(&scroll);
        suggestions_section.append(&gtk4::Separator::new(gtk4::Orientation::Horizontal));
        vbox.append(&suggestions_section);
    }

    // ── Name entry ─────────────────────────────────────────────────
    let entry = gtk4::Entry::builder()
        .placeholder_text("Filter or create new person")
        .hexpand(true)
        .build();
    if let Some(ref name) = current_name {
        entry.set_text(name);
    }
    vbox.append(&entry);

    // Filter suggestion rows as the user types; hide the section when nothing matches.
    if !suggestion_rows.is_empty() {
        entry.connect_changed({
            let suggestions_section = suggestions_section.clone();
            move |e| {
                let raw = e.text();
                let text = raw.trim().to_lowercase();
                let mut any_visible = false;
                for (row, name) in &suggestion_rows {
                    let visible = text.is_empty() || name.contains(text.as_str());
                    row.set_visible(visible);
                    if visible {
                        any_visible = true;
                    }
                }
                // Always show the section when the entry is blank (show all).
                suggestions_section.set_visible(text.is_empty() || any_visible);
            }
        });
    }

    // ── Action buttons ─────────────────────────────────────────────
    let btn_row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .build();

    let delete_btn = gtk4::Button::with_label("Delete");
    delete_btn.add_css_class("destructive-action");
    delete_btn.add_css_class("flat");

    let redraw_btn = gtk4::Button::with_label("Redraw box");
    redraw_btn.add_css_class("flat");

    let spacer = gtk4::Box::new(gtk4::Orientation::Horizontal, 0);
    spacer.set_hexpand(true);

    let cancel_btn = gtk4::Button::with_label("Cancel");
    cancel_btn.add_css_class("flat");
    let assign_btn = gtk4::Button::with_label("Assign");
    assign_btn.add_css_class("suggested-action");

    btn_row.append(&delete_btn);
    btn_row.append(&redraw_btn);
    btn_row.append(&spacer);
    btn_row.append(&cancel_btn);
    btn_row.append(&assign_btn);
    vbox.append(&btn_row);

    delete_btn.connect_clicked({
        let dialog = dialog.clone();
        let db = db.clone();
        let faces = faces.clone();
        let drawing_area = drawing_area.clone();
        move |_| {
            if let Ok(g) = db.lock() {
                let _ = g.delete_face_detection(face_id);
            }
            let mut f = faces.borrow_mut();
            if face_idx < f.len() {
                f.remove(face_idx);
            }
            drop(f);
            drawing_area.queue_draw();
            dialog.close();
        }
    });

    redraw_btn.connect_clicked({
        let dialog = dialog.clone();
        let draw_mode = draw_mode.clone();
        let replace_face = replace_face.clone();
        let parent = parent.clone();
        move |_| {
            replace_face.set(Some((face_id, face_idx)));
            draw_mode.set(true);
            parent.set_cursor_from_name(Some("crosshair"));
            dialog.close();
        }
    });

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
            assign_face_to_name(&db, &faces, &drawing_area, face_idx, face_id, &name);
            dialog.close();
        }
    });

    let header = adw::HeaderBar::new();
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&vbox));
    dialog.set_content(Some(&toolbar));
    dialog.present();
}
