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

#[derive(Clone)]
pub struct FaceOverlay {
    /// Image overlay used to draw bounding boxes.
    pub container: gtk4::Overlay,
    faces: Rc<RefCell<Vec<FaceDetection>>>,
    visible: Rc<Cell<bool>>,
    drawing_area: gtk4::DrawingArea,
    /// Embedding matrix rebuilt on each image load; shared with the click gesture.
    known_embeddings: Rc<RefCell<EmbeddingMatrix>>,
}

impl FaceOverlay {
    pub fn new(
        scrolled: &gtk4::ScrolledWindow,
        zoom: Rc<Cell<f64>>,
        img_dims: Rc<Cell<(i32, i32)>>,
        db: Arc<Mutex<maple_db::Database>>,
        tagging_top_k: usize,
    ) -> Self {
        let faces: Rc<RefCell<Vec<FaceDetection>>> = Rc::new(RefCell::new(vec![]));
        let visible: Rc<Cell<bool>> = Rc::new(Cell::new(false));
        let known_embeddings: Rc<RefCell<EmbeddingMatrix>> =
            Rc::new(RefCell::new(EmbeddingMatrix::empty()));

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

                    if face.person_id.is_some() {
                        cx.set_source_rgba(0.2, 0.85, 0.4, 0.9);
                    } else {
                        cx.set_source_rgba(0.2, 0.55, 1.0, 0.9);
                    }
                    cx.set_line_width(2.5);
                    cx.rectangle(sx, sy, sw, sh);
                    let _ = cx.stroke();

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

        let container = gtk4::Overlay::new();
        container.set_child(Some(scrolled));
        container.add_overlay(&drawing_area);

        let gesture = gtk4::GestureClick::new();
        gesture.connect_pressed({
            let faces = faces.clone();
            let visible = visible.clone();
            let zoom = zoom.clone();
            let img_dims = img_dims.clone();
            let scrolled = scrolled.clone();
            let db = db.clone();
            let drawing_area = drawing_area.clone();
            let known_embeddings = known_embeddings.clone();
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
                        tagging_top_k,
                    );
                }
            }
        });
        container.add_controller(gesture);

        Self {
            container,
            faces,
            visible,
            drawing_area,
            known_embeddings,
        }
    }

    pub fn load_for_image(&self, image_id: i64, db: &Arc<Mutex<maple_db::Database>>) {
        let new_faces = db.lock().unwrap().faces_for_image(image_id).unwrap_or_default();
        *self.faces.borrow_mut() = new_faces;
        *self.known_embeddings.borrow_mut() = EmbeddingMatrix::build(db);
        self.drawing_area.queue_draw();
    }

    pub fn set_visible(&self, v: bool) {
        self.visible.set(v);
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
    top_k: usize,
) {
    let matches = known.top_k(embedding, top_k);
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

    if !matches.is_empty() {
        let hint = gtk4::Label::new(Some("Suggestions"));
        hint.add_css_class("caption");
        hint.add_css_class("dim-label");
        hint.set_halign(gtk4::Align::Start);
        vbox.append(&hint);

        for (person_id, name, sim) in matches {
            let row = gtk4::Box::builder()
                .orientation(gtk4::Orientation::Horizontal)
                .spacing(8)
                .hexpand(true)
                .build();

            let person_btn = gtk4::Button::with_label(&name);
            person_btn.set_hexpand(true);
            person_btn.set_halign(gtk4::Align::Fill);
            person_btn.set_tooltip_text(Some("Assign this face"));

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
            vbox.append(&row);
        }
    }

    let entry = gtk4::Entry::builder()
        .placeholder_text("Add new person")
        .hexpand(true)
        .build();
    if let Some(ref name) = current_name {
        entry.set_text(name);
    }
    vbox.append(&entry);

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
