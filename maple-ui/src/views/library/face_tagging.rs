//! Standalone face-tagging wizard.
//!
//! Iterates every untagged face across the whole library, one at a time.
//!
//! Layout
//! ──────
//!   ┌─────────────────────────────────────────────────────────────┐
//!   │ [← Back]  Face Tagging                         [✓ Close]   │
//!   ├──────────────────────────────┬──────────────────────────────┤
//!   │                              │  Face 2 of 3  ·  Image 5/N  │
//!   │   Image with face boxes      │                              │
//!   │   Active face = orange       │  [Alice] 94.2 %              │
//!   │   Other faces = dim blue     │  [Bob]   81.3 %              │
//!   │                              │  [Carol] 79.1 %              │
//!   │                              │  ┌─────────────────────────┐ │
//!   │                              │  │ Enter name …            │ │
//!   │                              │  └─────────────────────────┘ │
//!   │                              │  [Skip]          [Assign New]│
//!   └──────────────────────────────┴──────────────────────────────┘
//!
//! Navigation
//! ──────────
//!   • Clicking a suggestion or "Assign New" tags the face and advances to
//!     the next untagged face (within the same image, then the next image).
//!   • "Skip" advances without tagging — the face stays untagged.
//!   • "Back" undoes the last action (un-assigns the face), returns to that
//!     face, and highlights the previously chosen button so the user can
//!     change or confirm.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;

use maple_db::FaceDetection;

use super::face_shared::{
    assign_face_to_name, assign_face_to_person, face_screen_rect, is_real_detection,
    next_untagged_index, unassign_face, EmbeddingMatrix,
};
use tracing::info;
use super::image_loader::load_image_async;

// ── History ───────────────────────────────────────────────────────

#[derive(Clone, Debug)]
enum TagAction {
    /// Assigned to an existing person.
    Person { person_id: i64 },
    /// Created a new person and assigned.
    NewPerson { person_id: i64, name: String },
    /// Skipped without assigning.
    Skipped,
}

#[derive(Clone)]
struct HistoryEntry {
    /// Which image (index into `image_ids`) this face belongs to.
    image_idx: usize,
    /// The face that was acted on.
    face_id: i64,
    face_idx: usize,
    action: TagAction,
}

// ── Internal state ───────────────────────────────────────────────

#[derive(Clone)]
struct TaggingState {
    db: Arc<Mutex<maple_db::Database>>,
    tagging_top_k: usize,

    /// IDs of all images that had at least one untagged face when the view opened.
    image_ids: Rc<RefCell<Vec<i64>>>,
    /// Current position within `image_ids`.
    image_idx: Rc<Cell<usize>>,
    /// Faces for the currently displayed image.
    current_faces: Rc<RefCell<Vec<FaceDetection>>>,
    /// Index into `current_faces` for the face being tagged right now.
    face_idx: Rc<Cell<usize>>,
    /// History stack — entries are added on every action, popped on Back.
    history: Rc<RefCell<Vec<HistoryEntry>>>,
    /// Embedding matrix built at session start; updated as faces are tagged.
    known_embeddings: Rc<RefCell<EmbeddingMatrix>>,

    // ── Widgets ───────────────────────────────────────────────────
    picture: gtk4::Picture,
    drawing_area: gtk4::DrawingArea,
    /// Natural size of the currently displayed image (pixels).
    img_dims: Rc<Cell<(i32, i32)>>,

    status_label: gtk4::Label,
    matches_box: gtk4::Box,
    name_entry: gtk4::Entry,
    assign_btn: gtk4::Button,
    skip_btn: gtk4::Button,
    back_btn: gtk4::Button,
    done_box: gtk4::Box,
    panel: gtk4::Box,
}

// ── Public entry point ────────────────────────────────────────────

/// Build and return the face-tagging navigation page.
///
/// Push this onto the `adw::NavigationView` when the user clicks "Tag Faces".
pub fn build_face_tagging_page(
    db: Arc<Mutex<maple_db::Database>>,
    tagging_top_k: usize,
) -> adw::NavigationPage {
    // ── Image display ─────────────────────────────────────────────
    let picture = gtk4::Picture::builder()
        .content_fit(gtk4::ContentFit::Contain)
        .hexpand(true)
        .vexpand(true)
        .build();

    let img_dims: Rc<Cell<(i32, i32)>> = Rc::new(Cell::new((0, 0)));

    let drawing_area = gtk4::DrawingArea::builder()
        .hexpand(true)
        .vexpand(true)
        .can_target(false)
        .build();

    let image_overlay = gtk4::Overlay::new();
    image_overlay.set_child(Some(&picture));
    image_overlay.add_overlay(&drawing_area);
    image_overlay.set_hexpand(true);
    image_overlay.set_vexpand(true);

    // ── Right panel ───────────────────────────────────────────────
    let status_label = gtk4::Label::builder()
        .halign(gtk4::Align::Start)
        .wrap(true)
        .build();
    status_label.add_css_class("heading");

    let matches_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(6)
        .build();

    let matches_scroll = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .build();
    matches_scroll.set_child(Some(&matches_box));

    let name_entry = gtk4::Entry::builder()
        .placeholder_text("Enter name…")
        .hexpand(true)
        .build();

    let assign_btn = gtk4::Button::with_label("Assign New");
    assign_btn.add_css_class("suggested-action");

    let skip_btn = gtk4::Button::with_label("Skip");
    skip_btn.set_tooltip_text(Some("Skip this face — it won't be tagged"));

    let back_btn = gtk4::Button::with_label("← Back");
    back_btn.add_css_class("flat");
    back_btn.set_sensitive(false);

    let action_row = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .spacing(8)
        .halign(gtk4::Align::Fill)
        .build();
    action_row.append(&back_btn);
    action_row.append(&skip_btn);
    action_row.append(&assign_btn);

    let panel = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .margin_top(12)
        .margin_bottom(12)
        .margin_start(12)
        .margin_end(12)
        .width_request(320)
        .hexpand(false)
        .vexpand(true)
        .build();
    panel.append(&status_label);
    panel.append(&matches_scroll);
    panel.append(&name_entry);
    panel.append(&action_row);

    // ── "All done" overlay ────────────────────────────────────────
    let done_box = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(12)
        .halign(gtk4::Align::Center)
        .valign(gtk4::Align::Center)
        .hexpand(true)
        .vexpand(true)
        .build();
    let done_icon = gtk4::Image::from_icon_name("emblem-ok-symbolic");
    done_icon.set_pixel_size(48);
    let done_lbl = gtk4::Label::new(Some("All faces tagged!"));
    done_lbl.add_css_class("title-2");
    let done_sub = gtk4::Label::new(Some("No untagged faces remain in the library."));
    done_sub.add_css_class("dim-label");
    done_box.append(&done_icon);
    done_box.append(&done_lbl);
    done_box.append(&done_sub);
    done_box.set_visible(false);

    // ── Main body ─────────────────────────────────────────────────
    let body = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Horizontal)
        .hexpand(true)
        .vexpand(true)
        .build();
    body.append(&image_overlay);
    body.append(&panel);
    body.append(&done_box);

    // ── Header ────────────────────────────────────────────────────
    let header = adw::HeaderBar::new();

    // ── Page ──────────────────────────────────────────────────────
    let toolbar = adw::ToolbarView::new();
    toolbar.add_top_bar(&header);
    toolbar.set_content(Some(&body));

    let page = adw::NavigationPage::builder()
        .title("Tag Faces")
        .child(&toolbar)
        .build();

    // ── State ─────────────────────────────────────────────────────
    let state = TaggingState {
        db,
        tagging_top_k,
        image_ids: Rc::new(RefCell::new(vec![])),
        image_idx: Rc::new(Cell::new(0)),
        current_faces: Rc::new(RefCell::new(vec![])),
        face_idx: Rc::new(Cell::new(0)),
        history: Rc::new(RefCell::new(vec![])),
        known_embeddings: Rc::new(RefCell::new(EmbeddingMatrix::empty())),
        picture,
        drawing_area,
        img_dims,
        status_label,
        matches_box,
        name_entry,
        assign_btn,
        skip_btn,
        back_btn,
        done_box,
        panel,
    };

    // Wire the draw function.
    wire_draw_func(&state);

    // Wire buttons.
    wire_buttons(&state);

    // Load data when the page is first shown.
    page.connect_map({
        let state = state.clone();
        let loaded = Rc::new(Cell::new(false));
        move |_| {
            if !loaded.get() {
                loaded.set(true);
                load_queue(&state);
            }
        }
    });

    page
}

// ── Data loading ─────────────────────────────────────────────────

/// Query all images with untagged faces and navigate to the first one.
fn load_queue(state: &TaggingState) {
    let ids = state
        .db
        .lock()
        .map(|g| g.images_with_untagged_faces().unwrap_or_default())
        .unwrap_or_default();

    info!("face tagging: loaded {} images with untagged faces", ids.len());

    *state.image_ids.borrow_mut() = ids;
    state.image_idx.set(0);
    state.history.borrow_mut().clear();
    state.back_btn.set_sensitive(false);
    *state.known_embeddings.borrow_mut() = EmbeddingMatrix::build(&state.db);

    navigate_to_image(state, 0, 0);
}

/// Load image at `img_idx` and navigate to the first untagged face at or after
/// `start_face_idx`.  Shows the "all done" panel if no more faces exist.
fn navigate_to_image(state: &TaggingState, img_idx: usize, start_face_idx: usize) {
    let ids = state.image_ids.borrow();

    // Advance through images until we find one with an untagged face.
    let mut idx = img_idx;
    loop {
        if idx >= ids.len() {
            drop(ids);
            show_done(state);
            return;
        }

        let image_id = ids[idx];
        let faces = state
            .db
            .lock()
            .map(|g| g.faces_for_image(image_id).unwrap_or_default())
            .unwrap_or_default();

        let face_start = if idx == img_idx { start_face_idx } else { 0 };
        if let Some(face_idx) = next_untagged_index(&faces, face_start) {
            let real_count = faces.iter().filter(|f| is_real_detection(f)).count();
            info!(
                "face tagging: navigating to image {} (id={}), face {}/{} ({}  total faces)",
                idx, image_id, face_idx, faces.len(), real_count,
            );
            drop(ids);
            state.image_idx.set(idx);
            *state.current_faces.borrow_mut() = faces;
            state.face_idx.set(face_idx);
            load_image_for_idx(state, image_id);
            rebuild_panel(state, None);
            return;
        }

        idx += 1;
    }
}

/// Load the image file into the Picture widget.
///
/// Uses the shared `load_image_async` so EXIF orientation is applied and
/// `img_dims` is populated from the *decoded* pixel dimensions (not the DB
/// metadata columns, which may be NULL).  The drawing area is queued for a
/// redraw once the texture arrives.
fn load_image_for_idx(state: &TaggingState, image_id: i64) {
    let path = state
        .db
        .lock()
        .ok()
        .and_then(|g| g.image_by_id(image_id).ok().flatten())
        .map(|img| img.path);

    let Some(path) = path else { return };

    let drawing_area = state.drawing_area.clone();
    load_image_async(path, &state.picture, &state.img_dims, move |_, _| {
        drawing_area.queue_draw();
    });
}

fn show_done(state: &TaggingState) {
    info!("face tagging: all faces tagged — showing done screen");
    state.panel.set_visible(false);
    state.done_box.set_visible(true);
    state.drawing_area.queue_draw();
}

// ── Panel rebuild ─────────────────────────────────────────────────

/// Rebuild the suggestion buttons for the current face.
///
/// `previous_action` is set when returning via Back — that action's button
/// is highlighted so the user can confirm or change their choice.
fn rebuild_panel(state: &TaggingState, previous_action: Option<&TagAction>) {
    state.panel.set_visible(true);
    state.done_box.set_visible(false);

    // Status line.
    let ids_count = state.image_ids.borrow().len();
    let img_idx = state.image_idx.get();
    let face_idx = state.face_idx.get();
    let faces = state.current_faces.borrow();
    let real_count = faces.iter().filter(|f| is_real_detection(f)).count();
    let real_pos = faces
        .iter()
        .take(face_idx + 1)
        .filter(|f| is_real_detection(f))
        .count();
    drop(faces);

    state.status_label.set_label(&format!(
        "Face {} of {}  ·  Image {} of {}",
        real_pos,
        real_count,
        img_idx + 1,
        ids_count,
    ));

    // Clear old suggestion buttons.
    while let Some(child) = state.matches_box.first_child() {
        state.matches_box.remove(&child);
    }
    state.name_entry.set_text("");

    // Get embedding for the current face.
    let embedding = {
        let faces = state.current_faces.borrow();
        faces
            .get(face_idx)
            .map(|f| f.embedding.clone())
            .unwrap_or_default()
    };

    let prev_person_id = previous_action.and_then(|a| match a {
        TagAction::Person { person_id } | TagAction::NewPerson { person_id, .. } => {
            Some(*person_id)
        }
        TagAction::Skipped => None,
    });
    let prev_new_name = previous_action.and_then(|a| match a {
        TagAction::NewPerson { name, .. } => Some(name.as_str()),
        _ => None,
    });

    let suggestions = state
        .known_embeddings
        .borrow()
        .top_k(&embedding, state.tagging_top_k);

    info!(
        "face tagging: face_idx={}, {} suggestions (embedding len={})",
        face_idx,
        suggestions.len(),
        embedding.len(),
    );
    for (pid, name, sim) in &suggestions {
        info!("  suggestion: person_id={}, name={:?}, similarity={:.4}", pid, name, sim);
    }

    if suggestions.is_empty() {
        let hint = gtk4::Label::new(Some("No suggestions — enter a name below."));
        hint.set_halign(gtk4::Align::Start);
        hint.add_css_class("dim-label");
        state.matches_box.append(&hint);
    } else {
        for (person_id, name, sim) in suggestions {
            let row = gtk4::Box::builder()
                .orientation(gtk4::Orientation::Horizontal)
                .spacing(8)
                .hexpand(true)
                .build();

            let btn = gtk4::Button::with_label(&name);
            btn.set_hexpand(true);
            btn.set_halign(gtk4::Align::Fill);

            // Highlight if this was the previously chosen person.
            if prev_person_id == Some(person_id) {
                btn.add_css_class("suggested-action");
            }

            btn.connect_clicked({
                let state = state.clone();
                let name = name.clone();
                move |_| {
                    on_pick_person(&state, person_id, &name);
                }
            });

            row.append(&btn);
            if sim.is_finite() {
                let pct = sim * 100.0;
                let sim_label = gtk4::Label::new(Some(&format!("{:.1}%", pct)));
                sim_label.set_width_chars(7);
                sim_label.set_xalign(1.0);

                // Style the tag based on similarity strength.
                if pct >= 80.0 {
                    sim_label.add_css_class("success");
                } else if pct >= 50.0 {
                    sim_label.add_css_class("warning");
                } else {
                    sim_label.add_css_class("dim-label");
                }

                row.append(&sim_label);
            }
            state.matches_box.append(&row);
        }
    }

    // Pre-fill entry if the previous action was "new person".
    if let Some(prev_name) = prev_new_name {
        state.name_entry.set_text(prev_name);
    }

    state.back_btn.set_sensitive(!state.history.borrow().is_empty());
    state.drawing_area.queue_draw();
}

// ── Actions ───────────────────────────────────────────────────────

/// User picked an existing person from the suggestions.
fn on_pick_person(state: &TaggingState, person_id: i64, name: &str) {
    let face_idx = state.face_idx.get();
    let (face_id, embedding) = {
        let faces = state.current_faces.borrow();
        let f = faces.get(face_idx);
        (f.map(|f| f.id), f.map(|f| f.embedding.clone()).unwrap_or_default())
    };
    let Some(face_id) = face_id else { return };

    info!(
        "face tagging: assigning face_id={} (idx={}) to existing person_id={} ({:?})",
        face_id, face_idx, person_id, name,
    );

    let dummy_da = gtk4::DrawingArea::new();
    assign_face_to_person(
        &state.db,
        &state.current_faces,
        &dummy_da,
        face_idx,
        face_id,
        person_id,
    );
    state.drawing_area.queue_draw();

    // Add this face's embedding to the matrix so subsequent faces can match it.
    state.known_embeddings.borrow_mut().add(person_id, name.to_owned(), &embedding);

    state.history.borrow_mut().push(HistoryEntry {
        image_idx: state.image_idx.get(),
        face_id,
        face_idx,
        action: TagAction::Person { person_id },
    });

    advance(state);
}

/// User clicked "Assign New" (or pressed Enter in the entry).
fn on_assign_new(state: &TaggingState) {
    let name = state.name_entry.text().trim().to_owned();
    if name.is_empty() {
        return;
    }

    let face_idx = state.face_idx.get();
    let face_id = {
        let faces = state.current_faces.borrow();
        faces.get(face_idx).map(|f| f.id)
    };
    let Some(face_id) = face_id else { return };

    let embedding = {
        let faces = state.current_faces.borrow();
        faces.get(face_idx).map(|f| f.embedding.clone()).unwrap_or_default()
    };

    info!(
        "face tagging: assigning face_id={} (idx={}) to new name {:?}",
        face_id, face_idx, name,
    );

    let dummy_da = gtk4::DrawingArea::new();
    let person_id = assign_face_to_name(
        &state.db,
        &state.current_faces,
        &dummy_da,
        face_idx,
        face_id,
        &name,
    );
    state.drawing_area.queue_draw();

    if let Some(pid) = person_id {
        // Add to matrix so subsequent faces in this session can match this person.
        state.known_embeddings.borrow_mut().add(pid, name.clone(), &embedding);

        state.history.borrow_mut().push(HistoryEntry {
            image_idx: state.image_idx.get(),
            face_id,
            face_idx,
            action: TagAction::NewPerson {
                person_id: pid,
                name: name.clone(),
            },
        });
    }

    advance(state);
}

/// User clicked "Skip".
fn on_skip(state: &TaggingState) {
    let face_idx = state.face_idx.get();
    let face_id = {
        let faces = state.current_faces.borrow();
        faces.get(face_idx).map(|f| f.id)
    };
    let Some(face_id) = face_id else { return };

    info!("face tagging: skipping face_id={} (idx={})", face_id, face_idx);

    state.history.borrow_mut().push(HistoryEntry {
        image_idx: state.image_idx.get(),
        face_id,
        face_idx,
        action: TagAction::Skipped,
    });

    advance(state);
}

/// Move to the next untagged face.
fn advance(state: &TaggingState) {
    let current_face_idx = state.face_idx.get();
    let current_img_idx = state.image_idx.get();

    // Look for the next untagged face within the current image (after current).
    let next_in_image = {
        let faces = state.current_faces.borrow();
        next_untagged_index(&faces, current_face_idx + 1)
    };

    if let Some(next_fi) = next_in_image {
        state.face_idx.set(next_fi);
        rebuild_panel(state, None);
    } else {
        // Move to next image.
        navigate_to_image(state, current_img_idx + 1, 0);
    }
}

/// Undo the last action and return to that face.
fn on_back(state: &TaggingState) {
    let entry = match state.history.borrow_mut().pop() {
        Some(e) => e,
        None => return,
    };

    info!(
        "face tagging: undoing {:?} for face_id={} (img_idx={}, face_idx={})",
        entry.action, entry.face_id, entry.image_idx, entry.face_idx,
    );

    let HistoryEntry {
        image_idx,
        face_id,
        face_idx,
        action,
    } = entry;

    // Un-assign the face in DB and in memory if it was assigned.
    match &action {
        TagAction::Person { .. } | TagAction::NewPerson { .. } => {
            // Re-load the image's faces so we have the latest state.
            let image_id = state.image_ids.borrow().get(image_idx).copied();
            if let Some(image_id) = image_id {
                let faces = state
                    .db
                    .lock()
                    .map(|g| g.faces_for_image(image_id).unwrap_or_default())
                    .unwrap_or_default();
                *state.current_faces.borrow_mut() = faces;
                // Un-assign by face_idx position.
                unassign_face(&state.db, &state.current_faces, face_id, face_idx);
                if state.image_idx.get() != image_idx {
                    // Navigating to a different image — reload picture.
                    load_image_for_idx(state, image_id);
                }
            }
        }
        TagAction::Skipped => {
            // Nothing to undo in DB; but if it's a different image, reload.
            let image_id = state.image_ids.borrow().get(image_idx).copied();
            if let Some(image_id) = image_id {
                if state.image_idx.get() != image_idx {
                    let faces = state
                        .db
                        .lock()
                        .map(|g| g.faces_for_image(image_id).unwrap_or_default())
                        .unwrap_or_default();
                    *state.current_faces.borrow_mut() = faces;
                    load_image_for_idx(state, image_id);
                }
            }
        }
    }

    state.image_idx.set(image_idx);
    state.face_idx.set(face_idx);
    rebuild_panel(state, Some(&action));
}

// ── Draw function ─────────────────────────────────────────────────

fn wire_draw_func(state: &TaggingState) {
    let faces = state.current_faces.clone();
    let face_idx = state.face_idx.clone();
    let img_dims = state.img_dims.clone();
    let picture = state.picture.clone();

    state.drawing_area.set_draw_func(move |da, cx, _w, _h| {
        let f = faces.borrow();
        if f.is_empty() {
            return;
        }
        let (img_w, img_h) = img_dims.get();
        if img_w == 0 || img_h == 0 {
            return;
        }
        let vw = da.width() as f64;
        let vh = da.height() as f64;
        if vw == 0.0 || vh == 0.0 {
            return;
        }
        let _ = picture.width(); // touch to ensure layout is fresh
        let active = face_idx.get();

        for (i, face) in f.iter().enumerate() {
            if !is_real_detection(face) {
                continue;
            }
            let Some((sx, sy, sw, sh)) =
                face_screen_rect(face.bbox, img_w, img_h, vw, vh, 1.0, 0.0, 0.0)
            else {
                continue;
            };

            if i == active {
                // Active face: bright orange outline + soft fill.
                cx.set_source_rgba(1.0, 0.55, 0.0, 0.95);
                cx.set_line_width(3.0);
                cx.rectangle(sx, sy, sw, sh);
                let _ = cx.stroke();
                cx.set_source_rgba(1.0, 0.55, 0.0, 0.12);
                cx.rectangle(sx, sy, sw, sh);
                let _ = cx.fill();
            } else if face.person_id.is_some() {
                // Tagged (not current): dim green.
                cx.set_source_rgba(0.2, 0.85, 0.4, 0.4);
                cx.set_line_width(1.5);
                cx.rectangle(sx, sy, sw, sh);
                let _ = cx.stroke();
            } else {
                // Untagged (not current): dim blue.
                cx.set_source_rgba(0.2, 0.55, 1.0, 0.4);
                cx.set_line_width(1.5);
                cx.rectangle(sx, sy, sw, sh);
                let _ = cx.stroke();
            }
        }
    });
}

// ── Button wiring ─────────────────────────────────────────────────

fn wire_buttons(state: &TaggingState) {
    state.assign_btn.connect_clicked({
        let state = state.clone();
        move |_| on_assign_new(&state)
    });

    state.name_entry.connect_activate({
        let state = state.clone();
        move |_| on_assign_new(&state)
    });

    state.skip_btn.connect_clicked({
        let state = state.clone();
        move |_| on_skip(&state)
    });

    state.back_btn.connect_clicked({
        let state = state.clone();
        move |_| on_back(&state)
    });
}
