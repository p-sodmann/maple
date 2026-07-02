//! Dedicated face-tagging window controller.
//!
//! Iterates through every real, untagged, non-skipped face across all present
//! images, showing each one as a square crop centred on the detection bounding
//! box with one full bbox-diameter of padding in every direction.  Pixels
//! outside the image bounds are filled with mid-grey so every face appears at a
//! consistent visual size regardless of proximity to the image edge.
//!
//! Any assignment action (Assign / Delete / Skip) removes the current face from
//! the queue and automatically advances to the next one; the window closes when
//! the queue is exhausted.

use std::cell::{Cell, RefCell};
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, Image, Rgb8Pixel, SharedPixelBuffer, SharedString};

use crate::face_crop::extract_crop;
use crate::face_overlay::{build_suggestions, is_real_detection, EmbeddingMatrix};
use crate::FaceTagWindow;

thread_local! {
    static FACE_TAG: RefCell<Option<FaceTagCtx>> = const { RefCell::new(None) };
}

/// Side length of the crop delivered to Slint (pixels, RGB8).
const CROP_PX: u32 = 480;

struct FaceEntry {
    path: PathBuf,
    face: maple_db::FaceDetection,
}

struct FaceTagCtx {
    window: FaceTagWindow,
    entries: Rc<RefCell<Vec<FaceEntry>>>,
    index: Rc<Cell<usize>>,
    known: Rc<RefCell<EmbeddingMatrix>>,
    #[allow(dead_code)]
    db: Arc<Mutex<maple_db::Database>>,
}

/// Open the face-tagging window over all currently untagged faces.
pub fn open(db: Arc<Mutex<maple_db::Database>>) {
    let entries = collect_untagged_faces(&db);
    if entries.is_empty() {
        tracing::info!("face_tag: no untagged faces");
        return;
    }

    if FACE_TAG.with(|c| c.borrow().is_none()) {
        match build(db.clone()) {
            Ok(ctx) => FACE_TAG.with(|c| *c.borrow_mut() = Some(ctx)),
            Err(e) => {
                tracing::error!("face_tag: build: {e}");
                return;
            }
        }
    }

    FACE_TAG.with(|cell| {
        let guard = cell.borrow();
        let Some(ctx) = guard.as_ref() else { return };
        *ctx.entries.borrow_mut() = entries;
        ctx.index.set(0);
        *ctx.known.borrow_mut() = EmbeddingMatrix::build(&db);
        load_face(ctx.window.as_weak(), &ctx.entries, &ctx.index, &ctx.known);
        ctx.window.show().unwrap_or_else(|e| tracing::error!("face_tag: show: {e}"));
    });
}

// ── Window construction ────────────────────────────────────────────

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<FaceTagCtx, slint::PlatformError> {
    let window = FaceTagWindow::new()?;
    let entries: Rc<RefCell<Vec<FaceEntry>>> = Rc::new(RefCell::new(Vec::new()));
    let index = Rc::new(Cell::new(0usize));
    let known: Rc<RefCell<EmbeddingMatrix>> = Rc::new(RefCell::new(EmbeddingMatrix::empty()));

    window.on_back({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                w.hide().ok();
            }
        }
    });

    window.on_prev_face({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        move || {
            let cur = index.get();
            if cur == 0 {
                return;
            }
            index.set(cur - 1);
            load_face(w.clone(), &entries, &index, &known);
        }
    });

    window.on_next_face({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        move || {
            let cur = index.get();
            if cur + 1 >= entries.borrow().len() {
                return;
            }
            index.set(cur + 1);
            load_face(w.clone(), &entries, &index, &known);
        }
    });

    window.on_face_assign_person({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        let db = db.clone();
        move |face_id, person_id| {
            let ok = db.lock().ok().is_some_and(|g| {
                let r = g.assign_face_to_person(face_id as i64, Some(person_id as i64))
                    .map_err(|e| tracing::warn!("face_tag: assign_to_person: {e}"))
                    .is_ok();
                if r {
                    let _ = g.update_person_representative(person_id as i64);
                }
                r
            });
            if ok {
                advance(&w, &entries, &index, &known);
            }
        }
    });

    window.on_face_assign_name({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        let db = db.clone();
        move |face_id, name| {
            let name = name.trim().to_owned();
            if name.is_empty() {
                return;
            }
            let embedding = entries
                .borrow()
                .iter()
                .find(|e| e.face.id == face_id as i64)
                .map(|e| e.face.embedding.clone())
                .unwrap_or_default();

            let result = db.lock().ok().and_then(|g| {
                let pid = g.upsert_person(&name)
                    .map_err(|e| tracing::warn!("face_tag: upsert_person: {e}"))
                    .ok()?;
                g.assign_face_to_person(face_id as i64, Some(pid))
                    .map_err(|e| tracing::warn!("face_tag: assign_face: {e}"))
                    .ok()?;
                let _ = g.update_person_representative(pid);
                Some(pid)
            });

            if let Some(person_id) = result {
                known.borrow_mut().add(person_id, name, &embedding);
                advance(&w, &entries, &index, &known);
            }
        }
    });

    window.on_face_delete({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        let db = db.clone();
        move |face_id| {
            if let Ok(g) = db.lock() {
                let _ = g.delete_face_detection(face_id as i64);
            }
            advance(&w, &entries, &index, &known);
        }
    });

    window.on_face_skip({
        let w = window.as_weak();
        let entries = entries.clone();
        let index = index.clone();
        let known = known.clone();
        let db = db.clone();
        move |face_id| {
            if let Ok(g) = db.lock() {
                let _ = g.mark_face_skipped(face_id as i64, true);
            }
            advance(&w, &entries, &index, &known);
        }
    });

    Ok(FaceTagCtx { window, entries, index, known, db })
}

// ── Navigation helpers ─────────────────────────────────────────────

/// Remove the current face entry and show the next, or close when empty.
fn advance(
    w: &slint::Weak<FaceTagWindow>,
    entries: &Rc<RefCell<Vec<FaceEntry>>>,
    index: &Rc<Cell<usize>>,
    known: &Rc<RefCell<EmbeddingMatrix>>,
) {
    let cur = index.get();
    entries.borrow_mut().remove(cur);
    let len = entries.borrow().len();
    if len == 0 {
        if let Some(w) = w.upgrade() {
            let _ = w.hide();
        }
        return;
    }
    index.set(cur.min(len - 1));
    load_face(w.clone(), entries, index, known);
}

/// Update window chrome and start async crop decode for the current entry.
fn load_face(
    w: slint::Weak<FaceTagWindow>,
    entries: &Rc<RefCell<Vec<FaceEntry>>>,
    index: &Rc<Cell<usize>>,
    known: &Rc<RefCell<EmbeddingMatrix>>,
) {
    let (n, i, face_id, bbox, embedding, path) = {
        let ents = entries.borrow();
        let i = index.get();
        let Some(e) = ents.get(i) else { return };
        (ents.len(), i, e.face.id, e.face.bbox, e.face.embedding.clone(), e.path.clone())
    };

    if let Some(win) = w.upgrade() {
        win.set_progress_text(format!("Face {} / {}", i + 1, n).into());
        win.set_face_panel_id(face_id as i32);
        win.set_face_name_entry(SharedString::new());
        win.set_face_suggestions(build_suggestions(&embedding, &known.borrow()));
        win.set_loading(true);
    }

    std::thread::spawn(move || {
        let result = extract_crop(&path, bbox, CROP_PX);
        let _ = w.upgrade_in_event_loop(move |win| {
            if let Ok(pixels) = result {
                let mut pb = SharedPixelBuffer::<Rgb8Pixel>::new(CROP_PX, CROP_PX);
                pb.make_mut_bytes().copy_from_slice(&pixels);
                win.set_face_crop(Image::from_rgb8(pb));
            }
            win.set_loading(false);
        });
    });
}

// ── Data loading ───────────────────────────────────────────────────

/// Collect every real untagged non-skipped face across all present images.
fn collect_untagged_faces(db: &Arc<Mutex<maple_db::Database>>) -> Vec<FaceEntry> {
    let Ok(guard) = db.lock() else { return vec![] };
    let image_ids = guard.images_with_untagged_faces().unwrap_or_default();
    let mut out = Vec::new();
    for image_id in image_ids {
        let Some(img) = guard.image_by_id(image_id).ok().flatten() else { continue };
        for face in guard.faces_for_image(image_id).unwrap_or_default() {
            if face.person_id.is_none() && !face.skipped && is_real_detection(&face) {
                out.push(FaceEntry { path: img.path.clone(), face });
            }
        }
    }
    out
}
