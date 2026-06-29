//! People-page controller.
//!
//! Loads all named persons from the DB and renders their representative face
//! crops asynchronously.  `load` is always called on the Slint event-loop
//! thread (from an `on_people_page_shown` callback), so it can upgrade the
//! Weak reference directly.  Background crop threads send only raw pixel bytes
//! back via `upgrade_in_event_loop` and rebuild the `Image` on arrival.

use std::sync::{Arc, Mutex};

use slint::{Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, VecModel};
use std::rc::Rc;

use crate::face_tag::extract_crop;
use crate::{AppWindow, PersonItem};

const CROP_PX: u32 = 240;

#[derive(Clone)]
pub struct PeoplePage {
    db: Arc<Mutex<maple_db::Database>>,
}

impl PeoplePage {
    pub fn new(db: Arc<Mutex<maple_db::Database>>) -> Self {
        Self { db }
    }

    /// Initial empty model bound before the first page-shown event.
    pub fn model(&self) -> ModelRc<PersonItem> {
        ModelRc::from(Rc::new(VecModel::<PersonItem>::default()))
    }

    /// Reload persons and kick off async crop renders.
    ///
    /// Must be called from the Slint event-loop thread.
    pub fn load(&self, window: slint::Weak<AppWindow>) {
        let persons = {
            let Ok(g) = self.db.lock() else { return };
            g.all_persons_with_representatives().unwrap_or_default()
        };

        // Build placeholder items (no image yet) and push to the window.
        // We are on the UI thread, so upgrade directly.
        let items: Vec<PersonItem> = persons
            .iter()
            .map(|p| PersonItem {
                person_id: p.id as i32,
                name: p.name.clone().into(),
                face_image: Default::default(),
                face_loaded: false,
            })
            .collect();

        if let Some(w) = window.upgrade() {
            let model = ModelRc::from(Rc::new(VecModel::from(items)));
            w.set_people_items(model);
        }

        // Spawn one thread per person that has a representative face.
        // The thread produces raw RGB pixels (Send), then re-enters the
        // event loop to insert the Image into the model row.
        for (idx, person) in persons.into_iter().enumerate() {
            let (Some(path), Some(bbox)) = (person.image_path, person.bbox) else {
                continue;
            };
            let w = window.clone();
            let expected_pid = person.id as i32;
            std::thread::spawn(move || {
                let Ok(pixels) = extract_crop(&path, bbox, CROP_PX) else {
                    return;
                };
                let _ = w.upgrade_in_event_loop(move |win| {
                    let model = win.get_people_items();
                    if let Some(mut item) = model.row_data(idx) {
                        if item.person_id == expected_pid {
                            let mut pb =
                                SharedPixelBuffer::<Rgb8Pixel>::new(CROP_PX, CROP_PX);
                            pb.make_mut_bytes().copy_from_slice(&pixels);
                            item.face_image = Image::from_rgb8(pb);
                            item.face_loaded = true;
                            model.set_row_data(idx, item);
                        }
                    }
                });
            });
        }
    }

    pub fn untagged_count(&self) -> usize {
        self.db
            .lock()
            .ok()
            .and_then(|g| g.untagged_face_count().ok())
            .unwrap_or(0)
    }
}
