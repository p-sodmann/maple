//! People-page controller.
//!
//! Loads all named persons from the DB and renders their representative face
//! crops asynchronously.  `load` is always called on the Slint event-loop
//! thread (from an `on_people_page_shown` callback), so it can upgrade the
//! Weak reference directly.  Background crop threads send only raw pixel bytes
//! back via `upgrade_in_event_loop` and rebuild the `Image` on arrival.
//!
//! Rendered crops are cached in memory for the life of the app (keyed by
//! person id, invalidated when the representative face id changes) so that
//! revisiting the People page doesn't re-decode every representative photo
//! from disk each time.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use slint::{Image, Model, ModelRc, Rgb8Pixel, SharedPixelBuffer, VecModel};
use std::rc::Rc;

use crate::face_crop::extract_crop;
use crate::services::people as people_service;
use crate::{AppWindow, PersonItem};

const CROP_PX: u32 = 240;

/// Cached crop pixels for one person, tagged with the representative face id
/// they were rendered from so a change in representative invalidates them.
/// `Arc<Mutex<_>>` (rather than `Rc<RefCell<_>>`) because it's written to
/// from inside `upgrade_in_event_loop`'s `Send`-bounded closure.
type CropCache = Arc<Mutex<HashMap<i64, (i64, Arc<Vec<u8>>)>>>;

#[derive(Clone)]
pub struct PeoplePage {
    db: Arc<Mutex<maple_db::Database>>,
    cache: CropCache,
}

impl PeoplePage {
    pub fn new(db: Arc<Mutex<maple_db::Database>>) -> Self {
        Self { db, cache: Arc::new(Mutex::new(HashMap::new())) }
    }

    /// Initial empty model bound before the first page-shown event.
    pub fn model(&self) -> ModelRc<PersonItem> {
        ModelRc::from(Rc::new(VecModel::<PersonItem>::default()))
    }

    /// Reload persons and kick off async crop renders.
    ///
    /// Must be called from the Slint event-loop thread.
    pub fn load(&self, window: slint::Weak<AppWindow>) {
        let persons = people_service::load_all_persons(&self.db);

        // Build placeholder items, filling in any already-cached crops
        // synchronously so revisiting the page doesn't flash blank tiles.
        // We are on the UI thread, so upgrade directly.
        let items: Vec<PersonItem> = persons
            .iter()
            .map(|p| {
                let cached = self.cache.lock().ok().and_then(|c| {
                    c.get(&p.id).and_then(|(face_id, pixels)| {
                        (Some(*face_id) == p.face_id).then(|| pixels.clone())
                    })
                });
                match cached {
                    Some(pixels) => PersonItem {
                        person_id: p.id as i32,
                        name: p.name.clone().into(),
                        face_image: image_from_rgb(&pixels),
                        face_loaded: true,
                    },
                    None => PersonItem {
                        person_id: p.id as i32,
                        name: p.name.clone().into(),
                        face_image: Default::default(),
                        face_loaded: false,
                    },
                }
            })
            .collect();

        if let Some(w) = window.upgrade() {
            let model = ModelRc::from(Rc::new(VecModel::from(items)));
            w.set_people_items(model);
        }

        // Spawn one thread per person whose representative crop isn't
        // already cached. The thread produces raw RGB pixels (Send), then
        // re-enters the event loop to cache them and insert the Image into
        // the model row.
        for (idx, person) in persons.into_iter().enumerate() {
            let (Some(path), Some(bbox), Some(face_id)) =
                (person.image_path, person.bbox, person.face_id)
            else {
                continue;
            };
            let already_cached = self.cache.lock().ok().is_some_and(|c| {
                c.get(&person.id).is_some_and(|(cached_face_id, _)| *cached_face_id == face_id)
            });
            if already_cached {
                continue;
            }

            let w = window.clone();
            let expected_pid = person.id as i32;
            let person_id = person.id;
            let cache = self.cache.clone();
            std::thread::spawn(move || {
                let Ok(pixels) = extract_crop(&path, bbox, CROP_PX) else {
                    return;
                };
                let _ = w.upgrade_in_event_loop(move |win| {
                    let pixels = Arc::new(pixels);
                    if let Ok(mut c) = cache.lock() {
                        c.insert(person_id, (face_id, pixels.clone()));
                    }
                    let model = win.get_people_items();
                    if let Some(mut item) = model.row_data(idx) {
                        if item.person_id == expected_pid {
                            item.face_image = image_from_rgb(&pixels);
                            item.face_loaded = true;
                            model.set_row_data(idx, item);
                        }
                    }
                });
            });
        }
    }

    pub fn untagged_count(&self) -> usize {
        people_service::load_untagged_face_count(&self.db)
    }
}

/// Build a Slint `Image` from a tightly-packed `CROP_PX × CROP_PX` RGB buffer.
fn image_from_rgb(pixels: &[u8]) -> Image {
    let mut pb = SharedPixelBuffer::<Rgb8Pixel>::new(CROP_PX, CROP_PX);
    pb.make_mut_bytes().copy_from_slice(pixels);
    Image::from_rgb8(pb)
}
