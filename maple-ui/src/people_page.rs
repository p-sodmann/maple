//! People-page controller.
//!
//! Loads all named persons from the DB and renders their representative face
//! crops asynchronously.  `load` is always called on the Slint event-loop
//! thread (from an `on_people_page_shown` callback), so it can upgrade the
//! Weak reference directly.  Background crop threads send only raw pixel bytes
//! back via `upgrade_in_event_loop` and rebuild the `Image` on arrival.
//!
//! Crops are persisted in the redb thumbnail cache (`ThumbnailCache::
//! get_face_crop`/`insert_face_crop`, see [`crate::rep_crop`]) so they
//! survive app restarts, and additionally memoized in memory for the life of
//! the app (keyed by person id, invalidated when the representative face id
//! changes) so revisiting the People page doesn't even pay the WebP decode
//! cost each time.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, Model, ModelRc, VecModel};
use std::rc::Rc;

use crate::rep_crop::{self, CropCache};
use crate::services::people as people_service;
use crate::transforms::trimmed_name;
use crate::{face_tag, services, AppCtx, AppWindow, PersonItem};

use maple_db::SearchQuery;

const CROP_PX: u32 = 240;

/// Build the page's model and wire its callbacks.
///
/// Called from `lib.rs` during startup — see [`AppCtx`] for the shared
/// handles the closures clone out of.
pub fn wire(window: &AppWindow, ctx: &AppCtx) {
    let people = PeoplePage::new(ctx.db.clone(), ctx.cache.clone(), ctx.thumb_quality);
    window.set_people_items(people.model());

    window.on_people_page_shown({
        let people = people.clone();
        let w = ctx.window.clone();
        move || {
            let w2 = w.clone();
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
            }
            people.load(w2);
        }
    });

    window.on_people_person_activated({
        let grid = ctx.grid.clone();
        let w = ctx.window.clone();
        let current_query = ctx.current_query.clone();
        let select_target = ctx.select_target.clone();
        move |person_id, name| {
            let q = SearchQuery::default().with_person(person_id as i64);
            *current_query.borrow_mut() = q.clone();
            grid.load(q);
            select_target.set(None);
            if let Some(win) = w.upgrade() {
                win.set_library_filter_name(name);
                win.set_library_active_collection_id(-1);
                win.set_page(crate::Page::Library);
            }
        }
    });

    window.on_people_edit_save({
        let db = ctx.db.clone();
        let people = people.clone();
        let w = ctx.window.clone();
        move |person_id, name| {
            let Some(name) = trimmed_name(&name) else { return };
            services::faces::rename_person(&db, person_id as i64, &name);
            if let Some(win) = w.upgrade() {
                people.load(win.as_weak());
            }
        }
    });

    window.on_people_edit_delete({
        let db = ctx.db.clone();
        let cache = ctx.cache.clone();
        let people = people.clone();
        let w = ctx.window.clone();
        move |person_id| {
            services::faces::delete_person(&db, &cache, person_id as i64);
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
                people.load(win.as_weak());
            }
        }
    });

    window.on_people_tag_faces({
        let db = ctx.db.clone();
        let people = people.clone();
        let w = ctx.window.clone();
        move || {
            face_tag::open(db.clone());
            // Refresh untagged count after the wizard is closed (best-effort;
            // the window fires this callback synchronously, so the count
            // updates when the user returns to the People page).
            if let Some(win) = w.upgrade() {
                win.set_people_untagged_count(people.untagged_count() as i32);
            }
        }
    });
}

#[derive(Clone)]
pub struct PeoplePage {
    db: Arc<Mutex<maple_db::Database>>,
    cache: Arc<maple_db::ThumbnailCache>,
    quality: u8,
    crop_cache: CropCache,
}

impl PeoplePage {
    pub fn new(db: Arc<Mutex<maple_db::Database>>, cache: Arc<maple_db::ThumbnailCache>, quality: u8) -> Self {
        Self { db, cache, quality, crop_cache: Arc::new(Mutex::new(HashMap::new())) }
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
                let cached = self.crop_cache.lock().ok().and_then(|c| {
                    c.get(&p.id).and_then(|(face_id, pixels)| {
                        (Some(*face_id) == p.face_id).then(|| pixels.clone())
                    })
                });
                match cached {
                    Some(pixels) => PersonItem {
                        person_id: p.id as i32,
                        name: p.name.clone().into(),
                        face_image: rep_crop::image_from_rgb(&pixels, CROP_PX),
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
            let already_cached = self.crop_cache.lock().ok().is_some_and(|c| {
                c.get(&person.id).is_some_and(|(cached_face_id, _)| *cached_face_id == face_id)
            });
            if already_cached {
                continue;
            }

            let w = window.clone();
            let expected_pid = person.id as i32;
            let person_id = person.id;
            let crop_cache = self.crop_cache.clone();
            let cache = self.cache.clone();
            let quality = self.quality;
            std::thread::spawn(move || {
                let redb_cached = cache.get_face_crop(face_id);
                let cache_for_store = cache.clone();
                let Ok(pixels) = rep_crop::extract_and_cache(
                    &path,
                    Some(bbox),
                    CROP_PX,
                    quality,
                    redb_cached,
                    |webp| {
                        if let Err(e) = cache_for_store.insert_face_crop(face_id, webp) {
                            tracing::warn!("insert_face_crop {face_id}: {e}");
                        }
                    },
                ) else {
                    return;
                };
                let _ = w.upgrade_in_event_loop(move |win| {
                    let pixels = Arc::new(pixels);
                    if let Ok(mut c) = crop_cache.lock() {
                        c.insert(person_id, (face_id, pixels.clone()));
                    }
                    let model = win.get_people_items();
                    if let Some(mut item) = model.row_data(idx) {
                        if item.person_id == expected_pid {
                            item.face_image = rep_crop::image_from_rgb(&pixels, CROP_PX);
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
