//! Service layer — DB queries + data-shape composition, no Slint `Weak`/
//! `ModelRc`/callback types. Window controllers call into these instead of
//! locking the DB inline; see the phased extraction plan in project memory.

pub mod collections;
pub mod faces;
pub mod images;
pub mod import;
pub mod people;
pub mod restructure;
pub mod settings;

#[cfg(test)]
mod poison_tests {
    use std::sync::{Arc, Mutex};

    use maple_db::{Database, SearchQuery};

    /// Open a throwaway library DB, seed it via `seed`, then poison its mutex
    /// the way a background worker would: a thread panics while holding the
    /// guard. Returns the handle plus the `TempDir` that must outlive it.
    ///
    /// Seeding happens *before* poisoning so the read tests assert against
    /// real rows — against the old `db.lock().ok()` code an empty DB would
    /// have made "returns empty" pass for the wrong reason.
    fn poisoned_db(seed: impl FnOnce(&Database)) -> (Arc<Mutex<Database>>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open db");
        seed(&db);
        let db = Arc::new(Mutex::new(db));

        // A panic while the guard is live sets the poison flag on unwind.
        // `join` catches it, so the test process survives — exactly what
        // happens when a `maple-db` worker thread dies mid-query.
        let handle = {
            let db = Arc::clone(&db);
            std::thread::spawn(move || {
                let _guard = db.lock().expect("first lock cannot be poisoned");
                panic!("simulated worker panic while holding the DB lock");
            })
        };
        assert!(handle.join().is_err(), "worker thread should have panicked");

        // Guard the guard: if this ever stops holding, the tests below would
        // pass vacuously against a healthy mutex.
        assert!(db.lock().is_err(), "mutex should now be poisoned");

        (db, dir)
    }

    #[test]
    fn reads_still_work_through_a_poisoned_mutex() {
        let dir = tempfile::tempdir().expect("tempdir");
        let img = dir.path().join("photo.jpg");
        std::fs::write(&img, b"not-really-a-jpeg").expect("write seed file");

        let (db, _dir) = poisoned_db(|d| {
            d.insert_image_with_raw(&img, &[7u8; 32], 17, None).expect("seed image");
            d.create_collection("Trip", "#336699", None).expect("seed collection");
        });

        // The symptom this guards against: the library rendering as empty
        // forever after one unrelated worker panic. These rows exist — a
        // poison-swallowing read would return nothing and look identical to
        // an empty library.
        assert_eq!(super::images::search_library(&db, &SearchQuery::default()).len(), 1);
        assert_eq!(super::collections::load_entries(&db).len(), 1);
        assert_eq!(super::collections::load_all_collections(&db).len(), 1);
    }

    #[test]
    fn writes_still_land_through_a_poisoned_mutex() {
        let (db, _dir) = poisoned_db(|_| {});

        assert!(
            super::collections::create_collection(&db, "Trip", "#336699", None),
            "create_collection must not silently no-op on a poisoned mutex"
        );

        // Read it back through a second, independent lock acquisition — this
        // is what proves the write reached SQLite rather than being dropped.
        let entries = super::collections::load_entries(&db);
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].name.as_str(), "Trip");

        let id = entries[0].id as i64;
        assert!(super::collections::rename_collection(&db, id, "Trip 2024"));
        assert_eq!(super::collections::load_entries(&db)[0].name.as_str(), "Trip 2024");

        // An Option-returning write path (Settings debug actions): `Some`
        // means the statement ran; `None` would mean the poison swallowed it.
        assert!(super::settings::clear_face_data(&db).is_some());
    }
}
