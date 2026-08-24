//! Post-copy DB insertion for the import flow.

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use maple_db::lock_db;

/// One copied file, ready to insert into the library DB. Deliberately
/// excludes UI-only fields (decoded thumbnail, selection state) carried on
/// the import window's own per-entry struct.
pub struct ImportEntry {
    /// Where the display file was copied to — a path inside the library, not
    /// the source path it was scanned under.
    pub path: PathBuf,
    /// Where this photo's companion RAW landed, if one was copied alongside
    /// the display file. `None` when the group had no raw, or the copy mode
    /// excluded it.
    pub raw_path: Option<PathBuf>,
    pub content_hash: [u8; 32],
    /// DINOv2 embedding computed during the SD-card scan, if stack detection
    /// was enabled and inference succeeded. Transferring it here means the
    /// background hasher never needs to recompute it for this image.
    pub embedding: Option<Vec<f32>>,
    /// Collections this photo should join on arrival — the import tags that
    /// were on the brush when the user marked it. Empty is the normal case.
    pub collections: Vec<i64>,
}

/// Insert freshly-copied files into the library DB. Best-effort per entry —
/// a metadata-read or insert failure for one file doesn't stop the rest.
///
/// `algorithm_key` identifies the embedding model (`StackSettings::algorithm_key`,
/// e.g. `"onnx:onnx-community/dinov2-small"`) under which any `embedding` on
/// an entry was computed.
pub fn insert_imported_images(
    db: &Arc<Mutex<maple_db::Database>>,
    entries: &[ImportEntry],
    algorithm_key: &str,
) {
    let guard = lock_db(db);
    for e in entries {
        let Ok(meta) = e.path.metadata() else { continue };
        if let Err(err) = guard.insert_image_with_raw(
            &e.path,
            &e.content_hash,
            meta.len(),
            e.raw_path.as_deref(),
        ) {
            tracing::warn!("insert_imported_images {}: {err}", e.path.display());
            continue;
        }
        // The row id is needed by both the embedding and the tags, so look
        // it up once — and not at all for the common entry that has neither.
        if e.embedding.is_none() && e.collections.is_empty() {
            continue;
        }
        let image_id = match guard.image_id_for_path(&e.path) {
            Ok(Some(id)) => id,
            Ok(None) => {
                tracing::warn!(
                    "insert_imported_images: no row found for just-inserted {}",
                    e.path.display()
                );
                continue;
            }
            Err(err) => {
                tracing::warn!(
                    "insert_imported_images: image_id_for_path failed for {}: {err}",
                    e.path.display()
                );
                continue;
            }
        };
        if let Some(embedding) = &e.embedding {
            let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
            if let Err(err) = guard.insert_image_hash(image_id, algorithm_key, &blob) {
                tracing::warn!(
                    "insert_imported_images: failed to store embedding for {}: {err}",
                    e.path.display()
                );
            }
        }
        // Best-effort per tag, like everything else here: a collection the
        // user deleted between marking the photo and copying it must not
        // cost them the import.
        for &collection_id in &e.collections {
            if let Err(err) = guard.add_image_to_collection(collection_id, image_id) {
                tracing::warn!(
                    "insert_imported_images: failed to tag {} with collection {collection_id}: {err}",
                    e.path.display()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use maple_db::Database;

    const KEY: &str = "onnx:test";

    /// A library DB plus a real file on disk to import into it —
    /// `insert_imported_images` reads the file's `metadata()`, so a path that
    /// does not exist is skipped entirely and every assertion below would
    /// pass vacuously.
    fn library(seed: impl FnOnce(&Database)) -> (Arc<Mutex<Database>>, tempfile::TempDir) {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = Database::open(&dir.path().join("library.db")).expect("open db");
        seed(&db);
        (Arc::new(Mutex::new(db)), dir)
    }

    fn photo(dir: &std::path::Path, name: &str) -> PathBuf {
        let path = dir.join(name);
        std::fs::write(&path, format!("pixels for {name}")).expect("write photo");
        path
    }

    fn entry(path: PathBuf, collections: Vec<i64>) -> ImportEntry {
        ImportEntry {
            path,
            raw_path: None,
            content_hash: [7; 32],
            embedding: None,
            collections,
        }
    }

    #[test]
    fn an_imported_photo_joins_the_collections_it_was_marked_with() {
        let mut holiday = 0;
        let (db, dir) = library(|db| {
            holiday = db.create_collection("Holiday", "#B5543E", None).expect("create");
        });
        let path = photo(dir.path(), "a.jpg");

        insert_imported_images(&db, &[entry(path.clone(), vec![holiday])], KEY);

        let guard = lock_db(&db);
        let id = guard.image_id_for_path(&path).expect("lookup").expect("row inserted");
        let joined: Vec<String> = guard
            .collections_for_image(id)
            .expect("collections")
            .into_iter()
            .map(|c| c.name)
            .collect();
        assert_eq!(joined, vec!["Holiday".to_string()]);
    }

    #[test]
    fn an_untagged_import_joins_nothing() {
        let (db, dir) = library(|db| {
            db.create_collection("Holiday", "#B5543E", None).expect("create");
        });
        let path = photo(dir.path(), "b.jpg");

        insert_imported_images(&db, &[entry(path.clone(), vec![])], KEY);

        let guard = lock_db(&db);
        let id = guard.image_id_for_path(&path).expect("lookup").expect("row inserted");
        assert!(guard.collections_for_image(id).expect("collections").is_empty());
    }

    #[test]
    fn a_tag_deleted_between_marking_and_copying_does_not_cost_the_import() {
        // The brush records ids at mark time; the user can delete that
        // collection before hitting Copy. Tagging is best-effort like
        // everything else here — the photo still has to land in the library.
        let (db, dir) = library(|_| {});
        let path = photo(dir.path(), "c.jpg");

        insert_imported_images(&db, &[entry(path.clone(), vec![9999])], KEY);

        let guard = lock_db(&db);
        let id = guard.image_id_for_path(&path).expect("lookup").expect("row inserted");
        assert!(guard.collections_for_image(id).expect("collections").is_empty());
    }

    #[test]
    fn two_photos_marked_under_different_brushes_land_in_different_collections() {
        let (mut holiday, mut portraits) = (0, 0);
        let (db, dir) = library(|db| {
            holiday = db.create_collection("Holiday", "#B5543E", None).expect("create");
            portraits = db.create_collection("Portraits", "#3388FF", None).expect("create");
        });
        let first = photo(dir.path(), "d.jpg");
        let second = photo(dir.path(), "e.jpg");

        insert_imported_images(
            &db,
            &[
                entry(first.clone(), vec![holiday]),
                entry(second.clone(), vec![portraits]),
            ],
            KEY,
        );

        let guard = lock_db(&db);
        let names = |path: &PathBuf| -> Vec<String> {
            let id = guard.image_id_for_path(path).expect("lookup").expect("row inserted");
            guard
                .collections_for_image(id)
                .expect("collections")
                .into_iter()
                .map(|c| c.name)
                .collect()
        };
        assert_eq!(names(&first), vec!["Holiday".to_string()]);
        assert_eq!(names(&second), vec!["Portraits".to_string()]);
    }
}
