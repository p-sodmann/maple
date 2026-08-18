//! End-to-end check of the "already imported" predicate the import scan
//! worker evaluates per photo (`maple-ui/src/import.rs`): real files → real
//! BLAKE3 content hashes → `seen_imported.bin` → `SeenSet::contains`.
//!
//! This covers the cross-crate wiring, which the unit tests in `maple-state`
//! cannot: they use synthetic hashes and never touch the named library file.
//! It is *not* the exactness regression test — at these set sizes the bloom's
//! false-positive rate is only ~0.06 %, so it would not reliably catch a
//! bloom-only `contains`. The deterministic guards for that live in
//! `maple-state/src/seen.rs` (`exact_under_total_bloom_saturation`,
//! `exactness_survives_roundtrip`).

use maple_state::SeenSet;

/// Write `count` files with distinct contents and return their content hashes.
fn make_files(dir: &std::path::Path, prefix: &str, count: usize) -> Vec<[u8; 32]> {
    (0..count)
        .map(|i| {
            let path = dir.join(format!("{prefix}_{i}.jpg"));
            std::fs::write(&path, format!("{prefix} image payload {i}")).unwrap();
            maple_import::content_hash(&path).unwrap()
        })
        .collect()
}

#[test]
fn only_already_imported_photos_report_as_seen() {
    let library = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();

    // A library that has already absorbed 2000 photos.
    let imported = make_files(source.path(), "old", 2000);
    let mut set = SeenSet::load_imported(library.path());
    for hash in &imported {
        set.insert(hash);
    }
    set.save_imported(library.path()).unwrap();

    // A fresh scan of a folder holding all of those plus 2000 brand-new ones.
    let fresh = make_files(source.path(), "new", 2000);
    let set = SeenSet::load_imported(library.path());
    assert_eq!(set.len(), 2000);

    let flagged_old = imported.iter().filter(|h| set.contains(h)).count();
    let flagged_new = fresh.iter().filter(|h| set.contains(h)).count();

    assert_eq!(flagged_old, 2000, "previously imported photos lost their badge");
    assert_eq!(flagged_new, 0, "never-imported photos were badged as imported");
}

#[test]
fn rescanning_the_same_folder_does_not_inflate_the_set() {
    let library = tempfile::tempdir().unwrap();
    let source = tempfile::tempdir().unwrap();
    let hashes = make_files(source.path(), "photo", 50);

    // Import, then import the very same files again.
    let mut set = SeenSet::load_imported(library.path());
    for hash in hashes.iter().chain(hashes.iter()) {
        set.insert(hash);
    }
    set.save_imported(library.path()).unwrap();

    assert_eq!(SeenSet::load_imported(library.path()).len(), 50);
}
