//! End-to-end check of the "already imported" predicate the import scan
//! worker evaluates per photo (`maple-ui/src/import.rs`): real files → real
//! BLAKE3 content hashes → the record on the medium → `SeenSet::contains`.
//!
//! This covers the cross-crate wiring, which the unit tests in `maple-state`
//! cannot: they use synthetic hashes and never touch the named library file.
//! It is *not* the exactness regression test — at these set sizes the bloom's
//! false-positive rate is only ~0.06 %, so it would not reliably catch a
//! bloom-only `contains`. The deterministic guards for that live in
//! `maple-state/src/seen.rs` (`exact_under_total_bloom_saturation`,
//! `exactness_survives_roundtrip`).

use maple_state::{Record, SeenSet};

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
    let mut set = SeenSet::load_replica(library.path(), Record::Imported);
    for hash in &imported {
        set.insert(hash);
    }
    set.save_replica(library.path(), Record::Imported).unwrap();

    // A fresh scan of a folder holding all of those plus 2000 brand-new ones.
    let fresh = make_files(source.path(), "new", 2000);
    let set = SeenSet::load_replica(library.path(), Record::Imported);
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
    let mut set = SeenSet::load_replica(library.path(), Record::Imported);
    for hash in hashes.iter().chain(hashes.iter()) {
        set.insert(hash);
    }
    set.save_replica(library.path(), Record::Imported).unwrap();

    assert_eq!(SeenSet::load_replica(library.path(), Record::Imported).len(), 50);
}

/// The point of keeping the record on the medium: a card imported on one
/// machine must not look untouched when it is plugged into another.
#[test]
fn a_card_carries_its_own_record_to_a_second_machine() {
    let card = tempfile::tempdir().unwrap();
    let first_machine = tempfile::tempdir().unwrap();
    let second_machine = tempfile::tempdir().unwrap();

    let hashes = make_files(card.path(), "dcim", 300);

    // Machine one imports the card.
    let mut imported = SeenSet::new();
    for hash in &hashes {
        imported.insert(hash);
    }
    assert!(
        imported
            .merge_save_to_source(card.path(), first_machine.path(), Record::Imported)
            .unwrap(),
        "a writable card should have taken the record"
    );

    // Machine two has never seen any of this.
    assert!(SeenSet::load_replica(second_machine.path(), Record::Imported).is_empty());

    // …but the card tells it.
    let seen = SeenSet::load_for_source(card.path(), second_machine.path(), Record::Imported);
    assert_eq!(hashes.iter().filter(|h| seen.contains(h)).count(), 300);

    // And a *different* card is still all new on machine one.
    let other_card = tempfile::tempdir().unwrap();
    let fresh = make_files(other_card.path(), "other", 50);
    let seen = SeenSet::load_for_source(other_card.path(), first_machine.path(), Record::Imported);
    assert_eq!(
        fresh.iter().filter(|h| seen.contains(h)).count(),
        0,
        "photos from a card never imported were badged as imported"
    );
}

/// Two cards imported on the same machine both leave a trace in the library
/// replica, which is what a card with no record of its own falls back to.
#[test]
fn the_library_replica_accumulates_across_cards() {
    let library = tempfile::tempdir().unwrap();
    let card_a = tempfile::tempdir().unwrap();
    let card_b = tempfile::tempdir().unwrap();

    let from_a = make_files(card_a.path(), "a", 20);
    let from_b = make_files(card_b.path(), "b", 20);

    for (card, hashes) in [(&card_a, &from_a), (&card_b, &from_b)] {
        let mut set = SeenSet::new();
        for hash in hashes.iter() {
            set.insert(hash);
        }
        set.merge_save_to_source(card.path(), library.path(), Record::Imported).unwrap();
    }

    let replica = SeenSet::load_replica(library.path(), Record::Imported);
    assert_eq!(replica.len(), 40, "the second card overwrote the first");
    assert!(from_a.iter().chain(from_b.iter()).all(|h| replica.contains(h)));
}
