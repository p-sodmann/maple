//! Merge-engine tests.
//!
//! Every test drives *two* real databases and syncs between them, because the
//! properties that matter here — convergence, and agreeing on a winner
//! without coordinating — are only observable with two independent clocks and
//! two independent device ids.

use std::path::PathBuf;

use super::wire::{Entity, Stamp, SyncBatch};
use crate::sync::collect::{DEFAULT_MAX_REVS, MAX_REVS_LIMIT};
use crate::Database;

// ── Harness ─────────────────────────────────────────────────────

/// Two libraries standing in for a workstation and a laptop.
struct Pair {
    _dir: tempfile::TempDir,
    a: Database,
    b: Database,
    /// Watermarks: how far each side has consumed of the other.
    a_seen: i64,
    b_seen: i64,
}

impl Pair {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        let a = Database::open(&dir.path().join("a.db")).unwrap();
        let b = Database::open(&dir.path().join("b.db")).unwrap();
        assert_ne!(a.device_id(), b.device_id(), "each install is its own device");
        Self { _dir: dir, a, b, a_seen: 0, b_seen: 0 }
    }

    /// Push everything new from A into B.
    fn a_to_b(&mut self) -> crate::ApplyReport {
        let batch = self.a.collect_changes(self.b_seen, DEFAULT_MAX_REVS).unwrap();
        let report = self.b.apply_batch(&batch).unwrap();
        self.b_seen = batch.next_rev;
        report
    }

    fn b_to_a(&mut self) -> crate::ApplyReport {
        let batch = self.b.collect_changes(self.a_seen, DEFAULT_MAX_REVS).unwrap();
        let report = self.a.apply_batch(&batch).unwrap();
        self.a_seen = batch.next_rev;
        report
    }

    /// Exchange in both directions until neither side has anything left.
    fn converge(&mut self) {
        for _ in 0..8 {
            let x = self.a_to_b();
            let y = self.b_to_a();
            if !x.changed() && !y.changed() {
                return;
            }
        }
        panic!("sync did not settle — the two sides are trading edits forever");
    }
}

fn hash(seed: u8) -> [u8; 32] {
    [seed; 32]
}

fn add_photo(db: &Database, name: &str, seed: u8) -> i64 {
    let path = PathBuf::from(format!("/photos/{name}"));
    db.insert_image(&path, &hash(seed), 1024).unwrap();
    db.image_id_for_path(&path).unwrap().unwrap()
}

fn collection_names(db: &Database) -> Vec<String> {
    let mut stmt = db
        .conn
        .prepare("SELECT name FROM collections ORDER BY name")
        .unwrap();
    let names = stmt
        .query_map([], |r| r.get::<_, String>(0))
        .unwrap()
        .filter_map(|r| r.ok())
        .collect();
    names
}

fn person_name(db: &Database, guid: &str) -> Option<String> {
    db.conn
        .query_row(
            "SELECT name FROM persons WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get(0),
        )
        .ok()
}

fn guid_of(db: &Database, table: &str, id: i64) -> String {
    db.conn
        .query_row(
            &format!("SELECT guid FROM {table} WHERE id = ?1"),
            rusqlite::params![id],
            |r| r.get(0),
        )
        .unwrap()
}

fn count(db: &Database, table: &str) -> i64 {
    db.conn
        .query_row(&format!("SELECT COUNT(*) FROM {table}"), [], |r| r.get(0))
        .unwrap()
}

// ── Basic replication ───────────────────────────────────────────

#[test]
fn a_collection_created_on_one_device_appears_on_the_other() {
    let mut p = Pair::new();
    p.a.create_collection("Iceland", "#3584e4", None).unwrap();

    p.converge();

    assert_eq!(collection_names(&p.b), vec!["Iceland"]);
}

#[test]
fn nothing_to_sync_produces_an_empty_batch() {
    let p = Pair::new();
    let batch = p.a.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    // A fresh library has no rows at all, so the watermark must not move.
    assert!(batch.is_empty());
    assert_eq!(batch.next_rev, 0);
}

#[test]
fn applying_the_same_batch_twice_changes_nothing_the_second_time() {
    let p = Pair::new();
    p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    add_photo(&p.a, "a.jpg", 1);

    let batch = p.a.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    let first = p.b.apply_batch(&batch).unwrap();
    let second = p.b.apply_batch(&batch).unwrap();

    assert!(first.changed());
    // Re-delivery happens routinely — a dropped connection re-sends from the
    // last acknowledged watermark — so it has to be a no-op, not a duplicate.
    assert!(!second.changed());
    assert_eq!(second.stale, batch.rows.len());
    assert_eq!(count(&p.b, "collections"), 1);
    assert_eq!(count(&p.b, "images"), 1);
}

#[test]
fn a_received_row_is_not_echoed_back() {
    let mut p = Pair::new();
    p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.converge();

    // B stamped nothing of its own, so a pull from B must be empty. If apply
    // re-stamped received rows with B's clock, this would loop forever.
    let back = p.b.collect_changes(p.a_seen, DEFAULT_MAX_REVS).unwrap();
    assert!(back.rows.is_empty(), "B echoed rows it merely received");
}

// ── Conflict resolution ─────────────────────────────────────────

#[test]
fn concurrent_edits_converge_on_one_winner() {
    let mut p = Pair::new();
    let pid = p.a.upsert_person("Ada").unwrap();
    let guid = guid_of(&p.a, "persons", pid);
    p.converge();

    // Both rename the same person while disconnected.
    p.a.rename_person(pid, "Ada Lovelace").unwrap();
    let b_id = p
        .b
        .conn
        .query_row(
            "SELECT id FROM persons WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    p.b.rename_person(b_id, "A. Lovelace").unwrap();

    p.converge();

    let a_name = person_name(&p.a, &guid).unwrap();
    let b_name = person_name(&p.b, &guid).unwrap();
    assert_eq!(a_name, b_name, "both sides must land on the same name");
    assert!(["Ada Lovelace", "A. Lovelace"].contains(&a_name.as_str()));
}

#[test]
fn a_newer_edit_beats_an_older_one_regardless_of_sync_direction() {
    let mut p = Pair::new();
    let pid = p.a.upsert_person("Ada").unwrap();
    let guid = guid_of(&p.a, "persons", pid);
    p.converge();

    let b_id = p
        .b
        .conn
        .query_row(
            "SELECT id FROM persons WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();

    // B edits, syncs, then A edits strictly afterwards.
    p.b.rename_person(b_id, "First").unwrap();
    p.converge();
    p.a.rename_person(pid, "Second").unwrap();
    p.converge();

    assert_eq!(person_name(&p.a, &guid).as_deref(), Some("Second"));
    assert_eq!(person_name(&p.b, &guid).as_deref(), Some("Second"));
}

#[test]
fn a_stamp_from_the_future_still_loses_to_a_later_local_edit() {
    let mut p = Pair::new();
    let pid = p.a.upsert_person("Ada").unwrap();
    let guid = guid_of(&p.a, "persons", pid);

    // Pretend A's clock is a year fast.
    let year = 365 * 24 * 60 * 60 * 1000i64;
    p.a.conn
        .execute(
            "UPDATE persons SET rev = rev + ?1 WHERE id = ?2",
            rusqlite::params![year, pid],
        )
        .unwrap();
    p.converge();

    // B's own clock is normal, but observing A's stamp must have dragged it
    // forward — otherwise B could never out-edit A again.
    let b_id = p
        .b
        .conn
        .query_row(
            "SELECT id FROM persons WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();
    p.b.rename_person(b_id, "Later").unwrap();
    p.converge();

    assert_eq!(person_name(&p.a, &guid).as_deref(), Some("Later"));
    assert_eq!(person_name(&p.b, &guid).as_deref(), Some("Later"));
}

// ── Deletes ─────────────────────────────────────────────────────

#[test]
fn a_delete_propagates() {
    let mut p = Pair::new();
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.converge();
    assert_eq!(count(&p.b, "collections"), 1);

    p.a.delete_collection(cid).unwrap();
    p.converge();

    assert_eq!(count(&p.b, "collections"), 0);
}

#[test]
fn a_delete_does_not_come_back_on_the_next_sync() {
    let mut p = Pair::new();
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.converge();
    p.a.delete_collection(cid).unwrap();
    p.converge();

    // The classic failure: B still holds the row, ships it back, and A
    // "restores" what the user deleted. The tombstone is what prevents it.
    p.converge();
    assert_eq!(count(&p.a, "collections"), 0);
    assert_eq!(count(&p.b, "collections"), 0);
}

#[test]
fn an_edit_after_a_delete_resurrects_the_row() {
    let mut p = Pair::new();
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    let guid = guid_of(&p.a, "collections", cid);
    p.converge();

    let b_id = p
        .b
        .conn
        .query_row(
            "SELECT id FROM collections WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();

    // A deletes; B renames strictly afterwards, without having seen the
    // delete. The later intent wins, so the collection comes back.
    //
    // The pause is load-bearing: the clock has millisecond resolution, so two
    // operations this close together would otherwise tie and be settled by
    // the device-id tiebreak. That is correct behaviour for a genuine tie,
    // but it is not the scenario under test — without real separation this
    // asserts nothing about ordering.
    p.a.delete_collection(cid).unwrap();
    std::thread::sleep(std::time::Duration::from_millis(2));
    p.b.rename_collection(b_id, "Iceland 2024").unwrap();

    p.converge();

    assert_eq!(collection_names(&p.a), vec!["Iceland 2024"]);
    assert_eq!(collection_names(&p.b), vec!["Iceland 2024"]);
}

#[test]
fn a_delete_after_an_edit_still_deletes() {
    let mut p = Pair::new();
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    let guid = guid_of(&p.a, "collections", cid);
    p.converge();

    let b_id = p
        .b
        .conn
        .query_row(
            "SELECT id FROM collections WHERE guid = ?1",
            rusqlite::params![guid],
            |r| r.get::<_, i64>(0),
        )
        .unwrap();

    p.b.rename_collection(b_id, "Iceland 2024").unwrap();
    p.b_to_a();
    p.a.delete_collection(cid).unwrap();

    p.converge();

    assert_eq!(count(&p.a, "collections"), 0);
    assert_eq!(count(&p.b, "collections"), 0);
}

#[test]
fn deleting_a_collection_takes_its_memberships_with_it() {
    let mut p = Pair::new();
    let iid = add_photo(&p.a, "a.jpg", 1);
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.a.add_image_to_collection(cid, iid).unwrap();
    p.converge();
    assert_eq!(count(&p.b, "collection_images"), 1);

    p.a.delete_collection(cid).unwrap();
    p.converge();

    // Only the parent is tombstoned; B's own cascade clears its children.
    assert_eq!(count(&p.b, "collection_images"), 0);
}

// ── Image identity ──────────────────────────────────────────────

#[test]
fn the_same_photo_imported_on_both_devices_becomes_one_row() {
    let mut p = Pair::new();
    // Same bytes, imported independently — so two different guids.
    add_photo(&p.a, "IMG_1234.jpg", 7);
    add_photo(&p.b, "IMG_1234.jpg", 7);

    p.converge();

    assert_eq!(count(&p.a, "images"), 1, "A grew a duplicate");
    assert_eq!(count(&p.b, "images"), 1, "B grew a duplicate");
    let a_guid: String = p
        .a
        .conn
        .query_row("SELECT guid FROM images", [], |r| r.get(0))
        .unwrap();
    let b_guid: String = p
        .b
        .conn
        .query_row("SELECT guid FROM images", [], |r| r.get(0))
        .unwrap();
    assert_eq!(a_guid, b_guid, "the two sides must agree on one identity");
}

#[test]
fn different_photos_are_not_unified() {
    let mut p = Pair::new();
    add_photo(&p.a, "a.jpg", 1);
    add_photo(&p.b, "b.jpg", 2);

    p.converge();

    assert_eq!(count(&p.a, "images"), 2);
    assert_eq!(count(&p.b, "images"), 2);
}

// ── Local-only state ────────────────────────────────────────────

#[test]
fn a_file_missing_on_one_device_stays_present_on_the_other() {
    let mut p = Pair::new();
    let path = PathBuf::from("/photos/a.jpg");
    p.a.insert_image(&path, &hash(1), 1024).unwrap();
    p.converge();

    // B's copy of the file goes away. That is a fact about B's disk.
    p.b.mark_missing(&PathBuf::from("/photos/a.jpg")).ok();
    p.converge();

    let a_status: String = p
        .a
        .conn
        .query_row("SELECT status FROM images", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        a_status, "present",
        "B's local disk state must not blank A's library"
    );
}

#[test]
fn a_local_path_is_never_overwritten_by_a_peers_path() {
    let mut p = Pair::new();
    let path = PathBuf::from("/workstation/photos/a.jpg");
    p.a.insert_image(&path, &hash(1), 1024).unwrap();
    p.converge();
    p.b.conn
        .execute(
            "UPDATE images SET path = '/laptop/pics/a.jpg', status = 'present'",
            [],
        )
        .unwrap();

    // Another round of edits from A must not relocate B's copy.
    let a_id = p.a.image_id_for_path(&path).unwrap().unwrap();
    p.a.update_image_hash_and_orientation(a_id, &hash(9), 6).unwrap();
    p.converge();

    let b_path: String = p
        .b
        .conn
        .query_row("SELECT path FROM images", [], |r| r.get(0))
        .unwrap();
    assert_eq!(b_path, "/laptop/pics/a.jpg");
    // ...while the replicated columns did travel.
    let b_orientation: i64 = p
        .b
        .conn
        .query_row("SELECT orientation FROM images", [], |r| r.get(0))
        .unwrap();
    assert_eq!(b_orientation, 6);
}

// ── Relationships ───────────────────────────────────────────────

#[test]
fn faces_and_their_person_links_replicate() {
    let mut p = Pair::new();
    let iid = add_photo(&p.a, "a.jpg", 1);
    let fid = p
        .a
        .insert_face_detection(iid, [0.1, 0.2, 0.3, 0.4], &[0.5f32; 512], 0.93)
        .unwrap();
    let pid = p.a.upsert_person("Ada").unwrap();
    p.a.assign_face_to_person(fid, Some(pid)).unwrap();

    p.converge();

    // The embedding rides along so the laptop needs no ONNX models, and the
    // face keeps its name because the link travels as a guid.
    let (name, conf, bbox_x1, emb_len): (String, f64, f64, i64) = p
        .b
        .conn
        .query_row(
            "SELECT p.name, f.confidence, f.bbox_x1, LENGTH(f.embedding)
             FROM face_detections f JOIN persons p ON p.id = f.person_id",
            [],
            |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?)),
        )
        .unwrap();
    assert_eq!(name, "Ada");
    assert!((conf - 0.93).abs() < 1e-6);
    assert!((bbox_x1 - 0.1).abs() < 1e-6);
    assert_eq!(emb_len, 512 * 4, "512 f32s should survive intact");
}

#[test]
fn a_membership_arriving_before_its_collection_is_retried_not_dropped() {
    let p = Pair::new();
    let iid = add_photo(&p.a, "a.jpg", 1);
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.a.add_image_to_collection(cid, iid).unwrap();

    // Deliver *only* the membership row, with its parents held back.
    let full = p.a.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    let orphan = SyncBatch {
        rows: full
            .rows
            .iter()
            .filter(|r| r.entity() == Entity::CollectionImage)
            .cloned()
            .collect(),
        tombstones: vec![],
        aliases: vec![],
        next_rev: full.next_rev,
    };
    let report = p.b.apply_batch(&orphan).unwrap();
    assert_eq!(report.deferred, 1);
    assert_eq!(count(&p.b, "collection_images"), 0);

    // The real batch carries the parents too, and the row lands.
    p.b.apply_batch(&full).unwrap();
    assert_eq!(count(&p.b, "collection_images"), 1);
}

#[test]
fn a_stack_and_its_members_replicate_despite_the_reference_cycle() {
    let mut p = Pair::new();
    let one = add_photo(&p.a, "a.jpg", 1);
    let two = add_photo(&p.a, "b.jpg", 2);
    let stack = p.a.create_stack().unwrap();
    p.a.set_image_stack(one, Some(stack)).unwrap();
    p.a.set_image_stack(two, Some(stack)).unwrap();
    p.a.set_stack_cover(stack, two).unwrap();

    p.converge();

    // images.stack_id → stacks.cover_image_id → images is a cycle no single
    // insertion order can satisfy; pass 4 is what resolves it.
    let members: i64 = p
        .b
        .conn
        .query_row(
            "SELECT COUNT(*) FROM images WHERE stack_id IS NOT NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(members, 2);

    let cover_hash: Vec<u8> = p
        .b
        .conn
        .query_row(
            "SELECT i.hash FROM stacks s JOIN images i ON i.id = s.cover_image_id",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(cover_hash, hash(2).to_vec(), "the chosen cover must survive");
}

#[test]
fn nested_collections_keep_their_parent() {
    let mut p = Pair::new();
    let parent = p.a.create_collection("Travel", "#3584e4", None).unwrap();
    p.a.create_collection("Iceland", "#e44", Some(parent)).unwrap();

    p.converge();

    let (child, parent_name): (String, String) = p
        .b
        .conn
        .query_row(
            "SELECT c.name, pa.name FROM collections c
             JOIN collections pa ON pa.id = c.parent_id",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert_eq!(child, "Iceland");
    assert_eq!(parent_name, "Travel");
}

// ── Batching ────────────────────────────────────────────────────

#[test]
fn a_batch_never_splits_a_stamp_group() {
    let db = Database::open(&tempfile::tempdir().unwrap().path().join("l.db")).unwrap();
    // `tombstone` stamps every id in one call with a single rev, so this
    // group is indivisible.
    let ids: Vec<i64> = (0..50)
        .map(|i| {
            let c = db
                .create_collection(&format!("c{i}"), "#fff", None)
                .unwrap();
            c
        })
        .collect();
    db.tombstone("collections", &ids).unwrap();

    // Ask for a single stamp's worth of changes, starting just below the
    // tombstone group.
    let before = db
        .conn
        .query_row("SELECT MIN(rev) - 1 FROM sync_tombstones", [], |r| {
            r.get::<_, i64>(0)
        })
        .unwrap();
    let batch = db.collect_changes(before, 1).unwrap();

    // All 50 must arrive together: acknowledging a watermark that sits inside
    // the group would silently lose the rest forever.
    assert_eq!(batch.tombstones.len(), 50);
}

#[test]
fn paging_through_many_batches_loses_nothing() {
    let p = Pair::new();
    for i in 0..40 {
        p.a.create_collection(&format!("c{i:02}"), "#fff", None).unwrap();
    }

    // Deliberately tiny batches to force many round trips.
    let mut seen = 0i64;
    loop {
        let batch = p.a.collect_changes(seen, 3).unwrap();
        if batch.is_empty() && batch.next_rev == seen {
            break;
        }
        p.b.apply_batch(&batch).unwrap();
        seen = batch.next_rev;
    }

    assert_eq!(count(&p.b, "collections"), 40);
}

#[test]
fn a_watermark_only_advances_over_delivered_changes() {
    let p = Pair::new();
    p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    let first = p.a.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    assert!(first.next_rev > 0);

    // Nothing new since — the watermark must hold, not creep forward.
    let second = p.a.collect_changes(first.next_rev, DEFAULT_MAX_REVS).unwrap();
    assert!(second.is_empty());
    assert_eq!(second.next_rev, first.next_rev);
}

// ── Derived data ────────────────────────────────────────────────

#[test]
fn apply_reports_which_derived_values_need_recomputing() {
    let p = Pair::new();
    let iid = add_photo(&p.a, "a.jpg", 1);
    let cid = p.a.create_collection("Iceland", "#3584e4", None).unwrap();
    p.a.add_image_to_collection(cid, iid).unwrap();
    let fid = p
        .a
        .insert_face_detection(iid, [0.1, 0.2, 0.3, 0.4], &[0.5f32; 512], 0.9)
        .unwrap();
    let pid = p.a.upsert_person("Ada").unwrap();
    p.a.assign_face_to_person(fid, Some(pid)).unwrap();

    let batch = p.a.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    let report = p.b.apply_batch(&batch).unwrap();

    // Centroids and representative ids are local rowids and never replicate,
    // so the caller has to refresh them for exactly these rows.
    assert_eq!(report.touched_collections.len(), 1);
    assert_eq!(report.touched_persons.len(), 1);
}

#[test]
fn derived_tables_are_not_replicated() {
    let mut p = Pair::new();
    let iid = add_photo(&p.a, "a.jpg", 1);
    p.a.insert_image_hash(iid, "onnx:test", &[0u8; 16]).unwrap();
    p.a.replace_exif_tags(iid, &[("Make".into(), "Fujifilm".into())])
        .unwrap();

    p.converge();

    // Both are cheap to recompute and are namespaced by model settings that
    // may differ between the two machines.
    assert_eq!(count(&p.b, "image_hashes"), 0);
    assert_eq!(count(&p.b, "image_exif_tags"), 0);
    assert_eq!(count(&p.b, "images"), 1);
}

// ── Wire-level guards ───────────────────────────────────────────

#[test]
fn a_tombstone_names_a_replicated_entity() {
    let db = Database::open(&tempfile::tempdir().unwrap().path().join("l.db")).unwrap();
    let cid = db.create_collection("Iceland", "#3584e4", None).unwrap();
    db.delete_collection(cid).unwrap();

    let batch = db.collect_changes(0, DEFAULT_MAX_REVS).unwrap();
    assert_eq!(batch.tombstones.len(), 1);
    assert_eq!(batch.tombstones[0].entity, Entity::Collection);
    assert!(batch.tombstones[0].stamp > Stamp::new(0, String::new()));
}

// ── Randomised convergence ──────────────────────────────────────

/// Deterministic xorshift — a fixed seed makes any failure reproducible,
/// which a real RNG would not.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

/// The complete replicated state, in a canonical order.
///
/// Reuses `collect_changes` from stamp zero: it already projects exactly the
/// columns that replicate and rewrites foreign keys as guids, so it *is* the
/// definition of "what both devices are supposed to agree on". Anything
/// machine-local — rowids, paths, `status` — is absent by construction.
fn projection(db: &Database) -> String {
    let mut batch = db.collect_changes(0, MAX_REVS_LIMIT).unwrap();

    // `origin_path` and `file_size` ride along on the wire so a receiver can
    // name a photo it has never seen and estimate a transfer, but they are
    // *advisory*: each device keeps its own copy's location, and converged
    // libraries are expected to disagree about them. Comparing them would
    // assert the opposite of the design.
    for row in &mut batch.rows {
        if let crate::sync::SyncRow::Image(img) = row {
            img.origin_path = String::new();
            img.file_size = 0;
        }
    }

    batch.rows.sort_by(|a, b| a.guid().cmp(b.guid()));
    batch.tombstones.sort_by(|a, b| a.guid.cmp(&b.guid));
    batch.aliases.sort_by(|a, b| a.alias.cmp(&b.alias));
    serde_json::to_string_pretty(&(&batch.rows, &batch.tombstones, &batch.aliases)).unwrap()
}

/// Apply one arbitrary user action to `db`, chosen by `rng`.
fn random_edit(db: &Database, rng: &mut Rng, tag: &str) {
    /// Pick a row deterministically.
    ///
    /// `ORDER BY RANDOM()` would make the whole test unreproducible: a seed
    /// that fails could never be re-run to find out *why*, and a fix could
    /// not be told apart from a lucky run.
    fn pick(db: &Database, table: &str, rng: &mut Rng) -> Option<i64> {
        let mut stmt = db
            .conn
            .prepare(&format!("SELECT id FROM {table} ORDER BY id"))
            .ok()?;
        let ids: Vec<i64> = stmt
            .query_map([], |r| r.get::<_, i64>(0))
            .ok()?
            .filter_map(|r| r.ok())
            .collect();
        if ids.is_empty() {
            return None;
        }
        Some(ids[rng.below(ids.len() as u64) as usize])
    }

    match rng.below(10) {
        0 | 1 => {
            let n = rng.next() % 1000;
            let path = PathBuf::from(format!("/{tag}/img{n}.jpg"));
            let _ = db.insert_image(&path, &hash((n % 251) as u8), 1024);
        }
        2 => {
            let n = rng.next() % 20;
            let _ = db.create_collection(&format!("coll{n}"), "#3584e4", None);
        }
        3 => {
            if let Some(id) = pick(db, "collections", rng) {
                let _ = db.rename_collection(id, &format!("renamed-{tag}-{}", rng.next() % 100));
            }
        }
        4 => {
            if let Some(id) = pick(db, "collections", rng) {
                let _ = db.delete_collection(id);
            }
        }
        5 => {
            let c = pick(db, "collections", rng);
            let i = pick(db, "images", rng);
            if let (Some(c), Some(i)) = (c, i) {
                let _ = db.add_image_to_collection(c, i);
            }
        }
        6 => {
            let c = pick(db, "collections", rng);
            let i = pick(db, "images", rng);
            if let (Some(c), Some(i)) = (c, i) {
                let _ = db.remove_image_from_collection(c, i);
            }
        }
        7 => {
            let n = rng.next() % 8;
            let _ = db.upsert_person(&format!("person{n}"));
        }
        8 => {
            if let Some(id) = pick(db, "images", rng) {
                let _ = db.insert_face_detection(id, [0.1, 0.1, 0.5, 0.5], &[0.25f32; 512], 0.9);
            }
        }
        _ => {
            let f = pick(db, "face_detections", rng);
            let p = pick(db, "persons", rng);
            if let (Some(f), Some(p)) = (f, p) {
                let _ = db.assign_face_to_person(f, Some(p));
            }
        }
    }
}

#[test]
fn random_concurrent_edits_always_converge() {
    for seed in 1..=60u64 {
        let mut p = Pair::new();
        let mut rng = Rng(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15));

        for round in 0..12 {
            // Both sides edit while disconnected...
            for _ in 0..rng.below(5) + 1 {
                random_edit(&p.a, &mut rng, "a");
            }
            for _ in 0..rng.below(5) + 1 {
                random_edit(&p.b, &mut rng, "b");
            }
            // ...and only sometimes reconnect, so divergence accumulates.
            if round % 3 != 2 {
                p.converge();
            }
        }
        p.converge();

        let a = projection(&p.a);
        let b = projection(&p.b);
        assert_eq!(a, b, "seed {seed}: the two libraries did not converge");
    }
}

#[test]
fn convergence_survives_tiny_batches() {
    let mut p = Pair::new();
    let mut rng = Rng(0xDEAD_BEEF);
    for _ in 0..40 {
        random_edit(&p.a, &mut rng, "a");
        random_edit(&p.b, &mut rng, "b");
    }

    // One stamp per batch — the most adversarial paging possible.
    for _ in 0..400 {
        let fwd = p.a.collect_changes(p.b_seen, 1).unwrap();
        p.b.apply_batch(&fwd).unwrap();
        p.b_seen = fwd.next_rev;

        let back = p.b.collect_changes(p.a_seen, 1).unwrap();
        p.a.apply_batch(&back).unwrap();
        p.a_seen = back.next_rev;

        if fwd.is_empty() && back.is_empty() {
            break;
        }
    }

    assert_eq!(projection(&p.a), projection(&p.b));
}
