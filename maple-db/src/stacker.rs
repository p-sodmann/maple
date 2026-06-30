//! Stack detection — group near-identical or semantically similar shots.
//!
//! [`update_stacks`] is called by the background hasher after each batch.
//! Only the newly-hashed images are compared against the full set (O(new × n)
//! rather than O(n²)), so the cost is proportional to the batch size rather
//! than the total library size.
//!
//! Existing stack assignments are preserved for already-hashed images.  The
//! union-find is seeded with those assignments so that already-grouped images
//! stay together without re-examining their pairs.

use std::collections::{hash_map::Entry, HashMap, HashSet};
use std::sync::{Arc, Mutex};

use anyhow::Result;
use maple_import::ImageHash;
use maple_state::{StackMode, StackSettings};
use tracing::{info, warn};

use crate::{lock_db, Database};

/// Update stacks for `new_ids` — images whose hash was just computed.
///
/// Only new images are compared against all hashed images (O(new × n)).
/// Existing stack assignments for already-hashed images are preserved and
/// seeded into the union-find so they are not silently broken.
pub fn update_stacks(
    db: &Arc<Mutex<Database>>,
    algorithm: &str,
    new_ids: &[i64],
    settings: &StackSettings,
) -> Result<usize> {
    if new_ids.is_empty() {
        return Ok(0);
    }

    // Load all hashed images with their current stack assignments.
    let rows = lock_db(db).images_with_hash_and_stack(algorithm)?;
    let n = rows.len();
    if n < 2 {
        return Ok(0);
    }

    let new_set: HashSet<i64> = new_ids.iter().copied().collect();

    info!(
        "Stacker: comparing {} new image(s) against {} total (algorithm={algorithm})",
        new_set.len(),
        n,
    );

    // Initialise union-find over all row indices.
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    // Seed union-find with existing stack assignments so that already-grouped
    // images stay together without re-examining their pairs.
    {
        let mut first_member: HashMap<i64, usize> = HashMap::new();
        for (idx, (_, _, stack_id)) in rows.iter().enumerate() {
            if let Some(sid) = stack_id {
                match first_member.entry(*sid) {
                    Entry::Occupied(e) => uf_union(&mut parent, &mut rank, *e.get(), idx),
                    Entry::Vacant(e) => { e.insert(idx); }
                }
            }
        }
    }

    // Compare each new image against ALL images to find similarity edges.
    //
    // FIXME: For large libraries this is still O(new × n) per call.  A future
    // optimisation could use a spatial index (BK-tree for pHash, HNSW for
    // embeddings) to cut this to O(new × log n).
    let edges = match settings.mode {
        StackMode::PHash => new_vs_all_phash(&rows, &new_set, settings.hash_size, settings.threshold),
        StackMode::Onnx  => new_vs_all_onnx(&rows, &new_set, settings.threshold),
    };

    if edges.is_empty() {
        return Ok(0);
    }

    for &(i, j) in &edges {
        uf_union(&mut parent, &mut rank, i, j);
    }

    // Collect components that contain at least one new image.
    let mut components: HashMap<usize, Vec<usize>> = HashMap::new();
    for idx in 0..n {
        let root = uf_find(&mut parent, idx);
        components.entry(root).or_default().push(idx);
    }

    let mut stacks_created = 0;

    for members in components.values() {
        if members.len() < 2 {
            continue;
        }

        // Skip components with no new images — their stacks are already correct.
        if !members.iter().any(|&i| new_set.contains(&rows[i].0)) {
            continue;
        }

        // Collect distinct existing stack ids in this component (order stable).
        let existing_stack_ids: Vec<i64> = {
            let mut seen = HashSet::new();
            members
                .iter()
                .filter_map(|&i| rows[i].2)
                .filter(|sid| seen.insert(*sid))
                .collect()
        };

        // Canonical stack: reuse the first existing one, or create a fresh one.
        let canonical = if let Some(&sid) = existing_stack_ids.first() {
            sid
        } else {
            let sid = lock_db(db).create_stack()?;
            stacks_created += 1;
            info!(stack_id = sid, size = members.len(), "created stack");
            sid
        };

        // Assign every member of the component to the canonical stack.
        let guard = lock_db(db);
        for &idx in members {
            let (image_id, _, current_sid) = &rows[idx];
            if *current_sid != Some(canonical) {
                if let Err(e) = guard.set_image_stack(*image_id, Some(canonical)) {
                    warn!("Stacker: failed to assign image {image_id} to stack {canonical}: {e}");
                }
            }
        }
    }

    Ok(stacks_created)
}

// ── Union-Find ────────────────────────────────────────────────────────────────

fn uf_find(parent: &mut [usize], x: usize) -> usize {
    if parent[x] != x {
        parent[x] = uf_find(parent, parent[x]);
    }
    parent[x]
}

fn uf_union(parent: &mut [usize], rank: &mut [u8], a: usize, b: usize) {
    let ra = uf_find(parent, a);
    let rb = uf_find(parent, b);
    if ra == rb {
        return;
    }
    match rank[ra].cmp(&rank[rb]) {
        std::cmp::Ordering::Less    => parent[ra] = rb,
        std::cmp::Ordering::Greater => parent[rb] = ra,
        std::cmp::Ordering::Equal   => { parent[rb] = ra; rank[ra] += 1; }
    }
}

// ── Edge discovery: new images vs all ────────────────────────────────────────

fn new_vs_all_phash(
    rows: &[(i64, Vec<u8>, Option<i64>)],
    new_set: &HashSet<i64>,
    hash_size: u32,
    threshold: f32,
) -> Vec<(usize, usize)> {
    let hashes: Vec<(usize, i64, Option<ImageHash>)> = rows
        .iter()
        .enumerate()
        .map(|(i, (id, blob, _))| (i, *id, ImageHash::from_bytes(blob).ok()))
        .collect();

    let mut edges = Vec::new();
    for &(i, id_i, ref h_i) in &hashes {
        if !new_set.contains(&id_i) {
            continue;
        }
        let Some(h_i) = h_i else { continue };
        for &(j, _, ref h_j) in &hashes {
            if i == j {
                continue;
            }
            let Some(h_j) = h_j else { continue };
            if maple_import::phash_similarity(h_i, h_j, hash_size) >= threshold {
                edges.push((i, j));
            }
        }
    }
    edges
}

fn new_vs_all_onnx(
    rows: &[(i64, Vec<u8>, Option<i64>)],
    new_set: &HashSet<i64>,
    threshold: f32,
) -> Vec<(usize, usize)> {
    let embeddings: Vec<(usize, i64, Vec<f32>)> = rows
        .iter()
        .enumerate()
        .map(|(i, (id, blob, _))| {
            let floats: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (i, *id, floats)
        })
        .collect();

    let mut edges = Vec::new();
    for (i, id_i, emb_i) in &embeddings {
        if !new_set.contains(id_i) {
            continue;
        }
        for (j, _, emb_j) in &embeddings {
            if i == j {
                continue;
            }
            if crate::models::image_cosine_similarity(emb_i, emb_j) >= threshold {
                edges.push((*i, *j));
            }
        }
    }
    edges
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;
    use std::path::PathBuf;

    const ALG: &str = "phash:8";

    // pHash similarity for hash_size=8: similarity = (64 - hamming_distance) / 64.
    // [0xFF; 8] vs [0xFF; 8] → dist=0  → sim=1.00  (match above threshold 0.90)
    // [0xFF; 8] vs [0x00; 8] → dist=64 → sim=0.00  (no match)

    fn tmp_db() -> (tempfile::TempDir, Arc<Mutex<Database>>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        (dir, Arc::new(Mutex::new(db)))
    }

    fn settings() -> StackSettings {
        StackSettings::default()
    }

    /// Insert an image and store a known pHash blob. Returns the DB image id.
    fn seed(db: &Arc<Mutex<Database>>, path: &str, hash_blob: [u8; 8]) -> i64 {
        let p = PathBuf::from(path);
        let guard = db.lock().unwrap();
        guard.insert_image(&p, &[0u8; 32], 1024).unwrap();
        let id = guard
            .all_images()
            .unwrap()
            .into_iter()
            .find(|r| r.path == p)
            .unwrap()
            .id;
        guard.insert_image_hash(id, ALG, &hash_blob).unwrap();
        id
    }

    fn stack_of(db: &Arc<Mutex<Database>>, id: i64) -> Option<i64> {
        db.lock()
            .unwrap()
            .images_with_hash_and_stack(ALG)
            .unwrap()
            .into_iter()
            .find(|(img_id, _, _)| *img_id == id)
            .and_then(|(_, _, sid)| sid)
    }

    #[test]
    fn two_new_similar_images_form_stack() {
        let (_dir, db) = tmp_db();
        let a = seed(&db, "/photos/a.jpg", [0xFF; 8]);
        let b = seed(&db, "/photos/b.jpg", [0xFF; 8]);

        let created = update_stacks(&db, ALG, &[a, b], &settings()).unwrap();

        assert_eq!(created, 1);
        let sa = stack_of(&db, a);
        let sb = stack_of(&db, b);
        assert!(sa.is_some());
        assert_eq!(sa, sb);
    }

    #[test]
    fn new_image_joins_existing_stack() {
        let (_dir, db) = tmp_db();
        let a = seed(&db, "/photos/a.jpg", [0xFF; 8]);
        let b = seed(&db, "/photos/b.jpg", [0xFF; 8]);
        update_stacks(&db, ALG, &[a, b], &settings()).unwrap();
        let existing_sid = stack_of(&db, a).unwrap();

        let c = seed(&db, "/photos/c.jpg", [0xFF; 8]);
        let created = update_stacks(&db, ALG, &[c], &settings()).unwrap();

        assert_eq!(created, 0, "no new stack created — c joins the existing one");
        assert_eq!(stack_of(&db, c), Some(existing_sid));
        // Existing members stay in place.
        assert_eq!(stack_of(&db, a), Some(existing_sid));
        assert_eq!(stack_of(&db, b), Some(existing_sid));
    }

    #[test]
    fn dissimilar_new_image_stays_unstacked() {
        let (_dir, db) = tmp_db();
        let a = seed(&db, "/photos/a.jpg", [0xFF; 8]);
        let b = seed(&db, "/photos/b.jpg", [0x00; 8]);

        let created = update_stacks(&db, ALG, &[a, b], &settings()).unwrap();

        assert_eq!(created, 0);
        assert!(stack_of(&db, a).is_none());
        assert!(stack_of(&db, b).is_none());
    }

    #[test]
    fn existing_stacks_unaffected_by_unrelated_new_image() {
        let (_dir, db) = tmp_db();
        let a = seed(&db, "/photos/a.jpg", [0xFF; 8]);
        let b = seed(&db, "/photos/b.jpg", [0xFF; 8]);
        update_stacks(&db, ALG, &[a, b], &settings()).unwrap();
        let existing_sid = stack_of(&db, a).unwrap();

        // d has a completely different hash — no similarity to a or b.
        let d = seed(&db, "/photos/d.jpg", [0x00; 8]);
        update_stacks(&db, ALG, &[d], &settings()).unwrap();

        assert_eq!(stack_of(&db, a), Some(existing_sid), "a unaffected");
        assert_eq!(stack_of(&db, b), Some(existing_sid), "b unaffected");
        assert!(stack_of(&db, d).is_none(), "d has no similar partner");
    }
}
