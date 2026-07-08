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
use maple_state::StackSettings;
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
    // optimisation could use a spatial index (e.g. HNSW) to cut this to
    // O(new × log n).
    let edges = new_vs_all_onnx(&rows, &new_set, settings.threshold);

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

// ── From-scratch clustering (no DB) ───────────────────────────────────────────

/// Cluster `embeddings` via a threshold union-find over cosine similarity.
///
/// Unlike [`update_stacks`], this does a full O(n²) all-pairs comparison —
/// intended for a one-off, from-scratch grouping (e.g. photos on an SD card
/// that aren't in the DB yet), not the DB's incremental new-vs-all update.
///
/// Returns clusters (each a list of indices into `embeddings`) with 2 or
/// more members; singletons are omitted.
pub fn cluster_embeddings(embeddings: &[Vec<f32>], threshold: f32) -> Vec<Vec<usize>> {
    let n = embeddings.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    for i in 0..n {
        for j in (i + 1)..n {
            if crate::models::image_cosine_similarity(&embeddings[i], &embeddings[j]) >= threshold {
                uf_union(&mut parent, &mut rank, i, j);
            }
        }
    }

    let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
    for i in 0..n {
        let root = uf_find(&mut parent, i);
        clusters.entry(root).or_default().push(i);
    }

    clusters.into_values().filter(|c| c.len() >= 2).collect()
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Database;
    use std::path::PathBuf;

    const ALG: &str = "onnx:test-model";

    // Cosine similarity of unit vectors, threshold 0.90 (StackSettings default):
    // [1.0, 0.0] vs [1.0, 0.0] → sim=1.00 (identical, match)
    // [1.0, 0.0] vs [0.0, 1.0] → sim=0.00 (orthogonal, no match)

    fn tmp_db() -> (tempfile::TempDir, Arc<Mutex<Database>>) {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::open(&dir.path().join("library.db")).unwrap();
        (dir, Arc::new(Mutex::new(db)))
    }

    fn settings() -> StackSettings {
        StackSettings::default()
    }

    /// Insert an image and store a known embedding. Returns the DB image id.
    fn seed(db: &Arc<Mutex<Database>>, path: &str, embedding: [f32; 2]) -> i64 {
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
        let blob: Vec<u8> = embedding.iter().flat_map(|f| f.to_le_bytes()).collect();
        guard.insert_image_hash(id, ALG, &blob).unwrap();
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
        let a = seed(&db, "/photos/a.jpg", [1.0, 0.0]);
        let b = seed(&db, "/photos/b.jpg", [1.0, 0.0]);

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
        let a = seed(&db, "/photos/a.jpg", [1.0, 0.0]);
        let b = seed(&db, "/photos/b.jpg", [1.0, 0.0]);
        update_stacks(&db, ALG, &[a, b], &settings()).unwrap();
        let existing_sid = stack_of(&db, a).unwrap();

        let c = seed(&db, "/photos/c.jpg", [1.0, 0.0]);
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
        let a = seed(&db, "/photos/a.jpg", [1.0, 0.0]);
        let b = seed(&db, "/photos/b.jpg", [0.0, 1.0]);

        let created = update_stacks(&db, ALG, &[a, b], &settings()).unwrap();

        assert_eq!(created, 0);
        assert!(stack_of(&db, a).is_none());
        assert!(stack_of(&db, b).is_none());
    }

    #[test]
    fn existing_stacks_unaffected_by_unrelated_new_image() {
        let (_dir, db) = tmp_db();
        let a = seed(&db, "/photos/a.jpg", [1.0, 0.0]);
        let b = seed(&db, "/photos/b.jpg", [1.0, 0.0]);
        update_stacks(&db, ALG, &[a, b], &settings()).unwrap();
        let existing_sid = stack_of(&db, a).unwrap();

        // d has an orthogonal embedding — no similarity to a or b.
        let d = seed(&db, "/photos/d.jpg", [0.0, 1.0]);
        update_stacks(&db, ALG, &[d], &settings()).unwrap();

        assert_eq!(stack_of(&db, a), Some(existing_sid), "a unaffected");
        assert_eq!(stack_of(&db, b), Some(existing_sid), "b unaffected");
        assert!(stack_of(&db, d).is_none(), "d has no similar partner");
    }

    #[test]
    fn cluster_embeddings_groups_similar_and_omits_singletons() {
        let embeddings = vec![
            vec![1.0, 0.0], // 0: identical to 1
            vec![1.0, 0.0], // 1
            vec![0.0, 1.0], // 2: orthogonal to everything, no partner
        ];

        let clusters = cluster_embeddings(&embeddings, 0.90);

        assert_eq!(clusters.len(), 1, "only the 0/1 pair should form a cluster");
        let mut cluster = clusters[0].clone();
        cluster.sort_unstable();
        assert_eq!(cluster, vec![0, 1]);
    }

    #[test]
    fn cluster_embeddings_empty_input_returns_no_clusters() {
        assert!(cluster_embeddings(&[], 0.90).is_empty());
    }
}
