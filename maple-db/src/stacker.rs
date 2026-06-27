//! Stack detection — group near-identical or semantically similar shots.
//!
//! [`update_stacks`] is called by the background hasher after each batch.  It
//! reads all rows from `image_hashes` for the active algorithm, computes
//! pairwise similarity, and writes `stack_id` assignments back via Union-Find.
//!
//! Because it operates on the full set of hashed images, stacks are correctly
//! formed even when the library was built up over multiple import sessions.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use maple_import::ImageHash;
use maple_state::{StackMode, StackSettings};
use tracing::{info, warn};

use crate::{lock_db, Database};

/// Recompute stacks for all images that have a hash under `algorithm`.
///
/// Existing `stack_id` assignments are cleared before the new clusters are
/// written, so removed pairs are handled correctly.
pub fn update_stacks(
    db: &Arc<Mutex<Database>>,
    algorithm: &str,
    settings: &StackSettings,
) -> Result<usize> {
    // Load all (image_id, hash_blob) pairs for this algorithm.
    let rows = lock_db(db).images_with_hash(algorithm)?;
    if rows.len() < 2 {
        return Ok(0);
    }

    info!("Stacker: comparing {} hashed images (algorithm={algorithm})", rows.len());

    let groups = match settings.mode {
        StackMode::PHash => cluster_phash(&rows, settings.hash_size, settings.threshold),
        StackMode::Onnx => cluster_onnx(&rows, settings.threshold),
    };

    // Clear all existing stack assignments for these images so stale groups
    // are removed.  We only touch images that have a hash for this algorithm,
    // so images hashed under a different algorithm are unaffected.
    {
        let guard = lock_db(db);
        for (image_id, _) in &rows {
            if let Err(e) = guard.set_image_stack(*image_id, None) {
                warn!("Stacker: failed to clear stack for image {image_id}: {e}");
            }
        }
    }

    let mut stacks_created = 0;
    for group in &groups {
        if group.len() < 2 {
            continue;
        }
        let stack_id = lock_db(db).create_stack()?;
        for &image_id in group {
            if let Err(e) = lock_db(db).set_image_stack(image_id, Some(stack_id)) {
                warn!("Stacker: failed to assign image {image_id} to stack {stack_id}: {e}");
            }
        }
        stacks_created += 1;
        info!(stack_id, size = group.len(), "created stack");
    }

    Ok(stacks_created)
}

// ── pHash clustering ──────────────────────────────────────────────────────────

fn cluster_phash(
    rows: &[(i64, Vec<u8>)],
    hash_size: u32,
    threshold: f32,
) -> Vec<Vec<i64>> {
    let hashes: Vec<(i64, ImageHash)> = rows
        .iter()
        .filter_map(|(id, blob)| {
            ImageHash::from_bytes(blob).ok().map(|h| (*id, h))
        })
        .collect();

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..hashes.len() {
        for j in (i + 1)..hashes.len() {
            let sim = maple_import::phash_similarity(&hashes[i].1, &hashes[j].1, hash_size);
            if sim >= threshold {
                edges.push((i, j));
            }
        }
    }

    union_find_groups(hashes.iter().map(|(id, _)| *id).collect(), edges)
}

// ── ONNX / cosine clustering ──────────────────────────────────────────────────

fn cluster_onnx(rows: &[(i64, Vec<u8>)], threshold: f32) -> Vec<Vec<i64>> {
    let embeddings: Vec<(i64, Vec<f32>)> = rows
        .iter()
        .map(|(id, blob)| {
            let floats: Vec<f32> = blob
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect();
            (*id, floats)
        })
        .collect();

    let mut edges: Vec<(usize, usize)> = Vec::new();
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            let sim = crate::models::image_cosine_similarity(
                &embeddings[i].1,
                &embeddings[j].1,
            );
            if sim >= threshold {
                edges.push((i, j));
            }
        }
    }

    union_find_groups(embeddings.iter().map(|(id, _)| *id).collect(), edges)
}

// ── Union-Find ────────────────────────────────────────────────────────────────

fn union_find_groups(node_ids: Vec<i64>, edges: Vec<(usize, usize)>) -> Vec<Vec<i64>> {
    let n = node_ids.len();
    let mut parent: Vec<usize> = (0..n).collect();
    let mut rank: Vec<u8> = vec![0; n];

    fn find(parent: &mut Vec<usize>, x: usize) -> usize {
        if parent[x] != x {
            parent[x] = find(parent, parent[x]);
        }
        parent[x]
    }

    fn union(parent: &mut Vec<usize>, rank: &mut Vec<u8>, a: usize, b: usize) {
        let ra = find(parent, a);
        let rb = find(parent, b);
        if ra == rb {
            return;
        }
        match rank[ra].cmp(&rank[rb]) {
            std::cmp::Ordering::Less => parent[ra] = rb,
            std::cmp::Ordering::Greater => parent[rb] = ra,
            std::cmp::Ordering::Equal => {
                parent[rb] = ra;
                rank[ra] += 1;
            }
        }
    }

    for (i, j) in edges {
        union(&mut parent, &mut rank, i, j);
    }

    let mut components: std::collections::HashMap<usize, Vec<i64>> =
        std::collections::HashMap::new();
    for (idx, &image_id) in node_ids.iter().enumerate() {
        let root = find(&mut parent, idx);
        components.entry(root).or_default().push(image_id);
    }

    components.into_values().filter(|g| g.len() >= 2).collect()
}
