//! Shared "centroid + nearest" math used to pick a representative item for a
//! group of embeddings — a person's representative face, a collection's
//! representative (cover) image.
//!
//! Both callers store L2-normalised embeddings, so cosine similarity is a
//! plain dot product ([`crate::faces::cosine_similarity`]).

use crate::faces::cosine_similarity;

/// Average `items`' embeddings, L2-normalise the mean, and return it together
/// with the id of the item whose own embedding is closest (by cosine
/// similarity) to that centroid.
///
/// Returns `(None, None)` when `items` is empty.
pub(crate) fn centroid_and_nearest(items: &[(i64, Vec<f32>)]) -> (Option<Vec<f32>>, Option<i64>) {
    if items.is_empty() {
        return (None, None);
    }

    let dim = items[0].1.len();
    let mut centroid = vec![0f32; dim];
    for (_, emb) in items {
        for (c, v) in centroid.iter_mut().zip(emb.iter()) {
            *c += v;
        }
    }
    let n = items.len() as f32;
    for c in &mut centroid {
        *c /= n;
    }

    let norm: f32 = centroid.iter().map(|v| v * v).sum::<f32>().sqrt().max(1e-9);
    for c in &mut centroid {
        *c /= norm;
    }

    let nearest = items
        .iter()
        .max_by(|(_, a), (_, b)| {
            cosine_similarity(a, &centroid)
                .partial_cmp(&cosine_similarity(b, &centroid))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(id, _)| *id);

    (Some(centroid), nearest)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_returns_none() {
        assert_eq!(centroid_and_nearest(&[]), (None, None));
    }

    #[test]
    fn single_item_is_its_own_nearest() {
        let (centroid, nearest) = centroid_and_nearest(&[(7, vec![1.0, 0.0])]);
        assert_eq!(nearest, Some(7));
        assert!(centroid.is_some());
    }

    #[test]
    fn picks_item_closest_to_mean() {
        // Two items far apart, one near the midpoint — the midpoint-ish one
        // should win regardless of insertion order.
        let items = vec![
            (1, vec![1.0, 0.0]),
            (2, vec![0.0, 1.0]),
            (3, vec![0.9, 0.436_f32]), // close to the normalised mean of (1) and (2)
        ];
        let (_, nearest) = centroid_and_nearest(&items);
        assert_eq!(nearest, Some(3));
    }
}
