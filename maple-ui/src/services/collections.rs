//! Collection-tree query shaping.

use std::collections::HashMap;

use slint::SharedString;

use crate::transforms::hex_to_color;
use crate::CollectionEntry;

/// DFS-flatten the collection tree into a display list with depth info.
///
/// `all_collections` returns rows sorted by name, so siblings within each
/// parent level already arrive in alphabetical order.
pub fn flatten_tree(colls: &[maple_db::Collection]) -> Vec<CollectionEntry> {
    // Map parent_id → indices of children (in the `colls` slice)
    let mut children: HashMap<i64, Vec<usize>> = HashMap::new();
    let mut root_indices: Vec<usize> = Vec::new();

    for (idx, c) in colls.iter().enumerate() {
        match c.parent_id {
            Some(pid) => children.entry(pid).or_default().push(idx),
            None => root_indices.push(idx),
        }
    }

    // Iterative DFS using a stack of (index, depth).
    // Roots are pushed in reverse order so the first root pops first.
    let mut result: Vec<CollectionEntry> = Vec::with_capacity(colls.len());
    let mut stack: Vec<(usize, i32)> = root_indices.into_iter().rev().map(|i| (i, 0)).collect();

    while let Some((idx, depth)) = stack.pop() {
        let c = &colls[idx];
        result.push(CollectionEntry {
            id: c.id as i32,
            name: SharedString::from(c.name.as_str()),
            color: hex_to_color(&c.color),
            image_count: c.image_count as i32,
            parent_id: c.parent_id.map(|id| id as i32).unwrap_or(-1),
            depth,
        });
        // Push children in reverse sorted order so they pop in sorted order.
        if let Some(kids) = children.get(&c.id) {
            for &kid_idx in kids.iter().rev() {
                stack.push((kid_idx, depth + 1));
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn coll(id: i64, name: &str, parent_id: Option<i64>) -> maple_db::Collection {
        maple_db::Collection {
            id,
            name: name.into(),
            color: "#336699".into(),
            created_at: 0,
            image_count: 0,
            parent_id,
        }
    }

    #[test]
    fn flatten_tree_orders_roots_before_children_alphabetically() {
        // As returned by `all_collections` (sorted by name): Nested is a
        // child of Trip, both are top-level-adjacent in the input slice.
        let colls = vec![coll(1, "Nested", Some(2)), coll(2, "Trip", None)];
        let entries = flatten_tree(&colls);
        let names: Vec<&str> = entries.iter().map(|e| e.name.as_str()).collect();
        assert_eq!(names, vec!["Trip", "Nested"]);
        assert_eq!(entries[0].depth, 0);
        assert_eq!(entries[1].depth, 1);
        assert_eq!(entries[1].parent_id, 2);
    }

    #[test]
    fn flatten_tree_marks_top_level_parent_id_as_negative_one() {
        let colls = vec![coll(1, "Solo", None)];
        let entries = flatten_tree(&colls);
        assert_eq!(entries[0].parent_id, -1);
    }
}
