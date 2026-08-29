//! Search query builder for the image library.
//!
//! Build a `SearchQuery` via method chaining and pass it to
//! `Database::search_images`.  Additional filter dimensions (date range,
//! ISO range, camera model, …) can be added here without touching call
//! sites that don't need them.

// ── Query model ──────────────────────────────────────────────────

/// Row ordering for a library listing.
///
/// The ordering has to live in SQL rather than being applied by the caller
/// afterwards: results are paged (`limit`/`offset`), so a client-side sort
/// would only ever order one page within itself and pages would interleave
/// wrongly.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub enum SearchOrder {
    /// Library-insertion order, newest import first.
    #[default]
    AddedDesc,
    /// Photo-taken date, newest shot first; falls back to the insertion
    /// timestamp for images whose EXIF carries no capture date. Drives the
    /// date-grouped library view, which needs same-day images adjacent.
    TakenDesc,
}

/// Parameters for filtering and paginating library images.
///
/// # Example
/// ```
/// # use maple_db::SearchQuery;
/// let q = SearchQuery::default()
///     .with_text("canon 50mm")
///     .with_limit(100);
/// ```
#[derive(Default, Clone, Debug)]
pub struct SearchQuery {
    /// Free-text search.  Every whitespace-separated token must appear as a
    /// substring of one of the image's EXIF fields, AI descriptions, person
    /// names, or comprehensive EXIF tag values.
    pub text: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// When set, restrict results to images in this collection.
    pub collection_id: Option<i64>,
    /// Dense embedding of the query text.  When present alongside `text`,
    /// `search_images` runs hybrid search: keyword + semantic vector results
    /// are merged with reciprocal rank fusion.  Populated by the UI from the
    /// resident sentence encoder; left `None` when no encoder is loaded.
    pub semantic_embedding: Option<Vec<f32>>,
    /// Number of nearest sentence vectors to retrieve for the semantic side
    /// (0 falls back to a built-in default).
    pub semantic_k: usize,
    /// When set, restrict results to images that have a face assigned to this person.
    pub person_id: Option<i64>,
    /// Row ordering.  Ignored by hybrid (semantic) search, which is ranked
    /// by relevance instead.
    pub order: SearchOrder,
    /// Hide rows whose bytes live on another device (`locality = 'remote'`).
    ///
    /// A master paired with a relay servant replicates every one of that
    /// servant's photos as metadata and receives none of the files, so its
    /// grid fills with tiles it can never load — a master runs no worker and
    /// has no route to a servant, so nothing on that machine can fetch them.
    /// This is how a user gets their own library back while the peer is in a
    /// mode that sends nothing. Off by default: the rows are real library
    /// entries and hiding them by default would be its own surprise.
    pub local_only: bool,
}

/// The `WHERE`-clause half of a [`SearchQuery`] — everything that narrows
/// *which rows*, with nothing about paging, ordering or ranking.
///
/// Carried as one value rather than as three positional arguments because
/// every listing path needs all of them and they are easy to transpose: two
/// `Option<i64>` and a `bool` next to a `limit`/`offset` pair is exactly the
/// signature that silently filters by the wrong thing.
#[derive(Default, Clone, Copy, Debug)]
pub(crate) struct Filters {
    pub collection_id: Option<i64>,
    pub person_id: Option<i64>,
    pub local_only: bool,
}

impl SearchQuery {
    /// The row-narrowing half of this query.
    pub(crate) fn filters(&self) -> Filters {
        Filters {
            collection_id: self.collection_id,
            person_id: self.person_id,
            local_only: self.local_only,
        }
    }

    /// Filter by free text.  Whitespace-separated tokens are ANDed together
    /// as prefix matches (e.g. `"nikon 50"` → rows where every token
    /// appears in at least one indexed field, as a prefix).
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        let t = text.into();
        self.text = if t.trim().is_empty() {
            None
        } else {
            Some(t.trim().to_owned())
        };
        self
    }

    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = Some(limit);
        self
    }

    pub fn with_offset(mut self, offset: usize) -> Self {
        self.offset = Some(offset);
        self
    }

    /// Restrict the listing to photos this device actually holds.
    pub fn with_local_only(mut self, local_only: bool) -> Self {
        self.local_only = local_only;
        self
    }

    pub fn with_collection(mut self, id: i64) -> Self {
        self.collection_id = Some(id);
        self
    }

    /// Attach a query embedding (and KNN depth) to enable hybrid search.
    pub fn with_semantic(mut self, embedding: Vec<f32>, k: usize) -> Self {
        self.semantic_embedding = Some(embedding);
        self.semantic_k = k;
        self
    }

    /// Filter by person: only images where a face is assigned to `person_id`.
    pub fn with_person(mut self, id: i64) -> Self {
        self.person_id = Some(id);
        self
    }

    /// Order the results (see [`SearchOrder`]).
    pub fn with_order(mut self, order: SearchOrder) -> Self {
        self.order = order;
        self
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.collection_id.is_none() && self.person_id.is_none()
    }
}

// ── ORDER BY formatting ──────────────────────────────────────────

/// `ORDER BY` clause for a listing, for a query aliasing `images` as `i`.
///
/// Every ordering ends in `i.id DESC`: `added_at`/`taken_at` have
/// second resolution and tie constantly within one bulk import, and a
/// non-total sort key makes `LIMIT`/`OFFSET` paging drop or repeat rows
/// across page boundaries (SQLite is free to break ties differently per
/// statement).
///
/// Each clause is backed by a V17 index (`idx_images_listing_added`,
/// `idx_images_listing_taken`).  The `TakenDesc` one indexes an expression,
/// and SQLite matches an expression index only when the query spells the
/// expression identically — keep `COALESCE(i.taken_at, i.added_at)` here and
/// the index DDL in `schema.rs` in step, or the planner drops back to sorting
/// the whole table into a temp b-tree for every page.
pub(crate) fn order_by_sql(order: SearchOrder) -> &'static str {
    match order {
        SearchOrder::AddedDesc => "ORDER BY i.added_at DESC, i.id DESC",
        SearchOrder::TakenDesc => "ORDER BY COALESCE(i.taken_at, i.added_at) DESC, i.id DESC",
    }
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_text_trims_and_rejects_blank() {
        assert!(SearchQuery::default().with_text("  ").text.is_none());
        assert_eq!(
            SearchQuery::default().with_text("  nikon  ").text.as_deref(),
            Some("nikon")
        );
    }
}
