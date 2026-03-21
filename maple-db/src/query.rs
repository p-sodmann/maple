//! Search query builder for the image library.
//!
//! Build a `SearchQuery` via method chaining and pass it to
//! `Database::search_images`.  Additional filter dimensions (date range,
//! ISO range, camera model, …) can be added here without touching call
//! sites that don't need them.

// ── Query model ──────────────────────────────────────────────────

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
    /// Free-text search matched against filename, make, model, and lens
    /// via FTS5 prefix matching.
    pub text: Option<String>,
    pub limit: Option<usize>,
    pub offset: Option<usize>,
    /// When set, restrict results to images in this collection.
    pub collection_id: Option<i64>,
}

impl SearchQuery {
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

    pub fn with_collection(mut self, id: i64) -> Self {
        self.collection_id = Some(id);
        self
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_none() && self.collection_id.is_none()
    }
}

// ── FTS5 query formatting ────────────────────────────────────────

/// Convert a user search string into a safe FTS5 `MATCH` expression.
///
/// Each whitespace-separated token becomes `"token"*` (prefix match).
/// Double-quotes within a token are doubled to escape them.
///
/// `"nikon 50mm"` → `"nikon"* "50mm"*`
pub fn build_fts_query(text: &str) -> String {
    text.split_whitespace()
        .filter(|t| !t.is_empty())
        .map(|t| format!("\"{}\"*", t.replace('"', "\"\"")))
        .collect::<Vec<_>>()
        .join(" ")
}

// ── Tests ────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fts_single_token() {
        assert_eq!(build_fts_query("nikon"), r#""nikon"*"#);
    }

    #[test]
    fn fts_multiple_tokens() {
        assert_eq!(build_fts_query("nikon 50mm"), r#""nikon"* "50mm"*"#);
    }

    #[test]
    fn fts_escapes_quotes() {
        assert_eq!(build_fts_query(r#"say "hello""#), r#""say"* """hello"""*"#);
    }

    #[test]
    fn with_text_trims_and_rejects_blank() {
        assert!(SearchQuery::default().with_text("  ").text.is_none());
        assert_eq!(
            SearchQuery::default().with_text("  nikon  ").text.as_deref(),
            Some("nikon")
        );
    }
}
