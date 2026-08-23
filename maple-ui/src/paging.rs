//! Page bookkeeping for the library grid's endless scrolling.
//!
//! The grid never holds the whole library: it appends fixed-size pages as
//! the viewport approaches the end of what is loaded. All of the decisions —
//! whether a fetch is due, at what offset, and when the listing is
//! exhausted — live here as plain arithmetic so they can be tested without a
//! window or a database.

/// Rows per page.
///
/// Sized so one page is several screens of tiles (a 160px cell on a 1440p
/// window shows ~50), which keeps fetches rare enough that the prefetch lead
/// hides them, while staying small enough that a page's thumbnail decode
/// finishes in about a second and a 50k-photo library costs ~300 decoded
/// buffers instead of 50,000.
pub const PAGE_SIZE: usize = 300;

/// Tracks how much of a query's result set has been requested, whether a
/// fetch is outstanding, and how far the view wants to be filled.
///
/// One fetch at a time: a page is only ever requested after the previous one
/// has landed, which is what lets the grid append pages blindly at the tail
/// (offset always equals the accumulated record count) and keeps model row
/// indices equal to record indices.
#[derive(Debug)]
pub struct PageCursor {
    /// Rows requested from the DB so far — the next page's `OFFSET`.
    requested: usize,
    /// A page fetch is in flight.
    fetching: bool,
    /// The DB returned a short page: there is nothing more to fetch.
    exhausted: bool,
    /// High-water mark of the row count the view has asked to have loaded.
    want: usize,
}

impl Default for PageCursor {
    fn default() -> Self {
        // The first page is always due: a freshly loaded grid shows it
        // before any scroll position exists to ask for more.
        Self { requested: 0, fetching: false, exhausted: false, want: PAGE_SIZE }
    }
}

impl PageCursor {
    /// Drop all paging state — a new query starts again at page 0.
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// Re-establish the cursor after an in-place refresh: `rows` are loaded
    /// and nothing is in flight.
    ///
    /// Unlike [`reset`](Self::reset) this keeps `want`, because the view has
    /// not moved — the user is still looking at wherever they had scrolled
    /// to, and forgetting how much they asked for would stop the grid filling
    /// in below them.
    ///
    /// `asked_for` is the limit the refresh query ran with. A short read is
    /// the end of the listing, exactly as it is for a page; a full one says
    /// nothing either way, so the next fetch is left to find out.
    pub fn refreshed(&mut self, rows: usize, asked_for: usize) {
        self.requested = rows;
        self.fetching = false;
        self.exhausted = rows < asked_for;
    }

    /// Raise the number of rows the view wants loaded. Monotonic: scrolling
    /// back up doesn't unload anything, so a lower request is not a reason
    /// to stop filling in what a deeper scroll already asked for.
    pub fn want(&mut self, rows: usize) {
        self.want = self.want.max(rows);
    }

    /// Offset of the next page to fetch, or `None` when one is already in
    /// flight, the listing is exhausted, or the view has enough rows.
    /// Marks the fetch as in flight.
    pub fn take_next_offset(&mut self) -> Option<usize> {
        if self.fetching || self.exhausted || self.requested >= self.want {
            return None;
        }
        let offset = self.requested;
        self.requested += PAGE_SIZE;
        self.fetching = true;
        Some(offset)
    }

    /// Record the arrival of the outstanding page, carrying `rows` records.
    /// A short page means the end of the listing.
    pub fn page_arrived(&mut self, rows: usize) {
        self.fetching = false;
        if rows < PAGE_SIZE {
            self.exhausted = true;
            // The short page didn't reach `requested`; pull it back to the
            // real row count so the offset stays truthful.
            self.requested = self.requested.saturating_sub(PAGE_SIZE - rows);
        }
    }

    /// Give up on the outstanding fetch and stop paging (used when a page
    /// arrives that cannot be appended in order — see `grid.rs`).
    pub fn abandon(&mut self) {
        self.fetching = false;
        self.exhausted = true;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_page_is_due_immediately() {
        let mut c = PageCursor::default();
        assert_eq!(c.take_next_offset(), Some(0));
    }

    #[test]
    fn only_one_fetch_at_a_time() {
        let mut c = PageCursor::default();
        c.want(10 * PAGE_SIZE);
        assert_eq!(c.take_next_offset(), Some(0));
        assert_eq!(c.take_next_offset(), None);
        c.page_arrived(PAGE_SIZE);
        assert_eq!(c.take_next_offset(), Some(PAGE_SIZE));
    }

    #[test]
    fn no_fetch_until_the_view_wants_more() {
        let mut c = PageCursor::default();
        assert_eq!(c.take_next_offset(), Some(0));
        c.page_arrived(PAGE_SIZE);
        // Want is still the initial one page — nothing further is due.
        assert_eq!(c.take_next_offset(), None);
        c.want(PAGE_SIZE + 1);
        assert_eq!(c.take_next_offset(), Some(PAGE_SIZE));
    }

    #[test]
    fn a_deep_want_is_filled_one_page_per_arrival() {
        let mut c = PageCursor::default();
        c.want(3 * PAGE_SIZE);
        for page in 0..3 {
            assert_eq!(c.take_next_offset(), Some(page * PAGE_SIZE));
            c.page_arrived(PAGE_SIZE);
        }
        assert_eq!(c.take_next_offset(), None);
    }

    #[test]
    fn want_is_monotonic() {
        let mut c = PageCursor::default();
        c.want(3 * PAGE_SIZE);
        c.want(1); // scrolled back up
        assert_eq!(c.take_next_offset(), Some(0));
        c.page_arrived(PAGE_SIZE);
        assert_eq!(c.take_next_offset(), Some(PAGE_SIZE));
    }

    #[test]
    fn short_page_ends_the_listing() {
        let mut c = PageCursor::default();
        c.want(10 * PAGE_SIZE);
        assert_eq!(c.take_next_offset(), Some(0));
        c.page_arrived(PAGE_SIZE - 40);
        assert_eq!(c.take_next_offset(), None);
        c.want(100 * PAGE_SIZE);
        assert_eq!(c.take_next_offset(), None);
    }

    #[test]
    fn empty_first_page_ends_the_listing() {
        let mut c = PageCursor::default();
        assert_eq!(c.take_next_offset(), Some(0));
        c.page_arrived(0);
        assert_eq!(c.take_next_offset(), None);
    }

    #[test]
    fn offsets_are_contiguous_across_pages() {
        // Offsets must tile the result set exactly: each page's offset is
        // the number of rows already delivered, which is what lets the grid
        // append at the tail without reindexing.
        let mut c = PageCursor::default();
        c.want(4 * PAGE_SIZE);
        let mut delivered = 0;
        while let Some(offset) = c.take_next_offset() {
            assert_eq!(offset, delivered);
            c.page_arrived(PAGE_SIZE);
            delivered += PAGE_SIZE;
        }
        assert_eq!(delivered, 4 * PAGE_SIZE);
    }

    #[test]
    fn reset_starts_over_at_page_zero() {
        let mut c = PageCursor::default();
        c.want(10 * PAGE_SIZE);
        c.take_next_offset();
        c.page_arrived(PAGE_SIZE);
        c.take_next_offset();

        c.reset();
        assert_eq!(c.take_next_offset(), Some(0));
    }

    #[test]
    fn abandon_stops_paging() {
        let mut c = PageCursor::default();
        c.want(10 * PAGE_SIZE);
        c.take_next_offset();
        c.abandon();
        assert_eq!(c.take_next_offset(), None);
        c.want(100 * PAGE_SIZE);
        assert_eq!(c.take_next_offset(), None);
    }

    fn scrolled_deep() -> PageCursor {
        let mut c = PageCursor::default();
        // Three pages loaded, the view asking for a fourth.
        for _ in 0..3 {
            c.want(PAGE_SIZE * 4);
            let offset = c.take_next_offset().expect("a page is due");
            assert_eq!(offset % PAGE_SIZE, 0);
            c.page_arrived(PAGE_SIZE);
        }
        c
    }

    #[test]
    fn a_refresh_keeps_what_the_view_asked_for() {
        // The difference between this and `reset`: the user has not moved, so
        // forgetting how far down they are would stop the grid filling in
        // below them until they scrolled again.
        let mut c = scrolled_deep();
        c.refreshed(PAGE_SIZE * 3, PAGE_SIZE * 3);

        assert_eq!(
            c.take_next_offset(),
            Some(PAGE_SIZE * 3),
            "the next page still continues where the loaded rows end"
        );
    }

    #[test]
    fn a_short_refresh_is_the_end_of_the_listing() {
        // Photos were deleted off the end while the grid was open.
        let mut c = scrolled_deep();
        c.refreshed(PAGE_SIZE * 3 - 10, PAGE_SIZE * 3);
        assert_eq!(c.take_next_offset(), None, "nothing left to fetch");
    }

    #[test]
    fn a_refresh_clears_an_in_flight_fetch() {
        // `refresh` orphans the page worker by taking a new generation, so a
        // cursor that still believed one was in flight would never fetch
        // again.
        let mut c = PageCursor::default();
        c.want(PAGE_SIZE * 2);
        c.take_next_offset().expect("first page");
        c.refreshed(PAGE_SIZE, PAGE_SIZE);
        assert_eq!(c.take_next_offset(), Some(PAGE_SIZE));
    }
}
