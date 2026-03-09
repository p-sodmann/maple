//! Debounced search entry for the library browser.
//!
//! Calls `on_change` on the GTK main thread after the user stops typing
//! for `DEBOUNCE_MS` milliseconds, preventing a DB query on every keystroke.

use std::cell::RefCell;
use std::rc::Rc;
use std::time::Duration;

use gtk4::glib;
use gtk4::prelude::*;

const DEBOUNCE_MS: u64 = 200;

/// Build a search entry that fires `on_change(text)` after debounce.
pub fn build_search_entry(on_change: impl Fn(String) + 'static) -> gtk4::SearchEntry {
    let entry = gtk4::SearchEntry::builder()
        .placeholder_text("Search filename, camera, lens…")
        .hexpand(true)
        .build();

    // Wrap in Rc so the closure can be cloned into the per-keystroke timer.
    let on_change = Rc::new(on_change);
    let pending: Rc<RefCell<Option<glib::SourceId>>> = Rc::new(RefCell::new(None));

    entry.connect_search_changed({
        let pending = pending.clone();
        move |e| {
            let text = e.text().to_string();

            // Cancel the previously scheduled invocation.
            if let Some(id) = pending.borrow_mut().take() {
                id.remove();
            }

            let on_change = on_change.clone();
            let pending_inner = pending.clone();
            let id = glib::timeout_add_local(Duration::from_millis(DEBOUNCE_MS), move || {
                // Clear the handle *before* returning Break so the outer handler
                // never sees a SourceId that GLib has already auto-removed.
                *pending_inner.borrow_mut() = None;
                on_change(text.clone());
                glib::ControlFlow::Break
            });
            *pending.borrow_mut() = Some(id);
        }
    });

    entry
}
