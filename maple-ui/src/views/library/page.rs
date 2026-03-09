//! Library browser page — wires the search bar, grid, and detail window.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;

use maple_db::SearchQuery;

use super::{detail_window, grid::LibraryGrid, search_bar};

/// Build the library browser page.
///
/// Opens a background metadata-filler thread so that newly imported images
/// get their EXIF fields populated while the user browses.
pub fn build_library_page(
    _nav_view: &adw::NavigationView,
    db: Arc<Mutex<maple_db::Database>>,
) -> adw::NavigationPage {
    // ── Grid ──────────────────────────────────────────────────────
    let library_grid = LibraryGrid::new(db.clone(), |image, window| {
        detail_window::open(&image, &window);
    });

    let scrolled = gtk4::ScrolledWindow::builder()
        .hscrollbar_policy(gtk4::PolicyType::Never)
        .vscrollbar_policy(gtk4::PolicyType::Automatic)
        .vexpand(true)
        .build();
    scrolled.set_child(Some(library_grid.widget()));

    // ── Search bar ────────────────────────────────────────────────
    let search_entry = search_bar::build_search_entry({
        let grid = library_grid.clone();
        move |text| {
            grid.load(SearchQuery::default().with_text(&text));
        }
    });

    let search_clamp = adw::Clamp::builder()
        .maximum_size(600)
        .child(&search_entry)
        .margin_top(8)
        .margin_bottom(8)
        .build();

    // ── Layout ────────────────────────────────────────────────────
    let content = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .build();
    content.append(&search_clamp);
    content.append(&scrolled);

    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&content));

    let page = adw::NavigationPage::builder()
        .title("Library")
        .child(&toolbar_view)
        .build();

    // ── Initial load + background metadata extraction ─────────────
    // Defer first load until the page is mapped so window geometry is settled.
    // `Rc<Cell<bool>>` lets us mutate a flag inside a `Fn` closure.
    let loaded = Rc::new(Cell::new(false));
    page.connect_map({
        let grid = library_grid.clone();
        let db = db.clone();
        move |_| {
            if !loaded.get() {
                loaded.set(true);
                grid.load(SearchQuery::default());
                maple_db::spawn_metadata_filler(db.clone());
            }
        }
    });

    page
}
