//! Debug tools window.
//!
//! A single non-modal window that stays open while the rest of the app is
//! used.  Clicking the debug button again raises the existing window instead
//! of opening a second one.
//!
//! The window uses an `adw::NavigationView` so additional debug pages can be
//! pushed without restructuring anything here.
//!
//! # Adding a new debug tool
//!   1. Build an `adw::NavigationPage` for the tool (see `build_similarity_page`).
//!   2. Push it from the root list by appending an `adw::ActionRow` with a
//!      `go-next-symbolic` suffix that calls `nav.push(&your_page)`.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use gtk4::prelude::*;
use libadwaita as adw;
use adw::prelude::*;

use maple_db::Database;

// ── Singleton ──────────────────────────────────────────────────────────────

thread_local! {
    static DEBUG_WIN: RefCell<Option<adw::Window>> = const { RefCell::new(None) };
}

/// Open the debug tools window, or raise it if it is already open.
pub fn open_debug(parent: &impl IsA<gtk4::Window>, db: &Arc<Mutex<Database>>) {
    // Raise an already-visible window rather than opening a second one.
    let existing = DEBUG_WIN.with(|cell| cell.borrow().clone());
    if let Some(win) = existing {
        if win.is_visible() {
            win.present();
            return;
        }
    }

    let db = db.clone();
    let nav = adw::NavigationView::new();
    let root_page = build_root_page(&nav, &db);
    nav.push(&root_page);

    let win = adw::Window::builder()
        .title("Debug Tools")
        .default_width(480)
        .default_height(420)
        .transient_for(parent)
        .build();
    win.set_content(Some(&nav));

    // Clear the singleton when the window is closed.
    win.connect_destroy(|_| {
        DEBUG_WIN.with(|cell| *cell.borrow_mut() = None);
    });

    DEBUG_WIN.with(|cell| *cell.borrow_mut() = Some(win.clone()));
    win.present();
}

// ── Pages ──────────────────────────────────────────────────────────────────

fn build_root_page(nav: &adw::NavigationView, db: &Arc<Mutex<Database>>) -> adw::NavigationPage {
    let tools_group = adw::PreferencesGroup::builder()
        .title("Tools")
        .build();

    let sim_row = adw::ActionRow::builder()
        .title("Similarity Query")
        .subtitle("Compute perceptual similarity between two images by ID")
        .activatable(true)
        .build();
    sim_row.add_suffix(&gtk4::Image::from_icon_name("go-next-symbolic"));
    tools_group.add(&sim_row);

    let pref_page = adw::PreferencesPage::new();
    pref_page.add(&tools_group);

    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&pref_page));

    let root = adw::NavigationPage::builder()
        .title("Debug Tools")
        .child(&toolbar_view)
        .build();

    sim_row.connect_activated({
        let nav = nav.clone();
        let db = db.clone();
        move |_| nav.push(&build_similarity_page(&db))
    });

    root
}

fn build_similarity_page(db: &Arc<Mutex<Database>>) -> adw::NavigationPage {
    // Fetch available algorithms upfront (fast — just a DISTINCT query).
    let algorithms: Vec<String> = db
        .lock()
        .ok()
        .and_then(|g| g.stored_algorithms().ok())
        .unwrap_or_default();

    // ── Input group ───────────────────────────────────────────────
    let input_group = adw::PreferencesGroup::builder()
        .title("Image IDs")
        .build();

    let id_a_row = adw::EntryRow::builder()
        .title("Image ID A")
        .input_purpose(gtk4::InputPurpose::Digits)
        .build();

    let id_b_row = adw::EntryRow::builder()
        .title("Image ID B")
        .input_purpose(gtk4::InputPurpose::Digits)
        .build();

    input_group.add(&id_a_row);
    input_group.add(&id_b_row);

    // ── Algorithm group ───────────────────────────────────────────
    let algo_group = adw::PreferencesGroup::builder()
        .title("Algorithm")
        .build();

    let algo_strings: Vec<&str> = algorithms.iter().map(|s| s.as_str()).collect();
    let algo_model = gtk4::StringList::new(&algo_strings);
    let algo_row = adw::ComboRow::builder()
        .title("Algorithm")
        .model(&algo_model)
        .build();

    if algorithms.is_empty() {
        algo_group.add(
            &adw::ActionRow::builder()
                .title("No hashes stored yet")
                .subtitle("Run the background hasher before querying similarity")
                .build(),
        );
    } else {
        algo_group.add(&algo_row);
    }

    // ── Result group ──────────────────────────────────────────────
    let result_group = adw::PreferencesGroup::builder()
        .title("Result")
        .build();

    let result_row = adw::ActionRow::builder()
        .title("Similarity")
        .subtitle("—")
        .build();

    let query_btn = gtk4::Button::builder()
        .label("Query")
        .css_classes(["suggested-action"])
        .valign(gtk4::Align::Center)
        .sensitive(!algorithms.is_empty())
        .build();
    result_row.add_suffix(&query_btn);
    result_group.add(&result_row);

    // ── Layout ────────────────────────────────────────────────────
    let pref_page = adw::PreferencesPage::new();
    pref_page.add(&input_group);
    pref_page.add(&algo_group);
    pref_page.add(&result_group);

    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&pref_page));

    let page = adw::NavigationPage::builder()
        .title("Similarity Query")
        .child(&toolbar_view)
        .build();

    // ── Query logic ───────────────────────────────────────────────
    query_btn.connect_clicked({
        let db = db.clone();
        let algorithms = algorithms.clone();
        move |_| {
            let id_a: Option<i64> = id_a_row.text().trim().parse().ok();
            let id_b: Option<i64> = id_b_row.text().trim().parse().ok();

            let algorithm = {
                let idx = algo_row.selected() as usize;
                algorithms.get(idx).cloned().unwrap_or_else(|| {
                    maple_state::Settings::load().stacks.algorithm_key()
                })
            };

            let subtitle = match (id_a, id_b) {
                (Some(a), Some(b)) => match db
                    .lock()
                    .ok()
                    .and_then(|g| g.similarity_for_images(a, b, &algorithm).ok())
                {
                    Some(Some(sim)) => format!("{sim:.6}   [{algorithm}]"),
                    Some(None) => {
                        format!("No hash stored for one or both IDs under [{algorithm}]")
                    }
                    None => "Database error".to_owned(),
                },
                _ => "Enter valid integer IDs in both fields".to_owned(),
            };
            result_row.set_subtitle(&subtitle);
        }
    });

    page
}
