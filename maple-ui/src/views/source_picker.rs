//! Folder picker view — select source and destination directories.

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use gtk4::prelude::*;
use gtk4::gio;
use libadwaita as adw;
use adw::prelude::*;

use super::image_browser;

/// Local UI state for folder selection.
struct PickerState {
    source: Option<PathBuf>,
    destination: Option<PathBuf>,
}

/// Build the source/destination folder picker page.
pub fn build_picker_page(
    nav_view: &adw::NavigationView,
    toast_overlay: &adw::ToastOverlay,
) -> adw::NavigationPage {
    // Restore previous session (validate that paths still exist)
    let mut session = maple_state::Session::load();
    session.validate_paths();

    let state = Rc::new(RefCell::new(PickerState {
        source: session.source.clone(),
        destination: session.destination.clone(),
    }));

    // ── Source row ───────────────────────────────────────────────
    let source_row = adw::ActionRow::builder()
        .title("Source Folder")
        .subtitle(match &session.source {
            Some(p) => p.display().to_string(),
            None => "Not selected".into(),
        })
        .build();

    let source_btn = gtk4::Button::builder()
        .icon_name("folder-open-symbolic")
        .valign(gtk4::Align::Center)
        .css_classes(["flat"])
        .build();
    source_row.add_suffix(&source_btn);
    source_row.set_activatable_widget(Some(&source_btn));

    // ── Destination row ─────────────────────────────────────────
    let has_source = session.source.is_some();
    let dest_row = adw::ActionRow::builder()
        .title("Destination Folder")
        .subtitle(match &session.destination {
            Some(p) => p.display().to_string(),
            None => "Not selected".into(),
        })
        .sensitive(has_source)
        .build();

    let dest_btn = gtk4::Button::builder()
        .icon_name("folder-open-symbolic")
        .valign(gtk4::Align::Center)
        .css_classes(["flat"])
        .build();
    dest_row.add_suffix(&dest_btn);
    dest_row.set_activatable_widget(Some(&dest_btn));

    // ── List box ────────────────────────────────────────────────
    let folder_list = gtk4::ListBox::builder()
        .selection_mode(gtk4::SelectionMode::None)
        .css_classes(["boxed-list"])
        .build();
    folder_list.append(&source_row);
    folder_list.append(&dest_row);

    // ── Scan button ─────────────────────────────────────────────
    let both_set = session.source.is_some() && session.destination.is_some();
    let scan_btn = gtk4::Button::builder()
        .label("Start Scan")
        .css_classes(["suggested-action", "pill"])
        .halign(gtk4::Align::Center)
        .sensitive(both_set)
        .build();

    // ── Layout ──────────────────────────────────────────────────
    let controls = gtk4::Box::builder()
        .orientation(gtk4::Orientation::Vertical)
        .spacing(24)
        .halign(gtk4::Align::Center)
        .build();
    controls.append(&folder_list);
    controls.append(&scan_btn);

    let clamp = adw::Clamp::builder()
        .maximum_size(500)
        .child(&controls)
        .build();

    let status_page = adw::StatusPage::builder()
        .icon_name("camera-photo-symbolic")
        .title("Maple")
        .description("Import your best shots.\nSelect source and destination folders to begin.")
        .child(&clamp)
        .build();

    let header = adw::HeaderBar::new();
    let toolbar_view = adw::ToolbarView::new();
    toolbar_view.add_top_bar(&header);
    toolbar_view.set_content(Some(&status_page));

    let page = adw::NavigationPage::builder()
        .title("Maple")
        .child(&toolbar_view)
        .build();

    // ── Source button signal ────────────────────────────────────
    {
        let state = state.clone();
        let source_row = source_row.clone();
        let dest_row = dest_row.clone();
        let scan_btn = scan_btn.clone();

        source_btn.connect_clicked(move |btn| {
            let dialog = gtk4::FileDialog::builder()
                .title("Select Source Folder")
                .build();

            let window = btn.root().and_downcast::<gtk4::Window>();
            let state = state.clone();
            let source_row = source_row.clone();
            let dest_row = dest_row.clone();
            let scan_btn = scan_btn.clone();

            dialog.select_folder(
                window.as_ref(),
                None::<&gio::Cancellable>,
                move |result| {
                    if let Ok(file) = result {
                        if let Some(path) = file.path() {
                            source_row.set_subtitle(&path.display().to_string());
                            state.borrow_mut().source = Some(path.clone());
                            dest_row.set_sensitive(true);
                            scan_btn.set_sensitive(state.borrow().destination.is_some());
                            save_session(&state.borrow());
                        }
                    }
                },
            );
        });
    }

    // ── Destination button signal ───────────────────────────────
    {
        let state = state.clone();
        let dest_row = dest_row.clone();
        let scan_btn = scan_btn.clone();

        dest_btn.connect_clicked(move |btn| {
            let dialog = gtk4::FileDialog::builder()
                .title("Select Destination Folder")
                .build();

            let window = btn.root().and_downcast::<gtk4::Window>();
            let state = state.clone();
            let dest_row = dest_row.clone();
            let scan_btn = scan_btn.clone();

            dialog.select_folder(
                window.as_ref(),
                None::<&gio::Cancellable>,
                move |result| {
                    if let Ok(file) = result {
                        if let Some(path) = file.path() {
                            dest_row.set_subtitle(&path.display().to_string());
                            state.borrow_mut().destination = Some(path.clone());
                            scan_btn.set_sensitive(state.borrow().source.is_some());
                            save_session(&state.borrow());
                        }
                    }
                },
            );
        });
    }

    // ── Scan button signal ──────────────────────────────────────
    {
        let nav_view = nav_view.clone();
        let toast_overlay = toast_overlay.clone();
        let state = state.clone();

        scan_btn.connect_clicked(move |_| {
            let st = state.borrow();
            let source = match st.source.clone() {
                Some(s) => s,
                None => return,
            };
            let destination = match st.destination.clone() {
                Some(d) => d,
                None => return,
            };
            drop(st);

            let browser_page =
                image_browser::build_browser_page(&source, &destination, &toast_overlay);
            nav_view.push(&browser_page);
        });
    }

    page
}

/// Persist current folder selections to disk.
fn save_session(state: &PickerState) {
    let session = maple_state::Session {
        source: state.source.clone(),
        destination: state.destination.clone(),
        config: maple_state::Config::default(),
    };
    if let Err(e) = session.save() {
        tracing::warn!("Failed to save session: {e}");
    }
}
