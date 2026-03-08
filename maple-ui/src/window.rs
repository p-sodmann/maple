use libadwaita as adw;

use crate::views::source_picker;

/// Build the main application window.
pub fn build_window(app: &adw::Application) -> adw::ApplicationWindow {
    let toast_overlay = adw::ToastOverlay::new();
    let nav_view = adw::NavigationView::new();

    // Initial page: folder picker
    let picker_page = source_picker::build_picker_page(&nav_view, &toast_overlay);
    nav_view.push(&picker_page);

    toast_overlay.set_child(Some(&nav_view));

    adw::ApplicationWindow::builder()
        .application(app)
        .title("Maple")
        .default_width(900)
        .default_height(600)
        .content(&toast_overlay)
        .build()
}
