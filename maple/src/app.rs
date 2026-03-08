use libadwaita as adw;
use adw::prelude::*;
use gtk4::gdk;

const APP_ID: &str = "dev.maple.Maple";

/// Build and run the GTK application. Returns the exit code.
pub fn run() -> i32 {
    let app = adw::Application::builder()
        .application_id(APP_ID)
        .build();

    app.connect_startup(|_| load_css());
    app.connect_activate(on_activate);
    app.run_with_args::<String>(&[]).into()
}

fn load_css() {
    let css = gtk4::CssProvider::new();
    css.load_from_string(include_str!("style.css"));
    gtk4::style_context_add_provider_for_display(
        &gdk::Display::default().expect("Could not get default display"),
        &css,
        gtk4::STYLE_PROVIDER_PRIORITY_APPLICATION,
    );
}

fn on_activate(app: &adw::Application) {
    let window = maple_ui::build_window(app);
    window.present();
}
