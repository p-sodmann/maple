//! maple-ui — GTK4 + libadwaita UI components.

pub mod hotkeys;
mod thumbnail;
mod views;
mod window;

pub use hotkeys::HotkeyManager;
pub use window::build_window;
