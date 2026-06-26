//! Drag-and-drop import — shared context and drop-target factory.
//!
//! Call `set_import_ctx` once from `window::build_window` after the nav view
//! and database are ready.  Any widget can then attach a drop target via
//! `widget.add_controller(make_drop_target())`.
//!
//! When files are dropped the target calls `open_import_for_files`, which
//! filters to supported image extensions, builds a browser page, and pushes
//! it onto the app's main navigation view.

use std::cell::RefCell;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use gtk4::gdk;
use gtk4::prelude::*;
use libadwaita as adw;

use crate::views::image_browser;

// ── App-wide import context ──────────────────────────────────────

/// Shared handles needed to open the import browser from any widget,
/// including the (separate) detail window.
#[derive(Clone)]
pub struct ImportCtx {
    pub nav_view: adw::NavigationView,
    pub toast_overlay: adw::ToastOverlay,
    pub db: Arc<Mutex<maple_db::Database>>,
}

thread_local! {
    static IMPORT_CTX: RefCell<Option<ImportCtx>> = const { RefCell::new(None) };
}

/// Register the app-wide import context.  Call once from `window::build_window`.
pub fn set_import_ctx(ctx: ImportCtx) {
    IMPORT_CTX.with(|c| *c.borrow_mut() = Some(ctx));
}

fn get_import_ctx() -> Option<ImportCtx> {
    IMPORT_CTX.with(|c| c.borrow().clone())
}

// ── Supported image extensions ───────────────────────────────────

fn is_image_path(path: &PathBuf) -> bool {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    matches!(
        ext.as_str(),
        "jpg" | "jpeg" | "png" | "gif" | "webp" | "tiff" | "tif"
            | "bmp" | "heic" | "heif" | "avif"
            | "raf" | "cr2" | "cr3" | "nef" | "arw" | "dng" | "orf" | "rw2" | "pef" | "srw"
    )
}

// ── Navigate to import browser for a list of dropped files ───────

/// Open the image browser pre-loaded with the given file paths.
///
/// Silently ignores non-image files.  Shows a toast if nothing remains.
pub fn open_import_for_files(files: Vec<PathBuf>) {
    let Some(ctx) = get_import_ctx() else { return };

    let image_files: Vec<PathBuf> = files.into_iter().filter(is_image_path).collect();

    if image_files.is_empty() {
        ctx.toast_overlay
            .add_toast(adw::Toast::new("No supported image files in drop"));
        return;
    }

    let settings = maple_state::Settings::load();
    let destination = settings.library_dir;

    let browser = image_browser::build_browser_page_from_files(
        image_files,
        &destination,
        &ctx.toast_overlay,
        ctx.db,
    );
    ctx.nav_view.push(&browser);
}

// ── Drop target factory ──────────────────────────────────────────

/// Create a `DropTarget` that accepts `GdkFileList` drops and opens the
/// import browser.  Attach it to any widget with `widget.add_controller(…)`.
pub fn make_drop_target() -> gtk4::DropTarget {
    let target =
        gtk4::DropTarget::new(gdk::FileList::static_type(), gdk::DragAction::COPY);
    target.connect_drop(|_, value, _, _| {
        let Ok(file_list) = value.get::<gdk::FileList>() else {
            return false;
        };
        let paths: Vec<PathBuf> = file_list
            .files()
            .iter()
            .filter_map(|f| f.path())
            .collect();
        open_import_for_files(paths);
        true
    });
    target
}
