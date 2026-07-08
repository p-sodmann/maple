//! Debug window controller — compare two images' stored embeddings by id.
//!
//! Looks up the DINOv2 embeddings already computed by the background hasher
//! for stack detection (`Database::similarity_for_images`); this is a lookup
//! against `image_hashes`, not a fresh ONNX inference pass.

use std::cell::RefCell;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, SharedString};

use crate::services::images as images_service;
use crate::DebugCompareWindow;

thread_local! {
    static DEBUG_COMPARE: RefCell<Option<DebugCompareWindow>> = const { RefCell::new(None) };
}

/// Open (or reuse) the debug compare window, syncing the current dark-mode state.
pub fn open(db: Arc<Mutex<maple_db::Database>>, is_dark: bool) {
    if DEBUG_COMPARE.with(|s| s.borrow().is_none()) {
        match build(db) {
            Ok(win) => DEBUG_COMPARE.with(|cell| *cell.borrow_mut() = Some(win)),
            Err(e) => {
                tracing::error!("Failed to build debug compare window: {e}");
                return;
            }
        }
    }
    DEBUG_COMPARE.with(|cell| {
        let guard = cell.borrow();
        if let Some(win) = guard.as_ref() {
            win.set_dark(is_dark);
            if let Err(e) = win.show() {
                tracing::error!("Failed to show debug compare window: {e}");
            }
        }
    });
}

/// Propagate a theme change to the debug compare window while it is open.
pub fn set_dark(dark: bool) {
    DEBUG_COMPARE.with(|s| {
        let guard = s.borrow();
        if let Some(win) = guard.as_ref() {
            win.set_dark(dark);
        }
    });
}

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<DebugCompareWindow, slint::PlatformError> {
    let window = DebugCompareWindow::new()?;

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_compare({
        let w = window.as_weak();
        move || {
            let Some(w) = w.upgrade() else { return };

            let parsed = w.get_image_id_a().trim().parse::<i64>().and_then(|a| {
                w.get_image_id_b().trim().parse::<i64>().map(|b| (a, b))
            });
            let (id_a, id_b) = match parsed {
                Ok(ids) => ids,
                Err(_) => {
                    w.set_result_is_error(true);
                    w.set_result_text(SharedString::from("Enter two numeric image ids."));
                    return;
                }
            };

            match images_service::compare_embeddings(&db, id_a, id_b) {
                Ok((name_a, name_b, score)) => {
                    w.set_result_is_error(false);
                    w.set_result_text(SharedString::from(format!(
                        "{name_a}  vs  {name_b}\nSimilarity: {:.1}%",
                        score * 100.0
                    )));
                }
                Err(msg) => {
                    w.set_result_is_error(true);
                    w.set_result_text(SharedString::from(msg));
                }
            }
        }
    });

    Ok(window)
}
