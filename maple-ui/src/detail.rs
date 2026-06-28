//! Detail / lightbox window controller (Slint).
//!
//! Slint port of `views/library/detail_window/`. Opens a second top-level
//! [`DetailWindow`] showing one full-resolution image with pointer-anchored
//! zoom + pan (the zoom math lives in `ui/detail.slint`), prev/next navigation
//! through the records the grid was displaying, a toggleable one-line EXIF
//! strip, and a collection-chips bar.
//!
//! Like the old GTK detail window — and the existing `DEBUG_WIN` pattern — the
//! window is a singleton held in a `thread_local!` so re-activating a grid cell
//! reuses the same window instead of stacking new ones. The strong handle lives
//! only in the thread-local; every callback captures a [`slint::Weak`] plus the
//! shared record list, so there is no reference cycle through Slint.
//!
//! ## Scope notes (Phase 4)
//! * Collection chips support **display + remove** (self-contained DB calls).
//!   **Adding** to a collection needs the collection picker/manager, which is
//!   deferred to Phase 7; the "+" affordance lands with it.
//! * The separate "ⓘ" info popup window (`info_window.rs`, full field list +
//!   AI descriptions) is **deferred** — the toggleable one-line EXIF strip
//!   ports `info_bar.rs` and covers the common case.

use std::cell::{Cell, RefCell};
use std::rc::Rc;
use std::sync::{Arc, Mutex};

use slint::{ComponentHandle, ModelRc, SharedString, VecModel};

use maple_db::LibraryImage;

use crate::image_loader;
use crate::{CollectionChip, DetailWindow};

thread_local! {
    /// The single live detail window (mirrors the old `DETAIL_CTX` singleton).
    static DETAIL: RefCell<Option<Detail>> = const { RefCell::new(None) };
}

/// The detail window and its shared state.
///
/// Not `Clone` — the generated `DetailWindow` handle isn't, and there should
/// only ever be one. The strong handle lives solely in the [`DETAIL`]
/// thread-local; callbacks capture a [`slint::Weak`] plus the shared `Rc`
/// fields, so no reference cycle forms through Slint.
struct Detail {
    window: DetailWindow,
    /// Snapshot of the records the grid was showing when this image was opened.
    records: Rc<RefCell<Vec<LibraryImage>>>,
    /// Position of the displayed image within `records`.
    index: Rc<Cell<usize>>,
    db: Arc<Mutex<maple_db::Database>>,
}

/// Open (or reuse) the detail window for `records[index]`.
///
/// Reuses the existing singleton window when present — updating its record list
/// and current index — otherwise builds one. Safe to call from a UI callback
/// while the main event loop is running.
pub fn open(records: Vec<LibraryImage>, index: usize, db: Arc<Mutex<maple_db::Database>>) {
    // Build the singleton on first use.
    if DETAIL.with(|d| d.borrow().is_none()) {
        match build(db) {
            Ok(d) => DETAIL.with(|cell| *cell.borrow_mut() = Some(d)),
            Err(e) => {
                tracing::error!("Failed to build detail window: {e}");
                return;
            }
        }
    }

    DETAIL.with(|cell| {
        let guard = cell.borrow();
        let Some(detail) = guard.as_ref() else {
            return;
        };
        let len = records.len();
        *detail.records.borrow_mut() = records;
        detail.index.set(index.min(len.saturating_sub(1)));

        show_current(detail);

        if let Err(e) = detail.window.show() {
            tracing::error!("Failed to show detail window: {e}");
        }
    });
}

/// Build a fresh detail window and wire its callbacks (once).
fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<Detail, slint::PlatformError> {
    let window = DetailWindow::new()?;
    let records: Rc<RefCell<Vec<LibraryImage>>> = Rc::new(RefCell::new(Vec::new()));
    let index = Rc::new(Cell::new(0usize));

    window.on_prev({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        move || navigate(&w, &records, &index, &db, -1)
    });
    window.on_next({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        move || navigate(&w, &records, &index, &db, 1)
    });

    // Close just hides the singleton so re-opening reuses it.
    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_toggle_fullscreen({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let fs = !w.window().is_fullscreen();
                w.window().set_fullscreen(fs);
                w.set_is_fullscreen(fs);
            }
        }
    });

    window.on_open_external({
        let records = records.clone();
        let index = index.clone();
        move || {
            let recs = records.borrow();
            if let Some(rec) = recs.get(index.get()) {
                if let Err(e) = open::that_detached(&rec.path) {
                    tracing::warn!("Failed to open {} externally: {e}", rec.path.display());
                }
            }
        }
    });

    window.on_remove_collection({
        let w = window.as_weak();
        let records = records.clone();
        let index = index.clone();
        let db = db.clone();
        move |coll_id| {
            let image_id = match records.borrow().get(index.get()) {
                Some(rec) => rec.id,
                None => return,
            };
            if let Ok(d) = db.lock() {
                if let Err(e) = d.remove_image_from_collection(coll_id as i64, image_id) {
                    tracing::warn!(
                        "Failed to remove image {image_id} from collection {coll_id}: {e}"
                    );
                }
            }
            if let Some(w) = w.upgrade() {
                w.set_collection_chips(load_chips(&db, image_id));
            }
        }
    });

    Ok(Detail { window, records, index, db })
}

/// Move `delta` steps through the record list (+1 next, -1 prev).
///
/// No-ops while a load is in flight so rapid key/click presses don't queue up
/// transitions (mirrors the old `is_loading` guard).
fn navigate(
    w: &slint::Weak<DetailWindow>,
    records: &Rc<RefCell<Vec<LibraryImage>>>,
    index: &Rc<Cell<usize>>,
    db: &Arc<Mutex<maple_db::Database>>,
    delta: i32,
) {
    let Some(window) = w.upgrade() else {
        return;
    };
    if window.get_loading() {
        return;
    }
    let len = records.borrow().len();
    if len == 0 {
        return;
    }
    let cur = index.get();
    let new = (cur as i64 + delta as i64).clamp(0, len as i64 - 1) as usize;
    if new == cur {
        return;
    }
    index.set(new);
    show_record(&window, records, index, db);
}

/// Display `records[index]` in `detail.window`.
fn show_current(detail: &Detail) {
    show_record(&detail.window, &detail.records, &detail.index, &detail.db);
}

/// Update window chrome for the current record and kick off the async decode.
fn show_record(
    window: &DetailWindow,
    records: &Rc<RefCell<Vec<LibraryImage>>>,
    index: &Rc<Cell<usize>>,
    db: &Arc<Mutex<maple_db::Database>>,
) {
    let rec = match records.borrow().get(index.get()) {
        Some(r) => r.clone(),
        None => return,
    };

    let filename = rec.meta.filename.clone().unwrap_or_else(|| "Image".to_owned());
    window.set_filename(filename.into());
    window.set_info_text(info_text(&rec).into());
    window.set_collection_chips(load_chips(db, rec.id));
    window.set_error_text(SharedString::new());
    window.set_loading(true);
    window.invoke_reset_view();

    image_loader::load_full_image(rec.path.clone(), window.as_weak());
}

/// Build the one-line EXIF summary shown in the info strip (port of
/// `info_bar.rs::fill_info_bar`).
fn info_text(image: &LibraryImage) -> String {
    let m = &image.meta;
    let mut fields: Vec<String> = Vec::new();

    match (&m.make, &m.model) {
        (Some(make), Some(model)) => fields.push(format!("{make} {model}")),
        (Some(make), None) => fields.push(make.clone()),
        _ => {}
    }
    if let Some(lens) = &m.lens {
        fields.push(lens.clone());
    }
    if let (Some(fl), Some(ap)) = (m.focal_length, m.aperture) {
        fields.push(format!("{fl:.0} mm  f/{ap:.1}"));
    }
    if let Some(iso) = m.iso {
        fields.push(format!("ISO {iso}"));
    }
    if let (Some(w), Some(h)) = (m.width, m.height) {
        fields.push(format!("{w} × {h}"));
    }

    fields.join("  ·  ")
}

/// Load the current image's collection memberships as chip data for the UI.
fn load_chips(db: &Arc<Mutex<maple_db::Database>>, image_id: i64) -> ModelRc<CollectionChip> {
    let collections = db
        .lock()
        .ok()
        .and_then(|d| d.collections_for_image(image_id).ok())
        .unwrap_or_default();

    let chips: Vec<CollectionChip> = collections
        .iter()
        .map(|c| CollectionChip {
            id: c.id as i32,
            name: c.name.clone().into(),
            color: parse_hex_color(&c.color),
        })
        .collect();

    ModelRc::from(Rc::new(VecModel::from(chips)))
}

/// Parse a `#rrggbb` hex string into a Slint colour. Falls back to neutral grey.
fn parse_hex_color(hex: &str) -> slint::Color {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return slint::Color::from_rgb_u8(r, g, b);
        }
    }
    slint::Color::from_rgb_u8(0x9a, 0x9a, 0x9a)
}
