//! Destination path template popup — configures the folder/filename
//! templates used by `copy_images` during import, and offers to restructure
//! files already in the library to match a changed template.
//!
//! Takes a `Database` handle (unlike most other popups it once didn't need
//! one for) — restructuring requires both enumerating existing library
//! files and updating their DB rows after a move. Opened today from
//! `SettingsWindow` and `ImportWindow`, both of which already hold `db`.

use std::cell::RefCell;
use std::rc::Rc;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use slint::{ComponentHandle, SharedString, Timer, TimerMode};

use maple_import::path_template::{render_filename_stem, render_folder, TemplateContext};
use maple_import::{ExifDateTime, PlannedMove};

use crate::PathTemplateWindow;

/// Progress/result messages from the background restructure worker.
enum RestructureMsg {
    Progress { done: usize, total: usize },
    Done { moved: usize, failed: usize },
}

/// Keeps the Slint window handle alive alongside the repeating timer that
/// polls restructure progress — dropping the timer would stop the polling.
struct PathTemplateHandle {
    window: PathTemplateWindow,
    _restructure_timer: Rc<RefCell<Option<Timer>>>,
}

thread_local! {
    static PATH_TEMPLATE: RefCell<Option<PathTemplateHandle>> = const { RefCell::new(None) };
}

/// Open (or reuse) the path template window, syncing the current dark-mode state.
pub fn open(db: Arc<Mutex<maple_db::Database>>, is_dark: bool) {
    if PATH_TEMPLATE.with(|s| s.borrow().is_none()) {
        match build(db) {
            Ok(handle) => PATH_TEMPLATE.with(|cell| *cell.borrow_mut() = Some(handle)),
            Err(e) => {
                tracing::error!("Failed to build path template window: {e}");
                return;
            }
        }
    }
    PATH_TEMPLATE.with(|cell| {
        let guard = cell.borrow();
        if let Some(handle) = guard.as_ref() {
            let win = &handle.window;
            win.set_dark(is_dark);
            populate(win);
            if let Err(e) = win.show() {
                tracing::error!("Failed to show path template window: {e}");
            }
        }
    });
}

/// Propagate a theme change to the path template window while it is open.
pub fn set_dark(dark: bool) {
    PATH_TEMPLATE.with(|s| {
        let guard = s.borrow();
        if let Some(handle) = guard.as_ref() {
            handle.window.set_dark(dark);
        }
    });
}

fn build(db: Arc<Mutex<maple_db::Database>>) -> Result<PathTemplateHandle, slint::PlatformError> {
    let window = PathTemplateWindow::new()?;

    // Plan computed by `on_save`, consumed by `on_restructure_confirmed` if
    // the user accepts. Lives for the singleton's lifetime; each Save
    // recomputes and overwrites it.
    let planned: Rc<RefCell<Vec<PlannedMove>>> = Rc::new(RefCell::new(Vec::new()));
    let restructure_timer: Rc<RefCell<Option<Timer>>> = Rc::new(RefCell::new(None));

    window.on_close_requested({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                let _ = w.hide();
            }
        }
    });

    window.on_preview({
        let w = window.as_weak();
        move || {
            if let Some(w) = w.upgrade() {
                update_preview(&w);
            }
        }
    });

    window.on_save({
        let w = window.as_weak();
        let db = db.clone();
        let planned = planned.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let mut settings = maple_state::Settings::load();
            settings.path_template.folder = w.get_folder_template().to_string();
            settings.path_template.filename = w.get_filename_template().to_string();
            if let Err(e) = settings.save() {
                tracing::error!("Failed to save path template settings: {e}");
                return;
            }
            crate::settings_window::refresh_path_template_display();

            let plan = crate::services::restructure::plan(
                &db,
                &settings.library_dir,
                &settings.path_template.folder,
                &settings.path_template.filename,
            );

            if plan.is_empty() {
                let _ = w.hide();
                return;
            }

            let n = plan.len();
            *planned.borrow_mut() = plan;

            let (verb, pronoun) = if n == 1 { ("doesn't", "it") } else { ("don't", "them") };
            w.set_confirm_message(SharedString::from(format!(
                "{n} file{} {verb} match your new naming. Move {pronoun} into place now?",
                if n == 1 { "" } else { "s" },
            )));
            w.set_confirm_open(true);
        }
    });

    window.on_restructure_confirmed({
        let w = window.as_weak();
        let db = db.clone();
        let planned = planned.clone();
        let restructure_timer = restructure_timer.clone();
        move || {
            let Some(w) = w.upgrade() else { return };
            let plan = planned.borrow().clone();
            if plan.is_empty() {
                let _ = w.hide();
                return;
            }

            w.set_restructuring(true);
            w.set_restructure_done(0);
            w.set_restructure_total(plan.len() as i32);
            w.set_restructure_summary(SharedString::new());

            let (tx, rx) = mpsc::channel::<RestructureMsg>();
            let db2 = db.clone();
            std::thread::spawn(move || {
                let summary = crate::services::restructure::execute(&db2, &plan, |done, total| {
                    let _ = tx.send(RestructureMsg::Progress { done, total });
                });
                let _ = tx.send(RestructureMsg::Done {
                    moved: summary.moved,
                    failed: summary.failed,
                });
            });

            let w_weak = w.as_weak();
            let timer = Timer::default();
            timer.start(TimerMode::Repeated, Duration::from_millis(30), move || {
                let Some(w) = w_weak.upgrade() else { return };
                loop {
                    match rx.try_recv() {
                        Ok(RestructureMsg::Progress { done, total }) => {
                            w.set_restructure_done(done as i32);
                            w.set_restructure_total(total as i32);
                        }
                        Ok(RestructureMsg::Done { moved, failed }) => {
                            w.set_restructuring(false);
                            let msg = if failed == 0 {
                                format!("Moved {moved} file{}.", if moved == 1 { "" } else { "s" })
                            } else {
                                format!("Moved {moved}, {failed} failed.")
                            };
                            w.set_restructure_summary(SharedString::from(msg));
                            // The library grid's cached records point at the
                            // pre-move paths — reload so it (and anything
                            // opened from it) sees the new locations.
                            if moved > 0 {
                                crate::grid::request_reload();
                            }
                            return;
                        }
                        Err(mpsc::TryRecvError::Empty) => break,
                        Err(mpsc::TryRecvError::Disconnected) => {
                            w.set_restructuring(false);
                            return;
                        }
                    }
                }
            });
            *restructure_timer.borrow_mut() = Some(timer);
        }
    });

    Ok(PathTemplateHandle { window, _restructure_timer: restructure_timer })
}

/// Populate the window with current settings values, an initial preview,
/// and reset any leftover restructure/confirm state from a previous run.
fn populate(window: &PathTemplateWindow) {
    let s = maple_state::Settings::load();
    window.set_folder_template(SharedString::from(s.path_template.folder));
    window.set_filename_template(SharedString::from(s.path_template.filename));
    window.set_confirm_open(false);
    window.set_restructuring(false);
    window.set_restructure_summary(SharedString::new());
    update_preview(window);
}

/// Render both templates against a fixed sample file and update the preview text.
fn update_preview(window: &PathTemplateWindow) {
    let ctx = TemplateContext {
        datetime: ExifDateTime::parse("2024:03:15 14:30:45"),
        original_stem: "IMG_1234",
        counter: 7,
        camera: Some("Fujifilm X100V"),
    };

    let folder = render_folder(&window.get_folder_template(), &ctx);
    let stem = render_filename_stem(&window.get_filename_template(), &ctx);
    let filename = if stem.is_empty() {
        "IMG_1234".to_owned()
    } else {
        stem
    };

    let mut preview = folder;
    preview.push(format!("{filename}.RAF"));
    window.set_preview_text(SharedString::from(preview.to_string_lossy().into_owned()));
}
