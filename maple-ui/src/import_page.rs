//! Import-page controller.
//!
//! Owns the wiring for the source picker embedded in AppWindow: the
//! favourites/recents location lists, the folder browser, and the hand-off to
//! the separate import browser window (`import.rs`) once a source is chosen.
//!
//! Called from `lib.rs` during startup — see [`crate::AppCtx`] for the shared
//! handles these functions clone out of.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

use slint::{ModelRc, SharedString, VecModel};

use crate::{import, AppCtx, AppWindow, ImportLocation};

/// Wire the embedded import page.
pub fn wire(window: &AppWindow, ctx: &AppCtx) {
    // Shared source-path state for the embedded picker.
    let import_source: Rc<RefCell<std::path::PathBuf>> =
        Rc::new(RefCell::new(std::path::PathBuf::new()));

    // Build the initial location lists.
    window.on_import_page_shown({
        let w = ctx.window.clone();
        let db = ctx.db.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move || {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_browse({
        let w = ctx.window.clone();
        let source = import_source.clone();
        move || {
            let picked = rfd::FileDialog::new()
                .set_title("Choose source folder")
                .pick_folder();
            if let Some(path) = picked {
                let name = folder_display_name(&path);
                let path_str = path.to_string_lossy().into_owned();
                *source.borrow_mut() = path;
                if let Some(w) = w.upgrade() {
                    w.set_import_source_name(SharedString::from(name));
                    w.set_import_source_path(SharedString::from(path_str));
                    w.set_import_source_count(0);
                }
            }
        }
    });

    window.on_import_clear({
        let w = ctx.window.clone();
        let source = import_source.clone();
        move || {
            *source.borrow_mut() = std::path::PathBuf::new();
            if let Some(w) = w.upgrade() {
                w.set_import_source_path(SharedString::default());
                w.set_import_source_name(SharedString::default());
                w.set_import_source_count(0);
            }
        }
    });

    window.on_import_location_selected({
        let w = ctx.window.clone();
        let db = ctx.db.clone();
        let source = import_source.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let mut locs = build_import_locations(&settings, &dirs, &starred);

            // Mark selected, deselect others.
            mark_selected(&mut locs, id.as_str());

            if let Some(loc) = locs.iter().find(|l| l.id == id.as_str()) {
                *source.borrow_mut() = std::path::PathBuf::from(loc.path.as_str());
                w.set_import_source_name(loc.name.clone().into());
                w.set_import_source_path(loc.path.clone().into());
                w.set_import_source_count(loc.count);
            }

            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_star_toggled({
        let w = ctx.window.clone();
        let db = ctx.db.clone();
        let settings = maple_state::Settings::load();
        let dirs = KnownDirs::from_os();
        move |id| {
            let Some(w) = w.upgrade() else { return };
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
            if let Some(loc) = locs.iter().find(|l| l.id == id.as_str()) {
                let path = loc.path.as_str();
                let g = maple_db::lock_db(&db);
                let _ = if starred.contains(path) {
                    g.remove_starred_path(path)
                } else {
                    g.add_starred_path(path)
                };
                // Explicit: the re-read below locks again, and this mutex is
                // not reentrant.
                drop(g);
            }
            let starred = starred_paths(&db);
            let locs = build_import_locations(&settings, &dirs, &starred);
            let (favs, recents) = partition_locations(locs);
            w.set_import_favorites(favs);
            w.set_import_recents(recents);
        }
    });

    window.on_import_start_scan({
        let db = ctx.db.clone();
        let source = import_source.clone();
        let w = ctx.window.clone();
        move || {
            let src = source.borrow().clone();
            if src.as_os_str().is_empty() {
                return;
            }
            let is_dark = w.upgrade().map(|w| w.get_dark()).unwrap_or(false);
            import::open_with_source(db.clone(), src, is_dark);
        }
    });
}

// ── Import location helpers ────────────────────────────────────────

/// Well-known user folders, resolved via OS "known folder" APIs
/// (`directories::UserDirs`) rather than manual home-dir joins, so
/// relocated folders (e.g. OneDrive-redirected Pictures on Windows)
/// are still found.
struct KnownDirs {
    pictures: Option<std::path::PathBuf>,
    desktop: Option<std::path::PathBuf>,
    downloads: Option<std::path::PathBuf>,
}

impl KnownDirs {
    fn from_os() -> Self {
        let user_dirs = directories::UserDirs::new();
        Self {
            pictures: user_dirs.as_ref().and_then(|u| u.picture_dir()).map(Into::into),
            desktop: user_dirs.as_ref().and_then(|u| u.desktop_dir()).map(Into::into),
            downloads: user_dirs.as_ref().and_then(|u| u.download_dir()).map(Into::into),
        }
    }
}

struct LocData {
    id:          String,
    name:        String,
    path:        String,
    count:       i32,
    is_starred:  bool,
    is_selected: bool,
}

/// Label a picked folder by its own name, falling back to the full path for
/// roots (`/`, `D:\`) that have no file name component.
fn folder_display_name(path: &Path) -> String {
    path.file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.to_string_lossy().into_owned())
}

/// Select the location with `id`, deselecting every other one.
fn mark_selected(locs: &mut [LocData], id: &str) {
    for loc in locs {
        loc.is_selected = loc.id == id;
    }
}

fn starred_paths(db: &std::sync::Arc<std::sync::Mutex<maple_db::Database>>) -> std::collections::HashSet<String> {
    maple_db::lock_db(db)
        .get_starred_paths()
        .unwrap_or_default()
        .into_iter()
        .collect()
}

fn build_import_locations(
    settings: &maple_state::Settings,
    dirs: &KnownDirs,
    starred: &std::collections::HashSet<String>,
) -> Vec<LocData> {
    let mut locs: Vec<LocData> = Vec::new();

    // Pictures directory
    if let Some(pictures) = &dirs.pictures {
        if pictures.is_dir() {
            let path = pictures.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "pictures".into(),
                name: "Pictures Library".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Desktop
    if let Some(desktop) = &dirs.desktop {
        if desktop.is_dir() {
            let path = desktop.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "desktop".into(),
                name: "Desktop".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Downloads
    if let Some(downloads) = &dirs.downloads {
        if downloads.is_dir() {
            let path = downloads.to_string_lossy().into_owned();
            locs.push(LocData {
                id: "downloads".into(),
                name: "Downloads".into(),
                is_starred: starred.contains(&path),
                path,
                count: 0,
                is_selected: false,
            });
        }
    }

    // Library directory (from settings) if different from Pictures
    let lib = settings.library_dir.clone();
    if lib.is_dir() && dirs.pictures.as_deref() != Some(lib.as_path()) {
        let path = lib.to_string_lossy().into_owned();
        locs.push(LocData {
            id: "library".into(),
            name: "Library Folder".into(),
            is_starred: starred.contains(&path),
            path,
            count: 0,
            is_selected: false,
        });
    }

    locs
}

fn partition_locations(locs: Vec<LocData>) -> (ModelRc<ImportLocation>, ModelRc<ImportLocation>) {
    let mut favs: Vec<ImportLocation> = Vec::new();
    let mut recents: Vec<ImportLocation> = Vec::new();
    for l in locs {
        let item = ImportLocation {
            id:          l.id.into(),
            name:        l.name.into(),
            path:        l.path.into(),
            count:       l.count,
            is_starred:  l.is_starred,
            is_selected: l.is_selected,
        };
        if item.is_starred {
            favs.push(item);
        } else {
            recents.push(item);
        }
    }
    (
        ModelRc::from(Rc::new(VecModel::from(favs))),
        ModelRc::from(Rc::new(VecModel::from(recents))),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use slint::Model;
    use std::collections::HashSet;

    fn loc(id: &str, starred: bool) -> LocData {
        LocData {
            id: id.into(),
            name: id.into(),
            path: format!("/{id}"),
            count: 0,
            is_starred: starred,
            is_selected: false,
        }
    }

    #[test]
    fn build_import_locations_finds_pictures_desktop_and_dedupes_library() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir(home.join("Pictures")).unwrap();
        std::fs::create_dir(home.join("Desktop")).unwrap();
        // No Downloads dir created — should be skipped.

        // Library dir same as Pictures -> should not duplicate.
        let settings = maple_state::Settings {
            library_dir: home.join("Pictures"),
            ..Default::default()
        };

        let dirs = KnownDirs {
            pictures: Some(home.join("Pictures")),
            desktop: Some(home.join("Desktop")),
            downloads: Some(home.join("Downloads")),
        };

        let starred = HashSet::new();
        let locs = build_import_locations(&settings, &dirs, &starred);

        let ids: Vec<&str> = locs.iter().map(|l| l.id.as_str()).collect();
        assert!(ids.contains(&"pictures"));
        assert!(ids.contains(&"desktop"));
        assert!(!ids.contains(&"downloads"));
        assert!(!ids.contains(&"library"));
    }

    #[test]
    fn build_import_locations_marks_starred_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path();
        std::fs::create_dir(home.join("Pictures")).unwrap();

        let settings = maple_state::Settings::default();
        let dirs = KnownDirs {
            pictures: Some(home.join("Pictures")),
            desktop: None,
            downloads: None,
        };
        let starred: HashSet<String> =
            [home.join("Pictures").to_string_lossy().into_owned()].into_iter().collect();

        let locs = build_import_locations(&settings, &dirs, &starred);
        let pictures = locs.iter().find(|l| l.id == "pictures").unwrap();
        assert!(pictures.is_starred);
    }

    #[test]
    fn mark_selected_selects_one_and_clears_the_rest() {
        let mut locs = vec![loc("pictures", false), loc("desktop", false), loc("downloads", false)];
        locs[0].is_selected = true;

        mark_selected(&mut locs, "desktop");

        assert!(!locs[0].is_selected);
        assert!(locs[1].is_selected);
        assert!(!locs[2].is_selected);
    }

    #[test]
    fn mark_selected_with_unknown_id_clears_everything() {
        let mut locs = vec![loc("pictures", false), loc("desktop", false)];
        locs[1].is_selected = true;

        mark_selected(&mut locs, "nope");

        assert!(locs.iter().all(|l| !l.is_selected));
    }

    #[test]
    fn partition_locations_splits_starred_into_favorites() {
        let locs = vec![loc("pictures", true), loc("desktop", false), loc("downloads", true)];

        let (favs, recents) = partition_locations(locs);

        assert_eq!(favs.row_count(), 2);
        assert_eq!(recents.row_count(), 1);
        assert_eq!(favs.row_data(0).unwrap().id, "pictures");
        assert_eq!(favs.row_data(1).unwrap().id, "downloads");
        assert_eq!(recents.row_data(0).unwrap().id, "desktop");
    }

    #[test]
    fn folder_display_name_uses_the_last_component() {
        assert_eq!(folder_display_name(Path::new("/home/user/Photos")), "Photos");
    }

    #[test]
    fn folder_display_name_falls_back_to_the_full_path_for_a_root() {
        // `/` has no file-name component — the label would otherwise be empty.
        assert_eq!(folder_display_name(Path::new("/")), "/");
    }
}
