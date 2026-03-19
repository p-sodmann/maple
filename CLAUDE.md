# Maple — Photo Library Manager

GTK4/libadwaita desktop app for importing, browsing, and organising photos.

## Build & Test

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

Requires GTK4 and libadwaita development headers (`libgtk-4-dev`, `libadwaita-1-dev` on Debian/Ubuntu).

ONNX Runtime (`ort`) is loaded dynamically — face detection/embedding features need `ORT_DYLIB_PATH` or a system-installed `libonnxruntime.so`.

## Workspace Crates

| Crate | Purpose |
|---|---|
| `maple` | Binary entry point (`main.rs` → `app.rs`) |
| `maple-ui` | GTK4/Adwaita UI: window, views, widgets |
| `maple-state` | Settings (settings.toml), Session (session.json), SeenSet (bloom filter) |
| `maple-import` | Recursive image scanner, BLAKE3 hasher, file copier, raw file support |
| `maple-db` | SQLite library database, background scanner, EXIF, AI tagging, face detection |
| `maple-embed`, `maple-cluster`, `maple-tournament` | Future stubs (not yet implemented) |

## Architecture

### Threading model
- **No tokio runtime** at the top level. All background work uses `std::thread::spawn` + `std::sync::mpsc` channels.
- GTK main loop polls results via `glib::timeout_add_local`.
- Database shared as `Arc<Mutex<maple_db::Database>>` across threads.
- UI-local state uses `Rc<RefCell<T>>` or `Rc<Cell<T>>` for GTK closure captures.

### Navigation flow
`main.rs` → `app::run()` → `window::build_window()` → `home::build_home_page()`
- Home → "Import Photos" → `source_picker::build_picker_page` → `image_browser::build_browser_page`
- Home → "Browse Library" → `library::build_library_page` → `detail_window::open` (on cell click)

### Key patterns
- **Generation counter**: `LibraryGrid` increments a counter on each `load()`; stale glib pollers self-terminate on mismatch.
- **Clone-shared structs**: Types like `LibraryGrid` wrap `Rc` internals and are cheaply cloned for closure captures.
- **Background workers**: AI tagger, face tagger, library scanner all follow the same spawn→loop→sleep→check-stop pattern.
- **Raw file support**: Only Fujifilm RAF currently. Always use `maple_import::loadable_image_bytes(path)` for loading images (handles raw preview extraction transparently). Check format with `maple_import::is_raw_format(path)`.

### Database
- SQLite in WAL mode, schema versioned via `PRAGMA user_version` (currently v5).
- One row per conceptual image; raw companions stored in `raw_path` column.
- FTS5 table `image_fts` for full-text search across EXIF fields, AI descriptions, and person names.

## Key Directories

```
maple-ui/src/views/          — all UI pages
maple-ui/src/views/library/  — library browser (grid, search, detail window, face tagging)
maple-db/src/models/         — ONNX inference framework (detection, embedding, session)
maple-import/src/            — scan, copy, hash, raw format support
```
