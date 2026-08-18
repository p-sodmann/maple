# Maple — Photo Library Manager

Cross-platform desktop app for importing, browsing, and organising photos.
UI built with **Slint** (no GTK/libadwaita runtime required).

## Build & Test

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

No system GTK headers needed — Slint ships its own renderer. A C toolchain is
still required (MSVC Build Tools on Windows, Xcode CLT on macOS,
`build-essential` on Linux): `rusqlite` (`bundled`), `sqlite-vec`, and `webp`
all compile vendored C sources via the `cc` crate.

ONNX Runtime (`ort`) backend is selected via a Cargo feature on `maple-db`,
forwarded through `maple-ui` and `maple` — **two separate build variants**:

```sh
# CPU build (default) — no flags needed:
cargo build --workspace

# GPU build — opt in explicitly, one package at a time:
cargo build --release -p maple --no-default-features --features gpu
```

- **`cpu` (default)**: `ort` downloads a platform-matched prebuilt onnxruntime
  binary at build time (`download-binaries` + `copy-dylibs`) and links it in.
  Needs network access the first time you build; CPU-only (no CUDA/TensorRT).
  `device = "cuda:0"` / `"tensorrt:0"` in settings.toml silently fall back to
  CPU (ort's normal EP-unavailable behavior) — this is expected on a `cpu`
  build.
- **`gpu`**: `ort` uses `load-dynamic` — no network needed to build, but the
  resulting binary loads onnxruntime *at runtime* instead, and needs a real
  shared library to even start (not just for GPU: CPU inference on a `gpu`
  build still requires this). Point `ORT_DYLIB_PATH` at a CUDA/TensorRT-
  enabled onnxruntime release (e.g. Microsoft's official `onnxruntime-gpu`
  build), or place the library where the OS's dynamic loader finds it by
  default (next to the `.exe` on Windows; `LD_LIBRARY_PATH`/rpath on Linux;
  `DYLD_*` paths on macOS).

Ship both variants as separate release artifacts per platform if you want to
offer GPU acceleration — there's no runtime fallback between the two within a
single binary; the backend is fixed at compile time.

## Workspace Crates

| Crate | Purpose |
|---|---|
| `maple` | Binary entry point (`main.rs` → `maple_ui::run()`) |
| `maple-ui` | Slint UI: windows, views, widgets; `ui/*.slint` compiled by `build.rs` |
| `maple-state` | Settings (settings.toml), Session (session.json), SeenSet (bloom filter) |
| `maple-import` | Recursive image scanner, BLAKE3 hasher, file copier, raw file support |
| `maple-db` | SQLite library database, background scanner, EXIF, AI tagging, face detection |
| `maple-embed`, `maple-cluster`, `maple-tournament` | Future stubs (not yet implemented) |

## Architecture

### Threading model
- **No tokio runtime** at the top level. All background work uses `std::thread::spawn` + `std::sync::mpsc` channels.
- Slint event loop drains channels via `slint::Timer` (repeated) or `slint::Weak::upgrade_in_event_loop` for one-shot loads.
- Database shared as `Arc<Mutex<maple_db::Database>>` across threads.
- UI-local state uses `Rc<RefCell<T>>` or `Rc<Cell<T>>` for Slint closure captures.

### Navigation flow
`main.rs` → `maple_ui::run()` → `AppWindow::new()` (home page)
- Home → "Import Photos" → `ImportWindow::new()` (rfd folder picker → scan → copy)
- Home → "Browse Library" → `LibraryPage` → `DetailWindow::open()` (on cell click)
- Library header → "Collections" → `CollectionsWindow::new()`
- Library header → "Settings" → `SettingsWindow::new()`

### Key patterns
- **Generation counter**: `LibraryGrid` increments a counter on each `load()`; stale `slint::Timer` pollers self-terminate on mismatch.
- **Clone-shared structs**: Types like `LibraryGrid` wrap `Rc` internals and are cheaply cloned for closure captures.
- **Singleton windows**: Each secondary window (`DetailWindow`, `ImportWindow`, `SettingsWindow`, `CollectionsWindow`) is held as `thread_local! { static X: RefCell<Option<T>> }`. Strong handle lives only there; all callbacks capture `slint::Weak`.
- **Context struct + `wire_*` functions**: each window builds one context holding its shared handles (`AppCtx` in `lib.rs`, `ImportCtx` in `import.rs`, `NavState` in `detail.rs`) and passes it to one `wire_*` function per feature block instead of wiring every callback in one scope. The context holds the window as a `slint::Weak` only — callbacks clone fields out of it, never a strong handle. Startup call order is load-bearing (paging wired before the first `grid.load`, `grid::register` before any window that calls `request_reload`).
- **Background workers**: AI tagger, face tagger, library scanner all follow the same spawn→loop→sleep→check-stop pattern.
- **Raw file support**: Only Fujifilm RAF currently. Always use `maple_import::loadable_image_bytes(path)` for loading images (handles raw preview extraction transparently). Check format with `maple_import::is_raw_format(path)`.

### Database
- SQLite in WAL mode, schema versioned via `PRAGMA user_version` (currently v5).
- One row per conceptual image; raw companions stored in `raw_path` column.
- FTS5 table `image_fts` for full-text search across EXIF fields, AI descriptions, and person names.

## Key Directories

```
maple-ui/ui/                 — Slint markup (app.slint, detail.slint, library.slint, …)
maple-ui/src/                — Rust UI controllers (grid.rs, detail.rs, import.rs, …)
maple-db/src/models/         — ONNX inference framework (detection, embedding, session)
maple-import/src/            — scan, copy, hash, raw format support
```
