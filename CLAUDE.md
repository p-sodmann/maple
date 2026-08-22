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
| `maple-sync` | Sync *transport*: pairing handshake, `sync_trust.json`, request signing |
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
- **Sync stamping**: every row in `maple_db::SYNCED_TABLES` carries `guid`/`rev`/`rev_dev`.
  Writers stamp *explicitly* via `Database::stamp()` — never via a trigger, since V17
  removed the last `AFTER UPDATE ON images` trigger for exactly that cost. Writes to
  machine-local columns (`status`, `path`, `raw_path`, `filename`, `locality`,
  `origin_device`) and to derived columns (centroids, `representative_*_id`) deliberately
  do **not** stamp; each carries a comment saying so. Deletes call
  `Database::tombstone(table, ids)` *before* the `DELETE`.
- **Merge engine** (`maple-db/src/sync/`): `collect_changes` reads local changes above a
  watermark, `apply_batch` merges a peer's. It lives in `maple-db`, not `maple-sync`,
  because merging needs transactional SQL over this schema — `maple-sync` is transport
  only. The property test `random_concurrent_edits_always_converge` drives two real
  databases through randomised concurrent edits; **keep it deterministic** (no
  `ORDER BY RANDOM()`), or a failing seed can't be reproduced to debug.
- **Pairing and signing** (`maple-sync/`): a mutual 6-digit-code handshake derives a
  shared secret from *both* codes, then seals a 32-byte long-term key stored in
  `config_dir()/sync_trust.json` (mode `0600`, written atomically — deliberately not
  `settings.toml`, whose `save` is non-atomic and eats the file's comments). Every later
  request is MAC'd with that key. Transport is plain HTTP with **no TLS**: signing stops
  impersonation and replay, not eavesdropping. Nothing in the crate samples the clock or
  the RNG — `now_ms` is an argument and randomness arrives through `RandomSource`
  (production: `Database::random_bytes`, SQLite `randomblob`), so every handshake and
  signature is reproducible in a test.
- **Relay** (`images.locality`, V20): a servant can browse the master's library while
  storing no originals. `locality='remote'` rows are `status='present'` and list
  normally; `all_paths()` filters them out so the 60-second scanner does not mark them
  missing and evict their thumbnails. Pixels come from two signed routes on the master —
  `GET /blob/thumb/{hash}` (rendered on a cache miss through a `ThumbRenderer` injected
  from `maple-ui`, because `maple-ui` depends on `maple-sync` and not the reverse) and
  `GET /blob/orig/{hash}` (streamed **verbatim**, so the receiver can verify its
  BLAKE3). The client handle lives in `maple-ui/src/remote.rs` as a process-wide
  `RemoteBlobs`, written and *cleared* by `SyncSupervisor::restart`. Thumbnails are
  cached locally; originals are memory-only — that is what makes it a relay.
- **Moving originals** (`maple-sync/src/transfer.rs`, P7): what makes `PeerMode::Full`
  and `Partial` mean something. **Both directions are driven by the servant** — a master
  has no client and does not know how to reach a servant, so "the master fetches the
  servant's originals" (§3.8) is really the servant asking `POST /sync/wanted` and
  uploading with `POST /blob/orig/{hash}`. Three rules that are easy to break:
  - **The receiver verifies.** A display file's hash is in the (signed) URL and is
    checked against the bytes before anything is written into a library. That content
    address is *why* the upload route can sign an empty body and stream a 100 MB raw to
    disk instead of buffering it to check a MAC. A **companion raw is unverifiable** —
    the schema hashes the display file only — and is accepted on the pairing's word; an
    `images.raw_hash` column would close that.
  - **Only a row that is already waiting can be filled in.** `Database::row_wanting`
    gates every upload, so a paired peer can complete a photo this library already
    replicated the metadata of and cannot invent, replace or misplace one.
  - **Rename and adopt happen under one database lock.** The 60-second scanner inserts
    any file no row claims and `images.path` is UNIQUE, so a scan landing between the
    two would take the path and leave `adopt_original` failing on the constraint. Bytes
    stage in `library_dir/.incoming` (hidden, so the scanner skips it) with no lock held;
    only the rename and the row update are locked.
  Anything that reads `images.path` to *open a file* must filter `locality = 'local'` —
  the AI tagger, face tagger, perceptual hasher, metadata filler and restructure planner
  all do. `ServerDeps::on_change` is the master's equivalent of the servant worker's:
  a master polls nothing, so without it a photo a servant sent sits unseen until the app
  restarts. (The 60-second scanner still refreshes nothing — that gap predates sync.)
- **Raw file support**: Only Fujifilm RAF currently. Always use `maple_import::loadable_image_bytes(path)` for loading images (handles raw preview extraction transparently). Check format with `maple_import::is_raw_format(path)`.

### Database
- SQLite in WAL mode, schema versioned via `PRAGMA user_version` (currently v20).
- Migrations live in `maple-db/src/schema.rs` as append-only `if version < N` steps that
  **replay history** — a fresh database runs every step in order, so a later step may
  undo an earlier one (V17 drops the FTS table V2 creates). Add new steps at the end;
  never edit an existing one.
- `PRAGMA foreign_keys` is enabled explicitly in `Database::open`, so `ON DELETE CASCADE`
  really cascades. Sync depends on this: one tombstone for a parent row propagates a
  delete, and each device's own cascade clears that device's children.
- One row per conceptual image; raw companions stored in `raw_path` column.
- Text search is `LIKE` across `images`, `ai_descriptions`, `persons` and
  `image_exif_tags` (`text_from_where` in `lib.rs`). The V2 `image_fts` table was never
  read and was dropped in V17 — its write triggers made every `images` insert ~7× more
  expensive and fired on every bulk `status`/`stack_id` update.

## Key Directories

```
maple-ui/ui/                 — Slint markup (app.slint, detail.slint, library.slint, …)
maple-ui/src/                — Rust UI controllers (grid.rs, detail.rs, import.rs, …)
maple-db/src/models/         — ONNX inference framework (detection, embedding, session)
maple-import/src/            — scan, copy, hash, raw format support
```
