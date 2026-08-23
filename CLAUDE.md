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
- **`load` vs `refresh`**: `load(query)` is *navigation* — new context, model cleared, back to
  page 0 and the top of the list. `refresh()` is an *event* — same query, re-reads only the
  rows already on screen and patches them in place, keeping the scroll offset, the decoded
  thumbnails and the selection. Everything that changes the library out of band goes through
  `grid::request_reload` (which calls `refresh`): the 60-second scanner, the metadata filler,
  an import, a sync pass, a rotation. Two rules make it safe: the model is replaced in a
  *single* `set_vec` (an empty model for even one frame collapses `viewport-height` in
  `library.slint` and Slint clamps the scroll to zero), and a tile is only reused when the
  row's `(id, hash)` still match, since a rotation mints a new hash for the same id. A
  refresh whose rows are unchanged does nothing at all — the scanner fires one every minute
  forever.
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
- **Discovery** (`maple-sync/src/discovery.rs`, P8): a master advertises
  `_maple-sync._tcp.local.` (`mdns-sd`, no async runtime) with `device_id`, `name` and
  `protocol` in TXT; a servant browses so the pairing modal can offer a pick-list and so
  the worker can **re-resolve on the failure path** — a moved master heals without
  re-pairing. A record is unauthenticated hearsay, so discovery may only choose *where*
  to dial, never *who* to trust: the only thing it is allowed to do afterwards is move an
  already-paired device id to a new address, and the credential does not travel with the
  record. Manual `host:port` stays the fallback for networks that block multicast
  (`discovery: None` is a fully working link). `ServiceDaemon` has **no `Drop`**, so
  `Advertiser` and `Browser` shut theirs down in their own — dropping one otherwise
  leaks a thread and two sockets per role switch. Everything downstream takes a
  `DeviceSource`, which a test implements in five lines; only those two structs touch a
  socket.
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
- **Import scan pipeline** (`maple-ui/src/import.rs`): three stages split along where
  the work blocks. **Reading is serial**, on one thread — a camera card is a single bus,
  and twelve readers on it each get a twelfth of the bandwidth and all finish late
  *together*, so the grid sits empty for minutes and then fills in a burst (measured: it
  made time-to-first-tile worse, not better). One reader also opens each file **once**;
  hashing and decoding used to open it separately, doubling traffic over the slowest link
  in the pipeline. `read_with_budget` hashes the *file* bytes — never a raw's preview, so
  the digest stays the identifier `SeenSet` and the library use. **Decoding fans out**
  across `[import] decode_threads` workers pulling from one bounded queue: pure CPU over
  bytes already in memory, so it cannot stall on the card. **Embedding is serial again**,
  behind everything, because it owns the single `&mut` ONNX session and the embedding
  cache; its results arrive later as `ScanMsg::Embedding`, after the tile is already on
  screen. Three rules: the tile is sent *before* the embed job (that queue blocks, and
  the user should already have the picture); every queue is bounded (`sync_channel`),
  since an unbounded one would pull a whole card into RAM; and each stage's original
  sender is dropped inside the `thread::scope` body, or the stages wait on each other
  forever. A file the card never returns from is **outlived, not cancelled** — the read
  runs on its own thread and `recv_timeout` walks away after `read_timeout_secs`, the
  same trade `image_loader.rs` makes. Its path is logged under the
  `maple::import::unreadable` target and the photo stays listed, selectable and copyable
  with no preview. A display file that will not *decode* falls back to the group's other
  files — a raw + JPEG pair lists the JPEG as the display file, so one corrupt JPEG would
  otherwise lose the preview for a photo whose raw is intact. That fallback reads from a
  decode thread, which the pipeline otherwise never does; it is safe only because it is
  rare (once per bad file, not once per photo). Two limits on it: the row keeps the
  **display file's** hash, since that is what the library row and `SeenSet` are keyed on,
  and it is not attempted after a *timeout* — the card is already not answering, and
  asking twice would double the stall.
- **Import previews are decoded on demand** (`maple-ui/src/import_previews.rs`): a card
  of a few thousand photos will not fit in memory as decoded frames (~196 KB each) and
  almost none of them are on screen, so the browser decodes what the user is *looking at*.
  Two ideas carry it. **Priority is evaluated when a worker picks up work, not when it is
  queued**: the queue holds wanted indices plus one `focus`, and a worker always takes the
  pending index nearest it, so scrolling re-prioritises everything already queued by
  writing a single number — no re-sorting, no cancelling, and no waiting out a backlog of
  photos that scrolled past before the ones now on screen get decoded. **Retention is
  two-tier**: `Retained` is an LRU of decoded frames capped by `[import]
  max_loaded_previews`, keyed on *when the photo was last in view* (scrolling back to one
  saves it), and eviction drops the pixels down to a ~15 KB WebP kept from the first
  decode — so scrolling back re-inflates from memory and never returns to the card.
  Reading still happens one file at a time (same bus argument as the scan); only the
  decode fans out. The scan itself no longer produces tile pixels at all: it sends the
  full listing up front (so the count and every filename are right from the first frame),
  then reads and hashes, and only decodes when burst detection needs pixels for the
  embedder. `ImportItem::loaded` therefore means "has a decoded preview", not "has been
  scanned", and goes false again on eviction. Requests are clamped to the retention cap —
  a window wider than what can be held would have the tail evicting the head forever.
- **The import record lives on the medium** (`maple-state/src/seen.rs`, P9): the scan's
  "already imported" badge reads `<source>/.maple_seen.bin`, written to the card itself
  so it carries its own history to the next machine — beside `.maple_embed_cache.bin`,
  which established the idiom. `library_dir/seen_imported.bin` stays as a
  *non-authoritative* replica, read only when the source has no readable record of its
  own (read-only card, network share, a folder that moved); a **corrupt** medium record
  falls back the same way as a missing one, since reading it as an empty set would send
  the user re-importing a whole card. `SeenSet` is **grow-only**, so saving is a
  read-merge-write union (`merge_save_to_source`) with no locking and no conflict
  resolution — that is what makes two importers running at once combine instead of
  clobber, and it is the same code path that folds a card's history into a library that
  has never seen it. Two invariants worth keeping: the all-zero `UNHASHED` sentinel the
  scan stores on a hash failure is refused by `insert` *and* dropped on load (it would
  otherwise badge every unreadable photo as imported), and `save_to` stages into a
  dot-prefixed scratch file and renames, so an ejected card mid-write leaves the previous
  record rather than a truncated one.
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
