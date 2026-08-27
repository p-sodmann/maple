# Maple — Photo Library Manager

Cross-platform desktop app for importing, browsing, and organising photos.
UI built with **Slint** (no GTK/libadwaita runtime required).

## Build & Test

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

### Session-detection lab

```sh
cargo run --release -p maple-db --bin session-lab -- <dir> --out /tmp/sessions.html
```

Runs every session engine over a real folder in one pass — one decode, every
engine sees the same frame — and reports what each cost and where each cut.
That frame is the **canonical preview** (`maple_import::preview`), byte-for-byte
what the import scan hands its own engines: the lab measured pristine pixels
once while the scan measured Lanczos3-resized ones, which quietly meant nothing
tuned here transferred.
Text gives per-engine ms/photo, bytes per signature, session-size statistics
and a pairwise boundary-F1 matrix (no ground truth needed: if a cheap engine
cuts a card where DINOv2 cuts it, the expensive one has nothing left to
justify).

`--out` writes an **interactive** page, not a static contact sheet: every
threshold, ensemble weight and time-curve point is a live slider and the
segmentation recomputes in the browser as you drag. `s` marks the session
under the cursor as a scene with draggable start/end handles, `x` flags an
outlier, and the result exports as a `--truth` file — which is how "that looks
about right" becomes a number the next run is scored against.

Two things make that work. Re-segmenting needs distances between arbitrary
pairs and a full matrix is `n²`, but segmentation only ever asks about pairs
*inside* one session, so the page ships a **banded** matrix (`--band`, default
48) quantised to a byte per pair; beyond the band a distance reads as 1.0,
which ends a session by drift, so a session longer than the band is capped in
the browser though not in Rust. And the JavaScript is a hand-written mirror of
`maple_import::session::segment`, which can rot — so the page carries Rust's
own answer and paints a red banner if its recomputation disagrees.

Other flags: `--dino` adds the baseline (downloads the model), `--ensemble
block-tile=2,time-gap=1` adds a weighted vote, `--cut <engine>=<f>` overrides a
threshold, `--time-points 1=0,60=0.5,600=0.85,3600=1` reshapes the time curve,
`--max-outliers N` sets how many non-matching frames a session may absorb.

### Dependency updates

```sh
./scripts/update-deps.py            # report only
./scripts/update-deps.py --apply    # bump manifests, cargo update, cargo check
```

Stdlib-only Python: reads the requirements out of every `Cargo.toml`, asks the
crates.io sparse index for the newest non-yanked release, and compares both
against `Cargo.lock`. A `lock` row is inside the existing requirement (`cargo
update` alone gets it); a `bump` row is semver-incompatible and rewrites the
version literal in the manifest, keeping the requirement's precision and the
comments around it. `=x.y.z` pins (`ort`) are reported and skipped unless
`--include-pinned` — the pin is deliberate, and ort's index carries only
prereleases.

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
- **One preview, and everything reads it** (`maple-import/src/preview.rs`): a photo's
  pixels are wanted by four things — the tile, the sharpness score, the session
  signature, sometimes the embedder — and the obvious implementation gives each whatever
  decode is in hand, leaving no answer to "which pixels did the detector actually see".
  So the pipeline narrows to one artefact: `preview::encode` makes a 256 px WebP
  (~15 KB against a decoded frame's ~196 KB), `preview::decode` turns it back into a
  frame, and **every check runs on the output of `decode`** while the WebP is the only
  thing kept. Two properties fall out. What was checked is what is shown — the signature
  that grouped a photo came from the same lossy pixels on screen, not a pristine decode
  that exists nowhere. And a recomputation *agrees*: re-deriving a signature or a
  sharpness score from a stored preview, days later or on another machine, reproduces
  the scan's own answer, because there was no other input it could have used. The
  round trip costs an encode plus a decode on the pipeline's *parallel* stage while the
  one serial reader spends ~100 ms/photo — nothing that shows on a clock. It lives in
  `maple-import` because the importer and `session-lab` must see byte-identical frames
  or a threshold tuned in the lab does not transfer; they had already drifted apart once
  (the lab downsampling with `image::thumbnail` against the scan's Lanczos3), which is
  the divergence the shared module exists to make impossible. `maple-ui/src/thumbnail.rs`
  keeps only the library's own cover crop and delegates the rest. One measured
  consequence, recorded in the module docs rather than discovered later: compression
  artefacts are high-frequency, which is what variance-of-Laplacian measures, so
  sharpness reads slightly high — pristine 966/157/27/4 over a progressively blurred
  image comes back as 930/192/58/20. Ordering survives, which was all the old
  auto-picking needed and is all the tournament's on-screen hint claims; the floor
  only makes two already-blurry photos harder to tell apart.
- **The medium remembers its own previews** (`maple-import/src/preview_cache.rs`):
  `.maple_previews.bin` beside `.maple_seen.bin` and `.maple_embed_cache.bin`, so a card
  carries to the next machine the previews already made from it. Unlike its neighbour
  `EmbeddingCache` it is keyed by **path, size and mtime — not content hash** — and the
  difference is the entire point: a content hash can only be computed by reading the
  whole file, which is precisely the cost a hit is meant to avoid. Keyed this way a hit
  never opens the file at all, so rescanning an unchanged card is a `stat` per photo
  instead of a ~100 ms read, and the *content hash rides in the record* for the same
  reason. The trade, stated in the module docs because the failure would be silent:
  `(path, size, mtime)` is taken as the file's identity, so a file replaced in place at
  the same size and mtime would be served the previous one's preview **and its hash**,
  and could be badged with the wrong import history. Camera cards write once and every
  editor moves the mtime — the same assumption `make` and `rsync` ship with. The file is
  **append-only** (tens of megabytes of previews; rewriting it per batch the way
  `EmbeddingCache` does would spend the savings back on the card), later records win,
  each record is length-prefixed so a card pulled mid-write costs exactly the truncated
  tail, and `flush` compacts whenever dead records outnumber live ones — which is what
  stops a card formatted and refilled a few times carrying every generation of
  `DSCF0001.JPG` forever. Writes go through one `preview_cache_stage` thread for the
  same reason there is one reader: a card is one bus, and N decoders appending to it
  would contend with the reads still going on.
- **Import scan pipeline** (`maple-ui/src/import.rs`): four stages split along where
  the work blocks. **Reading is serial**, on one thread — a camera card is a single bus,
  and twelve readers on it each get a twelfth of the bandwidth and all finish late
  *together*, so the grid sits empty for minutes and then fills in a burst (measured: it
  made time-to-first-tile worse, not better). One reader also opens each file **once**;
  hashing and decoding used to open it separately, doubling traffic over the slowest link
  in the pipeline — and increasingly it opens each file **not at all**: `read_one`
  consults the medium's own preview cache from the directory entry first, and a hit
  supplies the preview, the capture time *and* the content hash without a read.
  `read_with_budget` hashes the *file* bytes — never a raw's preview, so the digest
  stays the identifier `SeenSet` and the library use. **Decoding fans out**
  across `[import] decode_threads` workers pulling from one bounded queue: pure CPU over
  bytes already in memory, so it cannot stall on the card — and it is where the
  canonical preview is *made*, for the same reason EXIF is parsed here: the reader is the
  one serial stage and the slowest link, and this is pure CPU over bytes already in hand.
  **Signing is serial again**,
  behind everything: `signature_stage` owns the one `&mut dyn SessionEngine`, and that is
  load-bearing rather than incidental — `TimeGapEngine` latches its epoch on the first
  frame it sees, so one engine per decode thread would give each its own origin and make
  their signatures incomparable. Its results arrive later as `ScanMsg::Signature`, after
  the tile is already on screen, and frames reach it in *decode-completion* order, which
  is fine: a signature describes one photo and nothing else, and scan order is restored
  on the UI thread where each lands in its own entry. The DINOv2 embed stage still runs
  beside it when `[stacks] enabled` is on, but only to **store** an embedding with the
  copied photo so the library's own stacker need not compute it again — it no longer
  groups anything here. Cost is why: 26 ms/photo against the session engines' ~0.2 ms,
  and its bounded queue backpressures the decoders and thence the reader. EXIF is parsed
  on the *decode* thread from the bytes already in hand (`exif_read::read_bytes`), never
  on the reader — the reader is the one serial stage and the slowest link, and reopening
  each file for its timestamp would pay for session detection twice over. **Writing the
  medium's preview cache is serial too** (`preview_cache_stage`), and for the reader's
  own reason rather than the engine's: a card is one bus, so N decoders appending to it
  independently would contend with reads still in flight. It batches
  `PREVIEW_CACHE_FLUSH_EVERY` records per append and holds the cache lock across the
  write — the right way round, since a scan that is *writing* the cache is one whose
  lookups are missing anyway. Three rules:
  the tile is sent *before* the pixel jobs (those queues block, and the user should
  already have the picture); every queue is bounded (`sync_channel`), since an unbounded
  one would pull a whole card into RAM; and each stage's original sender is dropped
  inside the `thread::scope` body, or the stages wait on each other forever. On the UI
  side the drain is **time-boxed, not count-boxed** (`SCAN_DRAIN_BUDGET`), and only
  messages that actually paint are charged against the budget. A fixed ten-per-tick cap
  sized on a small folder turns a real card into a trickle: 954 photos are 1,910
  messages, six seconds of ticking before `finish_scan` — which runs on the *last*
  message and is the only thing that segments — can produce a single session. The `f`
  grid spent all of that displaying its literal "no sessions detected" string, which is
  how this was found; it now says "scanning" until the scan is actually over, because
  "none yet" and "none found" are different answers. A file the card never returns from is **outlived, not cancelled** — the read
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
- **Session detection is segmentation, not clustering** (`maple-import/src/session/`):
  the case is twenty pictures of one child in one room over four minutes — not a
  drive-mode burst. Those photos are *contiguous* in capture order, and that is the whole
  exploit: `segment` walks the sequence once (`n-1` comparisons, not `n²`), asks the local
  question "did the scene change *here*", and cannot chain a slow pan into one 200-photo
  group the way `maple_db::cluster_embeddings`'s union-find does — a session has two ends.
  Path order off a camera card already *is* capture order, so the sequence exists before
  any EXIF is read. **Time is a cost, not a cut**: a fixed gap threshold cannot work when
  one session comes at 2 s, 3 s, 40 s (repositioning the child), 2 s — so the gap only
  moves the *visual* evidence a cut requires (long gap → cheaper, short gap → dearer),
  with `hard_gap_secs` the one hard rule and hysteresis so a single badly framed frame
  does not fragment a run. `anchor_factor` is the anti-chaining rule: distance is measured
  to the frame the session *started* on as well as to the neighbour.
  What "the scene changed" means is a pluggable **engine** — a descriptor plus a distance,
  with `segment` knowing nothing about either. Their distances are **not comparable to
  each other**, so each carries its own `default_cut`. `ColorKmeansEngine` sees the palette
  and not where any of it is; `GridHistogramEngine` sees colour *and* its layout;
  `BlockTileEngine` counts the fraction of the frame that held still, which is the one
  built for a moving subject in a fixed scene; `maple_db::DinoEngine` is the baseline, in
  `maple-db` because `maple-import` must stay free of `ort`. `TimeGapEngine` is an engine
  too, and votes on the clock alone: a curve through `1 s→0.00, 1 min→0.50,
  10 min→0.85, 1 h→1.00`, interpolated in **log time** because the interesting range
  spans four orders of magnitude. It is a separate mechanism from the threshold shaping
  above — running both means time counts twice, so zero `tight_hold`/`long_drop` when an
  ensemble already has a time member.
  `EnsembleEngine` combines any of them with tunable weights. The trick that makes a vote
  possible at all: each member is put on a shared scale by `d/(d+cut)` — its own cut is
  what it means by "changed", so every member reads 0.5 at its own threshold and the
  weighted mean crosses 0.5 exactly when the vote does.
  **A session may absorb an outlier**: each photo is judged against the last frame
  *accepted* into the session, not its predecessor, so one shot of the cake in the middle
  of twenty of the child is bridged when the next frame comes back — up to `max_outliers`
  in a row. When patience runs out the cut lands *before* the first frame that stopped
  matching, since that is where the new scene actually started.
  Measured on a real folder: the cheap three cost 0.13–0.37 ms/photo and 80 B–1.5 KB per
  signature, DINOv2 26 ms and (unpooled) 385 KB — and read+decode is ~100 ms/photo, so the
  cheap engines are free and DINOv2 is not. `grid-histogram` and `block-tile` agree with
  each other at 0.95 boundary-F1 and with a `block-tile×2 + grid-histogram + time-gap`
  ensemble at 0.93; `color-kmeans` is the outlier at 0.70–0.74, which is the palette
  blindness showing.
- **Sessions are what the importer groups by** (`[sessions]` in settings.toml): they
  replaced the DINOv2 burst clustering, which cost 26 ms/photo, throttled the whole scan
  through its embed queue, defaulted to off, and chained transitively besides. `[stacks]
  enabled` now governs only post-import stacking in the library; when it is on the scan
  still computes an embedding, but purely to *store* it with the copied photo.
  `engine_from_spec` is the one place a **string** becomes an engine, so settings.toml,
  the lab's `--ensemble` and any future UI resolve a name identically. Two rules travel
  with it. A `cut` of `0` means "ask the engine", never "no threshold" — engine distances
  are not comparable to each other, so a number carried over from a different engine is
  meaningless rather than merely wrong. And `SegmentParams::for_spec` **zeroes
  `tight_hold`/`long_drop` when the spec already contains a time member**, because time
  as a *cost* plus time as a *vote* counts the one signal most likely to be wrong (a
  camera with the wrong clock, a card holding two days) twice over. The lab mirrors that
  in `params_for` **and again in its JavaScript** — the lab is where thresholds get
  chosen, so it has to segment the way the importer will or nothing tuned there transfers.
- **The `f` grid is where a session gets corrected** (`toggle_boundary` /
  `ensure_boundary` in `maple-ui/src/import.rs`): `f` swaps the preview for every photo
  at once with sessions drawn as tinted bands, `esc` or `f` goes back. There are two
  vocabularies for the same edit. **Keys toggle**: `[` sets the start at the current
  photo and `]` the stop, and pressing one where a boundary already sits removes it,
  merging. **Clicks ensure**: the first click opens a session on that photo and the next
  closes it there, so a session is drawn by its two ends — and a click saying "it starts
  here" about a photo that already starts one must *leave* it starting one, which is
  exactly what toggling would not do. That is why `ensure_boundary` exists beside
  `toggle_boundary` rather than reusing it. A grid click also moves the cursor, so
  opening a session does not cost the ability to navigate by clicking. The half-finished
  edit lives in `ctx.pending_cut` and is dropped when the view closes: it describes an
  intention, and until the second click there is no boundary to record. Opening the grid
  **during a scan parks on the scan's own frontier** (`scanned_count`), not on wherever
  the cursor was left — the reason to open it mid-scan is to watch the card come in, and
  that happens at the far end of what has been read. Four things make it small. A boundary is the only thing a session
  *is* — sessions **tile** the sequence, so inserting one splits and removing one merges,
  which means "set the start here" and "set the stop here" are a single toggle one photo
  apart and there is no merge key, no drag state and no separate model. Correcting a
  boundary **rebuilds the tournament** rather than patching it, which costs nothing
  because a rebuild carries every verdict forward and re-asks only about photos still
  undecided — the user was correcting the grouping, not giving up a decision. And the
  grid **suspends "Hide old images"** rather
  than working around it, so one model serves both views — a band drawn over a sequence
  with photos missing from the middle would lie about what it contains.
  Two traps this view sprang, both about *rows*. `ImportGridView` scrolls to
  `current-grid-row` = `current-row / columns`, **not** to `current-row`: with tiles
  abreast, strip row 500 sits on grid row 500/columns, so scrolling to the strip row
  overshoots by exactly that factor — straight to the bottom of a real card. The
  one-column filmstrip made the two the same number, so the bug could not appear until
  something asked for more than one column. And the grid is created fresh each time it
  opens, while a Slint `changed` handler does **not** fire for a value an element was
  born with — hence the `init => park()` beside `changed current-grid-row`, or it would
  open at the top however far down the card the cursor is.
- **Nothing is preselected; the keeper is chosen in a tournament**
  (`maple-ui/src/import_tournament.rs`): the importer used to auto-mark the sharpest
  photo of each detected session. That is a guess dressed as a decision — variance-of-
  Laplacian ranks a crisp badly-framed frame above the one where the child is looking at
  the camera, and once it is marked nobody looks again. So the marking is gone and the
  comparison is put in front of the user: two photos side by side, `1` keeps the left,
  `2` the right, `3` both.
  **It is a detour, not a mode.** The switch in the header only *enables* it; there is
  nothing to flip between, because the view follows the cursor. Land on a photo that
  belongs to a session and its bracket takes over; land on one in no session and it is
  the ordinary single-photo triage, because there is nothing to compare it against.
  `Tournament::enter` is the whole of that decision and `go_to` is the only way the
  cursor moves, so every navigation path gets the right view without knowing the
  tournament exists. This replaced a design where the tournament *was* a mode, which
  meant a card was either all comparisons or all single photos — and the photos in no
  session, a sixth of a real card, were never visited at all.
  Each session is a **bracket**. **The first round pairs up photos nobody has looked at
  yet** — (1,2), (3,4), (5,6) — so every photo gets its first comparison against a peer
  and nothing is anyone's incumbent; later rounds are keeper against keeper over whoever
  is still standing. Losing eliminates. `3` advances **both**, which is what makes "keep
  both" mean something: not "I cannot decide" but "these both go through" — and a photo
  it saves is still in the running and can still lose later. **Whoever is standing when
  the session runs out of questions is kept**, and the cursor walks on past it.
  Two rules make it terminate, and they are the whole correctness argument. A pair is
  put to the user **once** (`Bracket::met`), so building a round either finds an unasked
  pair — whose answer eliminates somebody or records a new meeting, both bounded — or
  finds none, which ends the session. There is no way to loop and no way to be asked the
  same question twice. Cost lands where it should: answer `1`/`2` throughout and a
  session of *n* takes *n* − 1 comparisons, the theoretical minimum for finding a single
  best; say "keep both" often and it costs more because more photos are still in the
  running, which is the user asking for it. The ceiling is the complete graph, so `k`
  (`keep_rest`) ends a session keeping whoever is standing — the bound on a session
  answered `3` all the way down, and worth having anyway for "these are all fine". An
  arrow steps over the whole session, and that is a **deferral**: nothing in it is
  marked passed, because moving on from a question is not answering it.
  A bracket shows two photos at a time, which leaves no sense of how far through a
  session of twenty you are or what you already threw out — so **the whole session is
  drawn as a strip under the panes** (`TourneyStrip`, `GroupView`), the two contestants
  badged `L` and `R`, the eliminated dimmed and crossed. Its thumbnails are cached per
  session because a verdict changes what the cards *say*, not what they show.
  The cursor **stands on the left contestant** for as long as a session is on screen
  (`publish_tournament` moves it). Keeping the two in lockstep by construction is what
  makes `enter` idempotent and what lets an undo reopen a session the cursor has already
  walked past — the restored pair pulls the cursor back rather than the cursor having to
  be repaired afterwards.
  One trap the bracket sprang, and the reason `decide` returns a `Decision` rather than
  a bare `Vec`: **a verdict that settles nobody is not a verdict that did nothing.** `3`
  advances both photos and eliminates neither, so mid-session it settles no one — and
  the UI gated its repaint on the settled list being non-empty (true while the pass
  still had an incumbent, where every verdict settled someone). So `3` advanced the
  bracket behind the user's back and never repainted: dead in the middle of a session,
  working at the end of one, where the session closes and its survivors settle. `acted`
  and `settled` are separate fields so the mistake is hard to write again. It is worth
  knowing that no test covers this: the bug lives in `apply_verdict`, which needs a live
  `ImportWindow`, so the type is the guard rather than the suite.
  Losing is a **skip**, not an absence — it sets `passed`, so `commit_skips` writes it
  to the medium's Skipped record and a re-scan does not offer it again; recording it as
  "no answer yet" would hand the whole card back next time. `u` undoes, and that is not
  a nicety: `1` and `2` are one key apart and every press eliminates a photo. Undo is a
  **whole-bracket snapshot** rather than an incremental rewind — a bracket is one
  session, so cloning it is nothing, and a decision touches `alive`, `met`, `queue` and
  the round counter at once.
  The brackets are **rebuilt, never resumed**, from the groups and the verdicts carried
  inside the `Tournament` itself (`carry`) rather than in a second copy beside it. That
  one rule buys three behaviours free: switching the feature off and on resumes,
  correcting a session boundary re-groups what is *left*, and a photo already in the
  library never enters a comparison whose result could not be acted on.
  **Paired zooming is the reason it can replace a sharpness score at all**, and it only
  works because the crop comes off the *original*: scaling up a 256 px canonical preview
  would show big soft blocks, which is the exact opposite of the judgement being asked
  for. So `PairRenderer` keeps two decodes alive (capped at `MAX_SOURCE_PX` = 4096 px,
  ~34 MB a side) on one worker thread and `maple_import::preview::render_region` cuts the
  visible rectangle out of one at pane resolution — Rust renders to exactly the pane's
  pixel size rather than letting Slint scale, or a zoomed pixel would be a resampled
  resample. Requests **coalesce** — a drag makes them faster than they can be served, so
  the queue is drained to its newest entry per side, the same "re-prioritise by
  overwriting, never by cancelling" trade `import_previews.rs` makes. There is **one**
  zoom and **one** centre, in Rust, and both panes are drawn from them; two panes each
  keeping their own and following each other would drift the first time a render was
  dropped. The centre is *normalised* (`crop_for` takes `cx`/`cy` in 0..1), which is what
  keeps a portrait and a landscape frame paired, and `clamp_center` reads **one** image
  rather than reconciling both — near an edge the shorter frame simply stops instead of
  dragging the other off whatever was being compared. Two smaller rules with visible
  consequences: a new pair resets to fit (`PairView::fit`), because a zoom is a question
  about *these* two frames; and a pane whose photo did **not** change is left alone, or
  `1` would drop the one thing being compared back to its placeholder and redraw the
  identical crop a moment later. The placeholder itself is the canonical preview inflated
  inline (~1 ms), because a raw's decode can take half a second and an empty frame for
  that long on every verdict reads as broken. The zoom **survives inside a session and is
  dropped between them**: twenty frames of one child in one room are framed alike, so
  re-zooming onto the eyes nineteen times is the tedium the feature exists to remove,
  while the next session is a different scene.
  One trap worth knowing before hanging anything else off a `changed` handler here: the
  panes report their own pixel size, and *everything* on screen is rendered to that
  number, so a report that never arrives leaves both panes blank forever. It is reported
  through derived `int` properties (the proven `row-count` pattern) rather than `changed
  width`, **and** `pane_size` falls back to half the window when nothing has been
  reported — a slightly soft render until the real number lands is a better failure than
  a feature that does not draw.
- **The zoom belongs to both views** (`wire_preview_zoom` in `maple-ui/src/import.rs`,
  `ZoomSurface` in `import.slint`): a comparison and a single photo ask the same question
  — is *this* frame sharp — so the single-photo view zooms with the same wheel, the same
  `0`/`+`/`-`, and the same geometry (`crop_for`/`zoom_at`/`clamp_center`); the pointer,
  the drag and the view's pixel size reach Rust through the one surface both views hang
  over their picture. Sharing became the point the moment the tournament stopped being a
  mode: the cursor walks in and out of sessions along one pass, so a zoom that lived only
  in the panes disappeared whenever the user landed on a photo in no session — a sixth of
  a real card, and the reading it invited was "the zoom is broken".
  What is deliberately *not* shared is the state. The pair's zoom survives across a
  session because twenty frames of one child are framed alike; the next photo along the
  strip is a new question, so this one resets to fit on every navigation — which is also
  what stops a walk through a card from holding a 34 MB decode of everything it passed.
  At fit the feature costs nothing: `preview-photo` is the ordinary 1200 px render and no
  original is opened. The first notch out of fit spawns the renderer, and from then until
  the photo changes every frame on screen is a crop cut to the view's exact pixel size —
  the same `PairRenderer`, one side of it asked for, its two-decode cache making a step
  back to the previous photo free. Three smaller rules with visible consequences. The
  canonical preview's own dimensions seed the geometry (`preview_src`), so the *first*
  notch already zooms towards the pointer rather than the middle: only the aspect is used
  and a 256 px copy has the same one. The fit render lands on a thread that cannot hold
  the (`Rc`-based) context, so it reads `current-index` and `preview-zoom-level` back off
  the window before painting — without that a slow 1200 px render drops over a crop the
  user has already zoomed to. And a rotation *drops* the renderer instead of reusing it,
  because the file has just been rewritten under the decode cached against its path.
- **Import previews are inflated on demand** (`maple-ui/src/import_previews.rs`): a card
  of a few thousand photos will not fit in memory as decoded frames (~196 KB each) and
  almost none of them are on screen, so the browser holds the canonical WebPs and decodes
  what the user is *looking at*. The service itself is now the **recovery** path, not the
  main one: the scan makes a preview for every photo it reads and the medium's cache
  holds the ones an earlier run made, so by the time the strip asks here it is a photo
  the scan has not reached yet, or one whose display file will not decode. Two ideas
  carry it. **Priority is evaluated when a worker picks up work, not when it is
  queued**: the queue holds wanted indices plus one `focus`, and a worker always takes the
  pending index nearest it, so scrolling re-prioritises everything already queued by
  writing a single number — no re-sorting, no cancelling, and no waiting out a backlog of
  photos that scrolled past before the ones now on screen get decoded. **Retention is
  two-tier**: `Retained` is an LRU of decoded frames capped by `[import]
  max_loaded_previews`, keyed on *when the photo was last in view* (scrolling back to one
  saves it), and eviction drops the pixels down to the ~15 KB canonical WebP — so
  scrolling back re-inflates from memory and never returns to the card. Every photo the
  scan touched has one, so the second tier is now essentially always populated; the arithmetic
  that buys it is ~15 KB × the whole card (a 5,000-photo card is ~75 MB resident, and the
  same on disk in `.maple_previews.bin`). Reading still happens one file at a time (same
  bus argument as the scan); only the decode fans out, and the WebP *encode* happens on
  a worker rather than the UI thread. The scan sends the full listing up front (so the
  count and every filename are right from the first frame), then reads, hashes and
  previews. `ImportItem::loaded` therefore means "has a decoded preview", not "has been
  scanned", and goes false again on eviction. One consequence with no other owner:
  `apply_scan_thumb` inflates a just-arrived preview itself when the row is inside the
  current window (`in_preview_window`), because the window is only re-requested when the
  viewport *moves* and a user watching a scan fill in is not scrolling. Requests are clamped to the retention cap —
  a window wider than what can be held would have the tail evicting the head forever.
  The window itself is reported by the strip only when the **viewport** moves, so anything
  that renumbers rows without scrolling — "Hide old images", above all — has to re-derive
  it in Rust (`preview_window_for`, centred on wherever the current photo landed). Re-using
  the stale one after a filter that shrank the strip names rows that no longer exist,
  `request_previews` bails on `first > last`, and every visible tile stays blank until the
  button is clicked a second time. Two traps met here, worth knowing before adding another
  `changed` handler: a Slint `changed` fires only when the property's value compares
  *unequal*, and `ModelRc`'s `PartialEq` is **pointer identity** — so `changed items` never
  fires at all against `VecModel::set_vec`, which mutates the model behind an unchanged
  `ModelRc`. Watch `row-count` (`items.length`, which tracks the model's own row-count
  notification) instead. And nothing in the UI may be the *only* thing that recomputes
  state Rust already knows changed.
  The strip **follows the selection but not the scroll**: Rust sets `current-row` beside
  `current-index` whenever *it* moves the current photo (an arrow key, a click, the
  filter landing somewhere new), and the strip's `changed current-row` parks that row one
  tile down from the top, so the photo just stepped past stays visible and stepping
  forward scrolls by exactly one tile. Both ends fall out of the clamp rather than
  needing a case. Nothing fires on `viewport-y` changing, which is what leaves a hand
  scroll alone. The row is deliberately *not* the index — hiding old photos renumbers
  every row without moving a photo, so sending the index would scroll to the wrong place,
  and `-1` (filtered out) means "leave the scroll where it is".
- **Import tags are collections** (`wire_tags` in `maple-ui/src/import.rs`): Maple has one
  labelling system, so a tag assigned while triaging *is* a `collections` row — no new
  table, and what the import writes is immediately visible in the Collections window,
  filterable in the library and replicated by sync. The current tags are a **brush, not a
  batch setting**: `s` opens the picker, `c` clears it, and `record_brush` stamps the
  brush onto a photo **at the moment it is marked**, not at copy time — which is what lets
  one triage pass produce two differently-tagged sets without copying twice. Unmarking
  drops the record, so a photo re-marked under a different brush cannot pick the old tags
  back up, and `c` means "stop tagging", never "untag". The brush survives a copy (the
  next pass keeps tagging); `brushed` does not, since the selection it belongs to is gone.
  Three smaller rules: a typed name that already exists is **adopted, not rejected**
  (`collections.name` is UNIQUE, and the user means that one), a new tag's colour is
  derived **from its name** so two devices that both create "Holiday" before they sync
  agree on what it looks like, and tagging is best-effort per tag — a collection deleted
  between marking and copying must not cost the user the import. The brush panel floats
  over the preview with **no `TouchArea` of its own**, so clicks fall through to the photo
  underneath and triage carries on around it; it stays visible when empty because
  invisible state that silently changes what an import writes would be a trap.
- **The import record lives on the medium** (`maple-state/src/seen.rs`, P9): the scan's
  "already imported" badge reads `<source>/.maple_seen.bin`, written to the card itself
  so it carries its own history to the next machine — beside `.maple_embed_cache.bin`,
  which established the idiom, and unioned on load with the `library_dir/seen_imported.bin`
  replica. **Neither side is authoritative, because neither side is complete**: the record
  is written to whatever folder was scanned, so a card scanned once at its root and once at
  `DCIM/101_FUJI` carries two *disjoint* records (measured on a real card: 20 and 46 hashes,
  intersection empty, against a replica holding all 69) — and the replica, which is the only
  place every decision lands, is per-machine, so a card arriving from another computer is
  described only by what it carries. `load_for_source` therefore merges the two rather than
  picking; reading either alone re-presents photos the user already decided about, the one
  failure this mechanism exists to prevent. A missing *or* corrupt record on either side
  contributes nothing instead of reading as an empty set that overrides the other, since
  that would send the user re-importing a whole card. The union is always safe because
  `SeenSet` is **grow-only** — there is no "forget" for one side to be more current about
  and so no winner to pick. That same property makes saving a
  read-merge-write union (`merge_save_to_source`) with no locking and no conflict
  resolution — that is what makes two importers running at once combine instead of
  clobber, and it is the same code path that folds a card's history into a library that
  has never seen it. Three invariants worth keeping: "Hide old images" is gated on the
  old-count being non-zero, so that count is incremented **per photo as the scan streams**
  and only recomputed in bulk by `refilter` at the end — deferring it entirely hides the
  button for the whole of a multi-minute card read, which is exactly when it is wanted. The
  all-zero `UNHASHED` sentinel the
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
maple-import/src/            — scan, copy, hash, canonical preview, raw format support
```
