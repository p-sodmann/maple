# Maple

Cross-platform desktop app for importing, browsing, and organising photos.
UI built with [Slint](https://slint.dev) — no GTK/libadwaita runtime required.

Features: recursive photo import with dedup, a fast library browser with
full-text and semantic search, face detection/recognition, AI-generated
image descriptions, burst/near-duplicate stack detection, and collections.

## Prerequisites

- **Rust** (stable, 2021 edition) — install via [rustup](https://rustup.rs)
- **A C toolchain** — `rusqlite` (bundled), `sqlite-vec`, and `webp` all
  compile vendored C sources via the `cc` crate:
  - Linux: `build-essential` (`sudo apt install build-essential`)
  - macOS: Xcode Command Line Tools (`xcode-select --install`)
  - Windows: MSVC Build Tools

No system GTK headers are needed — Slint ships its own renderer.

Maple builds in two variants depending on which ONNX Runtime execution
backend you want: **CPU** (default, simplest) or **GPU** (CUDA/TensorRT).
The backend is fixed at compile time — there's no runtime fallback between
the two within a single binary.

## Installation — CPU build (default)

This is the simplest option and what you want unless you specifically need
GPU-accelerated inference (face detection, AI tagging, stack detection,
semantic search all still work on CPU, just slower).

```sh
cargo build --release -p maple
```

`ort` (the ONNX Runtime crate) downloads a platform-matched prebuilt
onnxruntime binary at build time and links it into the binary. This means:

- **Network access is required the first time you build** (to fetch the
  onnxruntime binary).
- The result is CPU-only — no CUDA/TensorRT execution providers are
  available. If `settings.toml` has `device = "cuda:0"` or `"tensorrt:0"`
  under `[face]`, `[stacks]`, or `[semantic]`, it silently falls back to CPU
  (this is `ort`'s normal behavior when an execution provider is
  unavailable, not a bug).

Run it with:

```sh
cargo run --release -p maple
```

## Installation — GPU build (CUDA / TensorRT)

Opt in explicitly, one package at a time:

```sh
cargo build --release -p maple --no-default-features --features gpu
```

The GPU variant uses `ort`'s `load-dynamic` mode instead of linking a binary
at build time:

- **No network access is needed to build.**
- The resulting binary loads onnxruntime **at runtime** instead, and needs a
  real onnxruntime shared library to even start — this applies even if you
  only intend to run CPU inference on a `gpu` build, not just for GPU use.

You need to supply a CUDA/TensorRT-enabled onnxruntime release, e.g.
Microsoft's official [`onnxruntime-gpu`](https://github.com/microsoft/onnxruntime/releases)
build, either by:

1. Setting the `ORT_DYLIB_PATH` environment variable to point at the shared
   library, e.g.:
   ```sh
   ORT_DYLIB_PATH=/path/to/onnxruntime/lib/libonnxruntime.so \
     cargo run --release -p maple --no-default-features --features gpu
   ```
2. Or placing the library where the OS's dynamic loader finds it by default:
   - Linux: a directory on `LD_LIBRARY_PATH`, or set an rpath
   - Windows: next to the `.exe`
   - macOS: a `DYLD_*` library path

Then set `device = "cuda:0"` (or `"tensorrt:0"`) under the relevant section
(`[face]`, `[stacks]`, `[semantic]`) in `settings.toml`.

Ship both variants as separate release artifacts per platform if you want to
offer GPU acceleration — pick one backend per build.

## Model setup

Maple's AI features need ONNX models. Some are downloaded automatically;
face detection models must be supplied manually.

- **Face detection/recognition** (`[face]` in `settings.toml`) — download
  manually and point `detector_model` (and optionally `embedder_model`) at
  the file:
  - Detector: [atksh/onnx-facial-lmk-detector](https://github.com/atksh/onnx-facial-lmk-detector/releases)
    (default, single-pass detection + landmarks + aligned crops) or
    [InsightFace SCRFD](https://github.com/deepinsight/insightface/tree/master/model_zoo)
  - Embedder (optional, enables person grouping): an
    [InsightFace ArcFace](https://github.com/deepinsight/insightface/tree/master/model_zoo)
    model
- **Stack detection** (`[stacks]`) — downloads the DINOv2 ONNX export
  (`model_repo = "onnx-community/dinov2-small"`) from Hugging Face Hub
  automatically on first use.
- **Semantic search** (`[semantic]`) — downloads a sentence-transformer
  ONNX export (default `sentence-transformers/all-MiniLM-L6-v2`) from
  Hugging Face Hub automatically on first use, cached under
  `~/.config/maple/models/`.
- **AI image descriptions** (`[ai]`) — no local model; talks to an
  OpenAI-compatible server (e.g. [LM Studio](https://lmstudio.ai)) over HTTP,
  configured via `server_url` and `model`.

All settings live in `~/.config/maple/settings.toml`, written on first
launch from `maple-state/defaults.toml`.

## Build & Test

```sh
cargo build --workspace
cargo test --workspace
cargo clippy --workspace
```

## Workspace Crates

| Crate | Purpose |
|---|---|
| `maple` | Binary entry point |
| `maple-ui` | Slint UI: windows, views, widgets |
| `maple-state` | Settings (settings.toml), Session (session.json), SeenSet (bloom filter) |
| `maple-import` | Recursive image scanner, BLAKE3 hasher, file copier, raw file support |
| `maple-db` | SQLite library database, background scanner, EXIF, AI tagging, face detection |
