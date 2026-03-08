# Throppe — Implementation Plan

> Photo-import tournament app for Linux desktop.
> Pick the best shot from every burst; import only winners.

---

## 1. Project Identity

| Field | Value |
|-------|-------|
| Name | **Throppe** |
| Language | Rust (stable, 2021 edition) |
| UI toolkit | GTK 4 + libadwaita (`gtk4-rs` / `libadwaita-rs`) |
| Inference | ONNX Runtime (`ort` crate) |
| Target | Linux x86_64 — shipped as AppImage |
| License | TBD (proprietary or MIT) |

---

## 2. High-Level Flow

```
┌──────────────┐     ┌──────────────┐     ┌─────────────────┐
│  Select Src  │────▶│  Select Dst  │────▶│  Scan & Index   │
└──────────────┘     └──────────────┘     └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   Build/Load    │
                                          │   Embeddings    │
                                          └────────┬────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   Cluster into  │
                                          │   Groups        │
                                          └────────┬────────┘
                                                   │
                                    ┌──────────────▼──────────────┐
                                    │  Tournament (per group)     │
                                    │  1-vs-1 → last man standing │
                                    └──────────────┬──────────────┘
                                                   │
                                          ┌────────▼────────┐
                                          │   Import Winners│
                                          │   (auto-copy)   │
                                          └─────────────────┘
```

---

## 3. Work Packages

Each work package maps to a dedicated doc in `docs/` and one or more Rust modules/crates.

| # | Package | Doc | Key crate(s) / module(s) |
|---|---------|-----|--------------------------|
| WP-1 | **Architecture & crate layout** | [architecture.md](architecture.md) | workspace `Cargo.toml`, crate tree |
| WP-2 | **UI shell & navigation** | [ui-design.md](ui-design.md) | `throppe-ui` — GTK4 + Adw windows, views |
| WP-3 | **Embedding pipeline** | [embedding-pipeline.md](embedding-pipeline.md) | `throppe-embed` — ONNX, image pre-processing |
| WP-4 | **Similarity grouping** | [similarity-grouping.md](similarity-grouping.md) | `throppe-cluster` — cosine sim, connected components |
| WP-5 | **Tournament engine** | [tournament.md](tournament.md) | `throppe-tournament` — bracket, persistence |
| WP-6 | **Import engine** | [import-engine.md](import-engine.md) | `throppe-import` — async copy, verify, naming |
| WP-7 | **Persistence & state** | [persistence.md](persistence.md) | `throppe-state` — SQLite/JSON, resume |
| WP-8 | **Packaging & distribution** | [packaging.md](packaging.md) | AppImage, CI, ONNX model bundling |

---

## 4. Implementation Order & Milestones

### Phase 0 — Skeleton (Day 1)
- [ ] Initialize Cargo workspace with binary crate `throppe` and library crates.
- [ ] Verify GTK4 + libadwaita build on host.
- [ ] Empty GTK window launches.

### Phase 1 — Scan & Thumbnails (Day 2-3)
- [ ] Source/destination folder pickers.
- [ ] Recursive image scan (`jpg`, `jpeg`, `png`).
- [ ] Thumbnail generation (fast, off-main-thread via `image` crate + `tokio::spawn_blocking`).
- [ ] Show thumbnail grid in a scrolled view.

### Phase 2 — Embeddings (Day 3-5)
- [ ] Integrate ONNX Runtime (`ort` crate).
- [ ] Choose & bundle a small vision model (MobileNetV3 / EfficientNet-Lite / CLIP-ViT-B/32 visual encoder).
- [ ] Pre-process pipeline: decode → resize → normalize → `f32` tensor.
- [ ] Batch inference on a worker thread pool.
- [ ] Cache embeddings to disk (keyed by file path + mtime + size).
- [ ] Add Cuda Support for GPU

### Phase 3 — Clustering (Day 5-6)
- [ ] Compute pairwise cosine similarity matrix.
- [ ] Build adjacency graph at threshold (default 0.92).
- [ ] Extract connected components → groups.
- [ ] Singletons = auto-import (no tournament needed).

### Phase 4 — Tournament UI (Day 6-9)
- [ ] 1-vs-1 comparison view (A | B side-by-side).
- [ ] Group thumbnail strip below.
- [ ] Keyboard shortcuts: `←` / `→` to pick, `Z` to undo last.
- [ ] Bracket logic: seeding, byes for odd groups, advance winners.
- [ ] Persist every decision immediately.

### Phase 5 — Import Pipeline (Day 9-10)
- [ ] Async file copy triggered on "advance to next matchup."
- [ ] Collision-safe naming (`IMG_001.jpg` → `IMG_001_1.jpg` or hash suffix).
- [ ] Optional checksum verification (xxHash or BLAKE3).
- [ ] Progress indicator in the UI (non-blocking).
- [ ] "Preserve folder structure" toggle.

### Phase 6 — Polish & Packaging (Day 10-12)
- [ ] Full keyboard navigation audit.
- [ ] Dark/light theme via Adwaita.
- [ ] Error handling & user-facing messages.
- [ ] AppImage build (`appimagetool` + `linuxdeploy`).
- [ ] Bundle ONNX model inside AppImage.
- [ ] README, LICENSE.

---

## 5. Dependency Overview

| Purpose | Crate | Notes |
|---------|-------|-------|
| GTK 4 bindings | `gtk4` (≥ 0.9) | Features: `v4_12` |
| Libadwaita | `libadwaita` (≥ 0.7) | Features: `v1_4` |
| Async runtime | `tokio` | `rt-multi-thread`, `fs`, `sync` |
| ONNX inference | `ort` (≥ 2.0) | Bundles `onnxruntime` shared lib |
| Image decode | `image` | JPEG/PNG decode, resize |
| Fast resize | `fast_image_resize` | SIMD-accelerated thumbnail gen |
| Hashing | `xxhash-rust` or `blake3` | File verification |
| Serialization | `serde` + `serde_json` | State persistence |
| SQLite (optional) | `rusqlite` | Embedding cache + decisions |
| Logging | `tracing` + `tracing-subscriber` | Structured logging |
| CLI (optional) | `clap` | Dev/debug flags |

---

## 6. Model Choice

**Primary candidate:** MobileNetV3-Small (ImageNet) exported to ONNX.
- Input: `224×224×3` float32, normalized `[0,1]` with ImageNet mean/std.
- Output: 1024-dim embedding (take layer before final classifier).
- Size: ~5 MB ONNX file.
- Inference: <10 ms/image on CPU.

**Alternative:** CLIP ViT-B/32 visual encoder (~350 MB). Better semantic similarity but much larger. Can be offered as optional download.

---

## 7. Similarity Threshold Rationale

| Threshold | Behaviour |
|-----------|-----------|
| 0.98+ | Near-duplicates only (identical scene, tiny exposure diff) |
| 0.92–0.97 | Same scene/burst, slightly different framing | ← **default**
| 0.85–0.91 | Same subject, different angle |
| < 0.85 | Different images |

Threshold is user-configurable via a preferences panel or config file.

---

## 8. Risk Register

| Risk | Mitigation |
|------|-----------|
| ONNX Runtime linking issues on various distros | Bundle `.so` inside AppImage; `ort` crate handles download |
| GTK4/Adwaita version mismatch | Pin minimum GTK 4.10 / Adw 1.4; document runtime deps |
| Large SD cards (10k+ images) | Paginated scan, streaming embeddings, limit in-memory thumbnails |
| Non-JPEG images (RAW) | Phase 1 supports JPG/PNG; RAW deferred to future work |
| Model accuracy for grouping | Expose threshold slider; fallback to manual grouping |

---

## 9. Directory Layout (Planned)

```
throppe/
├── Cargo.toml                 # workspace root
├── throppe/                   # binary crate (main entry)
│   ├── Cargo.toml
│   └── src/
│       └── main.rs
├── throppe-ui/                # GTK4 UI components
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── app.rs             # AdwApplication setup
│       ├── window.rs          # main window
│       ├── views/
│       │   ├── mod.rs
│       │   ├── source_picker.rs
│       │   ├── scan_progress.rs
│       │   ├── tournament.rs  # 1-vs-1 comparison view
│       │   └── summary.rs     # final results
│       └── widgets/
│           ├── mod.rs
│           ├── image_card.rs
│           └── thumbnail_strip.rs
├── throppe-embed/             # embedding generation
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── model.rs           # ONNX session management
│       ├── preprocess.rs      # image → tensor
│       └── cache.rs           # embedding cache
├── throppe-cluster/           # similarity & grouping
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── cosine.rs          # cosine similarity
│       └── components.rs      # connected components
├── throppe-tournament/        # bracket logic
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       └── bracket.rs
├── throppe-import/            # file copy engine
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── copy.rs
│       ├── naming.rs          # collision-safe names
│       └── verify.rs          # checksum verification
├── throppe-state/             # persistence
│   ├── Cargo.toml
│   └── src/
│       ├── lib.rs
│       ├── session.rs         # session state
│       ├── decisions.rs       # tournament decisions
│       └── schema.rs          # DB/JSON schema
├── models/                    # ONNX model files (gitignored, bundled)
│   └── mobilenetv3_small.onnx
├── resources/                 # GResource XML, icons, CSS
│   ├── throppe.gresource.xml
│   └── style.css
├── packaging/                 # AppImage build scripts
│   ├── build-appimage.sh
│   ├── throppe.desktop
│   └── throppe.svg
└── docs/
    ├── plan.md                # ← you are here
    ├── architecture.md
    ├── ui-design.md
    ├── embedding-pipeline.md
    ├── similarity-grouping.md
    ├── tournament.md
    ├── import-engine.md
    ├── persistence.md
    └── packaging.md
```

---

## 10. Definition of Done

- [ ] App launches, user picks source + dest.
- [ ] Images are scanned, thumbnails shown.
- [ ] Embeddings computed (with progress bar).
- [ ] Groups formed; singletons auto-queued for import.
- [ ] Tournament flows through every group, one matchup at a time.
- [ ] Winners are imported automatically.
- [ ] All decisions + cache persisted; app can resume after restart.
- [ ] Ships as a working AppImage.
