# Phase 1 — Scan & Thumbnails

## What was done

Source/destination folder pickers, recursive image scanner, off-thread thumbnail
generation, and a scrollable thumbnail grid view.

## New / changed files

| File | Change |
|------|--------|
| `throppe-import/src/scan.rs` | **New** — recursive scanner for jpg/jpeg/png files |
| `throppe-import/src/lib.rs` | Exports `scan_images()` and `ImageFile` |
| `throppe-ui/src/thumbnail.rs` | **New** — decodes image → resizes → returns PNG bytes |
| `throppe-ui/src/views/mod.rs` | **New** — views sub-module |
| `throppe-ui/src/views/source_picker.rs` | **New** — AdwStatusPage with folder pickers + Scan button |
| `throppe-ui/src/views/thumbnail_grid.rs` | **New** — FlowBox grid populated by background worker |
| `throppe-ui/src/window.rs` | Refactored to use `AdwNavigationView` (picker → grid) |
| `throppe-ui/src/lib.rs` | Registers `thumbnail` and `views` modules |
| `throppe-ui/Cargo.toml` | Added `throppe-import`, `image` dependencies |

## Architecture

### UI navigation

```
AdwApplicationWindow
 └─ AdwToastOverlay
     └─ AdwNavigationView
         ├─ page "Throppe"      ← source_picker (initial)
         └─ page "Scan Results" ← thumbnail_grid (pushed on scan)
```

Back button in the grid header is automatic via `AdwNavigationView`.

### Folder picker view

- `AdwStatusPage` with camera icon + welcome text
- `AdwClamp(500px)` → `ListBox.boxed-list`
  - Source row (`AdwActionRow` + folder-open button → `FileDialog`)
  - Destination row (disabled until source chosen)
- "Start Scan" pill button (enabled when both folders set)

### Scan + thumbnail pipeline

```
┌──────────────────┐       mpsc::channel       ┌──────────────────┐
│  Worker thread   │  ─────────────────────────▶│  GTK main thread │
│                  │   ScanMsg::Count(n)        │  (timeout 32ms)  │
│  scan_images()   │   ScanMsg::Thumb{…}        │                  │
│  └► for each img │   ScanMsg::Done            │  update progress │
│     generate_    │   ScanMsg::Error(…)        │  append cards    │
│     thumbnail()  │                            │  to FlowBox      │
└──────────────────┘                            └──────────────────┘
```

- `std::sync::mpsc` channel carries `ScanMsg` enum
- `glib::timeout_add_local(32ms)` polls the channel on the GTK thread
- Each thumbnail is decoded with `image::open()`, resized with `thumbnail()`,
  encoded to PNG, then turned into a `gdk4::Texture` on the main thread

### Image card

Each card in the FlowBox is a vertical `gtk4::Box`:
- `gtk4::Picture` (180×180, ContentFit::Contain)
- `gtk4::Label` (filename, ellipsized)

## What launches

1. Picker page: select source folder → destination folder → "Start Scan".
2. Grid page: progress bar updates as thumbnails generate; images appear in a
   responsive flow grid.  Back button returns to the picker.

## Next: Phase 2

Integrate ONNX Runtime for embedding generation (MobileNetV3 or CLIP),
pre-processing pipeline, batch inference, and embedding cache.
