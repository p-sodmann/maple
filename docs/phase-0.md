# Phase 0 — Skeleton

## What was done

Cargo workspace with 1 binary + 6 library crates. GTK4 + libadwaita window launches.

## Crate map

```
throppe/          binary   — main(), boots tokio logger + GTK app
throppe-ui/       lib      — build_window() → AdwApplicationWindow with HeaderBar + StatusPage
throppe-embed/    lib      — stub (EmbedEngine placeholder)
throppe-cluster/  lib      — stub (Cluster placeholder)
throppe-tournament/ lib    — stub (Bracket placeholder)
throppe-import/   lib      — stub (ImportEngine placeholder)
throppe-state/    lib      — Config + Session structs with serde
```

## System requirements verified

- Rust 1.93.1 stable
- GTK 4.20.1 (`libgtk-4-dev`)
- libadwaita 1.8.0 (`libadwaita-1-dev`)
- `build-essential` for cc linker

## Key deps (workspace-level)

| Crate | Version | Purpose |
|-------|---------|---------|
| gtk4 | 0.9 (v4_14 feature) | UI bindings |
| libadwaita | 0.7 (v1_6 feature) | Adwaita widgets |
| tokio | 1 (rt-multi-thread) | Async runtime |
| image | 0.25 | Decode JPEG/PNG |
| serde/serde_json | 1 | Config/state serialization |
| blake3 | 1 | File checksums |
| tracing | 0.1 | Logging |
| anyhow | 1 | Error handling |

## Threading model (planned, not yet wired)

```
GTK main thread  ←  glib::Sender  ←  tokio runtime (spawn_blocking workers)
```

GTK never blocks. All heavy work (decode, inference, copy) goes through tokio.

## What launches

An `AdwApplicationWindow` (900×600) with:
- `AdwHeaderBar` titled "Throppe"
- `AdwStatusPage` with camera icon + placeholder text
- `AdwToastOverlay` wrapping everything (ready for notifications)

## Next: Phase 1

Wire up source/destination folder pickers, recursive image scan, thumbnail generation, and thumbnail grid view.
