# Maple — Corporate Identity & UI System

> Photo library manager. This document defines the visual identity and component
> system for the redesigned **light & dark** UI (Direction A — *Clean & minimal*).
> It is implementation-ready: every token has a name and a value, every component
> lists its states.

---

## 1. Brand

**Personality:** warm, editorial, quiet. The chrome recedes so the photos lead.
Clean sans UI, one muted maple accent used sparingly, a subtle film-strip heritage.

**Marks**
- **App / launcher icon:** the existing leaf-on-film mark (`assets/logo.png`). Kept as-is.
- **In-app mark:** a quieter geometric *film-frame* glyph — a rounded terracotta
  square (12px radius) with a 3-perforation strip on the left edge and a cream
  "frame" panel. Pairs with the **"Maple"** wordmark (Hanken Grotesk, 700, −0.02em).
- **Wordmark lockup:** mark + "Maple" at 17px in sidebars, 30px on hero/empty states.

**Voice rules**
1. Warm neutrals, never cold gray.
2. One accent — muted maple — used sparingly (active nav, primary action, selection).
3. Generous gutters; photos get the contrast.
4. Mono only for technical metadata (EXIF, paths, counts).
5. The same token names drive light and dark; only values swap.

---

## 2. Color tokens

Tokens are semantic. Reference the **name** in code; never hard-code a hex outside
this table. Light is the default theme.

| Token            | Light       | Dark        | Usage                                   |
|------------------|-------------|-------------|-----------------------------------------|
| `--bg`           | `#FAF8F5`   | `#1B1714`   | App background / content area           |
| `--surface`      | `#FFFFFF`   | `#241F1B`   | Cards, active nav, fields-on-paper      |
| `--sidebar`      | `#F4F0EA`   | `#141110`   | Left navigation rail                    |
| `--border`       | `#EBE5DD`   | `#322C26`   | Hairlines, dividers, field borders      |
| `--ink`          | `#29251F`   | `#F3EEE7`   | Primary text, titles                    |
| `--ink-2`        | `#6F6960`   | `#B0A698`   | Secondary text, inactive nav            |
| `--ink-3`        | `#A39A8E`   | `#766E63`   | Muted text, placeholders, metadata      |
| `--accent`       | `#B5543E`   | `#CB6B52`   | Primary action fill, selection ring     |
| `--accent-text`  | `#8E3E2C`   | `#E08A72`   | Accent text / icon on tint              |
| `--accent-tint`  | `#F2E2DA`   | `#3A2B25`   | Active-toggle bg, selected row, chips    |
| `--field`        | `#F4F0EA`   | `#241F1B`   | Search & input background               |
| `--shadow`       | `0 1px 2px rgba(0,0,0,.06)` | `0 1px 2px rgba(0,0,0,.30)` | Card / active-item lift |

**Status / utility (theme-independent unless noted)**
- Working / in-progress: `#C29A3E` on tint `#FAF1D8` (light) / `#3A3012` (dark).
- Success: `#6E8B6F`. Collection dots use a free palette (`#B5543E`, `#6E8B6F`,
  `#C29A3E`, …) — these are user data, not brand.

---

## 3. Typography

| Family            | Role                                   |
|-------------------|----------------------------------------|
| **Hanken Grotesk** | All UI & display. Weights 400/500/600/700/800. |
| **IBM Plex Mono**  | Metadata only — EXIF, file paths, counts, section eyebrows. 400/500. |

**Scale** (Hanken Grotesk unless noted)

| Style    | Size / Weight / Tracking      | Used for                          |
|----------|-------------------------------|-----------------------------------|
| Display  | 30–38px / 800 / −0.025em      | Empty states, hero                |
| Title    | 24px / 700 / −0.015em         | Screen title ("All Photos")       |
| Heading  | 19px / 600                    | Section headings                  |
| Body     | 15px / 400–500                | Default text, nav items           |
| Caption  | 13px / 500                    | Subtitles, counts                 |
| Eyebrow  | 11px / 600 / +0.06em / mono   | "COLLECTIONS", "RECENT" labels    |
| Meta     | 11–12px / 500 / mono          | EXIF, paths (`#A39A8E`)           |

---

## 4. Spacing, radius, layout

- **Spacing scale:** 4 · 8 · 12 · 16 · 22 · 32 (px).
- **Radius:** fields/buttons `9–10px`, cards `12px`, mark `12–15px`, chips/pills `20px`, dots `50%`.
- **App shell:** persistent left sidebar **228px** + flexible main column.
  - Sidebar: `--sidebar` bg, `1px --border` right edge, `18px 13px` padding.
  - Top bar: 62px, `1px --border` bottom, `0 22px` padding.
  - Content: `22px` gutters.
- **Photo grid (Direction A):** uniform squares, `grid-template-columns: repeat(6,1fr)`,
  `gap: 14px`, tile radius `10px`. Density adapts; 6 cols is the desktop default.

---

## 5. Iconography

Line icons, **1.8px** stroke, `currentColor`, round caps/joins, 18px in nav / 16px
inline. Built from primitives (rects, circles, simple paths). Set:
grid (Library), down-tray (Import), people (People), folder-stack (Collections),
sliders (Settings), magnifier (Search), sparkle (AI, filled), corner-brackets (Faces),
star (favorite), check (selection).

---

## 6. Components & states

### Sidebar nav item
- **Inactive:** transparent bg, `--ink-2` text/icon, 500 weight.
- **Active:** `--surface` bg + `--shadow`, `--accent-text` text/icon, 600 weight.
- **Sub-item** (e.g. *Tag faces* under People): indented 30px, 13px, optional count badge.

### Buttons
| Variant      | Fill / Border                         | Text            |
|--------------|---------------------------------------|-----------------|
| Primary      | `--accent` fill, accent shadow        | `#FFFFFF`       |
| Secondary    | `--surface`, `1px --border`           | `--ink`         |
| Ghost        | transparent                           | `--ink-2`       |
| Accent-soft  | `--accent-tint`                       | `--accent-text` |
| Icon         | 40×40, `--surface`, `1px --border`    | `--ink-2`       |

Disabled: 45% opacity, no shadow, `not-allowed`.

### Toggles (replaces the old "Tag with AI" / "Detect Faces" buttons)
These are **persistent on/off toggles**, not fire-once buttons — they reflect a
running background worker.
- **Off:** `--surface` bg, `1px --border`, `--ink-2` text, icon only.
- **On:** `--accent-tint` bg, `--accent-text` text, a **pulsing dot** + live progress
  (`412 / 2,481`, mono). Toggling off stops the worker.
- Switch control (settings): 38×22 pill, knob 18px; on = `--accent`.

### Search field
`--field` bg, `1px --border`, radius 10px, magnifier `--ink-3`, placeholder `--ink-3`.
Optional trailing `semantic` pill when the semantic index is loaded.

### Collection chip
Pill, color dot + name. Selected: `--accent-tint` bg / `--accent-text`. Unselected:
`--field` bg / `--ink-2`. "+ Add": dashed `--border`.

### Photo cell
Square, radius 10px. **Selected:** 2.5px `--accent` outline + 2px offset, accent
check (top-left). **Unselected hover:** subtle ring. Badges (bottom corners, mono on
`rgba(0,0,0,.4)`): `RAF` (raw), `⊞ 4` (stack count).

### Status pill
Working state: `#FAF1D8` bg, `1px #EEDFB0`, pulsing `#C29A3E` dot, label + mono count.

---

## 7. Navigation map

```
Sidebar
├─ Library        ← default screen (grid + search + AI/Faces toggles)
├─ Import         ← source-only picker (see §8)
├─ People         ← face groups
│   └─ Tag faces  ← (moved here from the library header) untagged-faces wizard, w/ count badge
├─ Collections    ← collection manager
└─ Settings (footer) · Theme toggle (footer)
```

**Changes from the old UI**
- "Tag with AI" and "Detect Faces" header buttons → **top-bar toggles** on Library.
- "Tag Faces" header button → **sub-item under People** in the sidebar.
- Full-screen page stack → **persistent sidebar shell**.

---

## 8. Screen specs

### Library
Top bar: search (left) · AI toggle · Faces toggle (right). Sub-header: "All Photos" +
count (left), grid/justified view switch (right). Body: uniform 6-col photo grid.

### Import  *(source folder only — no destination)*
The library has a fixed managed location, so import only needs a **source**.
- **Title:** "Import Photos" + "Choose where your photos come from."
- **Source selector:** if nothing chosen → dashed dropzone ("Drag a folder here or
  Browse"). If chosen → solid card showing the selected path + Change.
- **Favorites:** pinned, frequently-used sources (library root, network archive, …).
  Star toggles membership. Section eyebrow "FAVORITES".
- **Recent locations:** history, newest first, with photo counts. Section eyebrow "RECENT".
- Each row: folder icon · name + path (mono) · count · star · selection check.
- **Start Scan** (primary) — enabled only once a source is selected; disabled otherwise.

### Detail viewer / Face-tagging / Settings
To be designed next on these same tokens. Detail viewer keeps: zoom/pan, face overlay
toggle, collections, rotate, info, stacks, fullscreen.

---

## 9. Theming

One source of truth: the token table (§2). Implement as a palette swapped at the root
(CSS custom properties, a Slint global, etc.). All components reference token **names**,
so light↔dark is a single palette switch with no per-component changes. A theme toggle
lives in the sidebar footer.
