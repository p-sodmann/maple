//! Pure data-shape transforms shared across window controllers.
//!
//! No DB access, no `slint::Weak`/`ModelRc`/callback types — only the
//! generated plain-data structs (`DateGroup`, …) and primitives.

use std::collections::HashMap;

use maple_db::{cosine_similarity, FaceDetection, LibraryImage, SearchHit};
use slint::SharedString;

use crate::date;
use crate::{DateGroup, FaceBox, FacePersonSuggestion};

/// Group `records[from..]` into contiguous same-day runs for the
/// date-grouped view, extending `groups`, which already covers
/// `records[..from]`. Pass `from = 0` and an empty `groups` to group a whole
/// list.
///
/// Assumes `records` is sorted so that same-day images are adjacent (the
/// grid asks the DB for `SearchOrder::TakenDesc` in date view); otherwise
/// the same day may appear as multiple separate groups.
///
/// A page boundary is not a day boundary: when a page continues the day the
/// previous one ended on, its leading records grow that existing group
/// instead of opening a second group for the same day, so every group stays
/// one contiguous `start..start + count` slice of the accumulated list.
pub fn append_date_groups(groups: &mut Vec<DateGroup>, records: &[LibraryImage], from: usize) {
    // Day of the last already-grouped record — the day the appended page
    // may be continuing.
    let mut current_day = from.checked_sub(1).and_then(|i| records.get(i)).map(record_day);

    for (i, rec) in records.iter().enumerate().skip(from) {
        let ts = rec.meta.taken_at.unwrap_or(rec.added_at);
        let day = date::day_number(ts);
        if current_day != Some(day) {
            groups.push(DateGroup {
                label: SharedString::from(date::day_label(ts)),
                start: i as i32,
                count: 0,
            });
            current_day = Some(day);
        }
        if let Some(g) = groups.last_mut() {
            g.count += 1;
        }
    }
}

fn record_day(rec: &LibraryImage) -> i64 {
    date::day_number(rec.meta.taken_at.unwrap_or(rec.added_at))
}

/// Caption shown under a tile during search (empty when not a search hit).
pub fn score_caption(hit: Option<&SearchHit>) -> String {
    match hit {
        Some(SearchHit::Direct { .. }) => "direct".to_owned(),
        Some(SearchHit::Semantic { similarity, .. }) => {
            let pct = (similarity * 100.0).clamp(0.0, 100.0);
            format!("{pct:.0}% match")
        }
        None => String::new(),
    }
}

/// Format a Unix timestamp (seconds) as `YYYY-MM-DD HH:MM UTC`.
pub fn format_unix_ts(ts: i64) -> String {
    if ts <= 0 {
        return "—".to_owned();
    }
    let s = ts as u64;
    let days = s / 86400;
    let rem = s % 86400;
    let h = rem / 3600;
    let m = (rem % 3600) / 60;
    let (y, mo, d) = date::days_to_ymd(days);
    format!("{y:04}-{mo:02}-{d:02}  {h:02}:{m:02} UTC")
}

pub fn truncate_value(s: &str) -> String {
    const MAX: usize = 90;
    let count = s.chars().count();
    if count > MAX {
        let t: String = s.chars().take(MAX - 1).collect();
        format!("{t}…")
    } else {
        s.to_owned()
    }
}

/// Parse a `#rrggbb` hex string into a Slint colour. Falls back to neutral grey.
pub fn hex_to_color(hex: &str) -> slint::Color {
    let s = hex.trim_start_matches('#');
    if s.len() == 6 {
        if let (Ok(r), Ok(g), Ok(b)) = (
            u8::from_str_radix(&s[0..2], 16),
            u8::from_str_radix(&s[2..4], 16),
            u8::from_str_radix(&s[4..6], 16),
        ) {
            return slint::Color::from_rgb_u8(r, g, b);
        }
    }
    slint::Color::from_rgb_u8(0x9a, 0x9a, 0x9a)
}

/// Render a Slint colour as the `#rrggbb` string the DB stores. Inverse of
/// [`hex_to_color`]; the alpha channel is dropped (collection colours are
/// always opaque).
pub fn color_to_hex(color: slint::Color) -> String {
    format!("#{:02x}{:02x}{:02x}", color.red(), color.green(), color.blue())
}

// ── Name / id sanitising ───────────────────────────────────────────

/// Trim a user-entered name, rejecting one that is empty or all whitespace.
///
/// Every rename/create path bails on a blank name rather than writing it, so
/// the callers read as `let Some(name) = trimmed_name(&raw) else { return }`.
pub fn trimmed_name(raw: &str) -> Option<String> {
    let name = raw.trim();
    (!name.is_empty()).then(|| name.to_owned())
}

/// Decode the `-1`-means-none sentinel the Slint side uses for optional row
/// ids (no parent collection, no active filter, …).
pub fn optional_id(id: i32) -> Option<i64> {
    (id >= 0).then_some(id as i64)
}

/// Position of `id` within `records`, defaulting to the first record when it
/// isn't there — opening the viewer on something is better than not opening.
pub fn record_index(records: &[LibraryImage], id: i64) -> usize {
    records.iter().position(|r| r.id == id).unwrap_or(0)
}

// ── Face overlay ───────────────────────────────────────────────────

/// `true` when this row is a real detection (not a zero-confidence sentinel).
pub fn is_real_detection(face: &FaceDetection) -> bool {
    face.confidence >= 0.0 && face.bbox != [0.0, 0.0, 0.0, 0.0]
}

/// Where the image sits inside the detail window's viewport — mirrors the
/// `geo-*` properties the Slint side computes from the zoom/pan state, so
/// face hit-testing and box drawing can be done in plain Rust.
#[derive(Clone, Copy, Debug)]
pub struct ViewportGeometry {
    pub img_left: f32,
    pub img_top: f32,
    pub disp_w: f32,
    pub disp_h: f32,
}

impl ViewportGeometry {
    /// `None` before the image has been laid out — nothing can be hit or
    /// drawn against a zero-sized viewport.
    fn valid(&self) -> Option<&Self> {
        (self.disp_w > 0.0 && self.disp_h > 0.0).then_some(self)
    }

    /// Viewport point → normalised image coordinates, clamped to the image.
    fn normalise(&self, vx: f32, vy: f32) -> (f32, f32) {
        (
            ((vx - self.img_left) / self.disp_w).clamp(0.0, 1.0),
            ((vy - self.img_top) / self.disp_h).clamp(0.0, 1.0),
        )
    }
}

/// Id of the topmost real face box containing the viewport point, if any.
pub fn hit_test_face(
    faces: &[FaceDetection],
    geo: ViewportGeometry,
    vp_x: f32,
    vp_y: f32,
) -> Option<i64> {
    geo.valid()?;
    faces.iter().find_map(|face| {
        if !is_real_detection(face) {
            return None;
        }
        let [x1, y1, x2, y2] = face.bbox;
        let bx = geo.img_left + x1 * geo.disp_w;
        let by = geo.img_top + y1 * geo.disp_h;
        let bw = (x2 - x1) * geo.disp_w;
        let bh = (y2 - y1) * geo.disp_h;
        if vp_x >= bx && vp_x <= bx + bw && vp_y >= by && vp_y <= by + bh {
            Some(face.id)
        } else {
            None
        }
    })
}

/// A dragged-out rectangle as a normalised `[min_x, min_y, max_x, max_y]`
/// bbox, or `None` for a box too small to be anything but an accidental
/// click (under 0.5% of an image side).
pub fn normalise_draw_box(
    geo: ViewportGeometry,
    vx0: f32,
    vy0: f32,
    vx1: f32,
    vy1: f32,
) -> Option<[f32; 4]> {
    geo.valid()?;
    let (nx0, ny0) = geo.normalise(vx0, vy0);
    let (nx1, ny1) = geo.normalise(vx1, vy1);
    // Sort corners so bbox is always [min_x, min_y, max_x, max_y].
    let (bx1, bx2) = if nx0 <= nx1 { (nx0, nx1) } else { (nx1, nx0) };
    let (by1, by2) = if ny0 <= ny1 { (ny0, ny1) } else { (ny1, ny0) };
    if (bx2 - bx1) < 0.005 || (by2 - by1) < 0.005 {
        return None;
    }
    Some([bx1, by1, bx2, by2])
}

/// In-memory embedding matrix for fast cosine-similarity search.
///
/// Built once per image load from all currently assigned face embeddings
/// (query side lives in `services::faces::load_embedding_matrix`), so DB
/// queries are not repeated while the user cycles through suggestions.
/// Persons with no embedding (tagged before the embedder was configured) are
/// included as fallback rows with `sim = f32::NEG_INFINITY`.
pub struct EmbeddingMatrix {
    data: Vec<f32>,
    dim: usize,
    rows: Vec<(i64, String)>,
    persons: Vec<(i64, String)>,
}

impl EmbeddingMatrix {
    pub fn empty() -> Self {
        Self { data: vec![], dim: 512, rows: vec![], persons: vec![] }
    }

    /// Build from pre-fetched embedding rows and the full person list.
    pub fn from_rows(known: Vec<(i64, String, Vec<f32>)>, persons: Vec<(i64, String)>) -> Self {
        let dim = known
            .iter()
            .find_map(|(_, _, e)| if !e.is_empty() { Some(e.len()) } else { None })
            .unwrap_or(512);

        let mut mat = Self {
            data: Vec::with_capacity(known.len() * dim),
            dim,
            rows: Vec::with_capacity(known.len()),
            persons,
        };
        for (pid, name, emb) in &known {
            mat.add(*pid, name.clone(), emb);
        }
        mat
    }

    /// Register a person and optionally append their embedding row.
    pub fn add(&mut self, person_id: i64, name: String, embedding: &[f32]) {
        if !self.persons.iter().any(|(pid, _)| *pid == person_id) {
            self.persons.push((person_id, name.clone()));
        }
        if embedding.is_empty() {
            return;
        }
        if self.dim == 0 {
            self.dim = embedding.len();
        }
        if embedding.len() != self.dim {
            return;
        }
        self.data.extend_from_slice(embedding);
        self.rows.push((person_id, name));
    }

    /// Look up a known person's name by id (no DB access — served from the
    /// full person list captured at build time).
    fn person_name(&self, person_id: i64) -> Option<&str> {
        self.persons.iter().find(|(pid, _)| *pid == person_id).map(|(_, name)| name.as_str())
    }

    /// Top-k persons by cosine similarity.
    ///
    /// When no ArcFace data is available, all known persons are returned with
    /// `sim = f32::NEG_INFINITY` so the UI can still show name buttons.
    pub fn top_k(&self, query: &[f32], k: usize) -> Vec<(i64, String, f32)> {
        if k == 0 {
            return vec![];
        }

        if query.is_empty() || self.dim == 0 || query.len() != self.dim {
            return self
                .persons
                .iter()
                .take(k)
                .map(|(pid, name)| (*pid, name.clone(), f32::NEG_INFINITY))
                .collect();
        }

        let mut best: HashMap<i64, (String, f32)> = HashMap::new();
        for (i, (pid, name)) in self.rows.iter().enumerate() {
            let row = &self.data[i * self.dim..(i + 1) * self.dim];
            let sim = cosine_similarity(query, row);
            let entry = best
                .entry(*pid)
                .or_insert_with(|| (name.clone(), f32::NEG_INFINITY));
            if sim > entry.1 {
                entry.0 = name.clone();
                entry.1 = sim;
            }
        }
        for (pid, name) in &self.persons {
            best.entry(*pid).or_insert_with(|| (name.clone(), f32::NEG_INFINITY));
        }

        let mut results: Vec<(i64, String, f32)> =
            best.into_iter().map(|(pid, (name, sim))| (pid, name, sim)).collect();
        results.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
        results.truncate(k);
        results
    }
}

/// How many top-k suggestions to fetch (enough for a compact panel).
const SUGGESTION_LIMIT: usize = 12;

/// Build the face label for a detection: person name, a suggestion, or "?".
fn face_label(face: &FaceDetection, known: &EmbeddingMatrix, threshold: f32) -> (String, bool, bool) {
    if let Some(pid) = face.person_id {
        let name = known.person_name(pid).map(str::to_owned).unwrap_or_else(|| "?".into());
        return (name, true, false);
    }
    if face.embedding.is_empty() {
        return ("?".into(), false, false);
    }
    let matches = known.top_k(&face.embedding, 1);
    if let Some((_pid, name, sim)) = matches.first() {
        if sim.is_finite() && *sim >= threshold {
            return (name.clone(), false, true);
        }
    }
    ("?".into(), false, false)
}

/// Convert all loaded faces into [`FaceBox`] structs for the Slint model.
/// `threshold` is the similarity cutoff below which a match is shown as "?"
/// rather than a confident suggestion (`settings.face.similarity_threshold`).
pub fn faces_to_boxes(faces: &[FaceDetection], known: &EmbeddingMatrix, threshold: f32) -> Vec<FaceBox> {
    faces
        .iter()
        .filter(|f| is_real_detection(f))
        .map(|f| {
            let [x1, y1, x2, y2] = f.bbox;
            let (label, is_assigned, is_suggestion) = face_label(f, known, threshold);
            FaceBox {
                face_id: f.id as i32,
                x1,
                y1,
                x2,
                y2,
                label: label.into(),
                is_assigned,
                is_suggestion,
            }
        })
        .collect()
}

/// Build person suggestions for the assignment panel (ranked by similarity).
pub fn faces_to_suggestions(embedding: &[f32], known: &EmbeddingMatrix) -> Vec<FacePersonSuggestion> {
    let raw = known.top_k(embedding, SUGGESTION_LIMIT);
    raw.into_iter()
        .map(|(pid, name, sim)| {
            let sim_text: slint::SharedString = if sim.is_finite() && sim >= 0.0 {
                format!("{:.0}%", sim * 100.0).into()
            } else {
                "".into()
            };
            FacePersonSuggestion { person_id: pid as i32, name: name.into(), sim_text }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use maple_db::ImageStatus;

    fn image_at(added_at: i64, taken_at: Option<i64>) -> LibraryImage {
        LibraryImage {
            id: 1,
            path: "photo.jpg".into(),
            raw_path: None,
            added_at,
            status: ImageStatus::Present,
            meta: maple_db::ImageMetadata { taken_at, ..Default::default() },
            hash: None,
            stack_id: None,
            stack_size: None,
            search_hit: None,
        }
    }

    /// Group a whole list in one go (the unpaged case).
    fn build_date_groups(records: &[LibraryImage]) -> Vec<DateGroup> {
        let mut groups = Vec::new();
        append_date_groups(&mut groups, records, 0);
        groups
    }

    #[test]
    fn build_date_groups_splits_on_day_boundary() {
        // Two shots on day 0, one shot a full day later.
        let records = vec![image_at(0, Some(0)), image_at(3600, Some(3600)), image_at(90_000, Some(90_000))];
        let groups = build_date_groups(&records);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].start, 0);
        assert_eq!(groups[0].count, 2);
        assert_eq!(groups[1].start, 2);
        assert_eq!(groups[1].count, 1);
    }

    #[test]
    fn append_date_groups_continues_a_day_across_a_page_boundary() {
        // Page 1 ends mid-day; page 2 opens with two more shots from that
        // same day, then rolls over.
        let page1 = vec![image_at(0, Some(0)), image_at(3600, Some(3600))];
        let page2 = vec![image_at(7200, Some(7200)), image_at(90_000, Some(90_000))];

        let mut records = page1.clone();
        let mut groups = build_date_groups(&records);
        assert_eq!(groups.len(), 1);

        let from = records.len();
        records.extend(page2);
        append_date_groups(&mut groups, &records, from);

        // The day-0 group grew rather than being duplicated.
        assert_eq!(groups.len(), 2);
        assert_eq!((groups[0].start, groups[0].count), (0, 3));
        assert_eq!((groups[1].start, groups[1].count), (3, 1));
        // Groups tile the accumulated list without gaps or overlap.
        assert_eq!(groups.iter().map(|g| g.count).sum::<i32>(), records.len() as i32);
    }

    #[test]
    fn append_date_groups_matches_a_full_rebuild() {
        // Same records, delivered in three pages, must group identically to
        // one unpaged listing.
        let all: Vec<LibraryImage> = (0..9)
            .map(|k: i64| {
                // Three per day, three days.
                let ts = (k / 3) * 86_400 + (k % 3) * 3600;
                image_at(ts, Some(ts))
            })
            .collect();

        let mut records = Vec::new();
        let mut groups = Vec::new();
        for page in all.chunks(2) {
            let from = records.len();
            records.extend_from_slice(page);
            append_date_groups(&mut groups, &records, from);
        }

        let rebuilt = build_date_groups(&all);
        let shape = |gs: &[DateGroup]| -> Vec<(String, i32, i32)> {
            gs.iter().map(|g| (g.label.to_string(), g.start, g.count)).collect()
        };
        assert_eq!(shape(&groups), shape(&rebuilt));
        assert_eq!(groups.len(), 3);
    }

    #[test]
    fn append_date_groups_on_an_empty_page_changes_nothing() {
        let records = vec![image_at(0, Some(0))];
        let mut groups = build_date_groups(&records);
        append_date_groups(&mut groups, &records, records.len());
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].count, 1);
    }

    #[test]
    fn build_date_groups_falls_back_to_added_at_when_taken_at_missing() {
        let records = vec![image_at(90_000, None)];
        let groups = build_date_groups(&records);
        assert_eq!(groups.len(), 1);
        assert_eq!(groups[0].count, 1);
    }

    #[test]
    fn score_caption_variants() {
        assert_eq!(score_caption(None), "");
        assert_eq!(
            score_caption(Some(&SearchHit::Direct { field: "filename".into(), snippet: None })),
            "direct"
        );
        assert_eq!(
            score_caption(Some(&SearchHit::Semantic { similarity: 0.873, sentence: String::new() })),
            "87% match"
        );
    }

    #[test]
    fn format_unix_ts_handles_zero_and_negative() {
        assert_eq!(format_unix_ts(0), "—");
        assert_eq!(format_unix_ts(-5), "—");
    }

    #[test]
    fn format_unix_ts_known_value() {
        // 2026-06-30 12:00:00 UTC
        assert_eq!(format_unix_ts(1782820800), "2026-06-30  12:00 UTC");
    }

    #[test]
    fn truncate_value_leaves_short_strings_unchanged() {
        assert_eq!(truncate_value("hello"), "hello");
    }

    #[test]
    fn truncate_value_truncates_long_strings_with_ellipsis() {
        let long = "a".repeat(120);
        let out = truncate_value(&long);
        assert_eq!(out.chars().count(), 90);
        assert!(out.ends_with('…'));
    }

    #[test]
    fn hex_to_color_parses_valid_hex() {
        let c = hex_to_color("#ff8800");
        assert_eq!((c.red(), c.green(), c.blue()), (0xff, 0x88, 0x00));
    }

    #[test]
    fn hex_to_color_falls_back_on_invalid_hex() {
        let c = hex_to_color("not-a-color");
        assert_eq!((c.red(), c.green(), c.blue()), (0x9a, 0x9a, 0x9a));
    }

    #[test]
    fn color_to_hex_round_trips_through_hex_to_color() {
        assert_eq!(color_to_hex(hex_to_color("#ff8800")), "#ff8800");
    }

    #[test]
    fn color_to_hex_zero_pads_each_channel() {
        assert_eq!(color_to_hex(slint::Color::from_rgb_u8(0x00, 0x0a, 0xff)), "#000aff");
    }

    #[test]
    fn trimmed_name_strips_surrounding_whitespace() {
        assert_eq!(trimmed_name("  Holiday  ").as_deref(), Some("Holiday"));
    }

    #[test]
    fn trimmed_name_rejects_blank_input() {
        assert_eq!(trimmed_name(""), None);
        assert_eq!(trimmed_name("   \t "), None);
    }

    #[test]
    fn optional_id_maps_the_negative_sentinel_to_none() {
        assert_eq!(optional_id(-1), None);
        assert_eq!(optional_id(0), Some(0));
        assert_eq!(optional_id(7), Some(7));
    }

    #[test]
    fn record_index_finds_the_matching_record() {
        let mut records = vec![image_at(0, None), image_at(0, None), image_at(0, None)];
        for (i, rec) in records.iter_mut().enumerate() {
            rec.id = i as i64 + 10;
        }
        assert_eq!(record_index(&records, 11), 1);
    }

    #[test]
    fn record_index_falls_back_to_the_first_record() {
        let records = vec![image_at(0, None)];
        assert_eq!(record_index(&records, 999), 0);
        assert_eq!(record_index(&[], 1), 0);
    }

    fn face_at(id: i64, person_id: Option<i64>, embedding: Vec<f32>) -> FaceDetection {
        FaceDetection {
            id,
            image_id: 1,
            bbox: [0.1, 0.1, 0.2, 0.2],
            embedding,
            person_id,
            confidence: 0.9,
            skipped: false,
        }
    }

    #[test]
    fn is_real_detection_rejects_zero_confidence_sentinel() {
        let mut f = face_at(1, None, vec![]);
        f.bbox = [0.0, 0.0, 0.0, 0.0];
        assert!(!is_real_detection(&f));
        assert!(is_real_detection(&face_at(1, None, vec![])));
    }

    #[test]
    fn embedding_matrix_top_k_ranks_by_similarity() {
        let known = vec![
            (1, "Alice".to_owned(), vec![1.0, 0.0]),
            (2, "Bob".to_owned(), vec![0.0, 1.0]),
        ];
        let persons = vec![(1, "Alice".to_owned()), (2, "Bob".to_owned())];
        let matrix = EmbeddingMatrix::from_rows(known, persons);

        let top = matrix.top_k(&[1.0, 0.0], 1);
        assert_eq!(top[0].0, 1);
        assert_eq!(top[0].1, "Alice");
        assert!((top[0].2 - 1.0).abs() < 1e-6);
    }

    #[test]
    fn embedding_matrix_top_k_falls_back_to_all_persons_without_query() {
        let persons = vec![(1, "Alice".to_owned()), (2, "Bob".to_owned())];
        let matrix = EmbeddingMatrix::from_rows(vec![], persons);
        let top = matrix.top_k(&[], 5);
        assert_eq!(top.len(), 2);
        assert!(top.iter().all(|(_, _, sim)| *sim == f32::NEG_INFINITY));
    }

    #[test]
    fn faces_to_boxes_labels_assigned_face_from_person_list() {
        let persons = vec![(1, "Alice".to_owned())];
        let matrix = EmbeddingMatrix::from_rows(vec![], persons);
        let faces = vec![face_at(10, Some(1), vec![])];
        let boxes = faces_to_boxes(&faces, &matrix, 0.5);
        assert_eq!(boxes.len(), 1);
        assert_eq!(boxes[0].label.as_str(), "Alice");
        assert!(boxes[0].is_assigned);
        assert!(!boxes[0].is_suggestion);
    }

    #[test]
    fn faces_to_boxes_filters_out_zero_confidence_sentinels() {
        let matrix = EmbeddingMatrix::empty();
        let mut sentinel = face_at(1, None, vec![]);
        sentinel.bbox = [0.0, 0.0, 0.0, 0.0];
        let boxes = faces_to_boxes(&[sentinel], &matrix, 0.5);
        assert!(boxes.is_empty());
    }

    /// A 200×100 image drawn at (50, 20) in the viewport.
    fn geo() -> ViewportGeometry {
        ViewportGeometry { img_left: 50.0, img_top: 20.0, disp_w: 200.0, disp_h: 100.0 }
    }

    #[test]
    fn hit_test_face_finds_the_box_under_the_pointer() {
        // Default bbox [0.1, 0.1, 0.2, 0.2] → viewport x 70..90, y 30..40.
        let faces = vec![face_at(7, None, vec![])];
        assert_eq!(hit_test_face(&faces, geo(), 80.0, 35.0), Some(7));
        // Corners count as inside; just outside does not.
        assert_eq!(hit_test_face(&faces, geo(), 70.0, 30.0), Some(7));
        assert_eq!(hit_test_face(&faces, geo(), 91.0, 35.0), None);
        assert_eq!(hit_test_face(&faces, geo(), 80.0, 20.0), None);
    }

    #[test]
    fn hit_test_face_ignores_sentinels_and_an_unlaid_out_viewport() {
        let mut sentinel = face_at(1, None, vec![]);
        sentinel.bbox = [0.0, 0.0, 0.0, 0.0];
        // The sentinel's box would otherwise cover the image's top-left corner.
        assert_eq!(hit_test_face(&[sentinel], geo(), 50.0, 20.0), None);

        let unlaid = ViewportGeometry { disp_w: 0.0, disp_h: 0.0, ..geo() };
        assert_eq!(hit_test_face(&[face_at(7, None, vec![])], unlaid, 80.0, 35.0), None);
    }

    #[test]
    fn normalise_draw_box_sorts_corners_of_a_backwards_drag() {
        // Dragged bottom-right → top-left; both orders give the same bbox.
        let forward = normalise_draw_box(geo(), 70.0, 30.0, 90.0, 40.0).unwrap();
        let backward = normalise_draw_box(geo(), 90.0, 40.0, 70.0, 30.0).unwrap();
        assert_eq!(forward, backward);
        let [x1, y1, x2, y2] = forward;
        assert!((x1 - 0.1).abs() < 1e-6 && (x2 - 0.2).abs() < 1e-6);
        assert!((y1 - 0.1).abs() < 1e-6 && (y2 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn normalise_draw_box_clamps_a_drag_that_left_the_image() {
        let [x1, y1, x2, y2] = normalise_draw_box(geo(), -500.0, -500.0, 90.0, 40.0).unwrap();
        assert_eq!((x1, y1), (0.0, 0.0));
        assert!((x2 - 0.2).abs() < 1e-6 && (y2 - 0.2).abs() < 1e-6);
    }

    #[test]
    fn normalise_draw_box_rejects_an_accidental_click() {
        // Under 0.5% of an image side in either axis.
        assert_eq!(normalise_draw_box(geo(), 70.0, 30.0, 70.5, 40.0), None);
        assert_eq!(normalise_draw_box(geo(), 70.0, 30.0, 90.0, 30.2), None);
        let unlaid = ViewportGeometry { disp_w: 0.0, disp_h: 0.0, ..geo() };
        assert_eq!(normalise_draw_box(unlaid, 70.0, 30.0, 90.0, 40.0), None);
    }

    #[test]
    fn faces_to_suggestions_formats_similarity_percent() {
        let known = vec![(1, "Alice".to_owned(), vec![1.0, 0.0])];
        let matrix = EmbeddingMatrix::from_rows(known, vec![(1, "Alice".to_owned())]);
        let sugs = faces_to_suggestions(&[1.0, 0.0], &matrix);
        assert_eq!(sugs[0].name.as_str(), "Alice");
        assert_eq!(sugs[0].sim_text.as_str(), "100%");
    }
}
