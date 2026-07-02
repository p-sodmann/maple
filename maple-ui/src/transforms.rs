//! Pure data-shape transforms shared across window controllers.
//!
//! No DB access, no `slint::Weak`/`ModelRc`/callback types — only the
//! generated plain-data structs (`DateGroup`, …) and primitives.

use maple_db::{LibraryImage, SearchHit};
use slint::SharedString;

use crate::date;
use crate::DateGroup;

/// Group `records` into contiguous same-day runs for the date-grouped view.
/// Assumes `records` is already sorted so that same-day images are adjacent
/// (see the `date_view` sort in `grid.rs::load()`); otherwise the same day
/// may appear as multiple separate groups.
pub fn build_date_groups(records: &[LibraryImage]) -> Vec<DateGroup> {
    let mut groups: Vec<DateGroup> = Vec::new();
    let mut current_day: Option<i64> = None;

    for (i, rec) in records.iter().enumerate() {
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

    groups
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
}
