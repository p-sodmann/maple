//! Engine 4 — the clock, and nothing else.
//!
//! A photographer does not stay on one subject indefinitely. Past a minute
//! the next frame is unlikely to belong to the same session, past ten
//! minutes it is very unlikely, and past an hour it is a different sitting
//! whatever the room looks like. That is real information, it costs
//! nothing, and no pixel-based engine has access to it.
//!
//! As an *engine* it is meant for [`super::EnsembleEngine`] — on its own it
//! would happily group two unrelated photos taken three seconds apart. Its
//! value is as a vote alongside something that looks at the picture.
//!
//! ## Not the same thing as the gap in [`SegmentParams`]
//!
//! [`super::SegmentParams`] already lets the gap *move the threshold* a
//! visual engine is judged against. This engine instead *is* a judgement,
//! with its own distance. Running both counts time twice — zero
//! `tight_hold` and `long_drop` when an ensemble already has a time
//! member. The two exist because they answer different questions: the
//! threshold shaping asks "how much visual evidence do I need here", and
//! this asks "how likely is this at all".
//!
//! ## The curve
//!
//! Control points, interpolated **in log time** because the interesting
//! range spans four orders of magnitude (a tenth of a second to an hour)
//! and a linear ramp would spend all its resolution above ten minutes:
//!
//! | gap | distance | reading |
//! |---|---|---|
//! | ≤ 1 s | 0.00 | certainly the same session |
//! | 1 min | 0.50 | unlikely |
//! | 10 min | 0.85 | very unlikely |
//! | ≥ 1 h | 1.00 | no |
//!
//! Every point is tunable — that table is a starting position, not a
//! finding.

use super::{Frame, Signature, SessionEngine};

const NAME: &str = "time-gap";

/// `(gap in seconds, distance)`, ascending by gap.
pub type Point = (f32, f32);

pub struct TimeGapEngine {
    points: Vec<Point>,
    /// Capture times arrive as seconds since 1970 — around 1.8e9, which
    /// an `f32` signature would quantise to ~128-second steps. So the
    /// first frame's time becomes the origin and everything is stored
    /// relative to it: over a day that is ~8 ms of resolution, over a
    /// month ~0.25 s. An import spans hours.
    epoch: Option<f64>,
}

impl Default for TimeGapEngine {
    fn default() -> Self {
        Self::new(vec![(1.0, 0.0), (60.0, 0.5), (600.0, 0.85), (3600.0, 1.0)])
    }
}

impl TimeGapEngine {
    /// This engine's spec name, as `engine_from_spec` and settings.toml
    /// spell it.
    pub const NAME: &'static str = NAME;

    pub fn new(mut points: Vec<Point>) -> Self {
        points.retain(|(g, _)| g.is_finite() && *g > 0.0);
        points.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        Self { points, epoch: None }
    }

    /// Distance for a gap of `secs`, by log-time interpolation between the
    /// control points. Flat below the first point and above the last.
    pub fn distance_for_gap(&self, secs: f32) -> f32 {
        let Some(&(first_gap, first_d)) = self.points.first() else {
            return 0.0;
        };
        let &(last_gap, last_d) = self.points.last().expect("non-empty checked above");
        // NaN is a bug upstream, not a short gap — answer with the most
        // conservative point rather than "certainly the same session".
        if secs.is_nan() {
            return last_d;
        }
        if secs <= first_gap {
            return first_d;
        }
        if secs >= last_gap {
            return last_d;
        }
        let lg = secs.ln();
        for pair in self.points.windows(2) {
            let ((g0, d0), (g1, d1)) = (pair[0], pair[1]);
            if secs <= g1 {
                let (l0, l1) = (g0.ln(), g1.ln());
                let t = if (l1 - l0).abs() < f32::EPSILON { 0.0 } else { (lg - l0) / (l1 - l0) };
                return d0 + t * (d1 - d0);
            }
        }
        last_d
    }
}

impl SessionEngine for TimeGapEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        let points = self
            .points
            .iter()
            .map(|(g, d)| format!("{}→{d:.2}", super::ensemble::human_secs(*g)))
            .collect::<Vec<_>>()
            .join(", ");
        format!("log-interpolated over {points}")
    }

    fn default_cut(&self) -> f32 {
        // Halfway up the curve, which the default points put at one
        // minute — the gap the brief calls "unlikely".
        0.5
    }

    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature> {
        let Some(taken) = frame.taken else {
            // An unknown time is not a time of zero. Empty means "abstain",
            // and `distance` answers with the cut so this engine casts no
            // vote either way.
            return Ok(Signature::new(NAME, Vec::new()));
        };
        let epoch = *self.epoch.get_or_insert(taken);
        Ok(Signature::new(NAME, vec![(taken - epoch) as f32]))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() {
            return 1.0;
        }
        match (a.values().first(), b.values().first()) {
            (Some(x), Some(y)) => self.distance_for_gap((y - x).abs()),
            // Exactly on the fence: no information, so no vote.
            _ => self.default_cut(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn engine() -> TimeGapEngine {
        TimeGapEngine::default()
    }

    #[test]
    fn the_default_curve_hits_the_stated_points() {
        let e = engine();
        assert!(e.distance_for_gap(0.5) < 0.01, "a half-second gap is nothing");
        assert!((e.distance_for_gap(60.0) - 0.5).abs() < 1e-5, "one minute: unlikely");
        assert!((e.distance_for_gap(600.0) - 0.85).abs() < 1e-5, "ten minutes: very unlikely");
        assert!((e.distance_for_gap(3600.0) - 1.0).abs() < 1e-5, "an hour: no");
    }

    #[test]
    fn beyond_an_hour_it_stays_at_zero_likelihood() {
        let e = engine();
        assert_eq!(e.distance_for_gap(7200.0), 1.0);
        assert_eq!(e.distance_for_gap(86_400.0), 1.0);
        assert_eq!(e.distance_for_gap(f32::INFINITY), 1.0);
    }

    #[test]
    fn the_curve_is_monotone() {
        let e = engine();
        let mut prev = -1.0;
        for secs in [0.1, 1.0, 5.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0, 9000.0] {
            let d = e.distance_for_gap(secs);
            assert!(d >= prev, "{secs}s gave {d}, less than the {prev} before it");
            prev = d;
        }
    }

    #[test]
    fn interpolation_is_in_log_time() {
        // Halfway between 60 s and 600 s in log time is ~190 s, not 330 s.
        // At the geometric midpoint the distance must be the midpoint of
        // the two control values.
        let e = engine();
        let mid = (60.0f32 * 600.0).sqrt();
        assert!((e.distance_for_gap(mid) - 0.675).abs() < 1e-3, "got {}", e.distance_for_gap(mid));
    }

    #[test]
    fn a_missing_capture_time_abstains() {
        let mut e = engine();
        let img = image::RgbImage::new(4, 4);
        let unknown = e.signature(&Frame::new(&img, None)).unwrap();
        let known = e.signature(&Frame::new(&img, Some(1_700_000_000.0))).unwrap();
        assert!(unknown.values().is_empty());
        assert_eq!(e.distance(&unknown, &known), e.default_cut());
        assert_eq!(e.distance(&unknown, &unknown), e.default_cut());
    }

    #[test]
    fn signatures_keep_sub_second_resolution_across_a_long_import() {
        // The reason times are stored relative to the first frame: as raw
        // unix seconds an f32 quantises to ~128-second steps, which would
        // erase every gap this engine is meant to read.
        let mut e = engine();
        let img = image::RgbImage::new(4, 4);
        let base = 1_700_000_000.0;
        let a = e.signature(&Frame::new(&img, Some(base))).unwrap();
        let b = e.signature(&Frame::new(&img, Some(base + 86_400.0 + 0.25))).unwrap();
        let gap = b.values()[0] - a.values()[0];
        assert!((gap - 86_400.25).abs() < 0.05, "a day and a quarter-second came back as {gap}");
    }

    #[test]
    fn the_points_are_tunable() {
        let strict = TimeGapEngine::new(vec![(1.0, 0.0), (10.0, 1.0)]);
        assert_eq!(strict.distance_for_gap(30.0), 1.0);
        assert!(engine().distance_for_gap(30.0) < 0.5);
    }

    #[test]
    fn unsorted_or_junk_points_are_repaired_not_trusted() {
        let e = TimeGapEngine::new(vec![(600.0, 0.85), (-5.0, 0.2), (60.0, 0.5), (1.0, 0.0)]);
        assert!((e.distance_for_gap(60.0) - 0.5).abs() < 1e-5);
        assert!(e.distance_for_gap(0.1) < 0.01);
    }

    #[test]
    fn distance_is_symmetric() {
        let mut e = engine();
        let img = image::RgbImage::new(4, 4);
        let a = e.signature(&Frame::new(&img, Some(1_700_000_000.0))).unwrap();
        let b = e.signature(&Frame::new(&img, Some(1_700_000_300.0))).unwrap();
        assert_eq!(e.distance(&a, &b), e.distance(&b, &a));
    }
}
