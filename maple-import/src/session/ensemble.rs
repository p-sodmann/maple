//! Engine 5 — a weighted vote of the others.
//!
//! Every engine here is deliberately partial: the palette engine cannot see
//! where anything is, the block tile cannot see colour, the time engine
//! cannot see the picture at all. Combining them is not a hedge, it is the
//! point — a scene change that fools one rarely fools three.
//!
//! ## Putting incomparable numbers on one scale
//!
//! A 0.2 from the block tile and a 0.2 from DINOv2 mean nothing to each
//! other, so raw distances cannot simply be averaged. Each member's
//! distance is first mapped through
//!
//! ```text
//! score = d / (d + cut)
//! ```
//!
//! which is `0` at zero distance, exactly `0.5` at that member's own cut,
//! and approaches `1` as it saturates. Every member now says the same
//! thing on the same scale — *how far past my own threshold is this* — and
//! the weighted mean of those scores is the ensemble's distance, with
//! [`default_cut`](SessionEngine::default_cut) of `0.5`. A member that
//! abstains lands on exactly 0.5 and moves the result nowhere, which is
//! what makes [`super::TimeGapEngine`]'s "no capture time" answer harmless.
//!
//! Weights are relative; only their ratios matter. Zero drops a member
//! from the vote without removing it from the report.
//!
//! ## Signature layout
//!
//! Members' signatures are packed into one `Vec<f32>` behind a small
//! header (`member count`, then each member's length), because
//! [`Signature`] is one flat vector by design — it is what makes
//! signatures cheap to cache and to hand across a thread. `distance`
//! unpacks both sides and pairs them up positionally.

use super::{Frame, Signature, SessionEngine};

const NAME: &str = "ensemble";

pub struct EnsembleEngine {
    members: Vec<Member>,
}

struct Member {
    engine: Box<dyn SessionEngine>,
    weight: f32,
    /// Latched at construction: the member's own cut is what puts it on
    /// the shared scale, and it must not drift underneath us.
    cut: f32,
}

impl EnsembleEngine {
    /// `members` are `(engine, weight)`. A member's own `default_cut` sets
    /// its scale; pass a tuned one with [`with_cuts`](Self::with_cuts).
    pub fn new(members: Vec<(Box<dyn SessionEngine>, f32)>) -> Self {
        Self {
            members: members
                .into_iter()
                .map(|(engine, weight)| {
                    let cut = engine.default_cut();
                    Member { engine, weight: weight.max(0.0), cut }
                })
                .collect(),
        }
    }

    /// Override the per-member cut used to normalise, by engine name.
    /// Names that match no member are ignored — the debug harness passes
    /// one flat list of `--cut` overrides for every engine it knows.
    pub fn with_cuts(mut self, cuts: &[(String, f32)]) -> Self {
        for member in self.members.iter_mut() {
            if let Some((_, cut)) = cuts.iter().find(|(n, _)| n == member.engine.name()) {
                member.cut = *cut;
            }
        }
        self
    }

    pub fn members(&self) -> impl Iterator<Item = (&str, f32, f32)> {
        self.members.iter().map(|m| (m.engine.name(), m.weight, m.cut))
    }

    /// The shared scale: `0` at no distance, `0.5` at this member's own
    /// cut, approaching `1` as it saturates.
    fn score(distance: f32, cut: f32) -> f32 {
        if cut <= 0.0 {
            return if distance > 0.0 { 1.0 } else { 0.0 };
        }
        distance / (distance + cut)
    }

    fn unpack(values: &[f32]) -> Option<Vec<&[f32]>> {
        let count = *values.first()? as usize;
        let lengths: &[f32] = values.get(1..1 + count)?;
        let mut at = 1 + count;
        let mut out = Vec::with_capacity(count);
        for len in lengths {
            let len = *len as usize;
            out.push(values.get(at..at + len)?);
            at += len;
        }
        Some(out)
    }
}

impl SessionEngine for EnsembleEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        if self.members.is_empty() {
            return "no members".into();
        }
        let total: f32 = self.members.iter().map(|m| m.weight).sum();
        self.members
            .iter()
            .map(|m| {
                let share = if total > 0.0 { 100.0 * m.weight / total } else { 0.0 };
                format!("{}×{:.2} ({share:.0}%, cut {:.3})", m.engine.name(), m.weight, m.cut)
            })
            .collect::<Vec<_>>()
            .join(" + ")
    }

    fn default_cut(&self) -> f32 {
        // Every member scores 0.5 at its own threshold, so the weighted
        // mean crosses 0.5 exactly when the vote does.
        0.5
    }

    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature> {
        let mut parts = Vec::with_capacity(self.members.len());
        for member in self.members.iter_mut() {
            parts.push(member.engine.signature(frame)?);
        }
        let mut values = Vec::with_capacity(1 + parts.len() + parts.iter().map(|p| p.values().len()).sum::<usize>());
        values.push(parts.len() as f32);
        values.extend(parts.iter().map(|p| p.values().len() as f32));
        for part in &parts {
            values.extend_from_slice(part.values());
        }
        Ok(Signature::new(NAME, values))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() {
            return 1.0;
        }
        let (Some(pa), Some(pb)) = (Self::unpack(a.values()), Self::unpack(b.values())) else {
            return 1.0;
        };
        if pa.len() != self.members.len() || pb.len() != self.members.len() {
            return 1.0;
        }

        let mut acc = 0.0;
        let mut total = 0.0;
        for ((member, x), y) in self.members.iter().zip(pa).zip(pb) {
            if member.weight <= 0.0 {
                continue;
            }
            let name = member.engine.name();
            let d = member
                .engine
                .distance(&Signature::new(name, x.to_vec()), &Signature::new(name, y.to_vec()));
            acc += member.weight * Self::score(d, member.cut);
            total += member.weight;
        }
        if total <= 0.0 {
            return self.default_cut();
        }
        (acc / total).clamp(0.0, 1.0)
    }
}

/// `90` → `"1m30s"`. Shared with [`super::TimeGapEngine`]'s `describe`.
pub(crate) fn human_secs(secs: f32) -> String {
    if secs < 60.0 {
        format!("{secs:g$}s", g = if secs.fract() == 0.0 { 0 } else { 1 })
    } else if secs < 3600.0 {
        let (m, s) = ((secs / 60.0).floor(), secs % 60.0);
        if s == 0.0 {
            format!("{m:.0}m")
        } else {
            format!("{m:.0}m{s:.0}s")
        }
    } else {
        let h = secs / 3600.0;
        if h.fract() == 0.0 {
            format!("{h:.0}h")
        } else {
            format!("{h:.1}h")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::session::{BlockTileEngine, ColorKmeansEngine, TimeGapEngine};
    use image::RgbImage;

    /// Returns a fixed distance, so a test can control exactly what one
    /// member votes.
    struct Fixed {
        name: &'static str,
        distance: f32,
        cut: f32,
    }

    impl SessionEngine for Fixed {
        fn name(&self) -> &'static str {
            self.name
        }
        fn describe(&self) -> String {
            "fixed".into()
        }
        fn default_cut(&self) -> f32 {
            self.cut
        }
        fn signature(&mut self, _frame: &Frame<'_>) -> anyhow::Result<Signature> {
            Ok(Signature::new(self.name, vec![0.0]))
        }
        fn distance(&self, _a: &Signature, _b: &Signature) -> f32 {
            self.distance
        }
    }

    fn fixed(name: &'static str, distance: f32, cut: f32) -> Box<dyn SessionEngine> {
        Box::new(Fixed { name, distance, cut })
    }

    fn pair(engine: &mut dyn SessionEngine) -> (Signature, Signature) {
        let img = RgbImage::new(8, 8);
        let frame = Frame::new(&img, Some(1_700_000_000.0));
        (engine.signature(&frame).unwrap(), engine.signature(&frame).unwrap())
    }

    #[test]
    fn a_member_exactly_at_its_own_cut_scores_one_half() {
        let mut e = EnsembleEngine::new(vec![(fixed("a", 0.3, 0.3), 1.0)]);
        let (x, y) = pair(&mut e);
        assert!((e.distance(&x, &y) - 0.5).abs() < 1e-6);
        assert_eq!(e.default_cut(), 0.5, "so the ensemble sits exactly on its own fence");
    }

    #[test]
    fn members_with_wildly_different_scales_still_weigh_equally() {
        // 0.05 against a 0.10 cut and 0.35 against a 0.70 cut are the same
        // statement — both are half their own threshold. Averaging the raw
        // numbers would let the second dominate.
        let mut e = EnsembleEngine::new(vec![
            (fixed("dino", 0.05, 0.10), 1.0),
            (fixed("tile", 0.35, 0.70), 1.0),
        ]);
        let (x, y) = pair(&mut e);
        let both = e.distance(&x, &y);

        let mut one = EnsembleEngine::new(vec![(fixed("dino", 0.05, 0.10), 1.0)]);
        let (p, q) = pair(&mut one);
        assert!((both - one.distance(&p, &q)).abs() < 1e-6, "{both} vs the single member");
    }

    #[test]
    fn weight_shifts_the_vote() {
        let far = || fixed("far", 0.9, 0.1);
        let near = || fixed("near", 0.0, 0.1);

        let mut even = EnsembleEngine::new(vec![(far(), 1.0), (near(), 1.0)]);
        let (a, b) = pair(&mut even);
        let balanced = even.distance(&a, &b);

        let mut lean = EnsembleEngine::new(vec![(far(), 3.0), (near(), 1.0)]);
        let (a, b) = pair(&mut lean);
        assert!(lean.distance(&a, &b) > balanced, "more weight on the dissenter must raise it");
    }

    #[test]
    fn a_zero_weight_member_is_dropped_from_the_vote() {
        let mut e = EnsembleEngine::new(vec![
            (fixed("loud", 1.0, 0.1), 0.0),
            (fixed("quiet", 0.0, 0.1), 1.0),
        ]);
        let (a, b) = pair(&mut e);
        assert_eq!(e.distance(&a, &b), 0.0);
    }

    #[test]
    fn an_all_zero_weight_ensemble_abstains_rather_than_dividing_by_zero() {
        let mut e = EnsembleEngine::new(vec![(fixed("a", 1.0, 0.1), 0.0)]);
        let (a, b) = pair(&mut e);
        assert_eq!(e.distance(&a, &b), e.default_cut());
    }

    #[test]
    fn real_members_pack_and_unpack_without_crossing_wires() {
        // The signature layout is the fragile part: three members of very
        // different lengths have to come back out in the right order.
        let mut e = EnsembleEngine::new(vec![
            (Box::new(ColorKmeansEngine::default()), 1.0),
            (Box::new(BlockTileEngine::default()), 1.0),
            (Box::new(TimeGapEngine::default()), 1.0),
        ]);
        let img = RgbImage::from_fn(64, 64, |x, y| image::Rgb([x as u8 * 4, y as u8 * 4, 90]));
        let other = RgbImage::from_pixel(64, 64, image::Rgb([250, 250, 250]));
        let a = e.signature(&Frame::new(&img, Some(1_700_000_000.0))).unwrap();
        let same = e.signature(&Frame::new(&img, Some(1_700_000_002.0))).unwrap();
        let far = e.signature(&Frame::new(&other, Some(1_700_009_000.0))).unwrap();

        assert!(e.distance(&a, &same) < 0.2, "same picture two seconds later: {}", e.distance(&a, &same));
        assert!(e.distance(&a, &far) > e.default_cut(), "different picture two hours later");
        assert_eq!(e.distance(&a, &far), e.distance(&far, &a), "symmetric");
    }

    #[test]
    fn a_time_member_with_no_capture_time_moves_the_result_nowhere() {
        let mut e = EnsembleEngine::new(vec![
            (Box::new(ColorKmeansEngine::default()), 1.0),
            (Box::new(TimeGapEngine::default()), 1.0),
        ]);
        let img = RgbImage::from_pixel(64, 64, image::Rgb([120, 60, 30]));
        let a = e.signature(&Frame::new(&img, None)).unwrap();
        let b = e.signature(&Frame::new(&img, None)).unwrap();

        let mut colour_only = EnsembleEngine::new(vec![(Box::new(ColorKmeansEngine::default()) as Box<dyn SessionEngine>, 1.0)]);
        let (p, q) = {
            let f = Frame::new(&img, None);
            (colour_only.signature(&f).unwrap(), colour_only.signature(&f).unwrap())
        };
        // The abstaining member sits at 0.5, which is also where the
        // ensemble's own fence is — so the vote is unchanged in verdict.
        let with_time = e.distance(&a, &b);
        assert!(with_time < e.default_cut(), "still a match: {with_time}");
        assert!(colour_only.distance(&p, &q) < colour_only.default_cut());
    }

    #[test]
    fn cuts_can_be_overridden_by_name() {
        let e = EnsembleEngine::new(vec![(fixed("a", 0.3, 0.3), 1.0)])
            .with_cuts(&[("a".to_owned(), 0.6), ("nobody".to_owned(), 0.1)]);
        assert_eq!(e.members().map(|(_, _, cut)| cut).collect::<Vec<_>>(), vec![0.6]);
    }

    #[test]
    fn a_mismatched_signature_reads_as_unrelated() {
        let mut e = EnsembleEngine::new(vec![(fixed("a", 0.0, 0.3), 1.0)]);
        let (a, _) = pair(&mut e);
        assert_eq!(e.distance(&a, &Signature::new("ensemble", vec![9.0, 9.0])), 1.0);
        assert_eq!(e.distance(&a, &Signature::new("other", vec![])), 1.0);
    }

    #[test]
    fn human_secs_reads_like_a_person_wrote_it() {
        assert_eq!(human_secs(1.0), "1s");
        assert_eq!(human_secs(60.0), "1m");
        assert_eq!(human_secs(90.0), "1m30s");
        assert_eq!(human_secs(600.0), "10m");
        assert_eq!(human_secs(3600.0), "1h");
    }
}
