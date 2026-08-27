//! Session detection — finding the stretches where the photographer stayed
//! on one subject.
//!
//! Not a burst detector. The case this is built for is twenty pictures of
//! one child in one room over four minutes: irregular gaps, the subject
//! moving and reframing, the background holding still. Those photos are
//! *contiguous* in capture order, which is the whole exploit — this is a
//! **segmentation** of a sequence, not a clustering of a set:
//!
//! - `n - 1` comparisons instead of `n²`. A 3000-photo card costs 3000
//!   neighbour distances, not 4.5 million pairwise ones.
//! - No global threshold. "Did the scene change *here*" is a local question
//!   with a local answer, unlike "is this pair similar", whose one number
//!   has to work across a whole card.
//! - The transitive chaining that fuses a slow pan into one 200-photo group
//!   under union-find (`maple_db::cluster_embeddings`) cannot happen: a
//!   session has two ends.
//!
//! ## Engines
//!
//! What "the scene changed" *means* is deliberately pluggable. An engine is
//! a descriptor plus a distance over it; [`segment`] is shared and knows
//! nothing about either. Their distances are **not comparable to each
//! other** — each carries its own [`SessionEngine::default_cut`], and a cut
//! tuned for one means nothing to another.
//!
//! | Engine | Sees | Blind to |
//! |---|---|---|
//! | [`ColorKmeansEngine`] | the palette | where anything is |
//! | [`GridHistogramEngine`] | colour *and* its layout | shape, texture |
//! | [`BlockTileEngine`] | which parts of the frame held still | colour |
//! | [`TimeGapEngine`] | the clock alone | the picture entirely |
//! | [`EnsembleEngine`] | a weighted vote of the above | — |
//! | `maple_db::DinoEngine` | semantics | nothing cheap |
//!
//! An engine sees a [`Frame`], not just pixels, because the useful signals
//! are not all visual — [`TimeGapEngine`] reads nothing but the capture
//! time. Everything an engine might reasonably want lives on that struct so
//! adding the next signal does not churn the trait again.

use image::RgbImage;

mod block_tile;
mod color_kmeans;
mod ensemble;
mod grid_histogram;
mod time_gap;

pub use block_tile::BlockTileEngine;
pub use color_kmeans::ColorKmeansEngine;
pub use ensemble::EnsembleEngine;
pub use grid_histogram::GridHistogramEngine;
pub use time_gap::TimeGapEngine;

/// One photo, as an engine sees it.
pub struct Frame<'a> {
    /// The already-downscaled render. The import pipeline hands every
    /// engine the same ~256 px frame, which is also what makes a
    /// comparison between engines fair.
    pub rgb: &'a RgbImage,
    /// Capture time in fractional seconds since the epoch — `None` when
    /// the photo carries no usable EXIF.
    pub taken: Option<f64>,
}

impl<'a> Frame<'a> {
    pub fn new(rgb: &'a RgbImage, taken: Option<f64>) -> Self {
        Self { rgb, taken }
    }
}

/// One photo's descriptor, in whatever layout produced it.
///
/// The values are opaque outside the engine that made them: only
/// [`SessionEngine::distance`] knows the layout. The engine name rides
/// along so a mismatched pair is caught instead of silently returning a
/// meaningless number.
#[derive(Clone, Debug, PartialEq)]
pub struct Signature {
    engine: &'static str,
    values: Vec<f32>,
}

impl Signature {
    pub fn new(engine: &'static str, values: Vec<f32>) -> Self {
        Self { engine, values }
    }

    pub fn engine(&self) -> &'static str {
        self.engine
    }

    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Bytes of heap this signature holds — reported by the debug harness,
    /// since a card of 3000 photos holds one of these per photo.
    pub fn heap_bytes(&self) -> usize {
        self.values.len() * std::mem::size_of::<f32>()
    }
}

/// A way of deciding how much two frames differ.
///
/// `&mut self` on [`signature`](SessionEngine::signature) because the
/// DINOv2 engine owns an ONNX session and `ort` wants `&mut` to run it, and
/// because [`TimeGapEngine`] latches an epoch off the first frame it sees.
///
/// `Send` because the import scan builds the engine on the scan thread and
/// runs it on a stage of its own — one engine, moved once. It is
/// deliberately *not* `Sync`: `signature` takes `&mut self` (`TimeGapEngine`
/// latches an epoch on the first frame), so the engine is a sequential
/// thing and sharing one across decode threads would be a bug, not an
/// optimisation.
pub trait SessionEngine: Send {
    /// Stable identifier, also stamped into every [`Signature`].
    fn name(&self) -> &'static str;

    /// One line naming the configuration actually in force, for the log
    /// and the debug report.
    fn describe(&self) -> String;

    /// Distance at which this engine calls it a scene change, before
    /// [`segment`] applies the gap and streak adjustments.
    ///
    /// Engine-specific by necessity — a 0.2 from the block tile and a 0.2
    /// from DINOv2 are unrelated quantities.
    fn default_cut(&self) -> f32;

    /// Describe one frame.
    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature>;

    /// `0.0` = indistinguishable, `1.0` = unrelated. Must be symmetric.
    fn distance(&self, a: &Signature, b: &Signature) -> f32;
}

// ── Segmentation ──────────────────────────────────────────────────

/// Why [`segment`] ended a session before a given photo.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CutReason {
    /// The frame simply looks like somewhere else, and so did the ones
    /// after it — nothing came back.
    Scene,
    /// Each step was small but they added up: the frame no longer
    /// resembles the one the session started on. This is the pan that
    /// union-find chains into a single group.
    Drift,
    /// Longer than [`SegmentParams::hard_gap_secs`] passed. Whatever the
    /// pixels say, that is a different sitting.
    Gap,
}

/// One decision about one photo, kept so the debug harness can show its
/// work.
#[derive(Clone, Copy, Debug)]
pub struct Link {
    /// The photo being judged. A cut here means `at` starts a new session.
    pub at: usize,
    /// Which photo it was compared against — the last one accepted into
    /// the current session, which is not always `at - 1`: an outlier that
    /// got bridged over sits between them.
    pub from: usize,
    pub distance: f32,
    /// Distance back to the frame the current session started on.
    pub anchor_distance: f32,
    /// Seconds between `from` and `at`; `None` when either has no capture
    /// time.
    pub gap_secs: Option<f32>,
    /// The threshold `distance` was actually judged against, after the gap
    /// and streak adjustments.
    pub threshold: f32,
    pub cut: Option<CutReason>,
    /// This photo did not fit, but a later one matched the session again,
    /// so it was absorbed rather than allowed to end the run.
    pub bridged: bool,
}

/// A contiguous run of photos, as indices into the input sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Session {
    pub start: usize,
    /// Exclusive.
    pub end: usize,
}

impl Session {
    pub fn len(&self) -> usize {
        self.end - self.start
    }

    pub fn is_empty(&self) -> bool {
        self.end <= self.start
    }

    pub fn contains(&self, idx: usize) -> bool {
        idx >= self.start && idx < self.end
    }
}

/// The result of one walk down the sequence.
#[derive(Clone, Debug)]
pub struct Segmentation {
    /// Every session, including one-photo ones — a caller wanting only
    /// real groups filters on `len() >= 2`. They are kept because "this
    /// photo stands alone" is an answer, and the debug report needs to
    /// show it.
    pub sessions: Vec<Session>,
    /// One entry per photo after the first, in order.
    pub links: Vec<Link>,
    /// Photos that are inside a session without matching it — the shot of
    /// the cake in the middle of twenty shots of the child. Sorted.
    pub outliers: Vec<usize>,
}

impl Segmentation {
    /// The session `idx` belongs to.
    pub fn session_of(&self, idx: usize) -> Option<&Session> {
        self.sessions.iter().find(|s| s.contains(idx))
    }

    /// Sessions of two or more photos — the ones worth showing a user.
    pub fn groups(&self) -> impl Iterator<Item = &Session> {
        self.sessions.iter().filter(|s| s.len() >= 2)
    }

    pub fn is_outlier(&self, idx: usize) -> bool {
        self.outliers.binary_search(&idx).is_ok()
    }
}

/// How hard it is to end a session.
///
/// Time enters as a **cost, not a cut**. A fixed gap threshold cannot work
/// on the case this is for: twenty shots of one child come at 2 s, 3 s,
/// then 40 s while you reposition, then 2 s again. Cutting at a fixed 30 s
/// splits that session; not cutting at all merges two rooms shot 8 s apart.
/// So the gap does not decide — it moves the *visual* evidence required to
/// decide. A long gap makes cutting cheap, a short gap makes it expensive,
/// and the pixels still have the final say until `hard_gap_secs`.
///
/// This is a separate mechanism from [`TimeGapEngine`], which votes on the
/// clock as one member of an ensemble. Running both means time counts
/// twice; zero `tight_hold` and `long_drop` when the ensemble already has
/// a time member.
#[derive(Clone, Copy, Debug)]
pub struct SegmentParams {
    /// Base distance for a cut. Comes from [`SessionEngine::default_cut`]
    /// unless the caller overrides it.
    pub cut: f32,
    /// Gap at which the threshold has moved halfway from `tight_hold` to
    /// `long_drop`.
    pub gap_scale_secs: f32,
    /// Fraction the threshold is raised by at a zero gap — two frames one
    /// second apart get the benefit of the doubt.
    pub tight_hold: f32,
    /// Fraction the threshold falls by as the gap grows without bound.
    pub long_drop: f32,
    /// Extra threshold once a session is `streak_len` long. Hysteresis: one
    /// badly framed shot in the middle of a run should not fragment it.
    pub streak_bonus: f32,
    pub streak_len: usize,
    /// Multiple of the threshold at which distance-from-anchor alone ends
    /// the session. This is the anti-chaining rule; `f32::INFINITY`
    /// disables it and gives plain single-link behaviour.
    pub anchor_factor: f32,
    /// A gap this long always cuts, whatever the frames look like.
    pub hard_gap_secs: f32,
    /// How many consecutive non-matching photos a session may absorb while
    /// waiting to see whether the sequence comes back to it.
    ///
    /// One is the useful default: you photograph the child, turn to shoot
    /// the cake once, then turn back. Zero restores strict behaviour —
    /// anything that does not fit ends the session on the spot.
    pub max_outliers: usize,
}

impl SegmentParams {
    /// Defaults with `cut` taken from the engine.
    pub fn for_engine(engine: &dyn SessionEngine) -> Self {
        Self { cut: engine.default_cut(), ..Self::default() }
    }

    /// Defaults for an engine built from `spec`, with the threshold shaping
    /// switched off when the spec already votes on the clock.
    ///
    /// Time is a *cost* here — a long gap makes a cut cheaper via
    /// `tight_hold`/`long_drop` — and [`TimeGapEngine`] is a *vote*. Both at
    /// once means a long gap cheapens the threshold **and** pushes the
    /// distance over it, which double-counts the one signal most likely to
    /// be wrong (a camera whose clock is off, a card holding two days).
    /// Whichever mechanism the spec chose, it gets used once.
    pub fn for_spec(engine: &dyn SessionEngine, spec: &str) -> Self {
        let mut p = Self::for_engine(engine);
        if spec_has_time(spec) {
            p.tight_hold = 0.0;
            p.long_drop = 0.0;
        }
        p
    }
}

impl Default for SegmentParams {
    fn default() -> Self {
        Self {
            cut: 0.25,
            gap_scale_secs: 30.0,
            tight_hold: 0.35,
            long_drop: 0.45,
            streak_bonus: 0.15,
            streak_len: 3,
            anchor_factor: 1.8,
            hard_gap_secs: 1800.0,
            max_outliers: 1,
        }
    }
}

/// The threshold a distance is judged against.
///
/// `gap` is `None` when either photo has no capture time — treated as
/// exactly the neutral point (`gap_scale_secs`), so a card with no EXIF
/// degrades to pure visual segmentation instead of behaving as if every
/// photo were simultaneous.
fn threshold_for(p: &SegmentParams, gap: Option<f32>, streak: usize) -> f32 {
    let gap = gap.unwrap_or(p.gap_scale_secs).max(0.0);
    let s = gap / (gap + p.gap_scale_secs); // 0 at no gap → 1 as gap → ∞
    let mut t = p.cut * ((1.0 + p.tight_hold) - (p.tight_hold + p.long_drop) * s);
    if streak >= p.streak_len {
        t *= 1.0 + p.streak_bonus;
    }
    t.max(0.0)
}

/// Walk the sequence once and cut it into sessions.
///
/// `times` are capture times in seconds (fractional, so a sub-second EXIF
/// stamp survives); `None` for a photo whose time is unknown. `signatures`
/// and `times` must be the same length and in capture order — which, on a
/// camera card, path order already is.
///
/// Every photo is judged against the **last one accepted into the current
/// session**, not against its immediate predecessor. That is what lets a
/// session absorb an outlier: a frame that does not fit is held aside, and
/// if the one after it matches the session again, the odd frame is
/// recorded in [`Segmentation::outliers`] rather than allowed to end the
/// run. Only when `max_outliers` frames in a row fail to match does the
/// session end — and it ends *before* the first of them, since that is
/// where the new scene actually started.
pub fn segment(
    engine: &dyn SessionEngine,
    signatures: &[Signature],
    times: &[Option<f64>],
    p: &SegmentParams,
) -> Segmentation {
    assert_eq!(signatures.len(), times.len(), "signatures and times must agree in length");
    let n = signatures.len();
    let mut sessions = Vec::new();
    let mut outliers = Vec::new();
    // Indexed by photo, so a retry after a cut overwrites its earlier
    // provisional decision instead of appending a second one.
    let mut links: Vec<Option<Link>> = vec![None; n];
    if n == 0 {
        return Segmentation { sessions, links: Vec::new(), outliers };
    }

    let gap_between = |a: usize, b: usize| match (times[a], times[b]) {
        (Some(x), Some(y)) => Some((y - x).abs() as f32),
        _ => None,
    };

    let mut start = 0usize;
    let mut last_good = 0usize;
    // Photos held aside, waiting to see whether the sequence comes back.
    let mut pending: Vec<usize> = Vec::new();
    let mut at = 1usize;

    while at < n {
        let distance = engine.distance(&signatures[last_good], &signatures[at]);
        let anchor_distance = if start == last_good {
            distance
        } else {
            engine.distance(&signatures[start], &signatures[at])
        };
        let gap = gap_between(last_good, at);
        let threshold = threshold_for(p, gap, last_good + 1 - start);

        let reason = if gap.is_some_and(|g| g >= p.hard_gap_secs) {
            Some(CutReason::Gap)
        } else if distance > threshold {
            Some(CutReason::Scene)
        } else if anchor_distance > threshold * p.anchor_factor {
            Some(CutReason::Drift)
        } else {
            None
        };

        let link = Link {
            at,
            from: last_good,
            distance,
            anchor_distance,
            gap_secs: gap,
            threshold,
            cut: reason,
            bridged: false,
        };

        match reason {
            // Fits: anything held aside was bridged over after all.
            None => {
                for held in pending.drain(..) {
                    outliers.push(held);
                    if let Some(l) = links[held].as_mut() {
                        l.bridged = true;
                        l.cut = None;
                    }
                }
                links[at] = Some(link);
                last_good = at;
                at += 1;
            }
            // Does not fit, but the session may still get it back.
            Some(_) if pending.len() < p.max_outliers => {
                pending.push(at);
                links[at] = Some(link);
                at += 1;
            }
            // Out of patience. The scene changed at the *first* frame that
            // stopped matching, not at this one.
            Some(_) => {
                let cut_at = pending.first().copied().unwrap_or(at);
                sessions.push(Session { start, end: cut_at });
                if cut_at == at {
                    links[at] = Some(link);
                } else if let Some(l) = links[cut_at].as_mut() {
                    // Its own decision stands; it is the boundary now.
                    l.cut = l.cut.or(Some(CutReason::Scene));
                    l.bridged = false;
                }
                start = cut_at;
                last_good = cut_at;
                pending.clear();
                // Re-judge everything after the new start against it.
                at = cut_at + 1;
            }
        }
    }

    // Frames still held aside at the end never came back, so they are not
    // outliers — they are the start of something the card stopped short of.
    if let Some(&first) = pending.first() {
        sessions.push(Session { start, end: first });
        if let Some(l) = links[first].as_mut() {
            l.cut = l.cut.or(Some(CutReason::Scene));
        }
        start = first;
    }
    sessions.push(Session { start, end: n });

    outliers.sort_unstable();
    Segmentation { sessions, links: links.into_iter().flatten().collect(), outliers }
}

// ── Shared frame preparation ──────────────────────────────────────

/// Area-average `frame` down to `out_w × out_h`, as RGB in `0.0..=255.0`.
///
/// Averaging *is* the blur — every source pixel lands in exactly one output
/// cell, so shrinking a 256 px render to 48×48 low-passes it on the way.
/// No separate blur pass, and no dependency on a resize crate.
pub(crate) fn downsample(frame: &RgbImage, out_w: usize, out_h: usize) -> Vec<[f32; 3]> {
    let (w, h) = (frame.width() as usize, frame.height() as usize);
    let mut sums = vec![[0f32; 3]; out_w * out_h];
    let mut counts = vec![0f32; out_w * out_h];
    if w == 0 || h == 0 || out_w == 0 || out_h == 0 {
        return sums;
    }

    let raw = frame.as_raw();
    for y in 0..h {
        let oy = y * out_h / h;
        let row = y * w * 3;
        for x in 0..w {
            let cell = oy * out_w + (x * out_w / w);
            let px = row + x * 3;
            sums[cell][0] += raw[px] as f32;
            sums[cell][1] += raw[px + 1] as f32;
            sums[cell][2] += raw[px + 2] as f32;
            counts[cell] += 1.0;
        }
    }

    // A frame smaller than the grid leaves cells empty; carry the last
    // filled one forward rather than emitting black, which would read as a
    // huge difference to every engine.
    let mut last = [0f32; 3];
    for (cell, n) in sums.iter_mut().zip(counts) {
        if n > 0.0 {
            cell[0] /= n;
            cell[1] /= n;
            cell[2] /= n;
            last = *cell;
        } else {
            *cell = last;
        }
    }
    sums
}

/// Rec. 601 luma, matching [`crate::sharpness`].
pub(crate) fn luma(rgb: [f32; 3]) -> f32 {
    0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
}

/// [`segment`] over a sequence where some photos have no signature.
///
/// A photo the card never gave up — a read timeout, a file that will not
/// decode — has no descriptor, so there is no honest answer to "did the
/// scene change here". It becomes **its own session** and splits the
/// sequence, rather than being dropped (which would silently join the
/// photos either side of it, asserting a continuity nobody measured) or
/// treated as maximally distant (which would assert a change nobody
/// measured either). Each surviving run is segmented on its own and the
/// results are stitched back into whole-sequence indices.
pub fn segment_with_holes(
    engine: &dyn SessionEngine,
    signatures: &[Option<Signature>],
    times: &[Option<f64>],
    p: &SegmentParams,
) -> Segmentation {
    assert_eq!(signatures.len(), times.len(), "signatures and times must agree in length");
    let mut sessions = Vec::new();
    let mut links = Vec::new();
    let mut outliers = Vec::new();
    let mut at = 0usize;
    while at < signatures.len() {
        if signatures[at].is_none() {
            sessions.push(Session { start: at, end: at + 1 });
            at += 1;
            continue;
        }
        let end = signatures[at..]
            .iter()
            .position(Option::is_none)
            .map(|off| at + off)
            .unwrap_or(signatures.len());
        let run: Vec<Signature> = signatures[at..end].iter().map(|s| s.clone().unwrap()).collect();
        let seg = segment(engine, &run, &times[at..end], p);
        sessions
            .extend(seg.sessions.iter().map(|s| Session { start: s.start + at, end: s.end + at }));
        links.extend(seg.links.iter().map(|l| Link { at: l.at + at, from: l.from + at, ..*l }));
        outliers.extend(seg.outliers.iter().map(|o| o + at));
        at = end;
    }
    outliers.sort_unstable();
    Segmentation { sessions, links, outliers }
}

// ── Building an engine from a spec string ────────────────────────

/// Build an engine from a spec: one name, or a weighted ensemble.
///
/// `"block-tile"` is that engine at its own default cut.
/// `"block-tile=2,grid-histogram=1,time-gap=1"` is the weighted vote.
/// A bare name inside a list weighs 1, so `"block-tile,time-gap"` works.
///
/// This is the one place a *string* becomes an engine, so settings.toml,
/// the lab's `--ensemble` flag and any future UI all resolve a name the
/// same way. `dino` is deliberately not here: it lives in `maple-db`,
/// which depends on this crate and not the reverse.
pub fn engine_from_spec(spec: &str) -> anyhow::Result<Box<dyn SessionEngine>> {
    let members = parse_spec(spec)?;
    if members.len() == 1 && members[0].1 == 1.0 {
        return named_engine(&members[0].0);
    }
    let built = members
        .iter()
        .map(|(name, weight)| named_engine(name).map(|e| (e, *weight)))
        .collect::<anyhow::Result<Vec<_>>>()?;
    Ok(Box::new(EnsembleEngine::new(built)))
}

/// Whether `spec` already votes on the clock.
///
/// Load-bearing: the segmentation *also* shapes its threshold by the gap
/// (`tight_hold`/`long_drop`), and running both means time counts twice —
/// a long gap would both cheapen the cut and vote for it. A caller building
/// [`SegmentParams`] for a spec that contains a time member must zero the
/// shaping; [`SegmentParams::for_spec`] does it.
pub fn spec_has_time(spec: &str) -> bool {
    parse_spec(spec)
        .map(|m| m.iter().any(|(name, _)| name == TimeGapEngine::NAME))
        .unwrap_or(false)
}

fn parse_spec(spec: &str) -> anyhow::Result<Vec<(String, f32)>> {
    let members: Vec<(String, f32)> = spec
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|part| match part.split_once('=') {
            None => Ok((part.to_ascii_lowercase(), 1.0)),
            Some((name, weight)) => weight
                .trim()
                .parse::<f32>()
                .map_err(|_| anyhow::anyhow!("session engine weight `{weight}` is not a number"))
                .map(|w| (name.trim().to_ascii_lowercase(), w)),
        })
        .collect::<anyhow::Result<_>>()?;
    if members.is_empty() {
        anyhow::bail!("no session engine named in `{spec}`");
    }
    Ok(members)
}

fn named_engine(name: &str) -> anyhow::Result<Box<dyn SessionEngine>> {
    Ok(match name {
        BlockTileEngine::NAME => Box::<BlockTileEngine>::default(),
        GridHistogramEngine::NAME => Box::<GridHistogramEngine>::default(),
        ColorKmeansEngine::NAME => Box::<ColorKmeansEngine>::default(),
        TimeGapEngine::NAME => Box::<TimeGapEngine>::default(),
        other => anyhow::bail!(
            "unknown session engine `{other}` (known: {}, {}, {}, {})",
            BlockTileEngine::NAME,
            GridHistogramEngine::NAME,
            ColorKmeansEngine::NAME,
            TimeGapEngine::NAME,
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An engine whose signature is one number and whose distance is the
    /// absolute difference — lets the segmentation rules be tested without
    /// any pixels at all.
    struct Scalar;

    impl SessionEngine for Scalar {
        fn name(&self) -> &'static str {
            "scalar"
        }
        fn describe(&self) -> String {
            "test".into()
        }
        fn default_cut(&self) -> f32 {
            0.5
        }
        fn signature(&mut self, _frame: &Frame<'_>) -> anyhow::Result<Signature> {
            unimplemented!("tests build signatures directly")
        }
        fn distance(&self, a: &Signature, b: &Signature) -> f32 {
            (a.values()[0] - b.values()[0]).abs()
        }
    }

    fn sig(v: f32) -> Signature {
        Signature::new("scalar", vec![v])
    }

    fn sigs(vs: &[f32]) -> Vec<Signature> {
        vs.iter().map(|&v| sig(v)).collect()
    }

    fn clock(n: usize) -> Vec<Option<f64>> {
        (0..n).map(|i| Some(i as f64 * 2.0)).collect()
    }

    /// Strict: no gap shaping, no streak bonus, no drift rule, no
    /// outliers — so a test exercises one rule at a time.
    fn params() -> SegmentParams {
        SegmentParams {
            cut: 0.5,
            tight_hold: 0.0,
            long_drop: 0.0,
            streak_bonus: 0.0,
            anchor_factor: f32::INFINITY,
            max_outliers: 0,
            ..SegmentParams::default()
        }
    }

    fn spans(seg: &Segmentation) -> Vec<(usize, usize)> {
        seg.sessions.iter().map(|s| (s.start, s.end)).collect()
    }

    #[test]
    fn a_stable_sequence_is_one_session() {
        let seg = segment(&Scalar, &sigs(&[0.0, 0.05, 0.1, 0.05]), &clock(4), &params());
        assert_eq!(spans(&seg), vec![(0, 4)]);
    }

    #[test]
    fn a_scene_change_cuts() {
        let seg = segment(&Scalar, &sigs(&[0.0, 0.05, 5.0, 5.05]), &clock(4), &params());
        assert_eq!(spans(&seg), vec![(0, 2), (2, 4)]);
        assert_eq!(seg.links.iter().find(|l| l.at == 2).unwrap().cut, Some(CutReason::Scene));
    }

    #[test]
    fn drift_ends_a_session_even_though_every_step_is_small() {
        // The pan that single-link union-find would chain into one group:
        // each frame is 0.3 from the last, well under the 0.5 cut, but the
        // sixth looks nothing like the first.
        let vals: Vec<f32> = (0..6).map(|i| i as f32 * 0.3).collect();
        let p = SegmentParams { anchor_factor: 1.8, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(6), &p);
        assert!(seg.sessions.len() > 1, "drift should have ended a session: {:?}", spans(&seg));
        assert!(seg.links.iter().any(|l| l.cut == Some(CutReason::Drift)));
    }

    #[test]
    fn a_long_pause_in_the_same_scene_does_not_cut() {
        // 40 seconds while the photographer repositions the child. The
        // frames still match, so the session survives.
        let times = vec![Some(0.0), Some(40.0), Some(43.0)];
        let p = SegmentParams { cut: 0.5, ..SegmentParams::default() };
        let seg = segment(&Scalar, &sigs(&[0.0, 0.05, 0.1]), &times, &p);
        assert_eq!(spans(&seg), vec![(0, 3)]);
    }

    #[test]
    fn the_gap_moves_the_threshold_it_does_not_replace_it() {
        let p = SegmentParams::default();
        let tight = threshold_for(&p, Some(0.0), 1);
        let neutral = threshold_for(&p, Some(p.gap_scale_secs), 1);
        let long = threshold_for(&p, Some(600.0), 1);
        assert!(tight > neutral && neutral > long, "{tight} > {neutral} > {long}");
        assert!(long > 0.0, "a long gap lowers the bar, it never removes it");
        assert_eq!(threshold_for(&p, None, 1), neutral, "unknown time is the neutral point");
    }

    #[test]
    fn a_hard_gap_cuts_identical_frames() {
        let seg = segment(&Scalar, &sigs(&[0.0, 0.0]), &[Some(0.0), Some(4000.0)], &params());
        assert_eq!(spans(&seg), vec![(0, 1), (1, 2)]);
        assert_eq!(seg.links[0].cut, Some(CutReason::Gap));
    }

    #[test]
    fn a_streak_resists_one_odd_frame() {
        // Four steady frames, then one 0.55 step — over the bare 0.5 cut,
        // under the streak-adjusted one.
        let vals = [0.0, 0.0, 0.0, 0.0, 0.55];
        let p = SegmentParams { streak_bonus: 0.5, streak_len: 3, ..params() };
        assert_eq!(spans(&segment(&Scalar, &sigs(&vals), &clock(5), &p)), vec![(0, 5)]);

        let strict = SegmentParams { streak_bonus: 0.0, ..p };
        assert_eq!(spans(&segment(&Scalar, &sigs(&vals), &clock(5), &strict)), vec![(0, 4), (4, 5)]);
    }

    #[test]
    fn one_odd_photo_is_absorbed_when_the_sequence_comes_back() {
        // Twenty shots of the child, one of the cake, then the child
        // again. With outliers allowed that is one session, not three.
        let vals = [0.0, 0.05, 9.0, 0.1, 0.05];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(5), &p);
        assert_eq!(spans(&seg), vec![(0, 5)]);
        assert_eq!(seg.outliers, vec![2]);
        assert!(seg.is_outlier(2) && !seg.is_outlier(3));
        assert!(seg.links.iter().find(|l| l.at == 2).unwrap().bridged);
    }

    #[test]
    fn an_absorbed_photo_is_not_what_the_next_one_is_judged_against() {
        // The point of bridging: photo 3 is compared to photo 1, the last
        // frame that actually belonged, not to the cake.
        let vals = [0.0, 0.05, 9.0, 0.1];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(4), &p);
        assert_eq!(seg.links.iter().find(|l| l.at == 3).unwrap().from, 1);
    }

    #[test]
    fn a_real_scene_change_still_cuts_at_its_first_frame() {
        // Two frames in a row fail to match, so this is not an outlier —
        // and the boundary belongs before the first of them, not the
        // second.
        let vals = [0.0, 0.05, 9.0, 9.05, 9.1];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(5), &p);
        assert_eq!(spans(&seg), vec![(0, 2), (2, 5)]);
        assert!(seg.outliers.is_empty(), "nothing came back, so nothing was an outlier");
    }

    #[test]
    fn max_outliers_zero_restores_strict_behaviour() {
        let vals = [0.0, 0.05, 9.0, 0.1, 0.05];
        let seg = segment(&Scalar, &sigs(&vals), &clock(5), &params());
        assert_eq!(spans(&seg), vec![(0, 2), (2, 3), (3, 5)]);
        assert!(seg.outliers.is_empty());
    }

    #[test]
    fn a_trailing_mismatch_starts_its_own_session() {
        // The card ends on a frame that never matched anything. It is not
        // an outlier — nothing came back to prove it was.
        let vals = [0.0, 0.05, 9.0];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(3), &p);
        assert_eq!(spans(&seg), vec![(0, 2), (2, 3)]);
        assert!(seg.outliers.is_empty());
    }

    #[test]
    fn every_photo_after_the_first_gets_exactly_one_link() {
        let vals = [0.0, 0.05, 9.0, 0.1, 5.0, 5.05, 0.0];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(7), &p);
        let mut ats: Vec<usize> = seg.links.iter().map(|l| l.at).collect();
        ats.sort_unstable();
        assert_eq!(ats, vec![1, 2, 3, 4, 5, 6], "one final decision per photo");
    }

    #[test]
    fn sessions_tile_the_sequence_without_gaps_or_overlaps() {
        let vals = [0.0, 9.0, 0.1, 5.0, 5.05, 0.0, 0.02];
        let p = SegmentParams { max_outliers: 1, ..params() };
        let seg = segment(&Scalar, &sigs(&vals), &clock(7), &p);
        let mut expect = 0;
        for s in &seg.sessions {
            assert_eq!(s.start, expect, "sessions must tile: {:?}", spans(&seg));
            assert!(s.end > s.start);
            expect = s.end;
        }
        assert_eq!(expect, 7);
    }

    #[test]
    fn empty_and_single_inputs_are_handled() {
        let seg = segment(&Scalar, &[], &[], &params());
        assert!(seg.sessions.is_empty() && seg.links.is_empty());
        let seg = segment(&Scalar, &[sig(0.0)], &[None], &params());
        assert_eq!(spans(&seg), vec![(0, 1)]);
        assert!(seg.links.is_empty());
    }

    // ── Spec parsing ─────────────────────────────────────────────

    #[test]
    fn a_bare_name_builds_that_engine_not_an_ensemble() {
        assert_eq!(engine_from_spec("block-tile").unwrap().name(), BlockTileEngine::NAME);
        assert_eq!(engine_from_spec("  Time-Gap ").unwrap().name(), TimeGapEngine::NAME);
    }

    #[test]
    fn a_weighted_list_builds_an_ensemble() {
        let e = engine_from_spec("block-tile=2,grid-histogram=1,time-gap=1").unwrap();
        assert_eq!(e.name(), "ensemble");
        // Every member reads 0.5 at its own threshold, so the vote does too.
        assert_eq!(e.default_cut(), 0.5);
    }

    #[test]
    fn a_bare_name_inside_a_list_weighs_one() {
        let a = engine_from_spec("block-tile,time-gap").unwrap().describe();
        let b = engine_from_spec("block-tile=1,time-gap=1").unwrap().describe();
        assert_eq!(a, b);
    }

    #[test]
    fn an_unknown_engine_names_the_ones_that_exist() {
        let err = match engine_from_spec("dino") {
            Err(e) => e.to_string(),
            Ok(e) => panic!("built an engine named {}", e.name()),
        };
        assert!(err.contains("block-tile"), "unhelpful: {err}");
        assert!(engine_from_spec("").is_err());
        assert!(engine_from_spec("block-tile=heavy").is_err());
    }

    #[test]
    fn a_spec_with_a_time_member_switches_off_threshold_shaping() {
        // Time is a cost via tight_hold/long_drop and a vote via the
        // engine. Both at once double-counts it.
        let spec = "block-tile=2,time-gap=1";
        let e = engine_from_spec(spec).unwrap();
        let p = SegmentParams::for_spec(e.as_ref(), spec);
        assert_eq!(p.tight_hold, 0.0);
        assert_eq!(p.long_drop, 0.0);

        // A pixel-only spec keeps the shaping — it is the only way the
        // clock reaches the decision at all.
        let spec = "block-tile";
        let e = engine_from_spec(spec).unwrap();
        let p = SegmentParams::for_spec(e.as_ref(), spec);
        assert!(p.tight_hold > 0.0 && p.long_drop > 0.0);
        assert_eq!(p.cut, BlockTileEngine::default().default_cut());
    }

    #[test]
    fn groups_skips_solo_photos() {
        let seg = segment(&Scalar, &sigs(&[0.0, 0.0, 9.0]), &clock(3), &params());
        let groups: Vec<_> = seg.groups().map(|s| (s.start, s.end)).collect();
        assert_eq!(groups, vec![(0, 2)]);
        assert_eq!(seg.session_of(2).map(|s| s.len()), Some(1));
    }
}
