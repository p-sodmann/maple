//! The import tournament: a photo's fate is decided by looking at it beside
//! the one it is competing with.
//!
//! # Why a pass, not a bracket
//!
//! The importer used to auto-mark the sharpest photo of every detected
//! session. That is a guess dressed as a decision: variance-of-Laplacian
//! ranks a badly-framed-but-crisp frame above the one where the child is
//! actually looking at the camera, and once it is marked nobody re-examines
//! it. So the marking is gone and the comparison is put in front of the
//! user instead — but it has to cost about one keystroke per photo, or a
//! 954-photo card is unusable.
//!
//! A real bracket does not fit. It needs ⌈log₂ n⌉ rounds, re-shows photos
//! the user has already judged, and has nowhere to put "keep both" — which
//! is the answer often enough that leaving it out would be the feature's
//! biggest flaw. So the tournament is a **single pass with an incumbent**:
//!
//! * The incumbent is on the left. The challenger — the next photo in
//!   capture order — is on the right.
//! * `1` keeps the left: the challenger is rejected, the incumbent stays.
//! * `2` keeps the right: the incumbent is rejected, the challenger takes
//!   its place.
//! * `3` takes both: the incumbent is **settled as kept** and the
//!   challenger takes its place.
//!
//! One invariant makes this coherent and is worth stating on its own:
//! **only the incumbent's fate is still open.** Everything behind it is
//! settled and is never asked about again. When a session runs out the
//! incumbent is settled as kept, so a session of *n* photos costs exactly
//! *n* − 1 keystrokes and every one of its photos ends up decided.
//!
//! The challenger becomes the incumbent on both `2` and `3` — never the
//! old one — because the scene drifts across a session and the more recent
//! frame is the more informative thing to compare the next one against.
//!
//! # Why the rounds are rebuilt rather than kept
//!
//! [`Tournament::build`] takes the groups and the set of photos already
//! settled, and skips the latter. That one rule buys three behaviours with
//! no extra machinery: switching the tournament off and back on resumes
//! where it stopped; correcting a session boundary in the `f` grid
//! re-groups what is *left* without re-asking about anything already
//! decided; and a photo already in the library never enters a comparison
//! it could not act on the result of.

use std::path::{Path, PathBuf};
use std::sync::mpsc;

// ── The state machine ─────────────────────────────────────────────

/// Which photo (or both) survives one comparison.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// `1` — the incumbent on the left; the challenger is rejected.
    Left,
    /// `2` — the challenger on the right; the incumbent is rejected.
    Right,
    /// `3` — both are worth keeping.
    Both,
}

impl Verdict {
    /// Map the literal key the user pressed. Anything else is not a verdict.
    pub fn from_key(k: i32) -> Option<Self> {
        match k {
            1 => Some(Verdict::Left),
            2 => Some(Verdict::Right),
            3 => Some(Verdict::Both),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Cursor {
    round: usize,
    /// Position of the challenger inside `rounds[round]`.
    next: usize,
    incumbent: usize,
}

/// One decision's worth of rewind information.
#[derive(Clone, Copy, Debug)]
struct Step {
    at: Cursor,
    settled_len: usize,
}

/// A pass over the detected groups, one comparison at a time.
pub struct Tournament {
    /// Each round is one session's still-undecided photos, in capture
    /// order. Only rounds of two or more are kept — there is nothing to
    /// compare a lone photo against.
    rounds: Vec<Vec<usize>>,
    /// `None` once every round is finished.
    at: Option<Cursor>,
    /// Verdicts this card was given before the current pass was built.
    ///
    /// The record has to outlive the pass that made it, or switching the
    /// mode off and on — or correcting one session boundary — would ask
    /// about every photo again. It rides *inside* the tournament rather
    /// than beside it so there is one copy of it and nothing to keep in
    /// step; [`Tournament::carry`] is how it reaches the next build.
    carried: Vec<(usize, bool)>,
    /// Entry index → kept, in the order this pass decided them. A `Vec`
    /// rather than a map because [`Tournament::undo`] rewinds it by
    /// truncation.
    settled: Vec<(usize, bool)>,
    steps: Vec<Step>,
}

impl Tournament {
    /// Build a pass over `groups`, carrying `carried` forward as already
    /// decided and skipping anything `ineligible` rejects outright.
    ///
    /// A group with fewer than two photos left is dropped: "nothing else
    /// belongs with this" is an answer, but it is not a comparison.
    pub fn build(
        groups: &[Vec<usize>],
        carried: Vec<(usize, bool)>,
        ineligible: impl Fn(usize) -> bool,
    ) -> Self {
        let rounds: Vec<Vec<usize>> = groups
            .iter()
            .map(|g| {
                g.iter()
                    .copied()
                    .filter(|&i| !carried.iter().any(|(c, _)| *c == i) && !ineligible(i))
                    .collect::<Vec<_>>()
            })
            .filter(|g: &Vec<usize>| g.len() >= 2)
            .collect();
        let at = Self::open(&rounds, 0);
        Tournament { rounds, at, carried, settled: Vec::new(), steps: Vec::new() }
    }

    /// Every verdict this card has been given, for the next [`build`].
    ///
    /// [`build`]: Tournament::build
    pub fn carry(&self) -> Vec<(usize, bool)> {
        let mut all = self.carried.clone();
        all.extend_from_slice(&self.settled);
        all
    }

    /// `(kept, passed over)` across the whole card, this pass and every
    /// earlier one — which is what the finished panel is reporting on.
    pub fn tally(&self) -> (usize, usize) {
        let kept = self.carried.iter().chain(&self.settled).filter(|(_, k)| *k).count();
        (kept, self.carried.len() + self.settled.len() - kept)
    }

    fn open(rounds: &[Vec<usize>], from: usize) -> Option<Cursor> {
        rounds
            .get(from)
            .map(|r| Cursor { round: from, next: 1, incumbent: r[0] })
    }

    /// The comparison on screen: `(left, right)` as entry indices, or
    /// `None` when the pass is over.
    pub fn pair(&self) -> Option<(usize, usize)> {
        let c = self.at?;
        Some((c.incumbent, self.rounds[c.round][c.next]))
    }

    pub fn finished(&self) -> bool {
        self.at.is_none()
    }

    /// Whether there was ever anything to compare.
    pub fn is_empty(&self) -> bool {
        self.rounds.is_empty()
    }

    /// How many photos this pass has decided, out of how many it covers.
    pub fn progress(&self) -> (usize, usize) {
        (self.settled.len(), self.rounds.iter().map(Vec::len).sum())
    }

    /// Which round is on screen, out of how many. 1-based for display;
    /// `(rounds, rounds)` once finished.
    pub fn round(&self) -> (usize, usize) {
        let n = self.rounds.len();
        (self.at.map(|c| c.round + 1).unwrap_or(n), n)
    }

    /// Whether an [`undo`](Self::undo) would do anything.
    pub fn can_undo(&self) -> bool {
        !self.steps.is_empty()
    }

    /// Record one comparison's verdict and advance.
    ///
    /// Returns the photos whose fate this settled, so the caller can move
    /// exactly those marks rather than re-deriving the whole selection.
    /// Empty when the pass is already over.
    pub fn decide(&mut self, v: Verdict) -> Vec<(usize, bool)> {
        let Some(c) = self.at else { return Vec::new() };
        self.steps.push(Step { at: c, settled_len: self.settled.len() });
        let from = self.settled.len();

        let challenger = self.rounds[c.round][c.next];
        let incumbent = match v {
            Verdict::Left => {
                self.settled.push((challenger, false));
                c.incumbent
            }
            Verdict::Right => {
                self.settled.push((c.incumbent, false));
                challenger
            }
            Verdict::Both => {
                self.settled.push((c.incumbent, true));
                challenger
            }
        };

        let next = c.next + 1;
        self.at = if next < self.rounds[c.round].len() {
            Some(Cursor { round: c.round, next, incumbent })
        } else {
            // The last one standing has nothing left to lose to.
            self.settled.push((incumbent, true));
            Self::open(&self.rounds, c.round + 1)
        };
        self.settled[from..].to_vec()
    }

    /// Take back the last verdict.
    ///
    /// Not a nicety: every keystroke here permanently decides a photo, and
    /// `1` and `2` are one key apart. Without this, a mis-hit silently
    /// costs a photo and there is nothing on screen that would ever show
    /// it. Returns the entries whose marks must be withdrawn.
    pub fn undo(&mut self) -> Vec<usize> {
        let Some(step) = self.steps.pop() else { return Vec::new() };
        let undone: Vec<usize> =
            self.settled.drain(step.settled_len..).map(|(i, _)| i).collect();
        self.at = Some(step.at);
        undone
    }
}

// ── Paired zoom ───────────────────────────────────────────────────

/// Furthest in the paired zoom will go, as a multiple of "fits the pane".
///
/// Past 1:1 on the cached decode there is no more detail to reveal, only
/// bigger pixels — and this is a sharpness judgement, so magnifying past
/// the evidence would actively mislead.
pub const MAX_ZOOM: f32 = 12.0;

/// The source-pixel rectangle a pane shows, as `(x, y, w, h)`.
///
/// At `zoom == 1` this is the whole image, so the tournament opens showing
/// each photo exactly as the single-photo preview would. `cx`/`cy` are the
/// wanted centre in *normalised* source coordinates, which is what makes
/// the zoom paired: two photos of different sizes, or one portrait and one
/// landscape, still land on the same relative part of the frame.
pub fn crop_for(
    src_w: u32,
    src_h: u32,
    view_w: u32,
    view_h: u32,
    zoom: f32,
    cx: f32,
    cy: f32,
) -> (u32, u32, u32, u32) {
    let (src_w, src_h) = (src_w.max(1), src_h.max(1));
    let (view_w, view_h) = (view_w.max(1), view_h.max(1));
    // The scale `image-fit: contain` would use, then the zoom on top of it.
    let fit = (view_w as f32 / src_w as f32).min(view_h as f32 / src_h as f32);
    let scale = fit * zoom.clamp(1.0, MAX_ZOOM);

    let rw = ((view_w as f32 / scale).round() as u32).clamp(1, src_w);
    let rh = ((view_h as f32 / scale).round() as u32).clamp(1, src_h);

    let x = place(cx * src_w as f32, rw, src_w);
    let y = place(cy * src_h as f32, rh, src_h);
    (x, y, rw, rh)
}

/// Left edge of a window of `len` centred on `centre`, kept inside `span`.
fn place(centre: f32, len: u32, span: u32) -> u32 {
    let max = span.saturating_sub(len) as f32;
    (centre - len as f32 / 2.0).clamp(0.0, max).round() as u32
}

/// Pull `(cx, cy)` back to what [`crop_for`] can actually show.
///
/// Panning at the edge would otherwise bank up: drag ten times past the
/// right edge and it takes ten drags back before the picture moves. Feeding
/// the clamped centre back is what stops that, and doing it against **one**
/// of the two images (the left one) rather than both is what stops the two
/// panes fighting over whose edge wins.
pub fn clamp_center(
    src_w: u32,
    src_h: u32,
    view_w: u32,
    view_h: u32,
    zoom: f32,
    cx: f32,
    cy: f32,
) -> (f32, f32) {
    let (x, y, rw, rh) = crop_for(src_w, src_h, view_w, view_h, zoom, cx, cy);
    (
        (x as f32 + rw as f32 / 2.0) / src_w.max(1) as f32,
        (y as f32 + rh as f32 / 2.0) / src_h.max(1) as f32,
    )
}

/// The centre that keeps the source point under `(fx, fy)` in place while
/// the zoom changes from `zoom` to `new_zoom`.
///
/// Zooming about the centre of the pane is the wrong default here: the
/// thing being checked is usually a face, and it is rarely in the middle.
/// `fx`/`fy` are fractions of the pane, so the caller hands over where the
/// pointer was and the detail under it stays under it.
///
/// The pane's letterboxing is ignored, which matters only at `zoom == 1`
/// where there is nothing to pan anyway — one notch in and the crop matches
/// the pane's aspect, so the approximation stops being one.
#[allow(clippy::too_many_arguments)]
pub fn zoom_at(
    src_w: u32,
    src_h: u32,
    view_w: u32,
    view_h: u32,
    zoom: f32,
    cx: f32,
    cy: f32,
    new_zoom: f32,
    fx: f32,
    fy: f32,
) -> (f32, f32) {
    let (x, y, rw, rh) = crop_for(src_w, src_h, view_w, view_h, zoom, cx, cy);
    let (fx, fy) = (fx.clamp(0.0, 1.0), fy.clamp(0.0, 1.0));
    // The source pixel the pointer is over, before anything moves.
    let px = x as f32 + fx * rw as f32;
    let py = y as f32 + fy * rh as f32;

    let (_, _, nw, nh) = crop_for(src_w, src_h, view_w, view_h, new_zoom, cx, cy);
    let want_x = px + nw as f32 * (0.5 - fx);
    let want_y = py + nh as f32 * (0.5 - fy);
    clamp_center(
        src_w,
        src_h,
        view_w,
        view_h,
        new_zoom,
        want_x / src_w.max(1) as f32,
        want_y / src_h.max(1) as f32,
    )
}

// ── Rendering the pair ────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Side {
    Left,
    Right,
}

/// What a pane wants to see. Carries the whole view state rather than a
/// delta, so a request that arrives out of order still describes something
/// coherent.
#[derive(Clone, Debug)]
pub struct PaneRequest {
    pub side: Side,
    /// Bumped on every state change; a reply carrying a stale token is
    /// dropped rather than painted over a newer one.
    pub token: u64,
    pub path: PathBuf,
    pub view_w: u32,
    pub view_h: u32,
    pub zoom: f32,
    pub cx: f32,
    pub cy: f32,
}

pub enum PaneMsg {
    Ready {
        side: Side,
        token: u64,
        rgb: Vec<u8>,
        w: u32,
        h: u32,
        /// Dimensions of the decode the crop came out of — the pan clamp
        /// needs them, and only the worker knows them.
        src_w: u32,
        src_h: u32,
    },
    Failed {
        side: Side,
        token: u64,
    },
}

/// Longest edge of the decode the tournament keeps in memory per photo.
///
/// The point of zooming here is to see whether the eyes are sharp, so a
/// 256 px canonical preview is useless — this has to come off the original.
/// A full 26 MP frame is ~78 MB of RGB and two of them are held at once, so
/// the decode is capped: 4096 px is ~34 MB a side and still more resolution
/// than a comparison pane can show at any zoom worth using.
const MAX_SOURCE_PX: u32 = 4096;

struct Decoded {
    path: PathBuf,
    rgb: Vec<u8>,
    w: u32,
    h: u32,
}

/// Renders both panes on one worker thread, keeping the decodes alive
/// between requests.
///
/// One thread, not two, and a cache of two decodes: the incumbent carries
/// forward to the next comparison, so a verdict costs exactly one new
/// decode. Requests **coalesce** — a drag produces them faster than they
/// can be served, and only the newest state for each side is worth
/// rendering, so the queue is drained to its last entry per side before any
/// work starts. That is the same trade
/// [`crate::import_previews`] makes for the filmstrip: re-prioritise by
/// overwriting, never by cancelling.
pub struct PairRenderer {
    tx: mpsc::Sender<PaneRequest>,
}

impl PairRenderer {
    pub fn spawn(out: mpsc::Sender<PaneMsg>) -> Self {
        let (tx, rx) = mpsc::channel::<PaneRequest>();
        std::thread::spawn(move || run(rx, out));
        PairRenderer { tx }
    }

    /// Ask for a pane. Dropping the result is deliberate: the window can
    /// close while a render is in flight and there is nothing to report.
    pub fn request(&self, req: PaneRequest) {
        let _ = self.tx.send(req);
    }
}

fn run(rx: mpsc::Receiver<PaneRequest>, out: mpsc::Sender<PaneMsg>) {
    let mut cache: Vec<Decoded> = Vec::new();
    while let Ok(first) = rx.recv() {
        // Keep only the newest request per side — everything older
        // describes a viewport the user has already scrolled past.
        let mut pending: Vec<PaneRequest> = vec![first];
        while let Ok(more) = rx.try_recv() {
            match pending.iter().position(|p| p.side == more.side) {
                Some(i) => pending[i] = more,
                None => pending.push(more),
            }
        }
        for req in pending {
            let msg = render(&mut cache, &req);
            if out.send(msg).is_err() {
                return;
            }
        }
    }
}

fn render(cache: &mut Vec<Decoded>, req: &PaneRequest) -> PaneMsg {
    let Some(src) = decoded(cache, &req.path) else {
        return PaneMsg::Failed { side: req.side, token: req.token };
    };
    let region = crop_for(src.w, src.h, req.view_w, req.view_h, req.zoom, req.cx, req.cy);
    match maple_import::preview::render_region(
        &src.rgb, src.w, src.h, region, req.view_w.max(1), req.view_h.max(1),
    ) {
        Ok((rgb, w, h)) => PaneMsg::Ready {
            side: req.side,
            token: req.token,
            rgb,
            w,
            h,
            src_w: src.w,
            src_h: src.h,
        },
        Err(e) => {
            tracing::warn!(target: "maple::import::tournament", "render {:?}: {e}", req.path);
            PaneMsg::Failed { side: req.side, token: req.token }
        }
    }
}

/// The decode for `path`, from the cache or freshly made.
///
/// Two entries: the pair on screen. The incumbent's decode survives a
/// verdict because it is still one of the two most recently asked for.
fn decoded<'a>(cache: &'a mut Vec<Decoded>, path: &Path) -> Option<&'a Decoded> {
    if let Some(i) = cache.iter().position(|d| d.path == path) {
        // Most recently used last, so the truncation below drops the other.
        let d = cache.remove(i);
        cache.push(d);
        return cache.last();
    }
    let (rgb, w, h) = match maple_import::preview::render_to_rgb(path, MAX_SOURCE_PX) {
        Ok(v) => v,
        Err(e) => {
            tracing::warn!(target: "maple::import::tournament", "decode {path:?}: {e}");
            return None;
        }
    };
    cache.push(Decoded { path: path.to_path_buf(), rgb, w, h });
    while cache.len() > 2 {
        cache.remove(0);
    }
    cache.last()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn groups(gs: &[&[usize]]) -> Vec<Vec<usize>> {
        gs.iter().map(|g| g.to_vec()).collect()
    }

    fn kept(t: &Tournament) -> Vec<usize> {
        t.carry().iter().filter(|(_, k)| *k).map(|(i, _)| *i).collect()
    }

    fn rejected(t: &Tournament) -> Vec<usize> {
        t.carry().iter().filter(|(_, k)| !*k).map(|(i, _)| *i).collect()
    }

    #[test]
    fn a_session_of_n_costs_n_minus_one_keystrokes_and_decides_all_n() {
        let mut t = Tournament::build(&groups(&[&[0, 1, 2, 3, 4]]), Vec::new(), |_| false);
        let mut presses = 0;
        while t.pair().is_some() {
            t.decide(Verdict::Left);
            presses += 1;
        }
        assert_eq!(presses, 4);
        assert_eq!(t.progress(), (5, 5));
        // Holding the incumbent every time keeps the first photo only.
        assert_eq!(kept(&t), vec![0]);
        assert_eq!(rejected(&t), vec![1, 2, 3, 4]);
    }

    #[test]
    fn the_right_hand_photo_takes_over_as_incumbent() {
        let mut t = Tournament::build(&groups(&[&[0, 1, 2]]), Vec::new(), |_| false);
        assert_eq!(t.pair(), Some((0, 1)));
        t.decide(Verdict::Right);
        // 0 is out and 1 is now the one being defended.
        assert_eq!(t.pair(), Some((1, 2)));
        t.decide(Verdict::Right);
        assert_eq!(kept(&t), vec![2]);
        assert_eq!(rejected(&t), vec![0, 1]);
    }

    #[test]
    fn taking_both_banks_the_incumbent_and_moves_on() {
        let mut t = Tournament::build(&groups(&[&[0, 1, 2]]), Vec::new(), |_| false);
        t.decide(Verdict::Both);
        assert_eq!(t.pair(), Some((1, 2)));
        t.decide(Verdict::Both);
        assert_eq!(kept(&t), vec![0, 1, 2]);
        assert!(rejected(&t).is_empty());
    }

    /// The invariant the whole design rests on: nothing behind the
    /// incumbent is ever asked about twice, and nothing is left undecided.
    #[test]
    fn every_photo_in_a_round_ends_up_decided_exactly_once() {
        let script = [Verdict::Both, Verdict::Left, Verdict::Right, Verdict::Both];
        let mut t = Tournament::build(&groups(&[&[10, 11, 12, 13, 14]]), Vec::new(), |_| false);
        for v in script {
            t.decide(v);
        }
        let mut seen: Vec<usize> = t.carry().iter().map(|(i, _)| *i).collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![10, 11, 12, 13, 14]);
        assert!(t.finished());
    }

    #[test]
    fn rounds_run_one_after_another() {
        let mut t = Tournament::build(&groups(&[&[0, 1], &[5, 6]]), Vec::new(), |_| false);
        assert_eq!(t.round(), (1, 2));
        assert_eq!(t.pair(), Some((0, 1)));
        t.decide(Verdict::Left);
        assert_eq!(t.round(), (2, 2));
        assert_eq!(t.pair(), Some((5, 6)));
        t.decide(Verdict::Left);
        assert!(t.finished());
        assert_eq!(t.pair(), None);
    }

    #[test]
    fn a_group_left_with_fewer_than_two_undecided_photos_is_not_a_round() {
        // Verdicts carried in from an earlier pass leave this group with
        // one photo — nothing to compare it against.
        let carried = vec![(0usize, true), (1usize, false)];
        let t = Tournament::build(&groups(&[&[0, 1, 2], &[7, 8]]), carried, |_| false);
        assert_eq!(t.pair(), Some((7, 8)));
        // Progress is about *this* pass; the tally is about the card.
        assert_eq!(t.progress(), (0, 2));
        assert_eq!(t.tally(), (1, 1));
    }

    /// The rule that makes rebuilding cheap: a rebuild carries the
    /// verdicts and re-asks nothing. Switching the mode off and on, or
    /// correcting one boundary, must not cost the pass its progress.
    #[test]
    fn rebuilding_resumes_rather_than_restarting() {
        let gs = groups(&[&[0, 1, 2, 3]]);
        let mut t = Tournament::build(&gs, Vec::new(), |_| false);
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), Some((0, 2)));

        let again = Tournament::build(&gs, t.carry(), |_| false);
        // 1 is decided and stays decided; the pass picks up at 2.
        assert_eq!(again.pair(), Some((0, 2)));
        assert_eq!(again.progress(), (0, 3));
        assert_eq!(again.tally(), (0, 1));
    }

    #[test]
    fn an_ineligible_photo_never_enters_a_comparison() {
        // Already in the library, or it never decoded — either way there
        // is no answer the user could act on.
        let t = Tournament::build(&groups(&[&[0, 1, 2]]), Vec::new(), |i| i == 1);
        assert_eq!(t.pair(), Some((0, 2)));
        assert_eq!(t.progress(), (0, 2));
    }

    #[test]
    fn a_pass_with_nothing_to_compare_is_finished_from_the_start() {
        let t = Tournament::build(&groups(&[&[0, 1]]), Vec::new(), |_| true);
        assert!(t.is_empty());
        assert!(t.finished());
        assert_eq!(t.pair(), None);
    }

    #[test]
    fn deciding_after_the_end_does_nothing() {
        let mut t = Tournament::build(&groups(&[&[0, 1]]), Vec::new(), |_| false);
        t.decide(Verdict::Left);
        let before = t.carry();
        assert!(t.decide(Verdict::Right).is_empty());
        assert_eq!(t.carry(), before);
    }

    #[test]
    fn undo_puts_back_exactly_the_photos_the_last_verdict_settled() {
        let mut t = Tournament::build(&groups(&[&[0, 1, 2]]), Vec::new(), |_| false);
        t.decide(Verdict::Both);
        // The second verdict ends the round, so it settles *two* photos —
        // the challenger and the last one standing. Undo must return both.
        let ended = t.decide(Verdict::Left);
        assert_eq!(ended.len(), 2);
        assert!(t.finished());

        let back = t.undo();
        assert_eq!(back.len(), 2);
        assert!(!t.finished());
        assert_eq!(t.pair(), Some((1, 2)));
        assert_eq!(kept(&t), vec![0]);
    }

    #[test]
    fn undo_walks_all_the_way_back_to_the_start() {
        let mut t = Tournament::build(&groups(&[&[0, 1, 2], &[8, 9]]), Vec::new(), |_| false);
        for v in [Verdict::Right, Verdict::Both, Verdict::Left] {
            t.decide(v);
        }
        assert!(t.finished());
        while t.can_undo() {
            t.undo();
        }
        assert_eq!(t.pair(), Some((0, 1)));
        assert!(t.carry().is_empty());
        assert_eq!(t.progress(), (0, 5));
    }

    // ── The renderer, end to end ──────────────────────────────────

    fn write_photo(dir: &std::path::Path, name: &str, w: u32, h: u32) -> PathBuf {
        // A hard checkerboard, so a crop of the wrong region or a render
        // that quietly returned a flat frame would be visible in the pixels.
        let img = image::RgbImage::from_fn(w, h, |x, y| {
            let c = if (x / 16 + y / 16) % 2 == 0 { 255 } else { 0 };
            image::Rgb([c, (x * 255 / w) as u8, (y * 255 / h) as u8])
        });
        let path = dir.join(name);
        img.save(&path).unwrap();
        path
    }

    fn drain(rx: &mpsc::Receiver<PaneMsg>, n: usize) -> Vec<PaneMsg> {
        let mut out = Vec::new();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while out.len() < n && std::time::Instant::now() < deadline {
            if let Ok(m) = rx.recv_timeout(std::time::Duration::from_millis(200)) {
                out.push(m);
            }
        }
        out
    }

    /// The whole point of the zoom: what comes back is a crop of the
    /// *original*, at the pane's own resolution, not a magnified preview.
    #[test]
    fn a_zoomed_pane_comes_back_at_pane_resolution_from_the_original() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_photo(dir.path(), "a.png", 1200, 900);
        let (tx, rx) = mpsc::channel();
        let r = PairRenderer::spawn(tx);

        r.request(PaneRequest {
            side: Side::Left,
            token: 1,
            path: path.clone(),
            view_w: 400,
            view_h: 300,
            zoom: 3.0,
            cx: 0.5,
            cy: 0.5,
        });
        let msgs = drain(&rx, 1);
        match msgs.first() {
            Some(PaneMsg::Ready { side, token, w, h, src_w, src_h, rgb }) => {
                assert_eq!(*side, Side::Left);
                assert_eq!(*token, 1);
                assert_eq!((*src_w, *src_h), (1200, 900));
                // Filled the pane exactly — a 400x300 window into the
                // photo, upscaled by nothing.
                assert_eq!((*w, *h), (400, 300));
                assert_eq!(rgb.len(), 400 * 300 * 3);
            }
            other => panic!("expected a rendered pane, got {}", other.is_some()),
        }

        // …and it is genuinely a *crop*. Both zoom levels come back at the
        // pane's size, so dimensions alone would pass even if the zoom
        // were ignored entirely — the pixels are what prove it.
        r.request(PaneRequest {
            side: Side::Left,
            token: 2,
            path,
            view_w: 400,
            view_h: 300,
            zoom: 1.0,
            cx: 0.5,
            cy: 0.5,
        });
        let fit = drain(&rx, 1);
        let (PaneMsg::Ready { rgb: zoomed, .. }, PaneMsg::Ready { rgb: whole, .. }) =
            (&msgs[0], &fit[0])
        else {
            panic!("both renders must succeed");
        };
        assert_eq!(zoomed.len(), whole.len());
        assert_ne!(zoomed, whole, "the zoom showed the same pixels as the fit");
    }

    #[test]
    fn a_file_that_will_not_decode_reports_failure_rather_than_hanging() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("broken.jpg");
        std::fs::write(&path, b"not an image").unwrap();
        let (tx, rx) = mpsc::channel();
        let r = PairRenderer::spawn(tx);
        r.request(PaneRequest {
            side: Side::Right,
            token: 9,
            path,
            view_w: 100,
            view_h: 100,
            zoom: 1.0,
            cx: 0.5,
            cy: 0.5,
        });
        match drain(&rx, 1).first() {
            Some(PaneMsg::Failed { side, token }) => {
                assert_eq!(*side, Side::Right);
                assert_eq!(*token, 9);
            }
            _ => panic!("a broken file must come back as Failed"),
        }
    }

    /// Both panes are served from one thread, and the incumbent's decode
    /// survives the verdict that keeps it — which is what makes a keystroke
    /// cost one decode rather than two.
    #[test]
    fn both_sides_are_served_and_a_repeat_of_one_of_them_still_answers() {
        let dir = tempfile::tempdir().unwrap();
        let a = write_photo(dir.path(), "a.png", 800, 600);
        let b = write_photo(dir.path(), "b.png", 600, 800);
        let (tx, rx) = mpsc::channel();
        let r = PairRenderer::spawn(tx);
        let req = |side, token, path: &PathBuf| PaneRequest {
            side,
            token,
            path: path.clone(),
            view_w: 200,
            view_h: 200,
            zoom: 1.0,
            cx: 0.5,
            cy: 0.5,
        };
        r.request(req(Side::Left, 1, &a));
        r.request(req(Side::Right, 1, &b));
        assert_eq!(drain(&rx, 2).len(), 2);

        // The verdict kept the left photo; only the challenger changed.
        r.request(req(Side::Left, 2, &a));
        r.request(req(Side::Right, 2, &a));
        let second = drain(&rx, 2);
        assert_eq!(second.len(), 2);
        assert!(second.iter().all(|m| matches!(m, PaneMsg::Ready { token: 2, .. })));
    }

    // ── Paired zoom ───────────────────────────────────────────────

    #[test]
    fn at_zoom_one_the_whole_photo_is_shown() {
        assert_eq!(crop_for(4000, 3000, 800, 600, 1.0, 0.5, 0.5), (0, 0, 4000, 3000));
        // A pane the wrong shape for the photo still shows all of it — the
        // letterbox is Slint's `image-fit: contain` problem, not ours.
        assert_eq!(crop_for(4000, 3000, 400, 900, 1.0, 0.5, 0.5), (0, 0, 4000, 3000));
    }

    #[test]
    fn zooming_in_halves_the_region_each_time_it_doubles() {
        let (_, _, w1, h1) = crop_for(4000, 2000, 800, 400, 2.0, 0.5, 0.5);
        let (_, _, w2, h2) = crop_for(4000, 2000, 800, 400, 4.0, 0.5, 0.5);
        assert_eq!((w1, h1), (2000, 1000));
        assert_eq!((w2, h2), (1000, 500));
    }

    #[test]
    fn a_region_is_kept_inside_the_photo() {
        // Asking for the far corner cannot produce a rectangle hanging off
        // the edge — the crop is pushed back in instead.
        let (x, y, w, h) = crop_for(4000, 2000, 800, 400, 4.0, 1.0, 1.0);
        assert_eq!((x + w, y + h), (4000, 2000));
        let (x, y, _, _) = crop_for(4000, 2000, 800, 400, 4.0, 0.0, 0.0);
        assert_eq!((x, y), (0, 0));
    }

    #[test]
    fn zoom_is_clamped_at_both_ends() {
        // Below 1 would mean showing less than the photo; above MAX_ZOOM
        // is magnified pixels, not detail.
        assert_eq!(
            crop_for(4000, 3000, 800, 600, 0.1, 0.5, 0.5),
            crop_for(4000, 3000, 800, 600, 1.0, 0.5, 0.5)
        );
        assert_eq!(
            crop_for(4000, 3000, 800, 600, 1000.0, 0.5, 0.5),
            crop_for(4000, 3000, 800, 600, MAX_ZOOM, 0.5, 0.5)
        );
    }

    #[test]
    fn the_same_normalised_centre_lands_on_the_same_relative_place_in_both() {
        // A landscape and a portrait frame, same zoom and centre: the two
        // crops sit at the same fraction of their own frame, which is what
        // makes the zoom "paired" for photos that are not the same shape.
        let (x1, y1, w1, h1) = crop_for(4000, 3000, 600, 800, 3.0, 0.25, 0.6);
        let (x2, y2, w2, h2) = crop_for(3000, 4000, 600, 800, 3.0, 0.25, 0.6);
        let c1 = ((x1 + w1 / 2) as f32 / 4000.0, (y1 + h1 / 2) as f32 / 3000.0);
        let c2 = ((x2 + w2 / 2) as f32 / 3000.0, (y2 + h2 / 2) as f32 / 4000.0);
        assert!((c1.0 - c2.0).abs() < 0.01, "{c1:?} vs {c2:?}");
        assert!((c1.1 - c2.1).abs() < 0.01, "{c1:?} vs {c2:?}");
    }

    /// Where they *cannot* agree, and why that is fine. A centre near the
    /// edge is reachable in one frame and not the other, so the two crops
    /// come apart — which is exactly why [`clamp_center`] reads one image
    /// rather than reconciling both. The pair stays locked wherever both
    /// can go, and the odd frame stops at its own edge instead of dragging
    /// the other back off the detail being compared.
    #[test]
    fn near_an_edge_the_shorter_frame_simply_stops() {
        let (_, y1, _, h1) = crop_for(4000, 3000, 600, 800, 3.0, 0.25, 0.8);
        assert_eq!(y1 + h1, 3000);
        let (_, y2, _, h2) = crop_for(3000, 4000, 600, 800, 3.0, 0.25, 0.8);
        assert!(y2 + h2 < 4000, "the taller frame still has room to pan");
    }

    #[test]
    fn zooming_about_the_middle_leaves_the_middle_alone() {
        let (cx, cy) = zoom_at(4000, 3000, 800, 600, 1.0, 0.5, 0.5, 4.0, 0.5, 0.5);
        assert!((cx - 0.5).abs() < 0.001, "{cx}");
        assert!((cy - 0.5).abs() < 0.001, "{cy}");
    }

    #[test]
    fn zooming_under_the_pointer_keeps_that_detail_under_the_pointer() {
        // A pane 800x600 over a 4000x3000 photo: at fit, the pane's top-left
        // quarter-point is the photo's quarter-point too.
        let (cx, cy) = zoom_at(4000, 3000, 800, 600, 1.0, 0.5, 0.5, 2.0, 0.25, 0.25);
        let (x, y, w, h) = crop_for(4000, 3000, 800, 600, 2.0, cx, cy);
        // The photo pixel that was under (0.25, 0.25) must still be there.
        let under_x = x as f32 + 0.25 * w as f32;
        let under_y = y as f32 + 0.25 * h as f32;
        assert!((under_x - 1000.0).abs() < 12.0, "{under_x}");
        assert!((under_y - 750.0).abs() < 12.0, "{under_y}");
    }

    #[test]
    fn zooming_back_out_to_fit_recentres_on_its_own() {
        // Nothing to pan at fit, so wherever the user was peering, zooming
        // out lands on the whole frame rather than a remembered corner.
        let (cx, cy) = zoom_at(4000, 3000, 800, 600, 8.0, 0.9, 0.1, 1.0, 0.9, 0.1);
        assert_eq!(crop_for(4000, 3000, 800, 600, 1.0, cx, cy), (0, 0, 4000, 3000));
    }

    #[test]
    fn panning_past_an_edge_does_not_bank_up() {
        // Ten drags off the right edge must not cost ten drags to come back.
        let mut cx = 0.5f32;
        for _ in 0..10 {
            cx = clamp_center(4000, 3000, 800, 600, 4.0, cx + 0.3, 0.5).0;
        }
        let (x, _, w, _) = crop_for(4000, 3000, 800, 600, 4.0, cx, 0.5);
        assert_eq!(x + w, 4000);
        // One drag back must actually move the picture.
        let back = clamp_center(4000, 3000, 800, 600, 4.0, cx - 0.1, 0.5).0;
        let (x2, _, _, _) = crop_for(4000, 3000, 800, 600, 4.0, back, 0.5);
        assert!(x2 < x, "{x2} !< {x}");
    }

    #[test]
    fn a_degenerate_pane_or_photo_is_not_a_panic() {
        assert_eq!(crop_for(0, 0, 0, 0, 1.0, 0.5, 0.5), (0, 0, 1, 1));
        let _ = clamp_center(0, 0, 0, 0, 1.0, f32::NAN, f32::NAN);
    }
}
