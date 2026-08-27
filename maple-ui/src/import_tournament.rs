//! The import tournament: a photo's fate is decided by looking at it beside
//! the one it is competing with.
//!
//! # The shape of it
//!
//! The importer used to auto-mark the sharpest photo of every detected
//! session. That is a guess dressed as a decision: variance-of-Laplacian
//! ranks a badly-framed-but-crisp frame above the one where the child is
//! actually looking at the camera, and once it is marked nobody re-examines
//! it. So the marking is gone and the comparison is put in front of the
//! user — `1` keeps the left, `2` the right, `3` keeps both.
//!
//! Each detected session is its own **bracket**, run to completion before
//! the next one starts:
//!
//! * **The first round pairs up photos nobody has looked at yet** — (1,2),
//!   (3,4), (5,6) … so every photo gets its first comparison against a
//!   peer. Nothing is anyone's "incumbent" and no photo is asked about
//!   twice before the rest have been asked about once.
//! * Losing eliminates. `3` advances **both**, which is what makes "keep
//!   both" mean something: it is not "I cannot decide", it is "these both
//!   go through".
//! * Later rounds are **keeper against keeper**, over whoever is still
//!   standing.
//! * It ends when there is no pair left to ask about — either one photo is
//!   standing, or everyone still standing has already met everyone else
//!   still standing. **Whoever is standing at the end is kept.**
//!
//! # The two rules that make it terminate
//!
//! A pair is only ever put to the user **once** (`Bracket::met`). Building
//! a round therefore either finds an unasked pair — in which case the
//! answer eliminates somebody or records a new meeting, both of which are
//! bounded — or finds none, which ends the bracket. There is no way to
//! loop, and no way to be asked the same question twice.
//!
//! Cost lands where it should. Answer `1`/`2` throughout and a session of
//! *n* takes *n* − 1 comparisons, the theoretical minimum for finding a
//! single best. Say "keep both" often and it costs more, because more
//! photos are still in the running — which is the user asking for it. The
//! ceiling is the complete graph, so a session where *every* answer is `3`
//! would run to n(n−1)/2; `keep_rest` (`k`) is the way out, and is worth
//! having anyway for "this session is fine, move on".
//!
//! # It is a detour, not a mode
//!
//! There is no separate pass over the card and nothing to switch between.
//! The user walks the card the way they always have, and **the view follows
//! the cursor**: land on a photo that belongs to a session and its bracket
//! takes over; land on one that belongs to no session and it is the
//! ordinary one-photo triage, because there is nothing to compare it
//! against. When a bracket runs out of questions the cursor steps past the
//! session and the walk continues.
//!
//! That is why [`Tournament::enter`] exists and why there is no "next
//! bracket": the *cursor* says which bracket is on screen, not a pointer
//! inside here. Making the tournament a mode of its own meant a card was
//! either all comparisons or all single photos, so the photos in no session
//! — a sixth of a real card — were never visited at all.
//!
//! # Why brackets are rebuilt rather than resumed
//!
//! [`Tournament::build`] takes the groups and the verdicts already given
//! (`carry`) and skips the latter. That one rule buys three behaviours with
//! no extra machinery: switching the tournament off and back on resumes
//! where it stopped; correcting a session boundary in the `f` grid
//! re-groups what is *left* without re-asking about anything already
//! decided; and a photo already in the library never enters a comparison
//! it could not act on the result of.


use std::collections::{HashSet, VecDeque};
use std::path::{Path, PathBuf};
use std::sync::mpsc;

// ── The state machine ─────────────────────────────────────────────

/// Which photo (or both) goes through one comparison.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Verdict {
    /// `1` — the left photo; the right is eliminated.
    Left,
    /// `2` — the right photo; the left is eliminated.
    Right,
    /// `3` — both go through to the next round.
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

/// What one photo of the current session is doing, for the strip under the
/// two panes.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum CardState {
    /// On screen on the left.
    Left,
    /// On screen on the right.
    Right,
    /// Still in the running, waiting for a later round.
    In,
    /// Lost a comparison. Not coming back.
    Out,
}

impl CardState {
    /// The number the UI switches on.
    pub fn code(self) -> i32 {
        match self {
            CardState::In => 0,
            CardState::Left => 1,
            CardState::Right => 2,
            CardState::Out => 3,
        }
    }
}

/// An ordered pair key, so `(a, b)` and `(b, a)` are the same meeting.
type Met = (usize, usize);

fn met_key(a: usize, b: usize) -> Met {
    if a <= b {
        (a, b)
    } else {
        (b, a)
    }
}

/// One session's bracket.
#[derive(Clone)]
struct Bracket {
    /// Every photo in the session, in capture order. Never changes — it is
    /// what the strip under the panes draws.
    members: Vec<usize>,
    /// Still in the running, in capture order.
    alive: Vec<usize>,
    /// Every pair already put to the user. A question is asked once.
    met: HashSet<Met>,
    /// This round's remaining comparisons.
    queue: VecDeque<(usize, usize)>,
    /// How many rounds have been built, for the status line.
    round: usize,
}

impl Bracket {
    fn new(members: Vec<usize>) -> Self {
        let mut b = Bracket {
            alive: members.clone(),
            members,
            met: HashSet::new(),
            queue: VecDeque::new(),
            round: 0,
        };
        b.open_round();
        b
    }

    /// Pair the survivors up for another round.
    ///
    /// Greedy and in capture order, which makes the first round exactly
    /// (1,2), (3,4), (5,6) …. A photo that cannot be paired — because it
    /// has already met everyone left unpaired — simply sits the round out;
    /// it is still alive and can be paired next time. An **empty** round is
    /// the bracket's end condition: there is no question left to ask.
    fn open_round(&mut self) {
        self.queue.clear();
        let mut taken = vec![false; self.alive.len()];
        for i in 0..self.alive.len() {
            if taken[i] {
                continue;
            }
            for j in i + 1..self.alive.len() {
                if taken[j] || self.met.contains(&met_key(self.alive[i], self.alive[j])) {
                    continue;
                }
                taken[i] = true;
                taken[j] = true;
                self.queue.push_back((self.alive[i], self.alive[j]));
                break;
            }
        }
        if !self.queue.is_empty() {
            self.round += 1;
        }
    }

    fn pair(&self) -> Option<(usize, usize)> {
        self.queue.front().copied()
    }

    fn eliminate(&mut self, idx: usize) {
        self.alive.retain(|&a| a != idx);
    }
}

/// What one keystroke did.
///
/// `acted` and `settled` are deliberately separate fields, and this is the
/// whole reason the type exists rather than a bare `Vec`. `3` advances both
/// photos and eliminates neither, so it settles **nobody** while very much
/// moving the pass on. A caller that reads an empty `settled` as "nothing
/// happened" advances the bracket and never repaints — which is exactly how
/// `3` came to look dead in the middle of a session and alive at the end of
/// one, where the bracket ends and its survivors settle.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Decision {
    /// There was a comparison on screen, and it has been answered.
    pub acted: bool,
    /// Photos whose fate this settled. **May be empty on a real verdict.**
    pub settled: Vec<(usize, bool)>,
    /// This session has no question left — the cursor should step past it.
    pub session_done: bool,
}

/// What one undo did. Same two fields for the same reason: undoing a `3`
/// puts the question back without withdrawing a single mark.
#[derive(Default, Debug, PartialEq, Eq)]
pub struct Rewind {
    pub acted: bool,
    pub withdrawn: Vec<usize>,
}

/// One decision's worth of rewind information.
///
/// A whole-bracket snapshot rather than an incremental rewind: a bracket is
/// one session — tens of photos — so cloning it is nothing, and a decision
/// touches `alive`, `met`, `queue` and the round counter at once. Anything
/// incremental would be four things to keep in step for no measurable
/// saving.
struct Step {
    at: usize,
    bracket: Bracket,
    settled_len: usize,
}

/// A pass over the detected sessions, one comparison at a time.
pub struct Tournament {
    brackets: Vec<Bracket>,
    /// Entry index → its bracket, for every photo still in one. What makes
    /// the cursor able to say which comparison is on screen.
    of: Vec<(usize, usize)>,
    /// The bracket on screen. `None` whenever the cursor is not standing in
    /// a session that still has a question.
    at: Option<usize>,
    /// Verdicts this card was given before the current pass was built.
    ///
    /// The record has to outlive the pass that made it, or switching the
    /// mode off and on — or correcting one session boundary — would ask
    /// about every photo again. It rides *inside* the tournament rather
    /// than beside it so there is one copy of it and nothing to keep in
    /// step; [`Tournament::carry`] is how it reaches the next build.
    carried: Vec<(usize, bool)>,
    /// Entry index → kept, in the order this pass decided them.
    settled: Vec<(usize, bool)>,
    steps: Vec<Step>,
}

/// The current session, for the strip under the panes.
pub struct GroupView<'a> {
    members: &'a [usize],
    alive: &'a [usize],
    pair: Option<(usize, usize)>,
}

impl GroupView<'_> {
    /// The whole session in capture order — including the photos already
    /// out, because seeing what was rejected is half of why the strip is
    /// there.
    pub fn members(&self) -> &[usize] {
        self.members
    }

    pub fn state(&self, idx: usize) -> CardState {
        match self.pair {
            Some((l, _)) if l == idx => return CardState::Left,
            Some((_, r)) if r == idx => return CardState::Right,
            _ => {}
        }
        if self.alive.contains(&idx) {
            CardState::In
        } else {
            CardState::Out
        }
    }

    /// How many are still in the running.
    pub fn alive(&self) -> usize {
        self.alive.len()
    }
}

impl Tournament {
    /// Build a pass over `groups`, carrying `carried` forward as already
    /// decided and skipping anything `ineligible` rejects outright.
    ///
    /// A group with fewer than two photos left is dropped: "nothing else
    /// belongs with this" is an answer, but it is not a comparison, and a
    /// photo with nothing to compare against belongs in the ordinary
    /// one-photo triage.
    pub fn build(
        groups: &[Vec<usize>],
        carried: Vec<(usize, bool)>,
        ineligible: impl Fn(usize) -> bool,
    ) -> Self {
        let brackets: Vec<Bracket> = groups
            .iter()
            .map(|g| {
                g.iter()
                    .copied()
                    .filter(|&i| !carried.iter().any(|(c, _)| *c == i) && !ineligible(i))
                    .collect::<Vec<_>>()
            })
            .filter(|g: &Vec<usize>| g.len() >= 2)
            .map(Bracket::new)
            .collect();
        // Every photo of a bracket points at it, decided ones included —
        // the cursor lands on *photos*, and a photo that has already lost
        // still belongs to the session whose bracket is running.
        let mut of: Vec<(usize, usize)> = brackets
            .iter()
            .enumerate()
            .flat_map(|(b, br)| br.members.iter().map(move |&m| (m, b)))
            .collect();
        of.sort_unstable();
        Tournament { brackets, of, at: None, carried, settled: Vec::new(), steps: Vec::new() }
    }

    /// Put the bracket containing `entry` on screen, if there is one with a
    /// question left.
    ///
    /// The one entry point: the cursor moved, so what should be shown?
    /// Returns whether a comparison is now on screen — `false` means the
    /// caller should show the ordinary one-photo view, either because this
    /// photo is in no session or because its session is already settled.
    pub fn enter(&mut self, entry: usize) -> bool {
        self.at = self
            .of
            .binary_search_by_key(&entry, |(e, _)| *e)
            .ok()
            .map(|i| self.of[i].1)
            .filter(|&b| self.brackets[b].pair().is_some());
        self.at.is_some()
    }

    /// Stop showing a comparison — the cursor has left, or the feature was
    /// switched off.
    pub fn leave(&mut self) {
        self.at = None;
    }

    /// Settle whoever is standing in the bracket on screen and take it off.
    fn close(&mut self) -> bool {
        let Some(at) = self.at else { return false };
        if self.brackets[at].pair().is_some() {
            return false;
        }
        for &idx in &self.brackets[at].alive.clone() {
            self.settled.push((idx, true));
        }
        self.at = None;
        true
    }

    /// Every verdict this card has been given, for the next [`build`].
    ///
    /// [`build`]: Tournament::build
    pub fn carry(&self) -> Vec<(usize, bool)> {
        let mut all = self.carried.clone();
        all.extend_from_slice(&self.settled);
        all
    }

    /// The comparison on screen: `(left, right)` as entry indices, or
    /// `None` when the cursor is not in a live session.
    pub fn pair(&self) -> Option<(usize, usize)> {
        self.brackets.get(self.at?)?.pair()
    }

    /// The session on screen, for the strip under the panes.
    pub fn group(&self) -> Option<GroupView<'_>> {
        let b = self.brackets.get(self.at?)?;
        Some(GroupView { members: &b.members, alive: &b.alive, pair: b.pair() })
    }

    /// How many photos this pass has decided, out of how many it covers.
    pub fn progress(&self) -> (usize, usize) {
        (self.settled.len(), self.brackets.iter().map(|b| b.members.len()).sum())
    }

    /// Which session is on screen, out of how many. 1-based for display;
    /// `(0, n)` when the cursor is not in one.
    pub fn round(&self) -> (usize, usize) {
        (self.at.map(|a| a + 1).unwrap_or(0), self.brackets.len())
    }

    /// Which round of the session on screen, 1-based. 0 when not in one.
    pub fn session_round(&self) -> usize {
        self.at.and_then(|a| self.brackets.get(a)).map(|b| b.round).unwrap_or(0)
    }

    /// Whether an [`undo`](Self::undo) would do anything.
    pub fn can_undo(&self) -> bool {
        !self.steps.is_empty()
    }

    fn snapshot(&mut self) -> bool {
        let Some(at) = self.at else { return false };
        let Some(bracket) = self.brackets.get(at).cloned() else { return false };
        self.steps.push(Step { at, bracket, settled_len: self.settled.len() });
        true
    }

    /// Record one comparison's verdict and advance.
    ///
    /// Returns the photos whose fate this settled, so the caller can move
    /// exactly those marks rather than re-deriving the whole selection. A
    /// loser is settled at once and is not coming back; a survivor is only
    /// settled when its bracket runs out of questions, because until then
    /// it can still lose one. Empty when the pass is already over.
    pub fn decide(&mut self, v: Verdict) -> Decision {
        if !self.snapshot() {
            return Decision::default();
        }
        let from = self.settled.len();
        let at = self.at.expect("snapshot only succeeds inside a bracket");
        let b = &mut self.brackets[at];
        let Some((left, right)) = b.queue.pop_front() else { return Decision::default() };
        b.met.insert(met_key(left, right));
        match v {
            Verdict::Left => {
                b.eliminate(right);
                self.settled.push((right, false));
            }
            Verdict::Right => {
                b.eliminate(left);
                self.settled.push((left, false));
            }
            Verdict::Both => {}
        }
        if self.brackets[at].queue.is_empty() {
            self.brackets[at].open_round();
        }
        let session_done = self.close();
        Decision { acted: true, settled: self.settled[from..].to_vec(), session_done }
    }

    /// End the current session here, keeping everyone still standing.
    ///
    /// The pressure valve for the one case the termination rule cannot
    /// bound on its own: answer `3` to everything and a session runs to the
    /// complete graph. It is also just the right thing to have — "these are
    /// all fine, move on" is a real answer, and grinding through the
    /// remaining rounds to say it would be theatre.
    pub fn keep_rest(&mut self) -> Decision {
        if !self.snapshot() {
            return Decision::default();
        }
        let from = self.settled.len();
        let at = self.at.expect("snapshot only succeeds inside a bracket");
        self.brackets[at].queue.clear();
        let session_done = self.close();
        Decision { acted: true, settled: self.settled[from..].to_vec(), session_done }
    }

    /// Take back the last verdict.
    ///
    /// Not a nicety: every keystroke here permanently eliminates a photo,
    /// and `1` and `2` are one key apart. Without this, a mis-hit silently
    /// costs a photo and there is nothing on screen that would ever show
    /// it. Returns the entries whose marks must be withdrawn.
    pub fn undo(&mut self) -> Rewind {
        let Some(step) = self.steps.pop() else { return Rewind::default() };
        let withdrawn: Vec<usize> =
            self.settled.drain(step.settled_len..).map(|(i, _)| i).collect();
        self.brackets[step.at] = step.bracket;
        self.at = Some(step.at);
        Rewind { acted: true, withdrawn }
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

    /// Build a pass and stand the cursor on the first session, which is
    /// what the UI does the moment the user walks into one.
    fn build(gs: &[&[usize]]) -> Tournament {
        let mut t = Tournament::build(&groups(gs), Vec::new(), |_| false);
        t.enter(gs[0][0]);
        t
    }

    fn kept(t: &Tournament) -> Vec<usize> {
        let mut v: Vec<usize> =
            t.carry().iter().filter(|(_, k)| *k).map(|(i, _)| *i).collect();
        v.sort_unstable();
        v
    }

    fn out(t: &Tournament) -> Vec<usize> {
        let mut v: Vec<usize> =
            t.carry().iter().filter(|(_, k)| !*k).map(|(i, _)| *i).collect();
        v.sort_unstable();
        v
    }

    /// Walk the card the way the user does: stand on each session in turn,
    /// answer until it runs out of questions, move on.
    fn walk(
        t: &mut Tournament,
        gs: &[&[usize]],
        mut f: impl FnMut(usize, usize) -> Verdict,
    ) -> usize {
        let mut asked = 0;
        for g in gs {
            t.enter(g[0]);
            while let Some((l, r)) = t.pair() {
                t.decide(f(l, r));
                asked += 1;
                assert!(asked < 5000, "the bracket did not terminate");
            }
        }
        asked
    }

    // ── The cursor decides what is on screen ──────────────────────

    /// The whole shape of the feature: it is a detour, not a mode. Land on
    /// a photo in a session and there is a comparison; land on one in no
    /// session and there is not, so the caller shows the single view.
    #[test]
    fn a_photo_in_no_session_has_no_comparison() {
        let mut t = Tournament::build(&groups(&[&[3, 4, 5]]), Vec::new(), |_| false);
        assert!(!t.enter(0), "photo 0 is in no session");
        assert_eq!(t.pair(), None);
        assert!(t.enter(4), "photo 4 is");
        assert_eq!(t.pair(), Some((3, 4)));
    }

    /// Any member, not just the first — the user arrives wherever the
    /// arrow keys or a filmstrip click put them.
    #[test]
    fn entering_on_any_member_puts_that_session_on_screen() {
        let gs = groups(&[&[0, 1, 2], &[7, 8, 9]]);
        for &entry in &[7usize, 8, 9] {
            let mut t = Tournament::build(&gs, Vec::new(), |_| false);
            assert!(t.enter(entry));
            assert_eq!(t.pair(), Some((7, 8)), "entered on {entry}");
        }
    }

    /// A session whose photos are all already decided is not a detour any
    /// more — walking back over it gives the ordinary single view.
    #[test]
    fn a_session_with_nothing_left_to_ask_is_not_entered() {
        let mut t = build(&[&[0, 1]]);
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), None, "the session closed itself");
        assert!(!t.enter(0));
        assert!(!t.enter(1));
        assert!(t.group().is_none());
    }

    #[test]
    fn leaving_takes_the_comparison_off_screen_without_deciding_anything() {
        let mut t = build(&[&[0, 1, 2]]);
        assert!(t.pair().is_some());
        t.leave();
        assert_eq!(t.pair(), None);
        assert!(t.carry().is_empty(), "skipping a session decides nothing");
        // …and it is still waiting when the user comes back.
        assert!(t.enter(1));
        assert_eq!(t.pair(), Some((0, 1)));
    }

    // ── The bracket ───────────────────────────────────────────────

    /// Round one pairs photos that have not been looked at yet, so every
    /// photo gets its first comparison against a peer instead of photo 0
    /// appearing in every question.
    #[test]
    fn the_first_round_pairs_photos_nobody_has_seen_yet() {
        let mut t = build(&[&[0, 1, 2, 3, 4, 5]]);
        assert_eq!(t.pair(), Some((0, 1)));
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), Some((2, 3)));
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), Some((4, 5)));
    }

    /// Answer `1`/`2` throughout and it costs the theoretical minimum for
    /// finding a single best: one comparison per elimination.
    #[test]
    fn eliminating_every_time_costs_n_minus_one_and_leaves_one_photo() {
        let gs: &[&[usize]] = &[&[0, 1, 2, 3, 4, 5, 6, 7]];
        let mut t = Tournament::build(&groups(gs), Vec::new(), |_| false);
        let asked = walk(&mut t, gs, |_, _| Verdict::Left);
        assert_eq!(asked, 7);
        assert_eq!(kept(&t), vec![0]);
        assert_eq!(out(&t), vec![1, 2, 3, 4, 5, 6, 7]);
    }

    /// The keeper-against-keeper part: a photo that wins its first round
    /// meets another winner, not the photo it already beat.
    #[test]
    fn later_rounds_are_keeper_against_keeper() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        t.decide(Verdict::Right); // 1 beats 0
        assert_eq!(t.session_round(), 1);
        t.decide(Verdict::Right); // 3 beats 2
        assert_eq!(t.pair(), Some((1, 3)));
        assert_eq!(t.session_round(), 2);
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), None);
        assert_eq!(kept(&t), vec![1]);
    }

    /// "Keep both" advances both — it is not "I cannot decide". A photo it
    /// saves is still in the running and can still lose later.
    #[test]
    fn keeping_both_advances_both_and_they_can_still_lose() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        t.decide(Verdict::Both); // 0 and 1 both go through
        t.decide(Verdict::Left); // 2 beats 3
        assert_eq!(t.pair(), Some((0, 2)));
        t.decide(Verdict::Right); // 2 beats 0 — 0 is out despite the "both"
        assert_eq!(t.pair(), Some((1, 2)));
        t.decide(Verdict::Left);
        assert_eq!(kept(&t), vec![1]);
        assert_eq!(out(&t), vec![0, 2, 3]);
    }

    /// The trap that made `3` look dead in the middle of a session: a
    /// verdict that settles nobody is **not** a verdict that did nothing.
    /// Anything gating a repaint on the settled list being non-empty
    /// advances the bracket behind the user's back.
    #[test]
    fn keeping_both_settles_nobody_but_still_moves_the_question_on() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        let before = t.pair();
        let d = t.decide(Verdict::Both);
        assert!(d.acted, "a `3` is a real verdict");
        assert!(d.settled.is_empty(), "nobody is out yet");
        assert!(!d.session_done);
        assert_ne!(t.pair(), before, "but the comparison must still advance");
    }

    /// And the same on the way back: undoing a `3` withdraws no marks
    /// while very much putting the question back.
    #[test]
    fn undoing_a_keep_both_restores_the_question_it_withdraws_no_marks_for() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        let before = t.pair();
        t.decide(Verdict::Both);
        let r = t.undo();
        assert!(r.acted, "the undo really happened");
        assert!(r.withdrawn.is_empty(), "a `3` marked nobody, so there is nobody to unmark");
        assert_eq!(t.pair(), before, "but the question has to come back");
        assert!(!t.can_undo());
    }

    /// The end condition, stated as the user did: everyone still standing
    /// has met everyone else still standing, so there is nothing left to
    /// ask and they are all kept.
    #[test]
    fn a_session_where_every_answer_is_keep_both_keeps_everything() {
        let gs: &[&[usize]] = &[&[0, 1, 2, 3]];
        let mut t = Tournament::build(&groups(gs), Vec::new(), |_| false);
        let asked = walk(&mut t, gs, |_, _| Verdict::Both);
        // The documented ceiling: the complete graph, once each.
        assert_eq!(asked, 4 * 3 / 2);
        assert_eq!(kept(&t), vec![0, 1, 2, 3]);
        assert!(out(&t).is_empty());
    }

    /// The rule the whole thing rests on. A repeated question would be
    /// both an insult and a way to loop forever.
    #[test]
    fn no_pair_is_ever_put_to_the_user_twice() {
        for script in [
            [Verdict::Both, Verdict::Left, Verdict::Both, Verdict::Right],
            [Verdict::Left, Verdict::Both, Verdict::Right, Verdict::Both],
            [Verdict::Both, Verdict::Both, Verdict::Both, Verdict::Both],
        ] {
            let mut t = build(&[&[0, 1, 2, 3, 4, 5]]);
            let mut seen: HashSet<Met> = HashSet::new();
            let mut i = 0;
            while let Some((l, r)) = t.pair() {
                assert!(seen.insert(met_key(l, r)), "asked ({l},{r}) twice");
                t.decide(script[i % script.len()]);
                i += 1;
                assert!(i < 200, "did not terminate");
            }
        }
    }

    #[test]
    fn every_photo_of_a_session_ends_up_kept_or_out_and_never_both() {
        let gs: &[&[usize]] = &[&[0, 1, 2, 3, 4], &[10, 11, 12]];
        let mut t = Tournament::build(&groups(gs), Vec::new(), |_| false);
        let script = [Verdict::Both, Verdict::Left, Verdict::Right];
        let mut i = 0;
        walk(&mut t, gs, |_, _| {
            i += 1;
            script[i % script.len()]
        });
        let mut all = kept(&t);
        all.extend(out(&t));
        all.sort_unstable();
        assert_eq!(all, vec![0, 1, 2, 3, 4, 10, 11, 12]);
    }

    /// A verdict that empties a session says so, because that is the
    /// caller's cue to walk the cursor past it.
    #[test]
    fn the_verdict_that_empties_a_session_says_so() {
        let mut t = build(&[&[0, 1, 2]]);
        assert!(!t.decide(Verdict::Left).session_done);
        let last = t.decide(Verdict::Left);
        assert!(last.session_done);
        assert_eq!(t.pair(), None);
    }

    #[test]
    fn keep_rest_ends_the_session_and_keeps_whoever_is_standing() {
        let mut t = build(&[&[0, 1, 2, 3], &[8, 9]]);
        t.decide(Verdict::Left); // 1 is out
        let d = t.keep_rest();
        assert!(d.session_done);
        assert_eq!(d.settled, vec![(0, true), (2, true), (3, true)]);
        assert_eq!(out(&t), vec![1]);
        // …and the next session is there when the cursor reaches it.
        assert!(t.enter(8));
        assert_eq!(t.pair(), Some((8, 9)));
    }

    #[test]
    fn a_group_left_with_fewer_than_two_undecided_photos_is_not_a_bracket() {
        let carried = vec![(0usize, true), (1usize, false)];
        let mut t = Tournament::build(&groups(&[&[0, 1, 2], &[7, 8]]), carried, |_| false);
        assert!(!t.enter(2), "one photo left is not a comparison");
        assert!(t.enter(7));
        assert_eq!(t.pair(), Some((7, 8)));
        // Progress is about *this* pass, not the verdicts carried in.
        assert_eq!(t.progress(), (0, 2));
    }

    #[test]
    fn a_pass_with_nothing_to_compare_never_enters_anything() {
        let mut t = Tournament::build(&groups(&[&[0, 1]]), Vec::new(), |_| true);
        assert!(!t.enter(0));
        assert_eq!(t.pair(), None);
    }

    #[test]
    fn an_ineligible_photo_never_enters_a_comparison() {
        // Already in the library, or it never decoded — either way there
        // is no answer the user could act on.
        let mut t = Tournament::build(&groups(&[&[0, 1, 2]]), Vec::new(), |i| i == 1);
        assert!(t.enter(0));
        assert_eq!(t.pair(), Some((0, 2)));
        assert_eq!(t.progress(), (0, 2));
    }

    #[test]
    fn deciding_outside_a_session_does_nothing() {
        let mut t = build(&[&[0, 1]]);
        t.decide(Verdict::Left);
        let before = t.carry();
        // Past the end, `acted` is what says nothing happened.
        assert!(!t.decide(Verdict::Right).acted);
        assert!(!t.keep_rest().acted);
        assert_eq!(t.carry(), before);
    }

    /// The rule that makes rebuilding cheap: a rebuild carries the
    /// verdicts and re-asks nothing.
    #[test]
    fn rebuilding_resumes_rather_than_restarting() {
        let gs = groups(&[&[0, 1, 2, 3]]);
        let mut t = Tournament::build(&gs, Vec::new(), |_| false);
        t.enter(0);
        t.decide(Verdict::Left); // 1 is out for good

        let mut again = Tournament::build(&gs, t.carry(), |_| false);
        assert!(again.enter(0));
        // 1 is decided and stays decided; the bracket re-forms around the
        // three photos still in question.
        assert_eq!(again.pair(), Some((0, 2)));
        assert_eq!(again.progress(), (0, 3));
    }

    // ── Undo ──────────────────────────────────────────────────────

    #[test]
    fn undo_puts_back_exactly_what_the_last_verdict_settled() {
        let mut t = build(&[&[0, 1, 2]]);
        t.decide(Verdict::Left); // 1 out
        assert_eq!(t.pair(), Some((0, 2)));

        // The verdict that ends the session settles the survivor too.
        let ended = t.decide(Verdict::Left);
        assert_eq!(ended.settled, vec![(2, false), (0, true)]);
        assert_eq!(t.pair(), None);

        let back = t.undo();
        assert_eq!(back.withdrawn, vec![2, 0]);
        assert_eq!(t.pair(), Some((0, 2)), "and the question comes back");
        assert_eq!(out(&t), vec![1]);
    }

    /// Undo has to be able to reach back into a session the cursor has
    /// already walked past, which is why it restores `at` itself rather
    /// than leaving the caller to find its way back.
    #[test]
    fn undo_reopens_a_session_the_cursor_has_left() {
        let mut t = build(&[&[0, 1]]);
        t.decide(Verdict::Left);
        assert_eq!(t.pair(), None);
        t.undo();
        assert_eq!(t.pair(), Some((0, 1)), "without anyone calling `enter`");
    }

    #[test]
    fn undo_walks_all_the_way_back_to_the_start() {
        let gs: &[&[usize]] = &[&[0, 1, 2], &[8, 9]];
        let mut t = Tournament::build(&groups(gs), Vec::new(), |_| false);
        walk(&mut t, gs, |_, _| Verdict::Left);
        while t.can_undo() {
            t.undo();
        }
        assert_eq!(t.pair(), Some((0, 1)));
        assert!(t.carry().is_empty());
        assert_eq!(t.progress(), (0, 5));
    }

    #[test]
    fn undo_restores_a_session_that_keep_rest_ended() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        t.decide(Verdict::Both);
        t.decide(Verdict::Both);
        let before = t.pair();
        t.keep_rest();
        assert_eq!(t.pair(), None);
        t.undo();
        assert_eq!(t.pair(), before);
        assert!(t.carry().is_empty());
    }

    // ── The strip under the panes ─────────────────────────────────

    /// The strip shows the whole session, including what has been thrown
    /// out — seeing what was rejected is half of why it is there.
    #[test]
    fn the_group_view_marks_left_right_still_in_and_out() {
        let mut t = build(&[&[0, 1, 2, 3]]);
        t.decide(Verdict::Left); // 1 is out
        let g = t.group().unwrap();
        assert_eq!(g.members(), &[0, 1, 2, 3]);
        assert_eq!(g.state(2), CardState::Left);
        assert_eq!(g.state(3), CardState::Right);
        assert_eq!(g.state(0), CardState::In, "a winner waiting for round two");
        assert_eq!(g.state(1), CardState::Out);
        assert_eq!(g.alive(), 3);
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
