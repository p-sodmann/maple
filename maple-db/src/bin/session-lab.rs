//! `session-lab` — try the session engines on a real card, then tune them
//! by hand until they agree with you.
//!
//! Session detection is a judgement call about photographs, and no unit
//! test on synthetic gradients can tell you whether twenty pictures of a
//! child in a living room come out as one session or four. So this runs
//! every engine over a real folder — one decode, every engine sees the
//! same frame — and reports what each cost and where each cut.
//!
//! `--out` writes an HTML report that is **interactive**: every threshold
//! is a live slider, ensemble weights are live, and the segmentation
//! recomputes in the browser as you drag. You mark the scenes you actually
//! meant with a hotkey and two draggable markers, and the page exports them
//! as a `--truth` file, which is what turns "that looks about right" into a
//! number the next run can be scored against.
//!
//! ```sh
//! cargo run --release -p maple-db --bin session-lab -- /Volumes/CARD/DCIM --out /tmp/sessions.html
//! cargo run --release -p maple-db --bin session-lab -- ~/photos --ensemble block-tile=2,time-gap=1
//! ```
//!
//! ## How the browser recomputes
//!
//! Re-running a segmentation needs distances between arbitrary pairs, and
//! a full matrix is `n²`. But segmentation only ever asks about pairs
//! inside one session — the previous accepted frame, and the anchor — so
//! the page ships a **banded** matrix: every pair within `--band` of each
//! other, quantised to a byte. Beyond the band the distance reads as 1.0,
//! which ends a session by drift; a session longer than the band is
//! therefore capped in the browser but not in Rust, and the report says so
//! when it happens.
//!
//! The JavaScript is a line-by-line mirror of [`maple_import::session::segment`],
//! which is a thing that can rot. So the page also carries the Rust result
//! and checks its own against it on load, and says so in red if they differ.

use std::collections::HashSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::RgbImage;
use maple_import::session::{
    segment_with_holes, BlockTileEngine, ColorKmeansEngine, EnsembleEngine, Frame,
    GridHistogramEngine, SegmentParams, Segmentation, SessionEngine, Signature, TimeGapEngine,
};


fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .init();
    if let Err(err) = run() {
        eprintln!("session-lab: {err:#}");
        std::process::exit(1);
    }
}

// ── Arguments ─────────────────────────────────────────────────────

struct Args {
    dir: PathBuf,
    limit: Option<usize>,
    out: Option<PathBuf>,
    truth: Option<PathBuf>,
    dino: bool,
    only: Option<Vec<String>>,
    cuts: Vec<(String, f32)>,
    ensemble: Vec<(String, f32)>,
    time_points: Vec<(f32, f32)>,
    band: usize,
    max_outliers: usize,
    thumb_px: u32,
}

const USAGE: &str = "\
usage: session-lab <dir> [options]

  --limit N               only the first N photos
  --engines a,b,c         restrict to these engines by name
  --dino                  include the DINOv2 baseline (downloads the model)
  --ensemble a=2,b=1      add a weighted ensemble of the named engines
  --cut <engine>=<f>      override an engine's cut distance (repeatable)
  --time-points g=d,...   the time curve, e.g. 1=0,60=0.5,600=0.85,3600=1
  --max-outliers N        frames a session may absorb and come back from (default 1)
  --band N                pairs per photo shipped to the browser (default 48)
  --truth <file>          filenames that start a session, one per line
  --out <file.html>       write the interactive report
  --thumb-px N            thumbnail size in the report (default 132)
";

fn parse_pairs<T: std::str::FromStr>(spec: &str) -> Result<Vec<(String, T)>>
where
    T::Err: std::fmt::Display,
{
    spec.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|item| {
            let (name, value) = item.split_once('=').context("wants <name>=<value>")?;
            let value = value
                .trim()
                .parse::<T>()
                .map_err(|e| anyhow::anyhow!("{item:?}: {e}"))?;
            Ok((name.trim().to_owned(), value))
        })
        .collect()
}

fn parse_args() -> Result<Args> {
    let mut args = std::env::args().skip(1);
    let mut out = Args {
        dir: PathBuf::new(),
        limit: None,
        out: None,
        truth: None,
        dino: false,
        only: None,
        cuts: Vec::new(),
        ensemble: Vec::new(),
        time_points: vec![(1.0, 0.0), (60.0, 0.5), (600.0, 0.85), (3600.0, 1.0)],
        band: 48,
        max_outliers: 1,
        thumb_px: 132,
    };
    let mut dir = None;
    while let Some(arg) = args.next() {
        let mut value = |name: &str| args.next().with_context(|| format!("{name} needs a value"));
        match arg.as_str() {
            "-h" | "--help" => {
                print!("{USAGE}");
                std::process::exit(0);
            }
            "--limit" => out.limit = Some(value("--limit")?.parse().context("--limit")?),
            "--out" => out.out = Some(PathBuf::from(value("--out")?)),
            "--truth" => out.truth = Some(PathBuf::from(value("--truth")?)),
            "--thumb-px" => out.thumb_px = value("--thumb-px")?.parse().context("--thumb-px")?,
            "--band" => out.band = value("--band")?.parse::<usize>().context("--band")?.max(2),
            "--max-outliers" => {
                out.max_outliers = value("--max-outliers")?.parse().context("--max-outliers")?
            }
            "--dino" => out.dino = true,
            "--engines" => {
                out.only = Some(value("--engines")?.split(',').map(|s| s.trim().to_owned()).collect())
            }
            "--cut" => out.cuts.extend(parse_pairs::<f32>(&value("--cut")?).context("--cut")?),
            "--ensemble" => {
                out.ensemble = parse_pairs::<f32>(&value("--ensemble")?).context("--ensemble")?
            }
            "--time-points" => {
                let points = parse_pairs::<f32>(&value("--time-points")?).context("--time-points")?;
                out.time_points = points
                    .into_iter()
                    .map(|(g, d)| Ok((g.parse::<f32>().map_err(|e| anyhow::anyhow!("{g:?}: {e}"))?, d)))
                    .collect::<Result<Vec<_>>>()?;
                anyhow::ensure!(!out.time_points.is_empty(), "--time-points needs at least one point");
            }
            other if other.starts_with('-') => anyhow::bail!("unknown option {other}\n\n{USAGE}"),
            other => dir = Some(PathBuf::from(other)),
        }
    }
    out.dir = dir.context(USAGE)?;
    Ok(out)
}

// ── Pipeline ──────────────────────────────────────────────────────

/// One photo, as far as the lab cares.
struct Photo {
    path: PathBuf,
    /// Fractional seconds since the epoch, `None` with no usable EXIF.
    taken: Option<f64>,
    /// Base64 JPEG for the report; empty when `--out` was not asked for,
    /// since holding a thumbnail per photo for a whole card is the one
    /// thing here that grows without bound.
    thumb: String,
    /// `false` when nothing decoded — the photo stays in the sequence (a
    /// hole would silently join the sessions on either side of it) but
    /// carries no signature.
    decoded: bool,
}

/// What kind of thing the browser has to do to recompute an engine's
/// distances live.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Distances come from the shipped band; the cut is what tunes.
    Pixels,
    /// Recomputed from the control points, so the whole curve tunes.
    Time,
    /// Composed from its members, so weights and member cuts tune.
    Ensemble,
}

struct EngineRun {
    name: String,
    describe: String,
    kind: Kind,
    cut: f32,
    /// `(member, weight, member cut)`, empty unless this is an ensemble.
    members: Vec<(String, f32, f32)>,
    signatures: Vec<Option<Signature>>,
    /// Wall time spent inside `signature()`, summed over every photo.
    elapsed: Duration,
    bytes: usize,
}

/// The parameters this engine is actually segmented with.
///
/// Mirrors `SegmentParams::for_spec`: an engine that already votes on the
/// clock gets no threshold shaping, because time as a *cost*
/// (`tight_hold`/`long_drop`) plus time as a *vote* counts it twice. The
/// lab is where thresholds get chosen, so it has to segment the way the
/// importer will — a number tuned here against different parameters would
/// simply not transfer.
fn params_for(run: &EngineRun, base: &SegmentParams) -> SegmentParams {
    let votes_on_time = run.name == TimeGapEngine::NAME
        || run.members.iter().any(|(name, _, _)| name == TimeGapEngine::NAME);
    SegmentParams {
        cut: run.cut,
        tight_hold: if votes_on_time { 0.0 } else { base.tight_hold },
        long_drop: if votes_on_time { 0.0 } else { base.long_drop },
        ..*base
    }
}

fn run() -> Result<()> {
    let args = parse_args()?;
    let (mut engines, memberships) = build_engines(&args)?;

    let groups = maple_import::scan_grouped(&args.dir)
        .with_context(|| format!("scanning {}", args.dir.display()))?;
    let mut paths: Vec<PathBuf> = groups.into_iter().map(|g| g.display.path).collect();
    if let Some(limit) = args.limit {
        paths.truncate(limit);
    }
    anyhow::ensure!(!paths.is_empty(), "no images under {}", args.dir.display());

    println!("session-lab: {} photos from {}", paths.len(), args.dir.display());
    println!(
        "decoding at {} px via the canonical preview, {} engine(s)\n",
        maple_import::preview::PREVIEW_PX,
        engines.len()
    );

    let mut runs: Vec<EngineRun> = engines
        .iter()
        .map(|e| EngineRun {
            name: e.name().to_owned(),
            describe: e.describe(),
            kind: kind_of(e.as_ref()),
            cut: tuned_cut(&args, e.as_ref()),
            members: Vec::new(),
            signatures: Vec::with_capacity(paths.len()),
            elapsed: Duration::ZERO,
            bytes: 0,
        })
        .collect();
    // Ensembles report their membership so the browser can rebuild the
    // vote from the members' own bands. Recorded at construction rather
    // than recovered here — `SessionEngine` is deliberately not
    // downcastable, and it should stay that way.
    for (run, members) in runs.iter_mut().zip(&memberships) {
        run.members = members.clone();
    }

    // One serial pass: read, decode once, then hand the same frame to
    // every engine. Serial for the same reason the real scan is — a card
    // is one bus — and because the timings are only comparable if nothing
    // else is competing for the cores.
    let mut photos = Vec::with_capacity(paths.len());
    let decode_start = Instant::now();
    for (i, path) in paths.iter().enumerate() {
        if i % 100 == 0 && i > 0 {
            println!("  {i}/{}", paths.len());
        }
        let taken = maple_import::exif_read::read(path).capture_secs();
        let rgb = load_frame(path);
        let thumb = match (&rgb, args.out.is_some()) {
            (Some(f), true) => encode_thumb(f, args.thumb_px),
            _ => String::new(),
        };

        for (engine, run) in engines.iter_mut().zip(runs.iter_mut()) {
            let sig = match &rgb {
                Some(rgb) => {
                    let frame = Frame::new(rgb, taken);
                    let started = Instant::now();
                    let sig = engine.signature(&frame);
                    run.elapsed += started.elapsed();
                    match sig {
                        Ok(sig) => {
                            run.bytes += sig.heap_bytes();
                            Some(sig)
                        }
                        Err(err) => {
                            eprintln!("  {}: {} failed: {err:#}", path.display(), engine.name());
                            None
                        }
                    }
                }
                None => None,
            };
            run.signatures.push(sig);
        }

        photos.push(Photo { path: path.clone(), taken, thumb, decoded: rgb.is_some() });
    }
    let decode_elapsed = decode_start.elapsed();

    let undecodable = photos.iter().filter(|p| !p.decoded).count();
    let timed = photos.iter().filter(|p| p.taken.is_some()).count();
    println!(
        "\nread + decode: {:.1}s total ({:.1} ms/photo){}",
        decode_elapsed.as_secs_f32(),
        decode_elapsed.as_secs_f32() * 1000.0 / photos.len() as f32,
        if undecodable > 0 { format!(", {undecodable} would not decode") } else { String::new() }
    );
    println!(
        "capture times: {timed}/{} from EXIF{}\n",
        photos.len(),
        if timed < photos.len() { " (the rest fall back to the neutral gap)" } else { "" }
    );

    let times: Vec<Option<f64>> = photos.iter().map(|p| p.taken).collect();
    let base = SegmentParams { max_outliers: args.max_outliers, ..SegmentParams::default() };
    let segmentations: Vec<Segmentation> = engines
        .iter()
        .zip(&runs)
        .map(|(engine, run)| {
            segment_with_holes(engine.as_ref(), &run.signatures, &times, &params_for(run, &base))
        })
        .collect();

    print_cost_table(&runs, &photos);
    print_session_table(&runs, &segmentations, &photos);
    print_agreement(&runs, &segmentations);
    if let Some(truth) = &args.truth {
        print_truth(truth, &runs, &segmentations, &photos)?;
    }
    for (run, seg) in runs.iter().zip(&segmentations) {
        print_sessions(run, seg, &photos);
    }

    if let Some(out) = &args.out {
        let bands = build_bands(&engines, &runs, args.band);
        let html = render_html(&args, &runs, &segmentations, &photos, &bands, &base);
        let size = html.len();
        std::fs::write(out, html).with_context(|| format!("writing {}", out.display()))?;
        println!("wrote {} ({:.1} MB)", out.display(), size as f32 / 1e6);
        println!("open it, drag the sliders, press ? for the keys");
    }
    Ok(())
}

fn tuned_cut(args: &Args, engine: &dyn SessionEngine) -> f32 {
    args.cuts
        .iter()
        .find(|(n, _)| n == engine.name())
        .map(|(_, v)| *v)
        .unwrap_or_else(|| engine.default_cut())
}

fn kind_of(engine: &dyn SessionEngine) -> Kind {
    match engine.name() {
        "time-gap" => Kind::Time,
        "ensemble" => Kind::Ensemble,
        _ => Kind::Pixels,
    }
}

fn cheap_engine(name: &str) -> Option<Box<dyn SessionEngine>> {
    match name {
        "color-kmeans" => Some(Box::new(ColorKmeansEngine::default())),
        "grid-histogram" => Some(Box::new(GridHistogramEngine::default())),
        "block-tile" => Some(Box::new(BlockTileEngine::default())),
        _ => None,
    }
}

type Membership = Vec<(String, f32, f32)>;
/// The engines to run, paired with each one's membership.
type EngineSet = (Vec<Box<dyn SessionEngine>>, Vec<Membership>);

fn build_engines(args: &Args) -> Result<EngineSet> {
    let time_points = args.time_points.clone();
    let mut all: Vec<Box<dyn SessionEngine>> = vec![
        Box::new(ColorKmeansEngine::default()),
        Box::new(GridHistogramEngine::default()),
        Box::new(BlockTileEngine::default()),
        Box::new(TimeGapEngine::new(time_points.clone())),
    ];
    if args.dino {
        let settings = maple_state::Settings::load().stacks;
        all.push(Box::new(
            maple_db::DinoEngine::load(&settings).context("loading the DINOv2 baseline")?,
        ));
    }

    if let Some(only) = &args.only {
        let known: Vec<&str> = all.iter().map(|e| e.name()).collect();
        for want in only {
            anyhow::ensure!(
                known.contains(&want.as_str()),
                "unknown engine {want:?} (have: {})",
                known.join(", ")
            );
        }
        all.retain(|e| only.iter().any(|n| n == e.name()));
    }

    let mut memberships: Vec<Membership> = vec![Vec::new(); all.len()];
    if !args.ensemble.is_empty() {
        let mut members: Vec<(Box<dyn SessionEngine>, f32)> = Vec::new();
        for (name, weight) in &args.ensemble {
            let engine: Box<dyn SessionEngine> = match name.as_str() {
                "time-gap" => Box::new(TimeGapEngine::new(time_points.clone())),
                "dinov2" => {
                    let settings = maple_state::Settings::load().stacks;
                    Box::new(maple_db::DinoEngine::load(&settings).context("ensemble member dinov2")?)
                }
                other => cheap_engine(other)
                    .with_context(|| format!("unknown ensemble member {other:?}"))?,
            };
            members.push((engine, *weight));
        }
        let ensemble = EnsembleEngine::new(members).with_cuts(&args.cuts);
        memberships.push(
            ensemble.members().map(|(n, w, c)| (n.to_owned(), w, c)).collect(),
        );
        all.push(Box::new(ensemble));
    }

    anyhow::ensure!(!all.is_empty(), "no engines selected");
    Ok((all, memberships))
}

/// The frame every engine sees — byte-for-byte what the import scan hands
/// its own engines.
///
/// It goes through the *canonical preview* rather than decoding straight to
/// a working size, and that round trip is the point: the importer computes
/// on the WebP it keeps, so a lab that measured pristine pixels would be
/// tuning thresholds against an image the scan never sees. These two
/// diverged once already — the lab downsampled with `image::thumbnail`
/// while the scan used Lanczos3 — which is why the shared definition now
/// lives in [`maple_import::preview`] and neither side has a copy.
fn load_frame(path: &Path) -> Option<RgbImage> {
    maple_import::preview::decode(&maple_import::preview::encode_path(path).ok()?).ok()
}

fn encode_thumb(frame: &RgbImage, px: u32) -> String {
    let small = image::DynamicImage::ImageRgb8(frame.clone()).thumbnail(px, px);
    let mut jpeg = Vec::new();
    if image::codecs::jpeg::JpegEncoder::new_with_quality(&mut jpeg, 70)
        .encode_image(&small)
        .is_err()
    {
        return String::new();
    }
    STANDARD.encode(&jpeg)
}


/// Every pair of photos within `band` of each other, as one quantised byte
/// each — what the browser needs to re-segment without an `n²` matrix.
fn build_bands(
    engines: &[Box<dyn SessionEngine>],
    runs: &[EngineRun],
    band: usize,
) -> Vec<(String, Vec<u8>)> {
    engines
        .iter()
        .zip(runs)
        // Time and ensemble distances are *derived* in the browser — from
        // the curve and from the members — which is what makes their
        // controls live. Shipping a band for them would freeze exactly the
        // thing they exist to let you tune.
        .filter(|(_, run)| run.kind == Kind::Pixels)
        .map(|(engine, run)| {
            let n = run.signatures.len();
            let mut bytes = vec![255u8; n * band];
            for i in 0..n {
                let Some(a) = &run.signatures[i] else { continue };
                for k in 1..=band {
                    let Some(j) = i.checked_add(k).filter(|j| *j < n) else { break };
                    let Some(b) = &run.signatures[j] else { continue };
                    let d = engine.distance(a, b).clamp(0.0, 1.0);
                    bytes[i * band + k - 1] = (d * 255.0).round() as u8;
                }
            }
            (run.name.clone(), bytes)
        })
        .collect()
}

// ── Text report ───────────────────────────────────────────────────

fn print_cost_table(runs: &[EngineRun], photos: &[Photo]) {
    let n = photos.len().max(1) as f32;
    println!("{:<16} {:>10} {:>11} {:>10}  configuration", "engine", "ms/photo", "bytes/sig", "cut");
    println!("{}", "─".repeat(104));
    for run in runs {
        let sigs = run.signatures.iter().filter(|s| s.is_some()).count().max(1);
        println!(
            "{:<16} {:>10.3} {:>11} {:>10.3}  {}",
            run.name,
            run.elapsed.as_secs_f32() * 1000.0 / n,
            run.bytes / sigs,
            run.cut,
            run.describe
        );
    }
    println!();
}

fn print_session_table(runs: &[EngineRun], segmentations: &[Segmentation], photos: &[Photo]) {
    println!(
        "{:<16} {:>9} {:>8} {:>7} {:>10} {:>7} {:>8} {:>9}",
        "engine", "sessions", "groups", "solo", "in groups", "median", "largest", "outliers"
    );
    println!("{}", "─".repeat(104));
    for (run, seg) in runs.iter().zip(segmentations) {
        let mut sizes: Vec<usize> = seg.groups().map(|s| s.len()).collect();
        sizes.sort_unstable();
        let grouped: usize = sizes.iter().sum();
        println!(
            "{:<16} {:>9} {:>8} {:>7} {:>9.0}% {:>7} {:>8} {:>9}",
            run.name,
            seg.sessions.len(),
            sizes.len(),
            seg.sessions.len() - sizes.len(),
            100.0 * grouped as f32 / photos.len().max(1) as f32,
            sizes.get(sizes.len() / 2).copied().unwrap_or(0),
            sizes.last().copied().unwrap_or(0),
            seg.outliers.len(),
        );
    }
    println!();
}

/// Boundary agreement between every pair of engines.
///
/// No ground truth needed, and it answers the question that actually
/// decides this: if a cheap engine cuts a real card where DINOv2 cuts it,
/// the expensive one has nothing left to justify. F1 over boundary
/// positions, the standard way to score one segmentation against another.
fn print_agreement(runs: &[EngineRun], segmentations: &[Segmentation]) {
    if runs.len() < 2 {
        return;
    }
    let cuts: Vec<HashSet<usize>> = segmentations.iter().map(boundaries).collect();
    println!("boundary agreement (F1, 1.00 = identical cuts)");
    print!("{:<16}", "");
    for run in runs {
        print!("{:>16}", run.name);
    }
    println!("\n{}", "─".repeat(16 + 16 * runs.len()));
    for (i, run) in runs.iter().enumerate() {
        print!("{:<16}", run.name);
        for j in 0..runs.len() {
            if i == j {
                print!("{:>16}", "—");
            } else {
                print!("{:>16.2}", boundary_f1(&cuts[i], &cuts[j]));
            }
        }
        println!();
    }
    println!();
}

fn boundaries(seg: &Segmentation) -> HashSet<usize> {
    seg.sessions.iter().skip(1).map(|s| s.start).collect()
}

fn boundary_f1(a: &HashSet<usize>, b: &HashSet<usize>) -> f32 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    let shared = a.intersection(b).count() as f32;
    let precision = if a.is_empty() { 0.0 } else { shared / a.len() as f32 };
    let recall = if b.is_empty() { 0.0 } else { shared / b.len() as f32 };
    if precision + recall <= 0.0 {
        0.0
    } else {
        2.0 * precision * recall / (precision + recall)
    }
}

/// Score against a hand-written answer key: one filename per line, each
/// naming a photo that *starts* a session. Blank lines and `#` comments
/// are ignored, which is the format the HTML report exports.
fn print_truth(
    path: &Path,
    runs: &[EngineRun],
    segmentations: &[Segmentation],
    photos: &[Photo],
) -> Result<()> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let wanted: HashSet<&str> = text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .collect();
    let truth: HashSet<usize> = photos
        .iter()
        .enumerate()
        .filter(|(i, p)| {
            *i > 0 && p.path.file_name().and_then(|n| n.to_str()).is_some_and(|n| wanted.contains(n))
        })
        .map(|(i, _)| i)
        .collect();

    println!("against {} ({} boundaries)", path.display(), truth.len());
    println!("{:<16} {:>10} {:>10} {:>10}", "engine", "precision", "recall", "F1");
    println!("{}", "─".repeat(50));
    for (run, seg) in runs.iter().zip(segmentations) {
        let got = boundaries(seg);
        let shared = got.intersection(&truth).count() as f32;
        println!(
            "{:<16} {:>10.2} {:>10.2} {:>10.2}",
            run.name,
            if got.is_empty() { 0.0 } else { shared / got.len() as f32 },
            if truth.is_empty() { 0.0 } else { shared / truth.len() as f32 },
            boundary_f1(&got, &truth)
        );
    }
    println!();
    Ok(())
}

fn print_sessions(run: &EngineRun, seg: &Segmentation, photos: &[Photo]) {
    println!("── {} ─────────────────────────────────", run.name);
    for session in seg.groups() {
        let span = match (photos[session.start].taken, photos[session.end - 1].taken) {
            (Some(a), Some(b)) => human_gap((b - a) as f32),
            _ => "?".into(),
        };
        let odd = (session.start..session.end).filter(|i| seg.is_outlier(*i)).count();
        println!(
            "  [{:>4}..{:<4}] {:>3} photos  {} → {}  over {span}{}",
            session.start,
            session.end - 1,
            session.len(),
            name_of(&photos[session.start]),
            name_of(&photos[session.end - 1]),
            if odd > 0 { format!("  ({odd} bridged)") } else { String::new() },
        );
    }
    println!();
}

fn name_of(photo: &Photo) -> String {
    photo.path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default()
}

fn human_gap(secs: f32) -> String {
    if secs < 60.0 {
        format!("{secs:.1}s")
    } else if secs < 3600.0 {
        format!("{:.0}m{:02.0}s", secs / 60.0, secs % 60.0)
    } else {
        format!("{:.1}h", secs / 3600.0)
    }
}

/// JSON string escaping. The report embeds its data as a `<script>` blob,
/// so `</script>` inside any value would end the block early — hence the
/// `<` escape, which is not otherwise required.
fn json_str(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '<' => out.push_str("\\u003c"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

fn json_num(v: f32) -> String {
    if v.is_finite() {
        format!("{v}")
    } else if v > 0.0 {
        "1e9".into()
    } else {
        "-1e9".into()
    }
}

// ── Interactive report ────────────────────────────────────────────

fn render_html(
    args: &Args,
    runs: &[EngineRun],
    segmentations: &[Segmentation],
    photos: &[Photo],
    bands: &[(String, Vec<u8>)],
    base: &SegmentParams,
) -> String {
    let mut h = String::with_capacity(photos.len() * 9000);
    h.push_str("<!doctype html><meta charset=\"utf-8\"><meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">");
    h.push_str("<title>session-lab</title><style>");
    h.push_str(APP_CSS);
    h.push_str("</style>");

    let _ = write!(h, "<body style=\"--tw:{}px\">", args.thumb_px);
    h.push_str(APP_HTML);

    // ── the data ──
    h.push_str("<script>window.LAB=");
    h.push('{');
    let _ = write!(h, "dir:{},", json_str(&args.dir.display().to_string()));
    let _ = write!(h, "band:{},framePx:{},", args.band, maple_import::preview::PREVIEW_PX);
    let _ = write!(h, "defaultEngine:{},", json_str(&runs.last().map(|r| r.name.clone()).unwrap_or_default()));

    h.push_str("params:{");
    let _ = write!(
        h,
        "gapScale:{},tightHold:{},longDrop:{},streakBonus:{},streakLen:{},anchorFactor:{},hardGapSecs:{},maxOutliers:{}",
        json_num(base.gap_scale_secs),
        json_num(base.tight_hold),
        json_num(base.long_drop),
        json_num(base.streak_bonus),
        base.streak_len,
        json_num(base.anchor_factor),
        json_num(base.hard_gap_secs),
        base.max_outliers
    );
    h.push_str("},");

    h.push_str("time:[");
    for (i, (g, d)) in args.time_points.iter().enumerate() {
        let _ = write!(h, "{}[{},{}]", if i > 0 { "," } else { "" }, json_num(*g), json_num(*d));
    }
    h.push_str("],");

    h.push_str("photos:[");
    for (i, p) in photos.iter().enumerate() {
        if i > 0 {
            h.push(',');
        }
        h.push('{');
        let _ = write!(h, "n:{}", json_str(&name_of(p)));
        match p.taken {
            // Seconds since the epoch need the precision: an f32 would
            // quantise 1.8e9 to two-minute steps and every gap with it.
            Some(t) => {
                let _ = write!(h, ",t:{t}");
            }
            None => h.push_str(",t:null"),
        }
        if !p.decoded {
            h.push_str(",d:0");
        }
        if !p.thumb.is_empty() {
            let _ = write!(h, ",j:{}", json_str(&p.thumb));
        }
        h.push('}');
    }
    h.push_str("],");

    h.push_str("engines:[");
    for (i, (run, seg)) in runs.iter().zip(segmentations).enumerate() {
        if i > 0 {
            h.push(',');
        }
        h.push('{');
        let _ = write!(h, "name:{},", json_str(&run.name));
        let _ = write!(h, "describe:{},", json_str(&run.describe));
        let _ = write!(
            h,
            "kind:{},cut:{},",
            json_str(match run.kind {
                Kind::Pixels => "pixels",
                Kind::Time => "time",
                Kind::Ensemble => "ensemble",
            }),
            json_num(run.cut)
        );
        h.push_str("members:[");
        for (k, (name, weight, cut)) in run.members.iter().enumerate() {
            let _ = write!(
                h,
                "{}{{name:{},weight:{},cut:{}}}",
                if k > 0 { "," } else { "" },
                json_str(name),
                json_num(*weight),
                json_num(*cut)
            );
        }
        h.push_str("],");
        // The Rust answer, for the browser to check its own mirror against.
        h.push_str("rust:[");
        for (k, s) in seg.sessions.iter().skip(1).enumerate() {
            let _ = write!(h, "{}{}", if k > 0 { "," } else { "" }, s.start);
        }
        h.push_str("]}");
    }
    h.push_str("],");

    h.push_str("bands:{");
    for (i, (name, bytes)) in bands.iter().enumerate() {
        let _ = write!(
            h,
            "{}{}:{}",
            if i > 0 { "," } else { "" },
            json_str(name),
            json_str(&STANDARD.encode(bytes))
        );
    }
    h.push_str("}};</script>");

    h.push_str("<script>");
    h.push_str(APP_JS);
    h.push_str("</script></body>");
    h
}

const APP_CSS: &str = r##"
:root{color-scheme:light dark;--bg:#faf9f7;--fg:#1b1a18;--muted:#6d6a66;--line:#dcd8d2;--card:#fff;--cut:#c2410c;--pick:#2563eb;--ok:#15803d;--warn:#b45309}
@media(prefers-color-scheme:dark){:root{--bg:#161513;--fg:#eceae6;--muted:#9b9691;--line:#33302c;--card:#1f1e1b;--cut:#fb923c;--pick:#60a5fa;--ok:#4ade80;--warn:#fbbf24}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.5 ui-sans-serif,-apple-system,"Segoe UI",sans-serif;display:grid;grid-template-columns:300px 1fr;height:100vh;overflow:hidden}
aside{border-right:1px solid var(--line);padding:16px;overflow-y:auto;background:var(--card)}
main{overflow-y:auto;padding:16px 20px}
h1{font-size:15px;margin:0 0 2px;letter-spacing:.01em}
h2{font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);margin:20px 0 8px;font-weight:600}
.sub{color:var(--muted);font-size:12px;margin-bottom:10px;word-break:break-all}
label{display:block;font-size:12px;margin:9px 0 1px;color:var(--muted);display:flex;justify-content:space-between;gap:8px}
label b{color:var(--fg);font-weight:600;font-variant-numeric:tabular-nums}
input[type=range]{width:100%;margin:0;accent-color:var(--pick)}
input[type=number]{width:70px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:4px;padding:2px 5px;font:inherit;font-size:12px}
select,button{width:100%;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px 8px;font:inherit;font-size:13px;cursor:pointer}
button:hover{border-color:var(--pick)}
button.row{width:auto;flex:1}
.btns{display:flex;gap:6px;margin-top:8px}
textarea{width:100%;height:120px;background:var(--bg);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:6px;font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;resize:vertical}
table.stats{width:100%;border-collapse:collapse;font-size:12px;font-variant-numeric:tabular-nums}
table.stats td{padding:2px 0;border-bottom:1px solid var(--line)}
table.stats td:last-child{text-align:right;font-weight:600}
.mismatch{background:var(--cut);color:#fff;padding:6px 8px;border-radius:6px;font-size:12px;margin:8px 0}
.hint{color:var(--muted);font-size:11px;margin-top:6px}
kbd{border:1px solid var(--line);border-bottom-width:2px;border-radius:4px;padding:0 4px;font:inherit;font-size:11px;background:var(--bg)}
section.session{background:var(--card);border:1px solid var(--line);border-radius:9px;padding:8px 10px 10px;margin:0 0 10px}
section.session.solo{opacity:.55}
section.session.scene{border-color:var(--pick);box-shadow:0 0 0 1px var(--pick) inset}
section.session>.head{color:var(--muted);font-size:11.5px;margin-bottom:6px;font-variant-numeric:tabular-nums;display:flex;gap:10px;flex-wrap:wrap}
section.session>.head .why{color:var(--cut)}
.strip{display:flex;flex-wrap:wrap;gap:6px}
.tile{position:relative;width:var(--tw);cursor:pointer;border-radius:5px}
.tile img{display:block;width:100%;height:auto;border-radius:4px;background:var(--line)}
.tile .cap{display:block;font-size:9.5px;color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
.tile.cur{outline:2px solid var(--pick);outline-offset:2px}
.tile.outlier img{outline:2px dashed var(--warn);outline-offset:-2px}
.tile.inscene img{filter:none}
.tile.dim img{opacity:.35}
.tile .miss{display:grid;place-items:center;height:var(--tw);font-size:10px;color:var(--muted);border:1px dashed var(--line);border-radius:4px}
.handle{position:absolute;top:0;bottom:14px;width:12px;background:var(--pick);border-radius:4px;cursor:ew-resize;opacity:.85;touch-action:none;display:grid;place-items:center;color:#fff;font-size:9px}
.handle.hs{left:-7px}
.handle.he{right:-7px}
#grid{padding-bottom:40vh}
.overlay{position:fixed;inset:0;background:rgba(0,0,0,.5);display:grid;place-items:center;z-index:9}
.overlay[hidden]{display:none}
.overlay .card{background:var(--card);border-radius:12px;padding:20px 24px;max-width:460px;border:1px solid var(--line)}
.overlay table{border-collapse:collapse;font-size:13px}
.overlay td{padding:3px 12px 3px 0}
"##;

const APP_HTML: &str = r##"
<aside>
  <h1>session-lab</h1>
  <div class="sub" id="dir"></div>
  <div id="mismatch"></div>

  <h2>engine</h2>
  <select id="engine"></select>
  <div class="hint" id="describe"></div>

  <h2>thresholds</h2>
  <div id="cuts"></div>

  <h2>ensemble weights</h2>
  <div id="weights"><div class="hint">no ensemble — pass <code>--ensemble</code></div></div>

  <h2>time curve</h2>
  <div id="time"></div>

  <h2>segmentation</h2>
  <div id="params"></div>

  <h2>result</h2>
  <table class="stats" id="stats"></table>

  <h2>your scenes</h2>
  <div class="btns">
    <button class="row" id="adopt">Adopt engine</button>
    <button class="row" id="clear">Clear</button>
  </div>
  <textarea id="truth" spellcheck="false"></textarea>
  <div class="btns"><button class="row" id="download">Download truth</button></div>
  <div class="hint">Press <kbd>?</kbd> for the keys.</div>
</aside>
<main><div id="grid"></div></main>
<div class="overlay" id="help" hidden><div class="card">
  <h2 style="margin-top:0">keys</h2>
  <table>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>move the current photo</td></tr>
    <tr><td><kbd>s</kbd></td><td>mark the current session as a scene, and edit it</td></tr>
    <tr><td><kbd>drag</kbd></td><td>the blue handles move the scene's start and end</td></tr>
    <tr><td><kbd>⇧</kbd><kbd>←</kbd> <kbd>⇧</kbd><kbd>→</kbd></td><td>move the scene's start</td></tr>
    <tr><td><kbd>←</kbd> <kbd>→</kbd></td><td>while editing: move the scene's end</td></tr>
    <tr><td><kbd>⏎</kbd></td><td>commit the scene</td></tr>
    <tr><td><kbd>esc</kbd></td><td>cancel the edit</td></tr>
    <tr><td><kbd>x</kbd></td><td>mark the current photo an outlier</td></tr>
    <tr><td><kbd>a</kbd></td><td>adopt every session from this engine</td></tr>
    <tr><td><kbd>?</kbd></td><td>this</td></tr>
  </table>
</div></div>
"##;

/// The browser-side mirror of [`maple_import::session::segment`].
///
/// Kept deliberately close to the Rust, name for name, because the only
/// thing that makes a tuning session trustworthy is that the page is
/// segmenting the way the library does. `checkMirror` compares its output
/// against the Rust result shipped with the page and complains loudly if
/// the two have drifted apart.
const APP_JS: &str = r##"
const L = window.LAB;
const W = L.band;
const N = L.photos.length;
const byName = Object.fromEntries(L.engines.map(e => [e.name, e]));

// Members can name an engine that is not itself displayed, so make sure
// every referenced engine is addressable.
for (const e of L.engines) for (const m of e.members) if (!byName[m.name]) byName[m.name] = {name:m.name, kind:'pixels', cut:m.cut, members:[]};

function unb64(s){const bin=atob(s);const a=new Uint8Array(bin.length);for(let i=0;i<bin.length;i++)a[i]=bin.charCodeAt(i);return a;}
const bands = Object.fromEntries(Object.entries(L.bands).map(([k,v]) => [k, unb64(v)]));

const state = {
  engine: L.defaultEngine,
  params: {...L.params},
  cuts: Object.fromEntries(L.engines.map(e => [e.name, e.cut])),
  weights: {},
  time: L.time.map(p => [...p]),
  scenes: [],
  outliers: new Set(),
  cur: 0,
  edit: null,
  seg: null,
};
for (const e of L.engines) for (const m of e.members) state.weights[e.name+'/'+m.name] = m.weight;
for (const e of L.engines) for (const m of e.members) if (state.cuts[m.name] === undefined) state.cuts[m.name] = m.cut;

// ── distances ─────────────────────────────────────────────────────

function curve(secs){
  const p = state.time;
  if (!p.length) return 0;
  if (Number.isNaN(secs)) return p[p.length-1][1];
  if (secs <= p[0][0]) return p[0][1];
  if (secs >= p[p.length-1][0]) return p[p.length-1][1];
  const lg = Math.log(secs);
  for (let i=0;i+1<p.length;i++){
    const [g0,d0]=p[i],[g1,d1]=p[i+1];
    if (secs <= g1){
      const l0=Math.log(g0), l1=Math.log(g1);
      const t = Math.abs(l1-l0) < 1e-9 ? 0 : (lg-l0)/(l1-l0);
      return d0 + t*(d1-d0);
    }
  }
  return p[p.length-1][1];
}

function dist(name, i, j){
  const e = byName[name];
  if (e.kind === 'time'){
    const a = L.photos[i].t, b = L.photos[j].t;
    // Matches TimeGapEngine: an unknown time abstains at the fixed 0.5,
    // not at the tuned cut.
    if (a == null || b == null) return 0.5;
    return curve(Math.abs(b-a));
  }
  if (e.kind === 'ensemble'){
    let acc=0, total=0;
    for (const m of e.members){
      const w = state.weights[e.name+'/'+m.name];
      if (!(w > 0)) continue;
      const c = state.cuts[m.name];
      const d = dist(m.name, i, j);
      acc += w * (c <= 0 ? (d > 0 ? 1 : 0) : d/(d+c));
      total += w;
    }
    if (total <= 0) return 0.5;
    return Math.min(1, Math.max(0, acc/total));
  }
  const k = j - i;
  if (k < 1) return 0;
  if (k > W) return 1;            // outside the band: reads as fully changed
  const band = bands[name];
  if (!band) return 1;
  if (L.photos[i].d === 0 || L.photos[j].d === 0) return 1;
  return band[i*W + k - 1] / 255;
}

// ── segmentation (mirror of maple_import::session::segment) ───────

function thresholdFor(p, gap, streak){
  const g = Math.max(gap === null ? p.gapScale : gap, 0);
  const s = g / (g + p.gapScale);
  let t = p.cut * ((1 + p.tightHold) - (p.tightHold + p.longDrop) * s);
  if (streak >= p.streakLen) t *= 1 + p.streakBonus;
  return Math.max(t, 0);
}

// Mirrors `params_for` in Rust: an engine that already votes on the clock
// gets no threshold shaping, or time would count twice — once as a cost
// and once as a vote. The sliders stay live for every other engine, which
// is why this is decided per engine rather than by hiding the controls.
function votesOnTime(name){
  if (name === 'time-gap') return true;
  const e = L.engines.find(e => e.name === name);
  return !!e && (e.members || []).some(m => m.name === 'time-gap');
}

function segment(name){
  const p = {...state.params, cut: state.cuts[name]};
  if (votesOnTime(name)){ p.tightHold = 0; p.longDrop = 0; }
  const sessions = [], outliers = [], links = new Array(N).fill(null);
  const gapBetween = (a,b) => {
    const x = L.photos[a].t, y = L.photos[b].t;
    return (x == null || y == null) ? null : Math.abs(y - x);
  };
  if (N === 0) return {sessions, links: [], outliers, p};

  let start = 0, lastGood = 0, pending = [], at = 1;
  while (at < N){
    const d = dist(name, lastGood, at);
    const ad = start === lastGood ? d : dist(name, start, at);
    const gap = gapBetween(lastGood, at);
    const th = thresholdFor(p, gap, lastGood + 1 - start);

    let reason = null;
    if (gap !== null && gap >= p.hardGapSecs) reason = 'gap';
    else if (d > th) reason = 'scene';
    else if (ad > th * p.anchorFactor) reason = 'drift';

    const link = {at, from: lastGood, d, ad, gap, th, cut: reason, bridged: false};
    if (reason === null){
      for (const held of pending){
        outliers.push(held);
        if (links[held]){ links[held].bridged = true; links[held].cut = null; }
      }
      pending = [];
      links[at] = link; lastGood = at; at++;
    } else if (pending.length < p.maxOutliers){
      pending.push(at); links[at] = link; at++;
    } else {
      const cutAt = pending.length ? pending[0] : at;
      sessions.push({start, end: cutAt});
      if (cutAt === at) links[at] = link;
      else if (links[cutAt]){ links[cutAt].cut = links[cutAt].cut || 'scene'; links[cutAt].bridged = false; }
      start = cutAt; lastGood = cutAt; pending = []; at = cutAt + 1;
    }
  }
  if (pending.length){
    const first = pending[0];
    sessions.push({start, end: first});
    if (links[first]) links[first].cut = links[first].cut || 'scene';
    start = first;
  }
  sessions.push({start, end: N});
  outliers.sort((a,b) => a-b);
  return {sessions, links, outliers, p};
}

// The page is only worth trusting if it segments the way the library does.
function checkMirror(){
  const bad = [];
  for (const e of L.engines){
    const mine = segment(e.name).sessions.slice(1).map(s => s.start);
    if (mine.length !== e.rust.length || mine.some((v,i) => v !== e.rust[i])) bad.push(e.name);
  }
  const box = document.getElementById('mismatch');
  if (bad.length){
    box.innerHTML = '<div class="mismatch">This page disagrees with Rust on: ' + bad.join(', ') +
      '. The JavaScript mirror of <code>segment()</code> has drifted — trust the terminal, not this.</div>';
  } else box.innerHTML = '';
}

// ── grid ──────────────────────────────────────────────────────────

const grid = document.getElementById('grid');
const tiles = L.photos.map((photo, i) => {
  const el = document.createElement('div');
  el.className = 'tile';
  el.dataset.i = i;
  el.innerHTML = (photo.j
      ? '<img loading="lazy" src="data:image/jpeg;base64,' + photo.j + '" alt="">'
      : '<div class="miss">no preview</div>')
    + '<span class="cap"></span>'
    + '<i class="handle hs" data-h="s">◀</i><i class="handle he" data-h="e">▶</i>';
  el.querySelector('.cap').textContent = i + ' · ' + photo.n;
  el.addEventListener('click', ev => {
    if (ev.target.classList.contains('handle')) return;
    state.cur = i; paint();
  });
  return el;
});

function humanGap(s){
  if (s == null) return 'no time';
  if (s < 60) return s.toFixed(1) + 's';
  if (s < 3600) return Math.floor(s/60) + 'm' + String(Math.round(s%60)).padStart(2,'0') + 's';
  return (s/3600).toFixed(1) + 'h';
}

function rebuild(){
  const seg = state.seg;
  const frag = document.createDocumentFragment();
  for (let n = 0; n < seg.sessions.length; n++){
    const s = seg.sessions[n];
    const sec = document.createElement('section');
    sec.className = 'session' + (s.end - s.start < 2 ? ' solo' : '');
    sec.dataset.start = s.start;
    const head = document.createElement('div');
    head.className = 'head';
    const link = seg.links[s.start];
    const span = (L.photos[s.start].t != null && L.photos[s.end-1].t != null && s.end-s.start > 1)
      ? humanGap(L.photos[s.end-1].t - L.photos[s.start].t) : '';
    head.innerHTML = '<span>#' + n + ' · ' + (s.end-s.start) + ' photo' + (s.end-s.start===1?'':'s') +
      (span ? ' · ' + span : '') + ' · [' + s.start + '..' + (s.end-1) + ']</span>' +
      (link && link.cut ? '<span class="why">▲ ' + link.cut + ' · d=' + link.d.toFixed(3) +
        ' vs ' + link.th.toFixed(3) + ' · drift ' + link.ad.toFixed(3) + ' · ' + humanGap(link.gap) + '</span>' : '');
    sec.appendChild(head);
    const strip = document.createElement('div');
    strip.className = 'strip';
    for (let i = s.start; i < s.end; i++) strip.appendChild(tiles[i]);
    sec.appendChild(strip);
    frag.appendChild(sec);
  }
  grid.replaceChildren(frag);
}

function paint(){
  const seg = state.seg;
  const engineOut = new Set(seg.outliers);
  const scene = state.edit != null ? state.scenes[state.edit] : null;
  for (let i = 0; i < N; i++){
    const el = tiles[i];
    el.classList.toggle('cur', i === state.cur);
    el.classList.toggle('outlier', engineOut.has(i) || state.outliers.has(i));
    const inScene = scene && i >= scene.start && i < scene.end;
    el.classList.toggle('inscene', !!inScene);
    el.classList.toggle('dim', !!scene && !inScene);
    el.querySelector('.hs').style.display = scene && i === scene.start ? 'grid' : 'none';
    el.querySelector('.he').style.display = scene && i === scene.end - 1 ? 'grid' : 'none';
  }
  for (const sec of grid.children){
    const start = +sec.dataset.start;
    sec.classList.toggle('scene', state.scenes.some(s => s.start === start));
  }
  const cur = tiles[state.cur];
  if (cur) cur.scrollIntoView({block:'nearest'});
  renderStats();
  renderTruth();
}

function recompute(){
  state.seg = segment(state.engine);
  rebuild();
  paint();
}

// ── stats ─────────────────────────────────────────────────────────

function f1(a, b){
  const A = new Set(a), B = new Set(b);
  if (!A.size && !B.size) return 1;
  let shared = 0;
  for (const v of A) if (B.has(v)) shared++;
  const p = A.size ? shared/A.size : 0, r = B.size ? shared/B.size : 0;
  return p + r <= 0 ? 0 : 2*p*r/(p+r);
}

function renderStats(){
  const seg = state.seg;
  const sizes = seg.sessions.filter(s => s.end-s.start >= 2).map(s => s.end-s.start).sort((x,y)=>x-y);
  const rows = [
    ['sessions', seg.sessions.length],
    ['groups', sizes.length],
    ['solo', seg.sessions.length - sizes.length],
    ['largest', sizes.length ? sizes[sizes.length-1] : 0],
    ['median group', sizes.length ? sizes[Math.floor(sizes.length/2)] : 0],
    ['bridged outliers', seg.outliers.length],
  ];
  if (state.scenes.length){
    rows.push(['— vs your scenes —', '']);
    rows.push(['boundary F1', f1(seg.sessions.slice(1).map(s=>s.start), state.scenes.map(s=>s.start).filter(v=>v>0)).toFixed(2)]);
  }
  document.getElementById('stats').innerHTML =
    rows.map(([k,v]) => '<tr><td>'+k+'</td><td>'+v+'</td></tr>').join('');
}

function renderTruth(){
  const lines = ['# session-lab truth for ' + L.dir, '# one filename per line: the photo that STARTS a session'];
  for (const s of [...state.scenes].sort((a,b)=>a.start-b.start)) lines.push(L.photos[s.start].n);
  if (state.outliers.size) lines.push('', ...[...state.outliers].sort((a,b)=>a-b).map(i => '# outlier: ' + L.photos[i].n));
  document.getElementById('truth').value = lines.join('\n');
}

// ── controls ──────────────────────────────────────────────────────

function slider(host, label, value, min, max, step, onInput, fmt){
  const wrap = document.createElement('div');
  const id = 'c' + Math.random().toString(36).slice(2);
  wrap.innerHTML = '<label for="'+id+'">'+label+' <b></b></label>';
  const input = document.createElement('input');
  input.type = 'range'; input.id = id; input.min = min; input.max = max; input.step = step; input.value = value;
  const out = wrap.querySelector('b');
  const show = () => out.textContent = (fmt || (v => (+v).toFixed(3)))(input.value);
  show();
  input.addEventListener('input', () => { show(); onInput(+input.value); schedule(); });
  wrap.appendChild(input);
  host.appendChild(wrap);
}

let queued = false;
function schedule(){
  if (queued) return;
  queued = true;
  requestAnimationFrame(() => { queued = false; recompute(); });
}

function buildControls(){
  document.getElementById('dir').textContent = L.dir + ' · ' + N + ' photos';

  const pick = document.getElementById('engine');
  pick.innerHTML = L.engines.map(e => '<option value="'+e.name+'">'+e.name+'</option>').join('');
  pick.value = state.engine;
  pick.addEventListener('change', () => {
    state.engine = pick.value;
    document.getElementById('describe').textContent = byName[state.engine].describe || '';
    recompute();
  });
  document.getElementById('describe').textContent = byName[state.engine].describe || '';

  const cuts = document.getElementById('cuts');
  for (const name of Object.keys(state.cuts))
    slider(cuts, name, state.cuts[name], 0, 1, 0.005, v => state.cuts[name] = v);

  const weights = document.getElementById('weights');
  const ens = L.engines.filter(e => e.kind === 'ensemble');
  if (ens.length){
    weights.innerHTML = '';
    for (const e of ens) for (const m of e.members){
      const key = e.name + '/' + m.name;
      slider(weights, m.name, state.weights[key], 0, 5, 0.05, v => state.weights[key] = v, v => (+v).toFixed(2));
    }
  }

  const time = document.getElementById('time');
  state.time.forEach((pt, i) => {
    const row = document.createElement('div');
    row.innerHTML = '<label>at <input type="number" step="1" min="0.05" value="'+pt[0]+'"> s <b></b></label>';
    const gapIn = row.querySelector('input');
    time.appendChild(row);
    slider(time, '', pt[1], 0, 1, 0.01, v => state.time[i][1] = v);
    gapIn.addEventListener('input', () => {
      const v = parseFloat(gapIn.value);
      if (Number.isFinite(v) && v > 0){ state.time[i][0] = v; schedule(); }
    });
    row.querySelector('b').textContent = '';
  });

  const params = document.getElementById('params');
  const P = state.params;
  slider(params, 'gap scale (s)', P.gapScale, 1, 300, 1, v => P.gapScale = v, v => (+v).toFixed(0));
  slider(params, 'tight hold', P.tightHold, 0, 1, 0.01, v => P.tightHold = v, v => (+v).toFixed(2));
  slider(params, 'long drop', P.longDrop, 0, 1, 0.01, v => P.longDrop = v, v => (+v).toFixed(2));
  slider(params, 'streak bonus', P.streakBonus, 0, 1, 0.01, v => P.streakBonus = v, v => (+v).toFixed(2));
  slider(params, 'streak length', P.streakLen, 1, 10, 1, v => P.streakLen = v, v => (+v).toFixed(0));
  slider(params, 'anchor factor', P.anchorFactor, 1, 6, 0.05, v => P.anchorFactor = v, v => (+v).toFixed(2));
  slider(params, 'hard gap (min)', P.hardGapSecs/60, 1, 240, 1, v => P.hardGapSecs = v*60, v => (+v).toFixed(0)+'m');
  slider(params, 'max outliers', P.maxOutliers, 0, 5, 1, v => P.maxOutliers = v, v => (+v).toFixed(0));

  document.getElementById('adopt').addEventListener('click', adoptAll);
  document.getElementById('clear').addEventListener('click', () => {
    state.scenes = []; state.outliers.clear(); state.edit = null; paint();
  });
  document.getElementById('download').addEventListener('click', () => {
    const blob = new Blob([document.getElementById('truth').value], {type:'text/plain'});
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'sessions-truth.txt';
    a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 1000);
  });
}

// ── scene editing ─────────────────────────────────────────────────

function sessionAt(i){
  return state.seg.sessions.find(s => i >= s.start && i < s.end) || {start:i, end:i+1};
}

function editScene(){
  const existing = state.scenes.findIndex(s => state.cur >= s.start && state.cur < s.end);
  if (existing >= 0){ state.edit = existing; paint(); return; }
  const s = sessionAt(state.cur);
  state.scenes.push({start: s.start, end: s.end});
  state.edit = state.scenes.length - 1;
  paint();
}

function commitScene(){
  if (state.edit == null) return;
  const me = state.scenes[state.edit];
  // A scene the user just drew wins over whatever it overlaps: trim the
  // others, and drop any it swallowed whole.
  state.scenes = state.scenes.filter((s, i) => {
    if (i === state.edit) return true;
    if (s.start >= me.start && s.end <= me.end) return false;
    if (s.start < me.start && s.end > me.start) s.end = me.start;
    if (s.start < me.end && s.end > me.end) s.start = me.end;
    return s.end > s.start;
  });
  state.scenes.sort((a,b) => a.start - b.start);
  state.edit = null;
  paint();
}

function adoptAll(){
  state.scenes = state.seg.sessions.map(s => ({start: s.start, end: s.end}));
  state.edit = null;
  paint();
}

function moveEdge(which, delta){
  if (state.edit == null) return;
  const s = state.scenes[state.edit];
  if (which === 's') s.start = Math.min(Math.max(0, s.start + delta), s.end - 1);
  else s.end = Math.max(Math.min(N, s.end + delta), s.start + 1);
  state.cur = which === 's' ? s.start : s.end - 1;
  paint();
}

let dragging = null;
document.addEventListener('pointerdown', ev => {
  const h = ev.target.closest('.handle');
  if (!h || state.edit == null) return;
  dragging = h.dataset.h;
  ev.preventDefault();
});
document.addEventListener('pointermove', ev => {
  if (!dragging || state.edit == null) return;
  const el = document.elementFromPoint(ev.clientX, ev.clientY);
  const tile = el && el.closest ? el.closest('.tile') : null;
  if (!tile) return;
  const i = +tile.dataset.i;
  const s = state.scenes[state.edit];
  if (dragging === 's') s.start = Math.min(i, s.end - 1);
  else s.end = Math.max(i + 1, s.start + 1);
  paint();
});
document.addEventListener('pointerup', () => { dragging = null; });

// ── keys ──────────────────────────────────────────────────────────

document.addEventListener('keydown', ev => {
  if (ev.target.matches('input,textarea,select')) return;
  const help = document.getElementById('help');
  switch (ev.key){
    case 'ArrowRight':
      if (state.edit != null) moveEdge(ev.shiftKey ? 's' : 'e', 1);
      else { state.cur = Math.min(N-1, state.cur+1); paint(); }
      ev.preventDefault(); break;
    case 'ArrowLeft':
      if (state.edit != null) moveEdge(ev.shiftKey ? 's' : 'e', -1);
      else { state.cur = Math.max(0, state.cur-1); paint(); }
      ev.preventDefault(); break;
    case 's': editScene(); break;
    case 'Enter': commitScene(); break;
    case 'Escape':
      if (!help.hidden) { help.hidden = true; break; }
      if (state.edit != null){ state.scenes.splice(state.edit, 1); state.edit = null; paint(); }
      break;
    case 'x':
      if (state.outliers.has(state.cur)) state.outliers.delete(state.cur); else state.outliers.add(state.cur);
      paint(); break;
    case 'a': adoptAll(); break;
    case '?': help.hidden = !help.hidden; break;
  }
});
document.getElementById('help').addEventListener('click', ev => {
  if (ev.target.id === 'help') ev.target.hidden = true;
});

buildControls();
recompute();
checkMirror();
"##;
