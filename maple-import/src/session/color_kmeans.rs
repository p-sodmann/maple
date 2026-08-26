//! Engine 1 — dominant colours, via k-means in CIELAB.
//!
//! Blur the frame down to a small grid (the area-average in
//! [`super::downsample`] *is* the blur), convert to Lab so that distances
//! between colours mean roughly what the eye means by them, then find `k`
//! cluster centres and keep them with the share of the picture each one
//! covers.
//!
//! What it sees is the **palette**, and nothing else — the same room at the
//! same time of day gives the same handful of colours in the same
//! proportions no matter where the child is standing, which is exactly the
//! invariance this case wants. What it cannot see is *where* any of it is:
//! a red jumper on the left and a red jumper on the right are the same
//! signature. [`super::GridHistogramEngine`] is the counterpart that
//! trades the other way.
//!
//! Determinism matters — the same photo must produce the same signature on
//! every run, or a cached signature and a fresh one would disagree. So
//! k-means++ seeding draws from a fixed-seed LCG rather than a real RNG,
//! and ties resolve by index.

use super::{Frame, SessionEngine, Signature, downsample};

const NAME: &str = "color-kmeans";

/// Lab ΔE at which two colours count as entirely unrelated. Roughly the
/// distance from mid-grey to a saturated primary; beyond it the distance
/// saturates at 1.0 and stops carrying information.
const LAB_SCALE: f32 = 50.0;

pub struct ColorKmeansEngine {
    /// Number of dominant colours kept.
    k: usize,
    /// Side of the square grid the frame is averaged down to before
    /// clustering. Small on purpose: it is the blur.
    grid: usize,
    iterations: usize,
}

impl Default for ColorKmeansEngine {
    fn default() -> Self {
        Self { k: 5, grid: 48, iterations: 12 }
    }
}

impl ColorKmeansEngine {
    pub fn new(k: usize, grid: usize, iterations: usize) -> Self {
        Self { k: k.max(1), grid: grid.max(2), iterations: iterations.max(1) }
    }
}

impl SessionEngine for ColorKmeansEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        format!("k={} on a {}×{} Lab grid, {} Lloyd iterations", self.k, self.grid, self.grid, self.iterations)
    }

    fn default_cut(&self) -> f32 {
        0.22
    }

    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature> {
        let samples: Vec<[f32; 3]> = downsample(frame.rgb, self.grid, self.grid)
            .into_iter()
            .map(srgb_to_lab)
            .collect();
        let mut clusters = kmeans(&samples, self.k, self.iterations);
        // Heaviest first, so the distance's greedy matching sees the
        // colours that actually carry the picture before the specks, and
        // so two signatures list their colours in comparable order.
        clusters.sort_by(|a, b| b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal));

        let mut values = Vec::with_capacity(self.k * 4);
        for c in clusters {
            values.extend_from_slice(&[c.center[0], c.center[1], c.center[2], c.weight]);
        }
        Ok(Signature::new(NAME, values))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() {
            return 1.0;
        }
        // Symmetric weighted nearest-centroid: how far every colour in one
        // palette has to travel to find itself in the other, weighted by
        // how much of the picture it covers. Symmetrised because a palette
        // with an extra colour must read as different from both sides.
        let one_way = |from: &[f32], to: &[f32]| -> f32 {
            let mut acc = 0.0;
            let mut mass = 0.0;
            for src in from.chunks_exact(4) {
                let (color, weight) = ([src[0], src[1], src[2]], src[3]);
                if weight <= 0.0 {
                    continue;
                }
                let nearest = to
                    .chunks_exact(4)
                    .filter(|dst| dst[3] > 0.0)
                    .map(|dst| lab_distance(color, [dst[0], dst[1], dst[2]]))
                    .fold(f32::INFINITY, f32::min);
                if nearest.is_finite() {
                    acc += weight * nearest;
                    mass += weight;
                }
            }
            if mass > 0.0 {
                acc / mass
            } else {
                0.0
            }
        };

        let (a, b) = (a.values(), b.values());
        let mean = 0.5 * (one_way(a, b) + one_way(b, a));
        (mean / LAB_SCALE).clamp(0.0, 1.0)
    }
}

// ── k-means ───────────────────────────────────────────────────────

struct Cluster {
    center: [f32; 3],
    /// Share of the sampled pixels in this cluster, `0.0..=1.0`.
    weight: f32,
}

fn kmeans(samples: &[[f32; 3]], k: usize, iterations: usize) -> Vec<Cluster> {
    if samples.is_empty() {
        return (0..k).map(|_| Cluster { center: [0.0; 3], weight: 0.0 }).collect();
    }

    let mut centers = kmeans_plus_plus(samples, k);
    let mut assignment = vec![0usize; samples.len()];
    for _ in 0..iterations {
        let mut moved = false;
        for (i, s) in samples.iter().enumerate() {
            let best = nearest(*s, &centers);
            if best != assignment[i] {
                assignment[i] = best;
                moved = true;
            }
        }
        let mut sums = vec![[0f32; 3]; centers.len()];
        let mut counts = vec![0f32; centers.len()];
        for (s, &c) in samples.iter().zip(&assignment) {
            sums[c][0] += s[0];
            sums[c][1] += s[1];
            sums[c][2] += s[2];
            counts[c] += 1.0;
        }
        for (i, (sum, n)) in sums.iter().zip(&counts).enumerate() {
            if *n > 0.0 {
                centers[i] = [sum[0] / n, sum[1] / n, sum[2] / n];
            }
        }
        // Converged: nothing changed hands, so further iterations are
        // identical. A whole card's worth of these adds up.
        if !moved {
            break;
        }
    }

    let mut counts = vec![0f32; centers.len()];
    for s in samples {
        counts[nearest(*s, &centers)] += 1.0;
    }
    let total = samples.len() as f32;
    centers
        .into_iter()
        .zip(counts)
        .map(|(center, n)| Cluster { center, weight: n / total })
        // An empty cluster is a real outcome (a picture with three colours
        // in it), and weight 0 is how the distance is told to ignore it.
        .collect()
}

/// k-means++ seeding off a fixed-seed LCG, so the same frame always yields
/// the same signature.
fn kmeans_plus_plus(samples: &[[f32; 3]], k: usize) -> Vec<[f32; 3]> {
    let mut rng = Lcg::new(0x9E37_79B9);
    let mut centers = vec![samples[rng.below(samples.len())]];
    let mut best: Vec<f32> = samples.iter().map(|s| squared(*s, centers[0])).collect();

    while centers.len() < k {
        let total: f32 = best.iter().sum();
        let pick = if total <= f32::EPSILON {
            // Every sample already sits on a centre — the picture has
            // fewer distinct colours than `k`. Spread the rest evenly so
            // the outcome stays deterministic instead of arbitrary.
            (centers.len() * samples.len() / k).min(samples.len() - 1)
        } else {
            let target = rng.unit() * total;
            let mut acc = 0.0;
            let mut chosen = samples.len() - 1;
            for (i, d) in best.iter().enumerate() {
                acc += d;
                if acc >= target {
                    chosen = i;
                    break;
                }
            }
            chosen
        };
        let center = samples[pick];
        for (b, s) in best.iter_mut().zip(samples) {
            *b = b.min(squared(*s, center));
        }
        centers.push(center);
    }
    centers
}

fn nearest(sample: [f32; 3], centers: &[[f32; 3]]) -> usize {
    let mut best = 0;
    let mut best_d = f32::INFINITY;
    for (i, c) in centers.iter().enumerate() {
        let d = squared(sample, *c);
        if d < best_d {
            best_d = d;
            best = i;
        }
    }
    best
}

fn squared(a: [f32; 3], b: [f32; 3]) -> f32 {
    let (dl, da, db) = (a[0] - b[0], a[1] - b[1], a[2] - b[2]);
    dl * dl + da * da + db * db
}

fn lab_distance(a: [f32; 3], b: [f32; 3]) -> f32 {
    squared(a, b).sqrt()
}

/// A 64-bit linear congruential generator (Knuth's MMIX constants).
///
/// Not for anything that needs real randomness — this exists to make
/// k-means++ reproducible, which a real RNG would defeat.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }

    fn below(&mut self, n: usize) -> usize {
        (self.next() >> 33) as usize % n.max(1)
    }
}

// ── sRGB → CIELAB ─────────────────────────────────────────────────

/// D65 sRGB to CIELAB. Input channels are `0.0..=255.0`.
fn srgb_to_lab(rgb: [f32; 3]) -> [f32; 3] {
    let lin = |c: f32| {
        let c = (c / 255.0).clamp(0.0, 1.0);
        if c <= 0.04045 {
            c / 12.92
        } else {
            ((c + 0.055) / 1.055).powf(2.4)
        }
    };
    let (r, g, b) = (lin(rgb[0]), lin(rgb[1]), lin(rgb[2]));

    // sRGB D65 matrix, then normalise by the D65 white point.
    let x = (0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047;
    let y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b;
    let z = (0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883;

    let f = |t: f32| {
        const DELTA: f32 = 6.0 / 29.0;
        if t > DELTA * DELTA * DELTA {
            t.cbrt()
        } else {
            t / (3.0 * DELTA * DELTA) + 4.0 / 29.0
        }
    };
    let (fx, fy, fz) = (f(x), f(y), f(z));
    [116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn solid(color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(64, 64, image::Rgb(color))
    }

    /// Left half one colour, right half another.
    fn split(left: [u8; 3], right: [u8; 3]) -> RgbImage {
        RgbImage::from_fn(64, 64, |x, _| image::Rgb(if x < 32 { left } else { right }))
    }

    #[test]
    fn identical_frames_are_distance_zero() {
        let mut e = ColorKmeansEngine::default();
        let a = e.signature(&Frame::new(&solid([200, 40, 40]), None)).unwrap();
        let b = e.signature(&Frame::new(&solid([200, 40, 40]), None)).unwrap();
        assert!(e.distance(&a, &b) < 1e-4, "got {}", e.distance(&a, &b));
    }

    #[test]
    fn the_same_frame_always_signs_the_same() {
        // The fixed-seed LCG is what this is protecting: a signature that
        // varied run to run would make a cached one disagree with a fresh
        // one.
        let img = split([10, 120, 200], [220, 210, 30]);
        let mut e = ColorKmeansEngine::default();
        assert_eq!(e.signature(&Frame::new(&img, None)).unwrap(), e.signature(&Frame::new(&img, None)).unwrap());
    }

    #[test]
    fn different_palettes_are_far_apart() {
        let mut e = ColorKmeansEngine::default();
        let red = e.signature(&Frame::new(&solid([220, 20, 20]), None)).unwrap();
        let blue = e.signature(&Frame::new(&solid([20, 20, 220]), None)).unwrap();
        assert!(e.distance(&red, &blue) > 0.5, "got {}", e.distance(&red, &blue));
    }

    #[test]
    fn moving_a_colour_around_the_frame_does_not_change_the_palette() {
        // The defining property of this engine — and its blind spot. Same
        // two colours, opposite sides.
        let mut e = ColorKmeansEngine::default();
        let a = e.signature(&Frame::new(&split([200, 30, 30], [30, 30, 200]), None)).unwrap();
        let b = e.signature(&Frame::new(&split([30, 30, 200], [200, 30, 30]), None)).unwrap();
        assert!(e.distance(&a, &b) < 0.05, "got {}", e.distance(&a, &b));
    }

    #[test]
    fn distance_is_symmetric() {
        let mut e = ColorKmeansEngine::default();
        let a = e.signature(&Frame::new(&solid([180, 90, 20]), None)).unwrap();
        let b = e.signature(&Frame::new(&split([10, 10, 10], [240, 240, 240]), None)).unwrap();
        assert!((e.distance(&a, &b) - e.distance(&b, &a)).abs() < 1e-6);
    }

    #[test]
    fn lab_of_white_and_black_are_the_ends_of_the_l_axis() {
        let white = srgb_to_lab([255.0, 255.0, 255.0]);
        let black = srgb_to_lab([0.0, 0.0, 0.0]);
        assert!((white[0] - 100.0).abs() < 0.5, "L* of white was {}", white[0]);
        assert!(black[0].abs() < 0.5, "L* of black was {}", black[0]);
        assert!(white[1].abs() < 0.5 && white[2].abs() < 0.5, "white should be neutral");
    }

    #[test]
    fn a_mismatched_engine_reads_as_unrelated() {
        let mut e = ColorKmeansEngine::default();
        let mine = e.signature(&Frame::new(&solid([10, 10, 10]), None)).unwrap();
        let theirs = Signature::new("something-else", vec![0.0; 20]);
        assert_eq!(e.distance(&mine, &theirs), 1.0);
    }
}
