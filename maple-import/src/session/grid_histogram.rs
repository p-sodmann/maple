//! Engine 2 — colour *and* where it is: a grid of local histograms.
//!
//! Split the frame into a coarse grid and give every cell its own small
//! per-channel histogram. Two frames match when each part of one looks like
//! the same part of the other — so unlike [`super::ColorKmeansEngine`],
//! turning around in the same room reads as a change even when the palette
//! is identical.
//!
//! Two details do most of the work:
//!
//! - **Bins are filled with linear interpolation**, splitting each pixel
//!   between the two bins it falls between. With 8 bins a hard assignment
//!   would move a whole bin's worth of mass across a boundary for a one-stop
//!   exposure nudge, and report a scene change where a cloud passed.
//! - **The grid is coarse.** 4×4 cells is enough to say "the bright part is
//!   top-left" without tracking a child across the frame — the subject
//!   moving between two cells should not end a session.

use super::{Frame, SessionEngine, Signature, downsample};

const NAME: &str = "grid-histogram";

pub struct GridHistogramEngine {
    /// Cells per side.
    grid: usize,
    /// Histogram bins per channel, per cell.
    bins: usize,
    /// Pixels per side sampled from each cell.
    cell_px: usize,
}

impl Default for GridHistogramEngine {
    fn default() -> Self {
        Self { grid: 4, bins: 8, cell_px: 8 }
    }
}

impl GridHistogramEngine {
    pub fn new(grid: usize, bins: usize, cell_px: usize) -> Self {
        Self { grid: grid.max(1), bins: bins.max(2), cell_px: cell_px.max(1) }
    }

    fn cell_stride(&self) -> usize {
        self.bins * 3
    }
}

impl SessionEngine for GridHistogramEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        format!(
            "{g}×{g} cells of {b} interpolated bins per channel, sampled at {p}×{p} per cell",
            g = self.grid,
            b = self.bins,
            p = self.cell_px
        )
    }

    fn default_cut(&self) -> f32 {
        0.30
    }

    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature> {
        let side = self.grid * self.cell_px;
        let pixels = downsample(frame.rgb, side, side);
        let stride = self.cell_stride();
        let mut values = vec![0f32; self.grid * self.grid * stride];

        for y in 0..side {
            let cy = y / self.cell_px;
            for x in 0..side {
                let cell = (cy * self.grid + x / self.cell_px) * stride;
                let px = pixels[y * side + x];
                for (ch, value) in px.iter().enumerate() {
                    let base = cell + ch * self.bins;
                    // Position on the bin axis, then split between the two
                    // nearest bin centres.
                    let pos = (value / 255.0).clamp(0.0, 1.0) * (self.bins - 1) as f32;
                    let low = pos.floor() as usize;
                    let frac = pos - low as f32;
                    values[base + low] += 1.0 - frac;
                    if low + 1 < self.bins {
                        values[base + low + 1] += frac;
                    }
                }
            }
        }

        // Normalise every channel histogram to sum 1, so the distance below is
        // bounded and a cell's own brightness scale drops out.
        for hist in values.chunks_exact_mut(self.bins) {
            let total: f32 = hist.iter().sum();
            if total > 0.0 {
                for v in hist.iter_mut() {
                    *v /= total;
                }
            }
        }
        Ok(Signature::new(NAME, values))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() || a.values().len() != b.values().len() {
            return 1.0;
        }
        // Half the L1 distance between two distributions is the total
        // variation distance: 0 when identical, 1 when disjoint. Averaging
        // it over every cell and channel keeps the whole result in 0..=1.
        let mut acc = 0.0;
        let mut n = 0usize;
        for (ha, hb) in a.values().chunks_exact(self.bins).zip(b.values().chunks_exact(self.bins)) {
            let l1: f32 = ha.iter().zip(hb).map(|(x, y)| (x - y).abs()).sum();
            acc += 0.5 * l1;
            n += 1;
        }
        if n == 0 {
            return 1.0;
        }
        (acc / n as f32).clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    fn solid(color: [u8; 3]) -> RgbImage {
        RgbImage::from_pixel(64, 64, image::Rgb(color))
    }

    fn split(left: [u8; 3], right: [u8; 3]) -> RgbImage {
        RgbImage::from_fn(64, 64, |x, _| image::Rgb(if x < 32 { left } else { right }))
    }

    #[test]
    fn identical_frames_are_distance_zero() {
        let mut e = GridHistogramEngine::default();
        let a = e.signature(&Frame::new(&split([200, 40, 40], [20, 60, 200]), None)).unwrap();
        let b = e.signature(&Frame::new(&split([200, 40, 40], [20, 60, 200]), None)).unwrap();
        assert!(e.distance(&a, &b) < 1e-5);
    }

    #[test]
    fn layout_matters_here_unlike_the_palette_engine() {
        // The same two colours, swapped sides. `ColorKmeansEngine` calls
        // these identical; this one must not.
        let mut e = GridHistogramEngine::default();
        let a = e.signature(&Frame::new(&split([200, 30, 30], [30, 30, 200]), None)).unwrap();
        let b = e.signature(&Frame::new(&split([30, 30, 200], [200, 30, 30]), None)).unwrap();
        assert!(e.distance(&a, &b) > 0.5, "got {}", e.distance(&a, &b));
    }

    #[test]
    fn a_small_exposure_shift_is_a_small_distance() {
        // What the interpolated binning buys: a nudge in brightness moves
        // a sliver of mass, not a whole bin.
        let mut e = GridHistogramEngine::default();
        let a = e.signature(&Frame::new(&solid([120, 120, 120]), None)).unwrap();
        let b = e.signature(&Frame::new(&solid([128, 128, 128]), None)).unwrap();
        let d = e.distance(&a, &b);
        assert!(d < 0.35, "an 8/255 exposure nudge read as {d}");

        let far = e.signature(&Frame::new(&solid([240, 240, 240]), None)).unwrap();
        assert!(e.distance(&a, &far) > d, "a real change should still read larger");
    }

    #[test]
    fn distance_is_symmetric_and_bounded() {
        let mut e = GridHistogramEngine::default();
        let a = e.signature(&Frame::new(&solid([0, 0, 0]), None)).unwrap();
        let b = e.signature(&Frame::new(&solid([255, 255, 255]), None)).unwrap();
        let d = e.distance(&a, &b);
        assert!((d - e.distance(&b, &a)).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&d), "got {d}");
        assert!(d > 0.9, "black against white should be near the top: {d}");
    }

    #[test]
    fn a_mismatched_signature_reads_as_unrelated() {
        let mut e = GridHistogramEngine::default();
        let mine = e.signature(&Frame::new(&solid([10, 10, 10]), None)).unwrap();
        assert_eq!(e.distance(&mine, &Signature::new(NAME, vec![0.0; 3])), 1.0);
    }
}
