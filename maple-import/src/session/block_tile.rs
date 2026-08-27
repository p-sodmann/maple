//! Engine 3 — how much of the frame held still.
//!
//! Built for the exact case that motivates all of this: twenty pictures of
//! one child in one room. Across those frames **most of the picture does
//! not change** — same wall, same sofa, same light — while a small region
//! moves. A single blended distance cannot tell that apart from "everything
//! shifted a little", so this engine does not blend: it reduces the frame
//! to a small contrast-normalised grey tile and counts the **fraction of
//! blocks that stayed put**.
//!
//! - Same scene, child moved → most blocks stable, a contiguous minority
//!   changed. Small distance.
//! - New room, or turned around → almost nothing lines up. Large distance.
//!
//! Contrast normalisation (subtract the mean, divide by the deviation) is
//! what makes it survive the exposure changing between frames, which is
//! otherwise the thing that would break a same-scene run. The cost is that
//! it is blind to colour entirely — a red wall and a green wall of the same
//! brightness are one tile. [`super::ColorKmeansEngine`] is the counterpart
//! that sees only colour.

use super::{Frame, SessionEngine, Signature, downsample, luma};

const NAME: &str = "block-tile";

pub struct BlockTileEngine {
    /// Blocks per side.
    tile: usize,
    /// How far a block may move, in standard deviations of its own frame,
    /// and still count as unchanged.
    tolerance: f32,
}

impl Default for BlockTileEngine {
    fn default() -> Self {
        Self { tile: 16, tolerance: 0.4 }
    }
}

impl BlockTileEngine {
    /// This engine's spec name, as `engine_from_spec` and settings.toml
    /// spell it.
    pub const NAME: &'static str = NAME;

    pub fn new(tile: usize, tolerance: f32) -> Self {
        Self { tile: tile.max(2), tolerance: tolerance.max(0.0) }
    }
}

impl SessionEngine for BlockTileEngine {
    fn name(&self) -> &'static str {
        NAME
    }

    fn describe(&self) -> String {
        format!("{t}×{t} contrast-normalised grey blocks, ±{tol}σ counts as unchanged", t = self.tile, tol = self.tolerance)
    }

    fn default_cut(&self) -> f32 {
        // A third of the frame changing is already a lot for one subject
        // moving inside a fixed scene.
        0.35
    }

    fn signature(&mut self, frame: &Frame<'_>) -> anyhow::Result<Signature> {
        let mut values: Vec<f32> =
            downsample(frame.rgb, self.tile, self.tile).into_iter().map(luma).collect();

        let n = values.len() as f32;
        let mean = values.iter().sum::<f32>() / n;
        let variance = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
        // A flat frame has no deviation to divide by; leaving it at zero
        // would turn rounding noise into structure, so it stays flat.
        let scale = variance.sqrt();
        let inv = if scale > 1e-3 { 1.0 / scale } else { 0.0 };
        for v in values.iter_mut() {
            *v = (*v - mean) * inv;
        }
        Ok(Signature::new(NAME, values))
    }

    fn distance(&self, a: &Signature, b: &Signature) -> f32 {
        if a.engine() != b.engine() || a.values().len() != b.values().len() || a.values().is_empty() {
            return 1.0;
        }
        let stable = a
            .values()
            .iter()
            .zip(b.values())
            .filter(|(x, y)| (*x - *y).abs() <= self.tolerance)
            .count();
        1.0 - stable as f32 / a.values().len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    /// A background of vertical stripes with a solid `subject` block
    /// occupying the given fraction of the width.
    fn scene(subject: [u8; 3], subject_frac: f32, stripe: u8) -> RgbImage {
        let cut = (64.0 * subject_frac) as u32;
        RgbImage::from_fn(64, 64, |x, y| {
            if x < cut {
                image::Rgb(subject)
            } else if (y / 4) % 2 == 0 {
                image::Rgb([stripe, stripe, stripe])
            } else {
                image::Rgb([255 - stripe, 255 - stripe, 255 - stripe])
            }
        })
    }

    #[test]
    fn identical_frames_are_distance_zero() {
        let mut e = BlockTileEngine::default();
        let a = e.signature(&Frame::new(&scene([200, 40, 40], 0.2, 40), None)).unwrap();
        let b = e.signature(&Frame::new(&scene([200, 40, 40], 0.2, 40), None)).unwrap();
        assert_eq!(e.distance(&a, &b), 0.0);
    }

    #[test]
    fn a_subject_moving_inside_a_held_scene_is_a_small_distance() {
        // The case the engine exists for: same stripes, the subject grew
        // from a fifth of the frame to a third. Most blocks are untouched.
        let mut e = BlockTileEngine::default();
        let a = e.signature(&Frame::new(&scene([200, 40, 40], 0.2, 40), None)).unwrap();
        let b = e.signature(&Frame::new(&scene([200, 40, 40], 0.33, 40), None)).unwrap();
        let d = e.distance(&a, &b);
        assert!(d < e.default_cut(), "subject motion read as a scene change: {d}");
    }

    #[test]
    fn a_different_scene_is_a_large_distance() {
        let mut e = BlockTileEngine::default();
        let stripes = e.signature(&Frame::new(&scene([200, 40, 40], 0.2, 40), None)).unwrap();
        // Horizontal bands instead of vertical stripes — nothing lines up.
        let bands = RgbImage::from_fn(64, 64, |x, _| {
            let v = if (x / 4) % 2 == 0 { 20 } else { 230 };
            image::Rgb([v, v, v])
        });
        let other = e.signature(&Frame::new(&bands, None)).unwrap();
        assert!(e.distance(&stripes, &other) > e.default_cut(), "got {}", e.distance(&stripes, &other));
    }

    #[test]
    fn an_exposure_change_alone_does_not_move_it() {
        // Contrast normalisation earning its place: the same scene, one
        // stop darker, must still be the same scene.
        let mut e = BlockTileEngine::default();
        let bright = e.signature(&Frame::new(&scene([200, 40, 40], 0.2, 40), None)).unwrap();
        let dim = {
            let img = scene([200, 40, 40], 0.2, 40);
            let dark = RgbImage::from_fn(64, 64, |x, y| {
                let p = img.get_pixel(x, y).0;
                image::Rgb([p[0] / 2, p[1] / 2, p[2] / 2])
            });
            e.signature(&Frame::new(&dark, None)).unwrap()
        };
        assert!(e.distance(&bright, &dim) < 0.1, "got {}", e.distance(&bright, &dim));
    }

    #[test]
    fn a_flat_frame_stays_flat_instead_of_amplifying_noise() {
        let mut e = BlockTileEngine::default();
        let grey = e.signature(&Frame::new(&RgbImage::from_pixel(64, 64, image::Rgb([128, 128, 128])), None)).unwrap();
        assert!(grey.values().iter().all(|v| *v == 0.0));
    }

    #[test]
    fn distance_is_symmetric_and_bounded() {
        let mut e = BlockTileEngine::default();
        let a = e.signature(&Frame::new(&scene([10, 10, 10], 0.5, 0), None)).unwrap();
        let b = e.signature(&Frame::new(&scene([250, 250, 250], 0.1, 200), None)).unwrap();
        let d = e.distance(&a, &b);
        assert!((d - e.distance(&b, &a)).abs() < 1e-6);
        assert!((0.0..=1.0).contains(&d));
    }
}
