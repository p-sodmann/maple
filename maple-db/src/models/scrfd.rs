//! SCRFD face detector backed by `ort` (ONNX Runtime).
//!
//! Implements the standard InsightFace SCRFD architecture
//! (e.g. `scrfd_10g_bnkps.onnx`) via the [`DetectionModel`] trait so it is
//! a drop-in replacement for [`OnnxFaceDetector`].
//!
//! # Model I/O
//!
//! - Input  `images`      : `[1, 3, 640, 640]` float32 — BGR, normalised
//!   `(pixel − 127.5) / 128.0` — letterboxed with black padding.
//! - Outputs (9 total, 3 strides × `[scores, boxes, keypoints]`):
//!   - `score_8`,  `score_16`,  `score_32`   : `[N, 1]` — confidence
//!   - `bbox_8`,   `bbox_16`,   `bbox_32`    : `[N, 4]` — distance offsets
//!   - `kps_8`,    `kps_16`,    `kps_32`     : `[N, 10]` — 5-point landmarks
//!
//! # Embedding support
//!
//! Unlike the atksh model, SCRFD does not produce pre-aligned face crops.
//! When an [`EmbeddingModel`] is configured, crops are extracted manually
//! from the source image (resized to 112 × 112) before being passed to the
//! embedder.
//!
//! [`OnnxFaceDetector`]: super::detection::OnnxFaceDetector

use std::io::Cursor;
use std::path::{Path, PathBuf};

use anyhow::{bail, Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader, RgbImage};
use maple_import::{is_raw_format, loadable_image_bytes};
use ndarray::{Array2, Array4, Axis};
use tracing::{debug, info, warn};

use super::{detection::DetectionModel, device::ModelDevice, embedding::EmbeddingModel, session::OnnxSession};
use crate::face_detector::DetectedFace;

// ── Constants ──────────────────────────────────────────────────────────────────

const IH: usize = 640;
const IW: usize = 640;
const KPS: usize = 5;

// ── ScrfdDetector ──────────────────────────────────────────────────────────────

/// ONNX-backed SCRFD face detector.
///
/// Optionally holds an [`EmbeddingModel`] for ArcFace identity embeddings.
pub struct ScrfdDetector {
    session: OnnxSession,
    embedder: Option<Box<dyn EmbeddingModel>>,
    fmc: usize,
    num_anchors: usize,
    strides: [usize; 3],
    /// When `Some`, 112×112 face crops are saved here as PNGs.
    debug_dir: Option<PathBuf>,
}

impl ScrfdDetector {
    /// Load a SCRFD model from `path`, targeting `device`.
    ///
    /// `debug_dir`: when `Some`, 112×112 face crop PNGs are written there.
    pub fn load(
        path: &Path,
        device: &ModelDevice,
        embedder: Option<Box<dyn EmbeddingModel>>,
        debug_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let session = OnnxSession::load(path, device)
            .with_context(|| format!("loading SCRFD detector: {}", path.display()))?;

        let num_outputs = session.output_names.len();
        let (fmc, num_anchors, strides) = match num_outputs {
            9 => (3usize, 2usize, [8usize, 16, 32]),
            _ => bail!(
                "unsupported SCRFD architecture: expected 9 outputs, got {num_outputs}"
            ),
        };

        debug!(
            inputs  = ?session.input_names,
            outputs = ?session.output_names,
            "SCRFD detector loaded"
        );

        Ok(Self { session, embedder, fmc, num_anchors, strides, debug_dir })
    }
}

impl DetectionModel for ScrfdDetector {
    fn detect(&mut self, path: &Path) -> Result<Vec<DetectedFace>> {
        let img = load_rgb(path)?;
        let (orig_w, orig_h) = (img.width(), img.height());

        let (det_scale, blob) = preprocess(&img)?;

        let input_name = &self.session.input_names[0];
        let tensor = ort::value::TensorRef::from_array_view(blob.view())
            .context("creating SCRFD input tensor")?;
        let outputs = self
            .session
            .session
            .run(ort::inputs![input_name.as_str() => tensor])
            .context("running SCRFD detector")?;

        const SCORE_THRESH: f32 = 0.4;
        const IOU_THRESH: f32 = 0.5;

        // ── Per-stride decode ──────────────────────────────────────────────
        let mut all_scores: Vec<Array2<f32>> = Vec::new();
        let mut all_boxes: Vec<Array2<f32>> = Vec::new();
        let mut all_kps: Vec<Array2<f32>> = Vec::new();

        for (idx, &stride) in self.strides.iter().enumerate() {
            let aw = IW / stride;
            let ah = IH / stride;
            let n = aw * ah * self.num_anchors;

            let anchors = anchor_centers(aw, ah, stride, self.num_anchors);

            // Output layout: scores[0..fmc], boxes[fmc..2*fmc], kps[2*fmc..3*fmc]
            let scores_name = &self.session.output_names[idx];
            let boxes_name  = &self.session.output_names[idx + self.fmc];
            let kps_name    = &self.session.output_names[idx + 2 * self.fmc];

            let (_, scores_raw) = outputs[scores_name.as_str()]
                .try_extract_tensor::<f32>()
                .with_context(|| format!("SCRFD scores stride {stride}"))?;
            let (_, boxes_raw) = outputs[boxes_name.as_str()]
                .try_extract_tensor::<f32>()
                .with_context(|| format!("SCRFD boxes stride {stride}"))?;
            let (_, kps_raw) = outputs[kps_name.as_str()]
                .try_extract_tensor::<f32>()
                .with_context(|| format!("SCRFD keypoints stride {stride}"))?;

            // Flatten batch dim (shape may be [N,1] or [1,N,1]).
            let scores_flat: Vec<f32> = scores_raw.iter().copied().collect();
            let boxes_flat: Vec<f32> = boxes_raw.iter().map(|&v| v * stride as f32).collect();
            let kps_flat: Vec<f32> = kps_raw.iter().map(|&v| v * stride as f32).collect();

            let scores = Array2::from_shape_vec((n, 1), scores_flat)
                .with_context(|| format!("reshaping scores stride {stride}"))?;
            let boxes_dist = Array2::from_shape_vec((n, 4), boxes_flat)
                .with_context(|| format!("reshaping boxes stride {stride}"))?;
            let kps_dist = Array2::from_shape_vec((n, 2 * KPS), kps_flat)
                .with_context(|| format!("reshaping keypoints stride {stride}"))?;

            let boxes_decoded = distance2boxes(&anchors, &boxes_dist);
            let kps_decoded = distance2kps(&anchors, &kps_dist);

            let likely: Vec<usize> = scores
                .iter()
                .enumerate()
                .filter(|(_, &s)| s >= SCORE_THRESH)
                .map(|(i, _)| i)
                .collect();

            if !likely.is_empty() {
                all_scores.push(scores.select(Axis(0), &likely));
                all_boxes.push(boxes_decoded.select(Axis(0), &likely));
                all_kps.push(kps_decoded.select(Axis(0), &likely));
            }
        }

        if all_scores.is_empty() {
            return Ok(vec![]);
        }

        // ── Concatenate strides ────────────────────────────────────────────
        let scores_views: Vec<_> = all_scores.iter().map(|a| a.view()).collect();
        let boxes_views: Vec<_> = all_boxes.iter().map(|a| a.view()).collect();
        let kps_views: Vec<_> = all_kps.iter().map(|a| a.view()).collect();

        let scores = ndarray::concatenate(Axis(0), &scores_views).context("concat scores")?;
        let boxes = ndarray::concatenate(Axis(0), &boxes_views).context("concat boxes")?;
        let kps = ndarray::concatenate(Axis(0), &kps_views).context("concat keypoints")?;

        // Scale pixel coords from 640×640 letterbox back to original dimensions.
        let boxes = boxes.mapv(|v| v * det_scale);
        let kps = kps.mapv(|v| v * det_scale);
        let _ = kps; // keypoints stored but not yet surfaced in DetectedFace

        // ── Sort by score descending, then NMS ────────────────────────────
        let scores_flat: Vec<f32> = scores.iter().copied().collect();
        let order = argsort_desc(&scores_flat);

        let boxes_sorted = boxes.select(Axis(0), &order);
        let scores_sorted = scores.select(Axis(0), &order);

        let keep = nms(&boxes_sorted, IOU_THRESH);

        let boxes_final = boxes_sorted.select(Axis(0), &keep);
        let scores_final = scores_sorted.select(Axis(0), &keep);

        info!(
            orig   = %format!("{orig_w}x{orig_h}"),
            scale  = det_scale,
            faces  = keep.len(),
            "SCRFD detector I/O"
        );

        // ── Build DetectedFace results ─────────────────────────────────────
        let mut result = Vec::with_capacity(keep.len());
        for i in 0..keep.len() {
            let conf = scores_final[[i, 0]];
            let x1 = (boxes_final[[i, 0]] / orig_w as f32).clamp(0.0, 1.0);
            let y1 = (boxes_final[[i, 1]] / orig_h as f32).clamp(0.0, 1.0);
            let x2 = (boxes_final[[i, 2]] / orig_w as f32).clamp(0.0, 1.0);
            let y2 = (boxes_final[[i, 3]] / orig_h as f32).clamp(0.0, 1.0);

            // ── Crop + resize to 112×112 (for debug saving and/or embedding) ──
            let need_crop = self.embedder.is_some() || self.debug_dir.is_some();
            let resized_crop = if need_crop {
                let px1 = (x1 * orig_w as f32) as u32;
                let py1 = (y1 * orig_h as f32) as u32;
                let pw = ((x2 - x1) * orig_w as f32).max(1.0) as u32;
                let ph = ((y2 - y1) * orig_h as f32).max(1.0) as u32;
                let cropped = image::imageops::crop_imm(&img, px1, py1, pw, ph).to_image();
                Some(image::imageops::resize(
                    &cropped,
                    112,
                    112,
                    image::imageops::FilterType::Triangle,
                ))
            } else {
                None
            };

            // ── Debug: save the 112×112 RGB crop as PNG ────────────────────
            if let (Some(ref dir), Some(ref crop_img)) = (&self.debug_dir, &resized_crop) {
                if let Err(e) = std::fs::create_dir_all(dir) {
                    warn!(dir = %dir.display(), "could not create aligned_faces dir: {e}");
                } else {
                    let stem =
                        path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
                    let file = dir.join(format!("{stem}_face_{i}.png"));
                    if let Err(e) = crop_img.save(&file) {
                        warn!(file = %file.display(), "could not save face crop: {e}");
                    }
                }
            }

            // ── Optional ArcFace embedding ─────────────────────────────────
            let embedding = if let (Some(ref mut embedder), Some(ref crop_img)) =
                (&mut self.embedder, &resized_crop)
            {
                // Build [112, 112, 3] BGR f32 array for the embedder.
                let mut crop = ndarray::Array3::<f32>::zeros((112, 112, 3));
                for (x, y, p) in crop_img.enumerate_pixels() {
                    crop[[y as usize, x as usize, 0]] = p[2] as f32; // B
                    crop[[y as usize, x as usize, 1]] = p[1] as f32; // G
                    crop[[y as usize, x as usize, 2]] = p[0] as f32; // R
                }
                embedder.embed_face_crop(crop.view())?
            } else {
                vec![]
            };

            result.push(DetectedFace { bbox: [x1, y1, x2, y2], embedding, confidence: conf });
        }

        Ok(result)
    }
}

// SAFETY: ScrfdDetector holds OnnxSession (Send) + optional EmbeddingModel (Send+Sync).
unsafe impl Send for ScrfdDetector {}
unsafe impl Sync for ScrfdDetector {}

// ── Image loading ──────────────────────────────────────────────────────────────

fn load_rgb(path: &Path) -> Result<RgbImage> {
    if is_raw_format(path) {
        let bytes = loadable_image_bytes(path)?;
        let reader = ImageReader::new(Cursor::new(bytes))
            .with_guessed_format()
            .context("guessing format for raw preview")?;
        let mut decoder = reader.into_decoder().context("decoding image header")?;
        let orientation = decoder.orientation().context("reading EXIF orientation")?;
        let mut dyn_img = DynamicImage::from_decoder(decoder).context("decoding image")?;
        dyn_img.apply_orientation(orientation);
        Ok(dyn_img.to_rgb8())
    } else {
        let reader = ImageReader::open(path)
            .with_context(|| format!("opening image: {}", path.display()))?;
        let mut decoder = reader.into_decoder().context("decoding image header")?;
        let orientation = decoder.orientation().context("reading EXIF orientation")?;
        let mut dyn_img = DynamicImage::from_decoder(decoder).context("decoding image")?;
        dyn_img.apply_orientation(orientation);
        Ok(dyn_img.to_rgb8())
    }
}

// ── Preprocessing ──────────────────────────────────────────────────────────────

/// Letterbox `img` into `640×640`, returning `(det_scale, NCHW f32 blob)`.
///
/// `det_scale` converts detection-space pixel coords back to original-image
/// pixel coords: `original_pixel = detection_pixel * det_scale`.
fn preprocess(img: &RgbImage) -> Result<(f32, Array4<f32>)> {
    let img_ratio = img.height() as f32 / img.width() as f32;
    let canvas_ratio = IH as f32 / IW as f32;
    let (nw, nh) = if img_ratio > canvas_ratio {
        let nh = IH as u32;
        let nw = (nh as f32 / img_ratio).floor().max(1.0) as u32;
        (nw, nh)
    } else {
        let nw = IW as u32;
        let nh = (nw as f32 * img_ratio).floor().max(1.0) as u32;
        (nw, nh)
    };
    // Scale that converts resized coords → original coords.
    let det_scale = img.height() as f32 / nh as f32;

    let mut canvas = RgbImage::new(IW as u32, IH as u32);
    let resized = image::imageops::resize(img, nw, nh, image::imageops::FilterType::Nearest);
    image::imageops::overlay(&mut canvas, &resized, 0, 0);

    // NCHW [1, 3, IH, IW] — channel order BGR, normalised (pixel − 127.5) / 128.
    let blob = Array4::from_shape_fn((1, 3, IH, IW), |(_, c, y, x)| {
        // Map output channel c (BGR order) to the RGB pixel channel index.
        let rgb_ch = [2usize, 1, 0][c]; // c=0→B(idx 2), c=1→G(idx 1), c=2→R(idx 0)
        (canvas[(x as u32, y as u32)][rgb_ch] as f32 - 127.5) / 128.0
    });

    Ok((det_scale, blob))
}

// ── Anchor generation ──────────────────────────────────────────────────────────

/// Generate `[W*H*repeat, 2]` anchor centre coordinates for one stride level.
fn anchor_centers(w: usize, h: usize, stride: usize, repeat: usize) -> Array2<f32> {
    let n = w * h * repeat;
    let mut buf = Vec::with_capacity(n * 2);
    for y in 0..h {
        for x in 0..w {
            for _ in 0..repeat {
                buf.push((stride * x) as f32);
                buf.push((stride * y) as f32);
            }
        }
    }
    Array2::from_shape_vec((n, 2), buf).expect("anchor_centers shape invariant")
}

// ── Box / keypoint decoding ────────────────────────────────────────────────────

/// Decode distance offsets into `[x1, y1, x2, y2]` pixel boxes.
fn distance2boxes(anchors: &Array2<f32>, dist: &Array2<f32>) -> Array2<f32> {
    let n = anchors.nrows();
    let mut out = Array2::<f32>::zeros((n, 4));
    for i in 0..n {
        out[[i, 0]] = anchors[[i, 0]] - dist[[i, 0]];
        out[[i, 1]] = anchors[[i, 1]] - dist[[i, 1]];
        out[[i, 2]] = anchors[[i, 0]] + dist[[i, 2]];
        out[[i, 3]] = anchors[[i, 1]] + dist[[i, 3]];
    }
    out
}

/// Decode distance offsets into `[x0,y0, x1,y1, …]` keypoint pixel coords.
fn distance2kps(anchors: &Array2<f32>, dist: &Array2<f32>) -> Array2<f32> {
    let n = anchors.nrows();
    let mut out = Array2::<f32>::zeros((n, 2 * KPS));
    for i in 0..n {
        for k in 0..KPS {
            out[[i, 2 * k]]     = anchors[[i, 0]] + dist[[i, 2 * k]];
            out[[i, 2 * k + 1]] = anchors[[i, 1]] + dist[[i, 2 * k + 1]];
        }
    }
    out
}

// ── Post-processing ────────────────────────────────────────────────────────────

/// Return indices that sort `scores` in descending order.
fn argsort_desc(scores: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx
}

/// Non-maximum suppression.  `boxes` must be sorted by score descending.
/// Returns indices (into `boxes`) of kept detections.
fn nms(boxes: &Array2<f32>, iou_thresh: f32) -> Vec<usize> {
    let n = boxes.nrows();
    let areas: Vec<f32> = (0..n)
        .map(|i| (boxes[[i, 2]] - boxes[[i, 0]] + 1.0) * (boxes[[i, 3]] - boxes[[i, 1]] + 1.0))
        .collect();

    let mut keep = Vec::new();
    let mut order: Vec<usize> = (0..n).collect();

    while let Some(&i) = order.first() {
        keep.push(i);
        let tail = order[1..].to_vec();
        order = tail
            .into_iter()
            .filter(|&j| {
                let xx1 = boxes[[i, 0]].max(boxes[[j, 0]]);
                let yy1 = boxes[[i, 1]].max(boxes[[j, 1]]);
                let xx2 = boxes[[i, 2]].min(boxes[[j, 2]]);
                let yy2 = boxes[[i, 3]].min(boxes[[j, 3]]);
                let w = (xx2 - xx1 + 1.0).max(0.0);
                let h = (yy2 - yy1 + 1.0).max(0.0);
                let inter = w * h;
                let iou = inter / (areas[i] + areas[j] - inter);
                iou <= iou_thresh
            })
            .collect();
    }

    keep
}
