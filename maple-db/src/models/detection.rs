//! Detection model trait and atksh-based ONNX face detector.
//!
//! # Model I/O (atksh joined model)
//!
//! - Input  `input`           : `[H, W, 3]` uint8   — BGR pixel values `[0, 255]`
//! - Output 0 `scores`        : `[N]`        f32    — confidence per face
//! - Output 1 `bboxes`        : `[N, 4]`     i64    — `[x1, y1, x2, y2]` in pixels
//! - Output 2 `keypoints`     : `[N, 5, 2]`         — 5 facial landmarks (pixels)
//! - Output 3 `aligned_imgs`  : `[N, 112, 112, 3]` u8 — pre-aligned BGR `[0, 255]` face crops
//! - Output 4 `landmarks`     : `[N, *, 2]`
//! - Output 5 `affine_matrices`: `[N, 2, 3]`
//!
//! `aligned_imgs[i]` is fed directly to the optional ArcFace embedder —
//! no manual cropping or resizing is required.
//!
//! Download: <https://github.com/atksh/onnx-facial-lmk-detector/releases>

use std::path::{Path, PathBuf};

use std::io::Cursor;

use anyhow::{Context, Result};
use image::{DynamicImage, ImageDecoder, ImageReader};
use maple_import::{is_raw_format, loadable_image_bytes};
use ndarray::{Array3, Array4, ArrayView2, ArrayView4};
use tracing::{debug, info, warn};

use super::{
    device::ModelDevice,
    embedding::EmbeddingModel,
    session::OnnxSession,
};
use crate::face_detector::DetectedFace;

// ── Trait ──────────────────────────────────────────────────────────────────────

/// Detects faces in an image and returns one [`DetectedFace`] per person.
///
/// Implementations must be `Send + Sync` so they can be moved into background
/// threads or shared behind `Arc`.
pub trait DetectionModel: Send + Sync {
    /// Run detection (and optional embedding) on the image at `path`.
    fn detect(&mut self, path: &Path) -> Result<Vec<DetectedFace>>;
}

// ── OnnxFaceDetector ──────────────────────────────────────────────────────────

/// ONNX-backed face detector using the atksh joined model.
///
/// Optionally holds an [`EmbeddingModel`] (typically [`OnnxFaceEmbedder`]).
/// When an embedder is present, `aligned_imgs` from the detector are fed
/// through it to produce 512-dim ArcFace identity embeddings.
pub struct OnnxFaceDetector {
    /// atksh joined detector: SCRFD detection + landmark alignment.
    pub(super) detector: OnnxSession,
    /// Optional ArcFace embedder for per-face identity vectors.
    pub(super) embedder: Option<Box<dyn EmbeddingModel>>,
    /// When `Some`, aligned 112×112 face crops are saved here as PNGs.
    debug_dir: Option<PathBuf>,
}

impl OnnxFaceDetector {
    /// Create from a pre-loaded `OnnxSession` and optional embedder.
    pub fn new(
        detector: OnnxSession,
        embedder: Option<Box<dyn EmbeddingModel>>,
        debug_dir: Option<PathBuf>,
    ) -> Self {
        debug!(
            inputs  = ?detector.input_names,
            outputs = ?detector.output_names,
            "atksh detector loaded"
        );
        Self { detector, embedder, debug_dir }
    }

    /// Load the atksh detector directly from a path.
    pub fn load(
        path: &Path,
        device: &ModelDevice,
        embedder: Option<Box<dyn EmbeddingModel>>,
        debug_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let session = OnnxSession::load(path, device)
            .with_context(|| format!("loading face detector: {}", path.display()))?;
        Ok(Self::new(session, embedder, debug_dir))
    }
}

impl DetectionModel for OnnxFaceDetector {
    fn detect(&mut self, path: &Path) -> Result<Vec<DetectedFace>> {
        const MAX_DIM: u32 = 1920;

        // ── Load image and explicitly apply EXIF orientation ───────────────
        // Keep orientation handling explicit so detector and UI use the same basis.
        // For raw files (RAF), decode from the embedded JPEG preview bytes.
        let img = if is_raw_format(path) {
            let bytes = loadable_image_bytes(path)?;
            let reader = ImageReader::new(Cursor::new(bytes))
                .with_guessed_format()
                .with_context(|| format!("guessing format for raw preview: {}", path.display()))?;
            let mut decoder = reader
                .into_decoder()
                .with_context(|| format!("decoding image header: {}", path.display()))?;
            let orientation = decoder.orientation().context("reading EXIF orientation")?;
            let mut dyn_img = DynamicImage::from_decoder(decoder).context("decoding image")?;
            dyn_img.apply_orientation(orientation);
            dyn_img.to_rgb8()
        } else {
            let reader = ImageReader::open(path)
                .with_context(|| format!("opening image: {}", path.display()))?;
            let mut decoder = reader
                .into_decoder()
                .with_context(|| format!("decoding image header: {}", path.display()))?;
            let orientation = decoder.orientation().context("reading EXIF orientation")?;
            let mut dyn_img = DynamicImage::from_decoder(decoder).context("decoding image")?;
            dyn_img.apply_orientation(orientation);
            dyn_img.to_rgb8()
        };
        let (orig_w, orig_h) = (img.width(), img.height());

        // ── Optionally resize oversized images, preserving aspect ratio ──
        let scale = if orig_w > MAX_DIM || orig_h > MAX_DIM {
            (MAX_DIM as f32 / orig_w as f32).min(MAX_DIM as f32 / orig_h as f32)
        } else {
            1.0
        };
        let new_w = (orig_w as f32 * scale).round() as u32;
        let new_h = (orig_h as f32 * scale).round() as u32;

        let resized = if scale < 1.0 {
            image::imageops::resize(&img, new_w, new_h, image::imageops::FilterType::Triangle)
        } else {
            img
        };

        // ── Build [H, W, 3] BGR uint8 array (no padding — model takes dynamic sizes) ──
        let mut img_arr = Array3::<u8>::zeros((new_h as usize, new_w as usize, 3));
        for (x, y, p) in resized.enumerate_pixels() {
            img_arr[[y as usize, x as usize, 0]] = p[2]; // B
            img_arr[[y as usize, x as usize, 1]] = p[1]; // G
            img_arr[[y as usize, x as usize, 2]] = p[0]; // R
        }

        // ── Run atksh detector ─────────────────────────────────────────────
        let input_name = &self.detector.input_names[0];
        let tensor =
            ort::value::TensorRef::from_array_view(img_arr.view()).context("creating detector input tensor")?;
        let outputs = self
            .detector
            .session
            .run(ort::inputs![input_name.as_str() => tensor])
            .context("running face detector")?;

        // Positional outputs: 0=scores, 1=bboxes, 2=keypoints, 3=aligned_imgs, …
        let scores_name = &self.detector.output_names[0];
        let bboxes_name = &self.detector.output_names[1];
        let aligned_name = &self.detector.output_names[3];

        let (scores_shape, scores_data) = outputs[scores_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("scores")?;
        let (_, bboxes_i64) = outputs[bboxes_name.as_str()]
            .try_extract_tensor::<i64>()
            .context("bboxes")?;
        let (_, aligned_u8) = outputs[aligned_name.as_str()]
            .try_extract_tensor::<u8>()
            .context("aligned_imgs")?;

        let n = scores_shape[0] as usize;
        let bboxes_data: Vec<f32> = bboxes_i64.iter().map(|&v| v as f32).collect();
        info!(
            input_shape = ?img_arr.shape(),
            orig = %format!("{orig_w}x{orig_h}"),
            resized = %format!("{new_w}x{new_h}"),
            scale,
            faces = n,
            "detector I/O"
        );
        if n > 0 {
            info!(raw_bbox_first = ?&bboxes_data[..4.min(bboxes_data.len())], "first bbox (input-tensor pixels)");
        }
        let bboxes = ArrayView2::from_shape((n, 4), &bboxes_data[..])
            .context("reshaping bboxes to [N, 4]")?;

        // atksh model outputs aligned_imgs as (N, 224, 224, 3) uint8 BGR.
        let aligned_u8_4d = ArrayView4::from_shape((n, 224, 224, 3), aligned_u8)
            .context("reshaping aligned_imgs to [N, 224, 224, 3]")?;

        // ── Debug: save aligned crops ──────────────────────────────────────
        if let Some(ref dir) = self.debug_dir {
            if let Err(e) = std::fs::create_dir_all(dir) {
                warn!(dir = %dir.display(), "could not create aligned_faces dir: {e}");
            } else {
                let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("img");
                for i in 0..n {
                    // aligned_imgs are BGR — swap to RGB for PNG output.
                    let crop_rgb = image::RgbImage::from_fn(224, 224, |x, y| {
                        let b = aligned_u8_4d[[i, y as usize, x as usize, 0]];
                        let g = aligned_u8_4d[[i, y as usize, x as usize, 1]];
                        let r = aligned_u8_4d[[i, y as usize, x as usize, 2]];
                        image::Rgb([r, g, b])
                    });
                    let file = dir.join(format!("{stem}_face_{i}.png"));
                    if let Err(e) = crop_rgb.save(&file) {
                        warn!(file = %file.display(), "could not save aligned face crop: {e}");
                    }
                }
            }
        }

        let mut result = Vec::with_capacity(n);

        for i in 0..n {
            let conf = scores_data[i];

            // Map pixel bbox from resized space back to original,
            // then normalise to [0, 1] relative to original dimensions.
            // Model outputs [x1, y1, x2, y2] in pixel coords of the input tensor.
            let x1 = (bboxes[[i, 0]] / scale / orig_w as f32).clamp(0.0, 1.0);
            let y1 = (bboxes[[i, 1]] / scale / orig_h as f32).clamp(0.0, 1.0);
            let x2 = (bboxes[[i, 2]] / scale / orig_w as f32).clamp(0.0, 1.0);
            let y2 = (bboxes[[i, 3]] / scale / orig_h as f32).clamp(0.0, 1.0);

            // ── Optional ArcFace embedding pass ───────────────────────────
            // ArcFace expects 112×112. The atksh model outputs 224×224 crops,
            // so we resize down before embedding.
            let embedding = if let Some(ref mut embedder) = self.embedder {
                // Build a temporary RgbImage (treating BGR bytes as generic 3-ch)
                // and resize to 112×112. Channel ordering is preserved by resize.
                let crop_224 = image::RgbImage::from_fn(224, 224, |x, y| {
                    image::Rgb([
                        aligned_u8_4d[[i, y as usize, x as usize, 0]], // B
                        aligned_u8_4d[[i, y as usize, x as usize, 1]], // G
                        aligned_u8_4d[[i, y as usize, x as usize, 2]], // R
                    ])
                });
                let crop_112 = image::imageops::resize(
                    &crop_224,
                    112,
                    112,
                    image::imageops::FilterType::Triangle,
                );
                // Convert to [112, 112, 3] BGR f32 [0, 255] for the embedder.
                let mut crop_arr = Array3::<f32>::zeros((112, 112, 3));
                for (x, y, p) in crop_112.enumerate_pixels() {
                    crop_arr[[y as usize, x as usize, 0]] = p[0] as f32;
                    crop_arr[[y as usize, x as usize, 1]] = p[1] as f32;
                    crop_arr[[y as usize, x as usize, 2]] = p[2] as f32;
                }
                embedder.embed_face_crop(crop_arr.view())?
            } else {
                vec![]
            };

            result.push(DetectedFace { bbox: [x1, y1, x2, y2], embedding, confidence: conf });
        }

        Ok(result)
    }
}
