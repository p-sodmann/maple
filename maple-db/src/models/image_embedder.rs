//! ONNX vision model embedder for image similarity / stack detection.
//!
//! [`OnnxImageEmbedder`] loads any ONNX vision encoder (e.g. DINOv2) and
//! produces L2-normalised dense embeddings from full images.  These embeddings
//! are used by the stacker to group semantically similar shots.
//!
//! # Preprocessing pipeline (BitImageProcessor / ImageNet convention)
//!
//! 1. Resize so the shortest edge = `shortest_edge` (Lanczos3).
//! 2. Center-crop to `image_size × image_size`.
//! 3. Convert to RGB f32 and divide by 255.
//! 4. Normalise per-channel: `(pixel - mean) / std`.
//! 5. Transpose HWC → CHW and add batch dimension → `[1, 3, H, W]`.

use std::path::Path;

use anyhow::{Context, Result};
use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};
use ndarray::Array3;

use super::{
    device::ModelDevice,
    preprocessor::{Preprocessor, PreprocessStep},
    session::OnnxSession,
};

/// Dense image embedding model backed by an ONNX vision encoder.
pub struct OnnxImageEmbedder {
    session: OnnxSession,
    preprocessor: Preprocessor,
    shortest_edge: u32,
    image_size: u32,
    embedding_dim: usize,
}

impl OnnxImageEmbedder {
    /// Load an ONNX vision model from `path`.
    ///
    /// `shortest_edge` and `image_size` control the resize/crop preprocessing;
    /// `mean` and `std` are the per-channel ImageNet normalisation parameters.
    pub fn load(
        path: &Path,
        device: &ModelDevice,
        shortest_edge: u32,
        image_size: u32,
        mean: [f32; 3],
        std: [f32; 3],
    ) -> Result<Self> {
        let session = OnnxSession::load(path, device)
            .with_context(|| format!("loading image embedder: {}", path.display()))?;

        // Infer embedding dimension from the model's first output.
        let embedding_dim = session
            .session
            .outputs()
            .first()
            .and_then(|o| o.dtype().tensor_shape())
            .and_then(|shape| shape.last().copied())
            .and_then(|d| usize::try_from(d).ok())
            .unwrap_or(768);

        // ImageNet preprocessing: scale to [0,1], normalise, HWC→CHW, batch.
        let preprocessor = Preprocessor::new()
            .with_step(PreprocessStep::LinearScale {
                scale: 1.0 / 255.0,
                offset: 0.0,
            })
            .with_step(PreprocessStep::Normalize { mean, std })
            .with_step(PreprocessStep::HwcToChw)
            .with_step(PreprocessStep::AddBatchDim);

        Ok(Self {
            session,
            preprocessor,
            shortest_edge,
            image_size,
            embedding_dim,
        })
    }

    /// Compute an L2-normalised embedding for `img`.
    pub fn embed(&mut self, img: &DynamicImage) -> Result<Vec<f32>> {
        let arr = self.preprocess(img)?;

        let tensor = self
            .preprocessor
            .run(arr)?
            .into_dimensionality::<ndarray::Ix4>()
            .context("image embedder preprocessing must produce a 4-D tensor [1,C,H,W]")?;

        let input_name = &self.session.input_names[0];
        let tensor_ref = ort::value::TensorRef::from_array_view(tensor.view())
            .context("creating image embedder input tensor")?;
        let outputs = self
            .session
            .session
            .run(ort::inputs![input_name.as_str() => tensor_ref])
            .context("running image embedder")?;

        let output_name = &self.session.output_names[0];
        let (_, raw_data) = outputs[output_name.as_str()]
            .try_extract_tensor::<f32>()
            .context("extracting image embedding tensor")?;
        let raw: Vec<f32> = raw_data.to_vec();

        Ok(l2_normalize(raw))
    }

    /// Dimensionality of the embedding vector.
    pub fn embedding_dim(&self) -> usize {
        self.embedding_dim
    }

    /// Resize to shortest_edge, then center-crop to image_size × image_size.
    /// Returns an `[H, W, C]` RGB f32 array in [0, 255].
    fn preprocess(&self, img: &DynamicImage) -> Result<Array3<f32>> {
        let (w, h) = img.dimensions();

        // Resize so shortest edge = self.shortest_edge.
        let (new_w, new_h) = if w <= h {
            let new_w = self.shortest_edge;
            let new_h = (h as f32 * new_w as f32 / w as f32).round() as u32;
            (new_w, new_h)
        } else {
            let new_h = self.shortest_edge;
            let new_w = (w as f32 * new_h as f32 / h as f32).round() as u32;
            (new_w, new_h)
        };

        let resized = img.resize_exact(new_w, new_h, FilterType::Lanczos3);

        // Center crop.
        let crop_size = self.image_size;
        let x = (new_w.saturating_sub(crop_size)) / 2;
        let y = (new_h.saturating_sub(crop_size)) / 2;
        let cropped = resized.crop_imm(x, y, crop_size, crop_size);
        let rgb = cropped.to_rgb8();

        let (cw, ch) = rgb.dimensions();
        let mut arr = Array3::<f32>::zeros((ch as usize, cw as usize, 3));
        for (px, py, pixel) in rgb.enumerate_pixels() {
            arr[[py as usize, px as usize, 0]] = pixel[0] as f32;
            arr[[py as usize, px as usize, 1]] = pixel[1] as f32;
            arr[[py as usize, px as usize, 2]] = pixel[2] as f32;
        }
        Ok(arr)
    }
}

/// L2-normalise a vector in-place; returns it.  Avoids divide-by-zero for
/// degenerate (all-zero) vectors.
fn l2_normalize(mut v: Vec<f32>) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter_mut().for_each(|x| *x /= norm);
    }
    v
}

/// Cosine similarity of two L2-normalised vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}
