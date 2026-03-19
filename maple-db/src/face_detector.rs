//! Face detection and embedding pipeline — public API.
//!
//! This module keeps a stable public surface (`FaceDetector`, `FaceTagger`,
//! `DetectedFace`, `spawn_face_tagger`) while delegating all inference to the
//! modular [`crate::models`] framework.
//!
//! # Upgrading from tract-onnx
//!
//! The public API is identical to the previous tract-onnx-based version.
//! Internally, inference now runs through `ort` (ONNX Runtime bindings) which
//! supports CPU, CUDA, and TensorRT execution providers.  Pass a
//! [`crate::models::ModelDevice`] to [`FaceDetector::with_device`] to select
//! the execution target; `FaceDetector::new` defaults to CPU.
//!
//! # Thread model
//!
//! [`FaceDetector`] is `Send`.  [`spawn_face_tagger`] moves it into a
//! background `std::thread` and polls `images_needing_face_detection` in a
//! loop, sleeping 60 s when the queue is empty.

use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Result;
use tracing::{info, warn};

use crate::{models, Database};

// ── Public types ───────────────────────────────────────────────────────────────

/// A single face detected in an image.
pub struct DetectedFace {
    /// Normalised bounding box `[x1, y1, x2, y2]` each in `[0, 1]`.
    pub bbox: [f32; 4],
    /// L2-normalised 512-dim ArcFace embedding, or empty if no embedder configured.
    pub embedding: Vec<f32>,
    /// Detector confidence score.
    pub confidence: f32,
}

// ── FaceDetector ──────────────────────────────────────────────────────────────

/// Wraps an atksh detection + optional ArcFace embedding model.
///
/// Build with [`new`](FaceDetector::new) (CPU) or
/// [`with_device`](FaceDetector::with_device) (GPU).
pub struct FaceDetector {
    inner: Box<dyn models::DetectionModel>,
}

impl FaceDetector {
    /// Load detector (and optional embedder) targeting the CPU using the
    /// default atksh backend.
    ///
    /// For GPU inference or to select the SCRFD backend use
    /// [`with_device`](FaceDetector::with_device).
    pub fn new(detector_path: &Path, embedder_path: Option<&Path>) -> Result<Self> {
        Self::with_device(
            detector_path,
            embedder_path,
            &models::ModelDevice::Cpu,
            maple_state::DetectorKind::Atksh,
            None,
        )
    }

    /// Load detector (and optional embedder) targeting `device`.
    ///
    /// `kind` selects the detector backend:
    /// - [`DetectorKind::Atksh`] — single-pass model (detection + aligned crops).
    /// - [`DetectorKind::Scrfd`] — standard InsightFace SCRFD model.
    ///
    /// `debug_dir`: when `Some`, 112×112 aligned face crops are saved as PNGs
    /// into that directory after each image is processed.
    ///
    /// ```ignore
    /// let device: models::ModelDevice = settings.face.device.parse().unwrap_or_default();
    /// let debug_dir = settings.debug.then(|| maple_state::config_dir().join("aligned_faces"));
    /// let fd = FaceDetector::with_device(
    ///     &detector_path, embedder_path, &device, settings.face.detector_type, debug_dir,
    /// )?;
    /// ```
    pub fn with_device(
        detector_path: &Path,
        embedder_path: Option<&Path>,
        device: &models::ModelDevice,
        kind: maple_state::DetectorKind,
        debug_dir: Option<PathBuf>,
    ) -> Result<Self> {
        let mut factory = models::ModelFactory::new().with_device(device.clone());
        if let Some(dir) = debug_dir {
            factory = factory.with_debug_dir(dir);
        }
        let inner = factory.build_face_detector(detector_path, embedder_path, kind)?;
        Ok(Self { inner })
    }

    /// Detect all faces in the image at `path`.
    ///
    /// Returns one [`DetectedFace`] per detected person.  When an embedder is
    /// configured, aligned crops from the detector are fed through it
    /// automatically.
    pub fn detect_and_embed(&mut self, path: &Path) -> Result<Vec<DetectedFace>> {
        self.inner.detect(path)
    }
}

// SAFETY: FaceDetector wraps Box<dyn DetectionModel: Send + Sync>.
unsafe impl Send for FaceDetector {}

// ── Background tagger ─────────────────────────────────────────────────────────

/// Handle to the running face tagger background thread.
pub struct FaceTagger {
    handle: crate::worker::WorkerHandle,
}

impl FaceTagger {
    /// Signal the thread to stop after the current image.
    pub fn stop(&self) {
        self.handle.stop();
    }
}

/// Spawn a background thread that runs face detection on every unprocessed
/// present image, storing bounding boxes and embeddings in the database.
///
/// The thread polls `images_needing_face_detection` in a loop, sleeping 60 s
/// when the queue is empty.  A sentinel row (confidence = -1.0, empty
/// embedding) is inserted for images with no detected face so they are not
/// reprocessed on the next pass.
///
/// Returns a [`FaceTagger`] whose [`stop`](FaceTagger::stop) method gracefully
/// terminates the thread after the current image finishes.
pub fn spawn_face_tagger(db: Arc<Mutex<Database>>, detector: FaceDetector) -> FaceTagger {
    let handle = crate::worker::spawn_db_worker(
        "face-tagger",
        db,
        detector,
        Duration::from_secs(60),
        // fetch
        |db_guard| {
            db_guard
                .images_needing_face_detection()
                .unwrap_or_else(|e| {
                    warn!("face_tagger: DB query failed: {e:?}");
                    vec![]
                })
        },
        // process
        |detector, db, (image_id, path)| match detector.detect_and_embed(&path) {
            Ok(faces) => {
                let count = faces.len();
                let guard = crate::lock_db(db);
                for face in faces {
                    if let Err(e) = guard.insert_face_detection(
                        image_id,
                        face.bbox,
                        &face.embedding,
                        face.confidence,
                    ) {
                        warn!(
                            path = %path.display(),
                            "face_tagger: insert failed: {e:?}"
                        );
                    }
                }
                // Sentinel: marks image as "processed, no face found".
                if count == 0 {
                    if let Err(e) = guard.insert_face_detection(
                        image_id,
                        [0.0, 0.0, 0.0, 0.0],
                        &[],
                        -1.0,
                    ) {
                        warn!("face_tagger: failed to insert sentinel for image {image_id}: {e:?}");
                    }
                }
                info!(
                    path = %path.display(),
                    faces = count,
                    "face tagger: done"
                );
            }
            Err(e) => {
                warn!(path = %path.display(), "face_tagger: detection failed: {e:?}");
            }
        },
    );

    FaceTagger { handle }
}
