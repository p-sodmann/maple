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

use std::path::Path;
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
    /// Load detector (and optional embedder) targeting the CPU.
    ///
    /// This is the backward-compatible constructor.  For GPU inference use
    /// [`with_device`](FaceDetector::with_device).
    pub fn new(detector_path: &Path, embedder_path: Option<&Path>) -> Result<Self> {
        Self::with_device(detector_path, embedder_path, &models::ModelDevice::Cpu)
    }

    /// Load detector (and optional embedder) targeting `device`.
    ///
    /// ```ignore
    /// let device: models::ModelDevice = settings.face.device.parse().unwrap_or_default();
    /// let fd = FaceDetector::with_device(&detector_path, embedder_path, &device)?;
    /// ```
    pub fn with_device(
        detector_path: &Path,
        embedder_path: Option<&Path>,
        device: &models::ModelDevice,
    ) -> Result<Self> {
        let inner = models::ModelFactory::new()
            .with_device(device.clone())
            .build_face_detector(detector_path, embedder_path)?;
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
    stop_tx: std::sync::mpsc::SyncSender<()>,
}

impl FaceTagger {
    /// Signal the thread to stop after the current image.
    pub fn stop(&self) {
        let _ = self.stop_tx.send(());
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
    use std::sync::mpsc::RecvTimeoutError;

    let (stop_tx, stop_rx) = std::sync::mpsc::sync_channel(1);

    std::thread::Builder::new()
        .name("face-tagger".to_owned())
        .spawn(move || {
            let mut detector = detector;
            info!("face tagger started");
            'outer: loop {
                let images = match db.lock().unwrap().images_needing_face_detection() {
                    Ok(v) => v,
                    Err(e) => {
                        warn!("face_tagger: DB query failed: {e}");
                        vec![]
                    }
                };

                if images.is_empty() {
                    info!("face tagger: no pending images, sleeping");
                    match stop_rx.recv_timeout(Duration::from_secs(60)) {
                        Ok(_) | Err(RecvTimeoutError::Disconnected) => break 'outer,
                        Err(RecvTimeoutError::Timeout) => continue 'outer,
                    }
                }

                info!(count = images.len(), "face tagger: processing images");

                for (image_id, path) in images {
                    match stop_rx.try_recv() {
                        Ok(_) | Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                            break 'outer
                        }
                        Err(std::sync::mpsc::TryRecvError::Empty) => {}
                    }

                    match detector.detect_and_embed(&path) {
                        Ok(faces) => {
                            let count = faces.len();
                            let guard = db.lock().unwrap();
                            for face in faces {
                                if let Err(e) = guard.insert_face_detection(
                                    image_id,
                                    face.bbox,
                                    &face.embedding,
                                    face.confidence,
                                ) {
                                    warn!(
                                        path = %path.display(),
                                        "face_tagger: insert failed: {e}"
                                    );
                                }
                            }
                            // Sentinel: marks image as "processed, no face found".
                            if count == 0 {
                                let _ = guard.insert_face_detection(
                                    image_id,
                                    [0.0, 0.0, 0.0, 0.0],
                                    &[],
                                    -1.0,
                                );
                            }
                            info!(
                                path = %path.display(),
                                faces = count,
                                "face tagger: done"
                            );
                        }
                        Err(e) => {
                            warn!(path = %path.display(), "face_tagger: detection failed: {e}");
                        }
                    }
                }
            }
            info!("face tagger stopped");
        })
        .expect("failed to spawn face tagger thread");

    FaceTagger { stop_tx }
}
