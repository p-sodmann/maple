//! Modular ONNX inference framework for maple-db.
//!
//! # Architecture
//!
//! ```text
//! ModelFactory  ──builds──▶  Box<dyn DetectionModel>
//!                              (OnnxFaceDetector)
//!                                  │
//!                                  ├── OnnxSession (detector)
//!                                  │     └── ort::Session + device EPs
//!                                  │
//!                                  └── Option<Box<dyn EmbeddingModel>>
//!                                        (OnnxFaceEmbedder)
//!                                            └── OnnxSession (embedder)
//! ```
//!
//! # Extending with new model types
//!
//! 1. Add a new trait to [`embedding`] (e.g. `TextEmbeddingModel`) or create
//!    a new sub-module.
//! 2. Implement the trait in a concrete struct backed by [`OnnxSession`].
//! 3. Expose a `build_*` method on [`ModelFactory`].
//!
//! The [`Preprocessor`] pipeline handles all tensor format conversions so that
//! concrete model structs stay focused on inference logic.

pub mod detection;
pub mod device;
pub mod embedding;
pub mod factory;
pub mod preprocessor;
pub mod session;

pub use detection::{DetectionModel, OnnxFaceDetector};
pub use device::ModelDevice;
pub use embedding::{EmbeddingModel, OnnxFaceEmbedder, TextEmbeddingModel};
pub use factory::ModelFactory;
pub use preprocessor::{Preprocessor, PreprocessStep};
pub use session::OnnxSession;
