//! maple-state — Persistence layer.
//!
//! Owns session state, embedding cache, tournament decisions.
//! Phase 0: Config struct only.

mod seen;
pub mod sync;

pub use seen::{Record, SeenSet};
pub use sync::{PeerMode, SyncRole, SyncSettings};

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Return the default config directory for the current platform.
///
/// | Platform | Path |
/// |---|---|
/// | Linux   | `$XDG_CONFIG_HOME/maple` or `~/.config/maple` |
/// | macOS   | `~/Library/Application Support/maple` |
/// | Windows | `%APPDATA%\maple` |
pub fn config_dir() -> PathBuf {
    directories::ProjectDirs::from("", "", "maple")
        .map(|dirs| dirs.config_dir().to_path_buf())
        .unwrap_or_else(|| {
            // Absolute last resort — no home dir at all (headless container).
            PathBuf::from(".maple")
        })
}

/// Default session file path.
fn session_path() -> PathBuf {
    config_dir().join("session.json")
}

/// Runtime configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Cosine similarity threshold for grouping (0.0–1.0).
    pub similarity_threshold: f32,
    /// Copy files preserving source subfolder structure.
    pub preserve_folder_structure: bool,
    /// Verify copied files with BLAKE3 checksum.
    pub verify_checksum: bool,
    /// Thumbnail longest-edge size in pixels.
    pub thumbnail_size: u32,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.92,
            preserve_folder_structure: false,
            verify_checksum: true,
            thumbnail_size: 256,
        }
    }
}

/// Minimal session info persisted between runs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Session {
    pub source: Option<PathBuf>,
    pub destination: Option<PathBuf>,
    pub config: Config,
}

impl Session {
    /// Load session from the default config path.
    /// Returns `Session::default()` if the file doesn't exist or is invalid.
    pub fn load() -> Self {
        Self::load_from(&session_path())
    }

    /// Load session from a specific path.
    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(json) => serde_json::from_str(&json).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Persist session to the default config path.
    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&session_path())
    }

    /// Persist session to a specific path.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }

    /// Validate that persisted paths still exist on disk.
    /// Clears any paths that no longer exist.
    pub fn validate_paths(&mut self) {
        if let Some(ref p) = self.source {
            if !p.is_dir() {
                self.source = None;
            }
        }
        if let Some(ref p) = self.destination {
            if !p.is_dir() {
                self.destination = None;
            }
        }
    }
}

// ── Settings (settings.toml) ────────────────────────────────────

/// Default settings file path.
fn settings_path() -> PathBuf {
    config_dir().join("settings.toml")
}

/// Which face detection backend to use.
///
/// Serialises to/from the lowercase string name (`"atksh"`, `"scrfd"`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DetectorKind {
    /// atksh joined ONNX model — single-pass SCRFD detection + landmark
    /// alignment + aligned 112×112 face-crop extraction.  Provides
    /// pre-aligned crops so an ArcFace embedder can be used without any
    /// extra preprocessing.
    ///
    /// Download: <https://github.com/atksh/onnx-facial-lmk-detector/releases>
    #[default]
    Atksh,
    /// Standard SCRFD detector (e.g. `scrfd_10g_bnkps.onnx` from InsightFace).
    /// Outputs raw bounding boxes + 5-point keypoints; face crops for
    /// ArcFace embedding are produced by manually cropping from the source image.
    ///
    /// Download: <https://github.com/deepinsight/insightface/tree/master/model_zoo>
    Scrfd,
}

/// Face detection and recognition settings.
///
/// Stored under `[face]` in `settings.toml`.
///
/// **Required:** `detector_model` — path to an ONNX face detector.  Which
/// model format to expect is controlled by `detector_type`.
///
/// **Optional:** `embedder_model` — an ArcFace model for 512-dim identity
/// embeddings and cosine-similarity person grouping.  Download from
/// <https://github.com/deepinsight/insightface/tree/master/model_zoo>.
/// Leave empty to run detection-only (no person grouping).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FaceSettings {
    /// Whether the face tagger starts automatically when the library opens.
    #[serde(default)]
    pub enabled: bool,
    /// Face detector backend.  Accepted values: `"atksh"` (default), `"scrfd"`.
    #[serde(default)]
    pub detector_type: DetectorKind,
    /// Path to the detector ONNX model (required).
    #[serde(default)]
    pub detector_model: PathBuf,
    /// Path to an ArcFace ONNX embedder (optional — enables person grouping).
    /// Leave empty to run detection without embedding.
    #[serde(default)]
    pub embedder_model: PathBuf,
    /// Cosine-similarity threshold for suggesting a person match (0.0–1.0).
    /// ArcFace-R100: same person typically ≥ 0.40.
    #[serde(default = "FaceSettings::default_similarity_threshold")]
    pub similarity_threshold: f32,
    /// Number of suggested persons shown in tagging mode.
    #[serde(default = "FaceSettings::default_tagging_top_k")]
    pub tagging_top_k: usize,
    /// Execution device for ONNX inference.
    /// Accepts: `"cpu"` (default), `"cuda:N"` (NVIDIA GPU index N),
    /// `"tensorrt:N"` (TensorRT, fastest for fixed-shape models).
    /// GPU devices only take effect on a binary built with the `gpu` Cargo
    /// feature (see `maple-db/Cargo.toml`) and a CUDA-enabled ONNX Runtime on
    /// `ORT_DYLIB_PATH`; on the default `cpu` build this always falls back to
    /// CPU regardless of this setting.
    #[serde(default = "FaceSettings::default_device")]
    pub device: String,
}

impl FaceSettings {
    fn default_similarity_threshold() -> f32 {
        0.40
    }

    fn default_device() -> String {
        "cpu".into()
    }

    fn default_tagging_top_k() -> usize {
        5
    }

    /// True when the detector model path is set and exists on disk.
    /// The embedder is optional — its absence disables person similarity only.
    pub fn models_available(&self) -> bool {
        !self.detector_model.as_os_str().is_empty() && self.detector_model.exists()
    }

    /// Return the embedder path if it is configured and exists on disk.
    pub fn embedder_path(&self) -> Option<&std::path::Path> {
        if !self.embedder_model.as_os_str().is_empty() && self.embedder_model.exists() {
            Some(&self.embedder_model)
        } else {
            None
        }
    }
}

impl Default for FaceSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            detector_type: DetectorKind::default(),
            detector_model: PathBuf::new(),
            embedder_model: PathBuf::new(),
            similarity_threshold: Self::default_similarity_threshold(),
            tagging_top_k: Self::default_tagging_top_k(),
            device: Self::default_device(),
        }
    }
}

/// AI model configuration for image description.
///
/// Stored under `[ai]` in `settings.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AiSettings {
    /// Whether the AI tagger should start automatically when the library opens.
    #[serde(default)]
    pub enabled: bool,
    /// Base URL of the OpenAI-compatible server (e.g. `http://localhost:1234`).
    #[serde(default = "AiSettings::default_server_url")]
    pub server_url: String,
    /// Model identifier as the server expects it (e.g. `llava-v1.6`).
    #[serde(default = "AiSettings::default_model")]
    pub model: String,
    /// System prompt sent with every image.
    #[serde(default = "AiSettings::default_prompt")]
    pub prompt: String,
}

impl AiSettings {
    fn default_server_url() -> String {
        "http://localhost:1234".into()
    }

    fn default_model() -> String {
        "local-model".into()
    }

    fn default_prompt() -> String {
        "Describe this image in detail. Include the main subjects, scene, \
         colors, mood, any text visible, and notable elements. Be thorough \
         to enable comprehensive search results."
            .into()
    }
}

impl Default for AiSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            server_url: Self::default_server_url(),
            model: Self::default_model(),
            prompt: Self::default_prompt(),
        }
    }
}

/// Import scan settings.
///
/// Stored under `[import]` in `settings.toml`.
///
/// The scan reads the medium on **one** thread and decodes on several: a
/// camera card is a single bus, and several readers on it are slower than
/// one, not faster. So the knob here is the size of the *decode* pool.
/// There is little point pushing it high — a scan exists to be looked at,
/// and nobody triages photos faster than a handful of cores can produce
/// them — but a fast internal disk and a lot of cores can take more.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportSettings {
    /// Photos decoded in parallel during an import scan. Default: 4.
    ///
    /// Clamped to at least 1. Each worker holds one full-resolution frame
    /// while it works (a 24 MP photo is ~72 MB), so this is a memory dial
    /// as much as a speed one.
    #[serde(default = "ImportSettings::default_decode_threads")]
    pub decode_threads: usize,
    /// Seconds one photo gets to be read off the medium before the scan
    /// gives up on it and moves on. Default: 30.
    ///
    /// The backstop for a card that stops answering. Raise it for a slow
    /// reader and very large raws; the photo is still listed and still
    /// copyable either way, it just has no preview.
    #[serde(default = "ImportSettings::default_read_timeout_secs")]
    pub read_timeout_secs: u64,
    /// Decoded previews held in memory at once. Default: 128.
    ///
    /// A 256 px preview is about 196 KB decoded, so the default is roughly
    /// 25 MB — many screenfuls of a one-column filmstrip. Past this the
    /// least-recently-seen preview is dropped down to its WebP copy (~15 KB)
    /// and re-inflated from there if it scrolls back into view, so eviction
    /// never sends the app back to the card.
    #[serde(default = "ImportSettings::default_max_loaded_previews")]
    pub max_loaded_previews: usize,
}

impl ImportSettings {
    fn default_decode_threads() -> usize {
        4
    }

    fn default_read_timeout_secs() -> u64 {
        30
    }

    fn default_max_loaded_previews() -> usize {
        128
    }

    /// Decoded-preview ceiling, never small enough to thrash.
    ///
    /// A cap below a screenful would evict previews the user is still
    /// looking at and immediately decode them again.
    pub fn retained_previews(&self) -> usize {
        self.max_loaded_previews.max(16)
    }

    /// Decode pool size, never zero.
    pub fn decoders(&self) -> usize {
        self.decode_threads.max(1)
    }

    /// How long one read may take before it is abandoned.
    pub fn read_timeout(&self) -> std::time::Duration {
        std::time::Duration::from_secs(self.read_timeout_secs.max(1))
    }
}

impl Default for ImportSettings {
    fn default() -> Self {
        Self {
            decode_threads: Self::default_decode_threads(),
            read_timeout_secs: Self::default_read_timeout_secs(),
            max_loaded_previews: Self::default_max_loaded_previews(),
        }
    }
}

/// Thumbnail cache settings.
///
/// Stored under `[thumbnails]` in `settings.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThumbnailSettings {
    /// WebP lossy quality for cached thumbnails (0–100). Default: 80.
    #[serde(default = "ThumbnailSettings::default_quality")]
    pub quality: u8,
    /// Thumbnail longest-edge size in pixels. Default: 200.
    /// Changing this value invalidates the cache (clear it via Settings).
    #[serde(default = "ThumbnailSettings::default_size")]
    pub size: u32,
}

impl ThumbnailSettings {
    fn default_quality() -> u8 {
        80
    }

    fn default_size() -> u32 {
        200
    }
}

impl Default for ThumbnailSettings {
    fn default() -> Self {
        Self {
            quality: Self::default_quality(),
            size: Self::default_size(),
        }
    }
}

/// Destination path template settings for imported files.
///
/// Stored under `[path_template]` in `settings.toml`.
///
/// Both `folder` and `filename` use `{TOKEN}` placeholders resolved from
/// each file's EXIF capture date (falling back to its filesystem mtime when
/// no EXIF date is present):
///   `{YYYY}` `{YY}` `{MM}` `{DD}` `{hh}` `{mm}` `{ss}` — date/time
///   `{original}` — source filename stem (no extension)
///   `{counter}`  — 1-based index within the current import, zero-padded to 4 digits
///   `{camera}`   — EXIF Make+Model, when present
///
/// The original file extension is always preserved regardless of
/// `filename` — it is never user-templatable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PathTemplateSettings {
    /// Subfolder path under the destination root, `/`-separated.
    /// Empty string = flat copy (no subfolders). Default: `"{YYYY}/{MM}"`.
    #[serde(default = "PathTemplateSettings::default_folder")]
    pub folder: String,
    /// Filename stem template. Default `"{original}"` keeps the source
    /// filename unchanged.
    #[serde(default = "PathTemplateSettings::default_filename")]
    pub filename: String,
}

impl PathTemplateSettings {
    fn default_folder() -> String {
        "{YYYY}/{MM}".into()
    }

    fn default_filename() -> String {
        "{original}".into()
    }
}

impl Default for PathTemplateSettings {
    fn default() -> Self {
        Self {
            folder: Self::default_folder(),
            filename: Self::default_filename(),
        }
    }
}

/// Collection hotkey settings.
///
/// Stored under `[collections]` in `settings.toml`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSettings {
    /// Key name (GTK key name) to add the current image in the detail
    /// viewer to the last-used collection.  Default: `"a"`.
    #[serde(default = "CollectionSettings::default_add_hotkey")]
    pub add_hotkey: String,
}

impl CollectionSettings {
    fn default_add_hotkey() -> String {
        "a".into()
    }
}

impl Default for CollectionSettings {
    fn default() -> Self {
        Self {
            add_hotkey: Self::default_add_hotkey(),
        }
    }
}

/// Semantic search settings.
///
/// Stored under `[semantic]` in `settings.toml`.
///
/// Encodes each sentence of every AI description into a dense vector with a
/// sentence-transformer model and stores it in a `sqlite-vec` table, so search
/// can rank images by vector distance (merged with keyword results).
///
/// `model` is a HuggingFace repo id.  The ONNX model and tokenizer are
/// downloaded automatically and cached under `~/.config/maple/models/`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticSettings {
    /// Whether the sentence embedder starts automatically when the library opens.
    #[serde(default)]
    pub enabled: bool,
    /// HuggingFace repo id of the sentence-transformer model.
    /// e.g. `"sentence-transformers/all-MiniLM-L6-v2"` (384-dim, fast) or
    /// `"sentence-transformers/all-mpnet-base-v2"` (768-dim, higher quality).
    #[serde(default = "SemanticSettings::default_model")]
    pub model: String,
    /// Path of the ONNX model file within the repo.
    #[serde(default = "SemanticSettings::default_onnx_file")]
    pub onnx_file: String,
    /// Path of the tokenizer file within the repo.
    #[serde(default = "SemanticSettings::default_tokenizer_file")]
    pub tokenizer_file: String,
    /// Execution device for ONNX inference (`"cpu"`, `"cuda:N"`, `"tensorrt:N"`).
    #[serde(default = "SemanticSettings::default_device")]
    pub device: String,
    /// How many nearest sentence vectors to retrieve per query before merging.
    #[serde(default = "SemanticSettings::default_knn_k")]
    pub knn_k: usize,
}

impl SemanticSettings {
    fn default_model() -> String {
        "sentence-transformers/all-MiniLM-L6-v2".into()
    }

    fn default_onnx_file() -> String {
        "onnx/model.onnx".into()
    }

    fn default_tokenizer_file() -> String {
        "tokenizer.json".into()
    }

    fn default_device() -> String {
        "cpu".into()
    }

    fn default_knn_k() -> usize {
        200
    }
}

impl Default for SemanticSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model: Self::default_model(),
            onnx_file: Self::default_onnx_file(),
            tokenizer_file: Self::default_tokenizer_file(),
            device: Self::default_device(),
            knn_k: Self::default_knn_k(),
        }
    }
}

/// Session detection settings — how an import scan decides where one
/// sitting of photos ends and the next begins.
///
/// Stored under `[sessions]` in `settings.toml`.
///
/// This is *segmentation*, not clustering: the scan walks the card in
/// capture order and asks, at each photo, whether the scene changed here.
/// It replaces the DINOv2 burst grouping in the importer — that cost
/// 26 ms/photo and throttled the whole scan through its embed queue, where
/// these engines cost about 0.2 ms and ride along behind the card read.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionSettings {
    /// Whether the import scan groups photos into sessions at all.
    #[serde(default = "SessionSettings::default_enabled")]
    pub enabled: bool,
    /// Which engine decides "the scene changed": one name, or a weighted
    /// ensemble like `"block-tile=2,grid-histogram=1,time-gap=1"`.
    ///
    /// Known names: `block-tile` (fraction of the frame that held still —
    /// the one built for a moving subject in a fixed scene), `grid-histogram`
    /// (colour and its layout), `color-kmeans` (palette only, blind to where
    /// any of it is), `time-gap` (votes on the clock alone).
    #[serde(default = "SessionSettings::default_engine")]
    pub engine: String,
    /// Distance above which the scene counts as changed. `0` means "use
    /// whatever this engine considers its own threshold", which is the only
    /// sane default because **engine distances are not comparable to each
    /// other** — 0.35 means something different to every one of them.
    #[serde(default)]
    pub cut: f32,
    /// A gap this long always ends a session, whatever the pixels say.
    /// The one hard rule; everything else about time is a matter of degree.
    #[serde(default = "SessionSettings::default_hard_gap_secs")]
    pub hard_gap_secs: f32,
    /// How many non-matching frames in a row a session may absorb before it
    /// really has ended.
    ///
    /// One shot of the cake in the middle of twenty of the child is bridged
    /// when the next frame comes back. When patience runs out the cut lands
    /// *before* the first frame that stopped matching — that is where the
    /// new scene actually started.
    #[serde(default = "SessionSettings::default_max_outliers")]
    pub max_outliers: usize,
    /// How much further a photo may drift from the frame its session
    /// *started* on than from its immediate neighbour. The anti-chaining
    /// rule: without it a slow pan walks a session across a whole room one
    /// tolerable step at a time.
    #[serde(default = "SessionSettings::default_anchor_factor")]
    pub anchor_factor: f32,
}

impl SessionSettings {
    fn default_enabled() -> bool {
        true
    }

    fn default_engine() -> String {
        "block-tile=2,grid-histogram=1,time-gap=1".to_owned()
    }

    fn default_hard_gap_secs() -> f32 {
        1800.0
    }

    fn default_max_outliers() -> usize {
        1
    }

    fn default_anchor_factor() -> f32 {
        1.8
    }
}

impl Default for SessionSettings {
    fn default() -> Self {
        Self {
            enabled: Self::default_enabled(),
            engine: Self::default_engine(),
            cut: 0.0,
            hard_gap_secs: Self::default_hard_gap_secs(),
            max_outliers: Self::default_max_outliers(),
            anchor_factor: Self::default_anchor_factor(),
        }
    }
}

/// Stack detection settings.
///
/// Stored under `[stacks]` in `settings.toml`.
///
/// After import, newly copied images are compared pairwise using dense
/// DINOv2 image embeddings and similar ones are assigned a shared `stack_id`
/// in the DB.  The library grid shows each stack as a single tile with a
/// count badge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StackSettings {
    /// Whether stack detection runs automatically after each import batch,
    /// and whether the import browser also detects bursts live during an
    /// SD-card scan (same toggle drives both).
    #[serde(default)]
    pub enabled: bool,
    /// Cosine-similarity threshold (0.0–1.0) between image embeddings.
    /// Images with similarity ≥ this value are placed in the same stack.
    #[serde(default = "StackSettings::default_threshold")]
    pub threshold: f32,
    /// HuggingFace repo id of the ONNX vision model.
    /// e.g. `"onnx-community/dinov2-small"`.
    /// The repo must contain an exported ONNX file at `onnx_file`.
    #[serde(default = "StackSettings::default_model_repo")]
    pub model_repo: String,
    /// Path of the ONNX model file within the repo.
    #[serde(default = "StackSettings::default_onnx_file")]
    pub onnx_file: String,
    /// Resize image so its shortest edge equals this value before cropping.
    /// Should match the model's `size.shortest_edge`.
    #[serde(default = "StackSettings::default_shortest_edge")]
    pub shortest_edge: u32,
    /// Center-crop size after resize.
    /// Should match the model's `crop_size`.
    #[serde(default = "StackSettings::default_image_size")]
    pub image_size: u32,
    /// Per-channel mean for ImageNet-style normalisation.
    #[serde(default = "StackSettings::default_image_mean")]
    pub image_mean: [f32; 3],
    /// Per-channel std for ImageNet-style normalisation.
    #[serde(default = "StackSettings::default_image_std")]
    pub image_std: [f32; 3],
    /// Execution device for ONNX inference.
    /// Accepts: `"cpu"` (default), `"cuda:N"`, `"tensorrt:N"`.
    #[serde(default = "StackSettings::default_device")]
    pub device: String,
}

impl StackSettings {
    fn default_threshold() -> f32 {
        0.90
    }

    fn default_model_repo() -> String {
        "onnx-community/dinov2-small".into()
    }

    fn default_onnx_file() -> String {
        "onnx/model.onnx".into()
    }

    fn default_shortest_edge() -> u32 {
        256
    }

    fn default_image_size() -> u32 {
        224
    }

    fn default_image_mean() -> [f32; 3] {
        [0.485, 0.456, 0.406]
    }

    fn default_image_std() -> [f32; 3] {
        [0.229, 0.224, 0.225]
    }

    fn default_device() -> String {
        "cpu".into()
    }
}

impl StackSettings {
    /// A stable string key that identifies the current model.
    ///
    /// Used as the `algorithm` column in `image_hashes`.  When the user
    /// changes `model_repo`, the key changes and old rows are ignored — the
    /// background hasher will recompute embeddings under the new key.
    ///
    /// Format: `"onnx:onnx-community/dinov2-small"`
    pub fn algorithm_key(&self) -> String {
        format!("onnx:{}", self.model_repo)
    }
}

impl Default for StackSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            threshold: Self::default_threshold(),
            model_repo: Self::default_model_repo(),
            onnx_file: Self::default_onnx_file(),
            shortest_edge: Self::default_shortest_edge(),
            image_size: Self::default_image_size(),
            image_mean: Self::default_image_mean(),
            image_std: Self::default_image_std(),
            device: Self::default_device(),
        }
    }
}

/// The bundled defaults.toml — written to disk on first launch so users
/// can discover and edit every setting.
const DEFAULTS_TOML: &str = include_str!("../defaults.toml");

/// Application settings loaded from `settings.toml`.
///
/// Missing keys fall back to defaults. The file is created with defaults
/// if it doesn't exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Enable debug output (e.g. saving aligned face crops to disk).
    /// When `true`, aligned face images are written to
    /// `~/.config/maple/aligned_faces/` for inspection.
    #[serde(default)]
    pub debug: bool,
    /// Number of full-resolution images to keep buffered around the
    /// current image in the browser view.
    #[serde(default = "Settings::default_preview_buffer_size")]
    pub preview_buffer_size: usize,
    /// Directory where library data files are stored
    /// (`seen_imported.bin`, `seen_skipped.bin`, …).
    /// Defaults to `~/.config/maple/`.
    #[serde(default = "Settings::default_library_dir")]
    pub library_dir: PathBuf,
    /// Path to the SQLite library database.
    /// Defaults to `~/.config/maple/library.db`.
    #[serde(default = "Settings::default_database_path")]
    pub database_path: PathBuf,
    /// Destination folder/filename templates for imported files.
    #[serde(default)]
    pub path_template: PathTemplateSettings,
    /// Import scan tuning.
    #[serde(default)]
    pub import: ImportSettings,
    /// AI image description settings.
    #[serde(default)]
    pub ai: AiSettings,
    /// Face detection / recognition settings.
    #[serde(default)]
    pub face: FaceSettings,
    /// Collection hotkey settings.
    #[serde(default)]
    pub collections: CollectionSettings,
    /// Semantic search settings.
    #[serde(default)]
    pub semantic: SemanticSettings,
    /// Thumbnail cache settings.
    #[serde(default)]
    pub thumbnails: ThumbnailSettings,
    /// Stack detection settings.
    #[serde(default)]
    pub stacks: StackSettings,
    /// Import session detection settings.
    #[serde(default)]
    pub sessions: SessionSettings,
    /// Machine-local sync configuration. Role, device name and per-peer mode
    /// deliberately live in the database instead — see [`sync`].
    #[serde(default)]
    pub sync: SyncSettings,
}

impl Settings {
    fn default_preview_buffer_size() -> usize {
        21
    }

    fn default_library_dir() -> PathBuf {
        config_dir()
    }

    fn default_database_path() -> PathBuf {
        config_dir().join("library.db")
    }

    /// Replace empty path fields with their runtime defaults.
    fn fill_empty_paths(&mut self) {
        if self.library_dir.as_os_str().is_empty() {
            self.library_dir = Self::default_library_dir();
        }
        if self.database_path.as_os_str().is_empty() {
            self.database_path = Self::default_database_path();
        }
    }

    /// Load settings from the default config path.
    /// Returns `Settings::default()` if the file doesn't exist or is invalid.
    pub fn load() -> Self {
        Self::load_from(&settings_path())
    }

    /// Load settings from a specific path.
    ///
    /// If the file doesn't exist, the bundled `defaults.toml` is written
    /// to disk so the user can discover and edit every available setting.
    pub fn load_from(path: &Path) -> Self {
        let contents = match std::fs::read_to_string(path) {
            Ok(c) => c,
            Err(_) => {
                // First boot — seed settings.toml from the bundled defaults.
                if let Some(parent) = path.parent() {
                    let _ = std::fs::create_dir_all(parent);
                }
                let _ = std::fs::write(path, DEFAULTS_TOML);
                DEFAULTS_TOML.to_owned()
            }
        };
        let mut settings: Self = match toml::from_str(&contents) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("maple: failed to parse settings.toml: {e}\nFalling back to defaults.");
                Self::default()
            }
        };
        settings.fill_empty_paths();
        settings
    }

    /// Persist settings to the default config path.
    pub fn save(&self) -> anyhow::Result<()> {
        self.save_to(&settings_path())
    }

    /// Persist settings to a specific path.
    pub fn save_to(&self, path: &Path) -> anyhow::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let toml_str = toml::to_string_pretty(self)?;
        std::fs::write(path, toml_str)?;
        Ok(())
    }
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            debug: false,
            preview_buffer_size: Self::default_preview_buffer_size(),
            library_dir: Self::default_library_dir(),
            database_path: Self::default_database_path(),
            path_template: PathTemplateSettings::default(),
            import: ImportSettings::default(),
            ai: AiSettings::default(),
            face: FaceSettings::default(),
            collections: CollectionSettings::default(),
            semantic: SemanticSettings::default(),
            thumbnails: ThumbnailSettings::default(),
            stacks: StackSettings::default(),
            sessions: SessionSettings::default(),
            sync: SyncSettings::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_values() {
        let cfg = Config::default();
        assert!((cfg.similarity_threshold - 0.92).abs() < f32::EPSILON);
        assert!(!cfg.preserve_folder_structure);
        assert!(cfg.verify_checksum);
        assert_eq!(cfg.thumbnail_size, 256);
    }

    #[test]
    fn config_roundtrip_json() {
        let cfg = Config {
            similarity_threshold: 0.85,
            preserve_folder_structure: true,
            verify_checksum: false,
            thumbnail_size: 512,
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: Config = serde_json::from_str(&json).unwrap();
        assert!((parsed.similarity_threshold - 0.85).abs() < f32::EPSILON);
        assert!(parsed.preserve_folder_structure);
        assert!(!parsed.verify_checksum);
        assert_eq!(parsed.thumbnail_size, 512);
    }

    #[test]
    fn session_default_has_no_paths() {
        let s = Session::default();
        assert!(s.source.is_none());
        assert!(s.destination.is_none());
    }

    #[test]
    fn session_roundtrip_json() {
        let s = Session {
            source: Some(PathBuf::from("/photos/src")),
            destination: Some(PathBuf::from("/photos/dst")),
            config: Config::default(),
        };
        let json = serde_json::to_string_pretty(&s).unwrap();
        let parsed: Session = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.source.unwrap(), PathBuf::from("/photos/src"));
        assert_eq!(parsed.destination.unwrap(), PathBuf::from("/photos/dst"));
    }

    #[test]
    fn session_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.json");

        let s = Session {
            source: Some(PathBuf::from("/tmp/src")),
            destination: Some(PathBuf::from("/tmp/dst")),
            config: Config::default(),
        };
        s.save_to(&path).unwrap();

        let loaded = Session::load_from(&path);
        assert_eq!(loaded.source, Some(PathBuf::from("/tmp/src")));
        assert_eq!(loaded.destination, Some(PathBuf::from("/tmp/dst")));
    }

    #[test]
    fn session_load_missing_file_returns_default() {
        let loaded = Session::load_from(Path::new("/nonexistent/session.json"));
        assert!(loaded.source.is_none());
        assert!(loaded.destination.is_none());
    }

    #[test]
    fn session_validate_paths_clears_missing() {
        let dir = tempfile::tempdir().unwrap();
        let mut s = Session {
            source: Some(dir.path().to_path_buf()),
            destination: Some(PathBuf::from("/nonexistent/path")),
            config: Config::default(),
        };
        s.validate_paths();
        assert!(s.source.is_some()); // dir exists
        assert!(s.destination.is_none()); // cleared
    }

    #[test]
    fn settings_default_values() {
        let s = Settings::default();
        assert_eq!(s.preview_buffer_size, 21);
        assert_eq!(s.library_dir, config_dir());
        assert_eq!(s.face.tagging_top_k, 5);
    }

    #[test]
    fn settings_roundtrip_toml() {
        let s = Settings {
            preview_buffer_size: 11,
            library_dir: PathBuf::from("/my/library"),
            database_path: PathBuf::from("/my/library/library.db"),
            path_template: PathTemplateSettings {
                folder: "{YYYY}/{MM}/{DD}".into(),
                filename: "{YYYY}{MM}{DD}_{counter}".into(),
            },
            ..Settings::default()
        };
        let toml_str = toml::to_string_pretty(&s).unwrap();
        let parsed: Settings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.preview_buffer_size, 11);
        assert_eq!(parsed.library_dir, PathBuf::from("/my/library"));
        assert_eq!(parsed.database_path, PathBuf::from("/my/library/library.db"));
        assert_eq!(parsed.path_template.folder, "{YYYY}/{MM}/{DD}");
        assert_eq!(parsed.path_template.filename, "{YYYY}{MM}{DD}_{counter}");
    }

    #[test]
    fn defaults_toml_parses_and_fills_paths() {
        let mut settings: Settings = toml::from_str(DEFAULTS_TOML).unwrap();
        // Empty strings in the file mean "use runtime default".
        assert!(settings.library_dir.as_os_str().is_empty());
        assert!(settings.database_path.as_os_str().is_empty());
        settings.fill_empty_paths();
        assert_eq!(settings.library_dir, config_dir());
        assert_eq!(settings.database_path, config_dir().join("library.db"));
        assert_eq!(settings.preview_buffer_size, 21);
        assert_eq!(settings.path_template.folder, "{YYYY}/{MM}");
        assert_eq!(settings.path_template.filename, "{original}");
        assert!(!settings.ai.enabled);
        assert!(!settings.face.enabled);
    }

    #[test]
    fn settings_missing_file_writes_defaults_toml() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.preview_buffer_size, 21);
        assert_eq!(loaded.library_dir, config_dir());
        // The file should now exist with the bundled defaults content.
        let on_disk = std::fs::read_to_string(&path).unwrap();
        assert_eq!(on_disk, DEFAULTS_TOML);
    }

    #[test]
    fn settings_missing_file_returns_default() {
        let loaded = Settings::load_from(Path::new("/nonexistent/settings.toml"));
        assert_eq!(loaded.preview_buffer_size, 21);
        assert_eq!(loaded.library_dir, config_dir());
    }

    #[test]
    fn config_dir_is_absolute_even_without_home() {
        // P1: on Windows, $HOME is not set and $XDG_CONFIG_HOME doesn't exist.
        // The old fallback `PathBuf::from(".")` yields a *relative* path,
        // which breaks every derived path (DB, settings, thumb cache).
        // After the fix, `directories::ProjectDirs` returns a correct absolute
        // platform path on all OSes.
        //
        // NOTE: mutates env vars — run with --test-threads=1 if this races.
        let old_home = std::env::var("HOME").ok();
        let old_xdg = std::env::var("XDG_CONFIG_HOME").ok();
        // SAFETY: single-threaded test context; vars restored before return.
        unsafe {
            std::env::remove_var("HOME");
            std::env::remove_var("XDG_CONFIG_HOME");
        }
        let dir = config_dir();
        unsafe {
            match old_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
            match old_xdg {
                Some(v) => std::env::set_var("XDG_CONFIG_HOME", v),
                None => std::env::remove_var("XDG_CONFIG_HOME"),
            }
        }
        assert!(
            dir.is_absolute(),
            "config_dir() returned a relative path when HOME is unset: {}",
            dir.display()
        );
        assert_eq!(
            dir.file_name().and_then(|n| n.to_str()),
            Some("maple"),
            "config_dir() must end with 'maple'"
        );
    }

    #[test]
    fn settings_save_and_load_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");

        let s = Settings {
            preview_buffer_size: 13,
            library_dir: PathBuf::from("/custom/lib"),
            database_path: PathBuf::from("/custom/lib/library.db"),
            ..Settings::default()
        };
        s.save_to(&path).unwrap();

        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.preview_buffer_size, 13);
        assert_eq!(loaded.library_dir, PathBuf::from("/custom/lib"));
        assert_eq!(loaded.database_path, PathBuf::from("/custom/lib/library.db"));
    }

    #[test]
    fn sync_section_roundtrips() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");

        let s = Settings {
            sync: SyncSettings {
                listen_addr: "192.168.1.20:9000".into(),
                interval_secs: 45,
            },
            ..Settings::default()
        };
        s.save_to(&path).unwrap();

        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.sync.listen_addr, "192.168.1.20:9000");
        assert_eq!(loaded.sync.interval_secs, 45);
    }

    #[test]
    fn an_import_section_round_trips_and_defaults_sanely() {
        let s = ImportSettings::default();
        assert_eq!(s.decoders(), 4);
        assert_eq!(s.read_timeout(), std::time::Duration::from_secs(30));
        assert_eq!(s.retained_previews(), 128);

        // Neither knob may come back as zero: no decoders would mean no
        // scan at all, and a zero timeout would abandon every photo.
        let zeroed = ImportSettings {
            decode_threads: 0,
            read_timeout_secs: 0,
            max_loaded_previews: 0,
        };
        assert_eq!(zeroed.decoders(), 1);
        assert_eq!(zeroed.read_timeout(), std::time::Duration::from_secs(1));
        // A ceiling below a screenful would evict previews still on screen
        // and immediately decode them again.
        assert_eq!(zeroed.retained_previews(), 16);

        let parsed: Settings = toml::from_str(
            "[import]\ndecode_threads = 9\nread_timeout_secs = 120\n",
        )
        .unwrap();
        assert_eq!(parsed.import.decoders(), 9);
        assert_eq!(parsed.import.read_timeout().as_secs(), 120);
    }

    #[test]
    fn settings_with_no_import_section_still_scan() {
        let parsed: Settings = toml::from_str("debug = false\n").unwrap();
        assert_eq!(parsed.import.decoders(), 4);
    }

    #[test]
    fn settings_without_a_sync_section_load_defaults() {
        // Every existing installation's settings.toml predates `[sync]`, so
        // loading one must yield defaults rather than failing outright and
        // dropping the user back to a wholly default configuration.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("settings.toml");
        std::fs::write(&path, "preview_buffer_size = 7\n").unwrap();

        let loaded = Settings::load_from(&path);
        assert_eq!(loaded.preview_buffer_size, 7, "the rest of the file must still parse");
        assert_eq!(loaded.sync.listen_addr, SyncSettings::default().listen_addr);
        assert_eq!(loaded.sync.interval_secs, SyncSettings::default().interval_secs);
    }

    #[test]
    fn bundled_defaults_parse_and_match_the_sync_defaults() {
        // defaults.toml is written to disk verbatim on first launch, so a
        // value there that disagrees with the Rust default would silently
        // become the real default for every new installation.
        let parsed: Settings = toml::from_str(DEFAULTS_TOML).expect("defaults.toml must parse");
        assert_eq!(parsed.sync.listen_addr, SyncSettings::default().listen_addr);
        assert_eq!(parsed.sync.interval_secs, SyncSettings::default().interval_secs);
    }
}
