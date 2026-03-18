//! maple-state — Persistence layer.
//!
//! Owns session state, embedding cache, tournament decisions.
//! Phase 0: Config struct only.

mod seen;

pub use seen::SeenSet;

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Return the default config directory: `~/.config/maple`.
pub fn config_dir() -> PathBuf {
    let base = std::env::var("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".into());
            PathBuf::from(home).join(".config")
        });
    base.join("maple")
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

/// Face detection and recognition settings.
///
/// Stored under `[face]` in `settings.toml`.
///
/// **Required:** `detector_model` — the atksh joined ONNX model that performs
/// face detection + landmark detection + aligned-crop extraction in a single
/// pass.  Download from <https://github.com/atksh/onnx-facial-lmk-detector/releases>.
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
    /// Path to the atksh joined ONNX model (required for face detection).
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
    /// Requires a CUDA-enabled ONNX Runtime on `ORT_DYLIB_PATH` for GPU.
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

/// Application settings loaded from `settings.toml`.
///
/// Missing keys fall back to defaults. The file is created with defaults
/// if it doesn't exist.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Settings {
    /// Number of full-resolution images to keep buffered around the
    /// current image in the browser view.
    #[serde(default = "Settings::default_preview_buffer_size")]
    pub preview_buffer_size: usize,
    /// Directory where library data files are stored
    /// (`seen_imported.bin`, `seen_rejected.bin`, …).
    /// Defaults to `~/.config/maple/`.
    #[serde(default = "Settings::default_library_dir")]
    pub library_dir: PathBuf,
    /// Path to the SQLite library database.
    /// Defaults to `~/.config/maple/library.db`.
    #[serde(default = "Settings::default_database_path")]
    pub database_path: PathBuf,
    /// `strftime`-style format string for organising imported files into
    /// subdirectories based on EXIF date.  Set to an empty string to
    /// disable (flat copy).  Defaults to `"%Y/%m"` → `2024/01/`.
    #[serde(default = "Settings::default_folder_format")]
    pub folder_format: String,
    /// AI image description settings.
    #[serde(default)]
    pub ai: AiSettings,
    /// Face detection / recognition settings.
    #[serde(default)]
    pub face: FaceSettings,
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

    fn default_folder_format() -> String {
        "%Y/%m".into()
    }

    /// Load settings from the default config path.
    /// Returns `Settings::default()` if the file doesn't exist or is invalid.
    pub fn load() -> Self {
        Self::load_from(&settings_path())
    }

    /// Load settings from a specific path.
    pub fn load_from(path: &Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
            Err(_) => {
                let settings = Self::default();
                // Write the default file so the user can discover/edit it.
                let _ = settings.save_to(path);
                settings
            }
        }
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
            preview_buffer_size: Self::default_preview_buffer_size(),
            library_dir: Self::default_library_dir(),
            database_path: Self::default_database_path(),
            folder_format: Self::default_folder_format(),
            ai: AiSettings::default(),
            face: FaceSettings::default(),
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
            ..Settings::default()
        };
        let toml_str = toml::to_string_pretty(&s).unwrap();
        let parsed: Settings = toml::from_str(&toml_str).unwrap();
        assert_eq!(parsed.preview_buffer_size, 11);
        assert_eq!(parsed.library_dir, PathBuf::from("/my/library"));
        assert_eq!(parsed.database_path, PathBuf::from("/my/library/library.db"));
    }

    #[test]
    fn settings_missing_file_returns_default() {
        let loaded = Settings::load_from(Path::new("/nonexistent/settings.toml"));
        assert_eq!(loaded.preview_buffer_size, 21);
        assert_eq!(loaded.library_dir, config_dir());
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
}
