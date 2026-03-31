//! TOML configuration for the CS2 bot.

use serde::Deserialize;
use std::path::PathBuf;

/// Top-level configuration.
#[derive(Debug, Deserialize, Clone)]
pub struct Config {
    pub general: GeneralConfig,
    pub models: ModelConfig,
    pub screen: ScreenConfig,
    pub aim: AimConfig,
    pub nav: NavConfig,
    pub movement: MovementConfig,
    pub sensitivity: SensitivityConfig,
}

#[derive(Debug, Deserialize, Clone)]
pub struct GeneralConfig {
    /// Target tick rate in Hz.
    #[serde(default = "default_tick_rate")]
    pub tick_rate: u32,
    /// Dry-run mode: log actions without sending inputs.
    #[serde(default)]
    pub dry_run: bool,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ModelConfig {
    /// Path to the YOLO ONNX model.
    pub yolo_path: PathBuf,
    /// Path to the RepVIT ONNX model.
    pub repvit_path: PathBuf,
    /// Use CUDA execution provider.
    #[serde(default)]
    pub use_cuda: bool,
    /// CUDA device ID.
    #[serde(default)]
    pub cuda_device_id: i32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct ScreenConfig {
    /// Capture width.
    #[serde(default = "default_width")]
    pub width: u32,
    /// Capture height.
    #[serde(default = "default_height")]
    pub height: u32,
}

#[derive(Debug, Deserialize, Clone)]
pub struct AimConfig {
    /// FOV cone in degrees — only engage targets within this angle from crosshair.
    #[serde(default = "default_fov")]
    pub fov_degrees: f64,
    /// WindMouse gravity parameter.
    #[serde(default = "default_wind_gravity")]
    pub wind_gravity: f64,
    /// WindMouse wind parameter.
    #[serde(default = "default_wind_wind")]
    pub wind_wind: f64,
    /// WindMouse max step size.
    #[serde(default = "default_wind_max_step")]
    pub wind_max_step: f64,
    /// WindMouse min distance for wind to apply.
    #[serde(default = "default_wind_min_dist")]
    pub wind_min_dist: f64,
    /// Enable One-Euro filter for aim jitter reduction.
    #[serde(default)]
    pub one_euro: bool,
    /// One-Euro filter min cutoff frequency.
    #[serde(default = "default_one_euro_min_cutoff")]
    pub one_euro_min_cutoff: f64,
    /// One-Euro filter beta (speed coefficient).
    #[serde(default = "default_one_euro_beta")]
    pub one_euro_beta: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct NavConfig {
    /// Path to the nav mesh JSON file (awpy export).
    pub nav_file: PathBuf,
    /// Map index for RepVIT.
    #[serde(default)]
    pub map_idx: i64,
    /// Path smoothing method: "funnel" or "catmull".
    #[serde(default = "default_smooth_method")]
    pub smooth_method: String,
    /// Distance to waypoint before advancing to next (in game units).
    #[serde(default = "default_waypoint_threshold")]
    pub waypoint_threshold: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct MovementConfig {
    /// Walking speed in game units per second.
    #[serde(default = "default_walk_speed")]
    pub walk_speed: f64,
}

#[derive(Debug, Deserialize, Clone)]
pub struct SensitivityConfig {
    /// In-game mouse sensitivity.
    #[serde(default = "default_sens")]
    pub game_sensitivity: f64,
    /// FOV scale factor (used for delta calculation: delta = pixel_offset * sens_factor / fov_scale).
    #[serde(default = "default_fov_scale")]
    pub fov_scale: f64,
}

// Default value functions
fn default_tick_rate() -> u32 { 60 }
fn default_width() -> u32 { 384 }
fn default_height() -> u32 { 288 }
fn default_fov() -> f64 { 45.0 }
fn default_wind_gravity() -> f64 { 9.0 }
fn default_wind_wind() -> f64 { 3.0 }
fn default_wind_max_step() -> f64 { 10.0 }
fn default_wind_min_dist() -> f64 { 12.0 }
fn default_one_euro_min_cutoff() -> f64 { 1.0 }
fn default_one_euro_beta() -> f64 { 0.007 }
fn default_smooth_method() -> String { "catmull".to_string() }
fn default_waypoint_threshold() -> f64 { 64.0 }
fn default_walk_speed() -> f64 { 250.0 }
fn default_sens() -> f64 { 2.0 }
fn default_fov_scale() -> f64 { 90.0 }

impl Config {
    /// Load configuration from a TOML file.
    pub fn load(path: &std::path::Path) -> anyhow::Result<Self> {
        let contents = std::fs::read_to_string(path)?;
        let config: Config = toml::from_str(&contents)?;
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_deserialize() {
        let toml_str = r#"
[general]
tick_rate = 60
dry_run = true

[models]
yolo_path = "models/yolo.onnx"
repvit_path = "models/repvit.onnx"
use_cuda = false

[screen]
width = 384
height = 288

[aim]
fov_degrees = 45.0
wind_gravity = 9.0
wind_wind = 3.0
wind_max_step = 10.0
wind_min_dist = 12.0
one_euro = false

[nav]
nav_file = "maps/dust2.json"
map_idx = 0
smooth_method = "catmull"
waypoint_threshold = 64.0

[movement]
walk_speed = 250.0

[sensitivity]
game_sensitivity = 2.0
fov_scale = 90.0
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert!(config.general.dry_run);
        assert_eq!(config.screen.width, 384);
        assert_eq!(config.screen.height, 288);
        assert!((config.aim.fov_degrees - 45.0).abs() < 1e-10);
    }
}
