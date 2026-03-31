// Public API items used by consumers/extensions but not all used in main.rs yet.
#![allow(dead_code)]
//! CS2 Deathmatch Bot — Main Entry Point
//!
//! A high-performance bot for offline/bot sandbox play that uses:
//! - YOLO model for enemy detection
//! - RepVIT model for navigation/positioning
//! - WindMouse for human-like mouse movement
//! - Kalman filtering for position smoothing
//! - Nav mesh pathfinding for world navigation

mod aim_controller;
mod config;
mod detection;
mod game_loop;
mod input_sim;
mod math_utils;
mod navmesh;
mod navigation;
mod onnx_inference;
mod screen_capture;

use anyhow::Result;
use clap::Parser;
use config::Config;
use game_loop::{BotController, TickData};
use onnx_inference::{RepVitModel, YoloModel};
use screen_capture::ScreenCapturer;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

/// CS2 Deathmatch Bot
#[derive(Parser, Debug)]
#[command(name = "cs2_bot", version, about = "CS2 Deathmatch Bot for offline sandbox play")]
struct Cli {
    /// Path to config.toml
    #[arg(short, long, default_value = "config.toml")]
    config: PathBuf,

    /// Map name (for logging/identification)
    #[arg(short, long, default_value = "unknown")]
    map: String,

    /// Path to nav mesh JSON file (overrides config)
    #[arg(long)]
    nav_file: Option<PathBuf>,

    /// Dry-run mode: log actions without sending inputs
    #[arg(long)]
    dry_run: bool,
}

fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("cs2_bot=info".parse()?),
        )
        .init();

    let cli = Cli::parse();
    info!(map = %cli.map, config = ?cli.config, "CS2 Bot starting");

    // Load config
    let mut config = Config::load(&cli.config)?;

    // CLI overrides
    if let Some(nav_file) = cli.nav_file {
        config.nav.nav_file = nav_file;
    }
    if cli.dry_run {
        config.general.dry_run = true;
    }

    info!(
        tick_rate = config.general.tick_rate,
        dry_run = config.general.dry_run,
        screen = format!("{}x{}", config.screen.width, config.screen.height),
        "Configuration loaded"
    );

    // Load ONNX models
    let mut yolo_model = YoloModel::load(&config.models.yolo_path, config.models.use_cuda)?;
    let mut repvit_model = RepVitModel::load(&config.models.repvit_path, config.models.use_cuda)?;

    // Initialize screen capture
    let mut capturer = ScreenCapturer::new(config.screen.width, config.screen.height)?;

    // Initialize bot controller
    let mut bot = BotController::new(config.clone())?;

    let tick_duration = Duration::from_secs_f64(1.0 / config.general.tick_rate as f64);

    info!("Entering main loop (press Ctrl+C to stop)");

    // Main game loop
    loop {
        let tick_start = Instant::now();

        // 1. Grab frame
        let frame = match capturer.grab_frame() {
            Ok(f) => f,
            Err(e) => {
                warn!("Frame grab failed: {e}");
                std::thread::sleep(tick_duration);
                continue;
            }
        };

        // 2. Convert frame to tensor format
        let chw_data = screen_capture::bgra_to_chw_f32(&frame.data, frame.width, frame.height);

        // 3. Run YOLO inference (could be spawn_blocking in async context)
        let yolo_output = match yolo_model.infer(&chw_data) {
            Ok(out) => out,
            Err(e) => {
                warn!("YOLO inference failed: {e}");
                std::thread::sleep(tick_duration);
                continue;
            }
        };

        // 4. Run RepVIT inference
        let minimap_data = chw_data.clone(); // Placeholder: in production, crop minimap region
        let repvit_output = match repvit_model.infer(
            &chw_data,
            [1, 3, config.screen.height as usize, config.screen.width as usize],
            &minimap_data,
            [1, 3, config.screen.height as usize, config.screen.width as usize],
            config.nav.map_idx,
        ) {
            Ok(out) => out,
            Err(e) => {
                warn!("RepVIT inference failed: {e}");
                std::thread::sleep(tick_duration);
                continue;
            }
        };

        // 5. Process detections
        let mut detections = detection::parse_yolo_output(
            &yolo_output,
            config.screen.width,
            config.screen.height,
        );
        detection::nms(&mut detections, 0.45, 0.25);

        // 6. Update navigation (Kalman smoothing)
        let (position, yaw, pitch) = bot
            .nav_processor_mut()
            .process(repvit_output.position, repvit_output.angles);

        // 7. Run bot logic
        let tick_data = TickData {
            detections,
            position,
            yaw,
            pitch,
        };

        if let Err(e) = bot.tick(&tick_data) {
            error!("Bot tick error: {e}");
        }

        // 8. Maintain tick rate
        let elapsed = tick_start.elapsed();
        if elapsed < tick_duration {
            std::thread::sleep(tick_duration - elapsed);
        }
    }
}
