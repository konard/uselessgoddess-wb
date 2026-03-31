//! Game loop: tick-based state machine for the CS2 bot.
//!
//! States:
//! - Navigating: following nav mesh path toward objective
//! - Acquiring: enemy detected, aiming with WindMouse
//! - Engaging: on target, shooting
//! - Idle: no target, no path (recalculating)

use crate::aim_controller::{wind_mouse, OneEuroFilter, WindMouseParams};
use crate::config::Config;
use crate::detection::{self, Detection};
use crate::input_sim::{KeyboardInput, MoveDirection, MouseButton, RawMouse};
use crate::math_utils::Vec2;
use crate::navmesh::NavMesh;
use crate::navigation::NavigationProcessor;
use anyhow::Result;
use tracing::{debug, info, warn};

/// Bot behavioral state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BotState {
    /// Following nav mesh path.
    Navigating,
    /// Enemy detected, moving crosshair toward target.
    Acquiring,
    /// On target, firing.
    Engaging,
    /// No target, no path — recalculating.
    Idle,
}

/// Current tick data from perception.
pub struct TickData {
    /// Detected enemies after NMS.
    pub detections: Vec<Detection>,
    /// Current world position (smoothed).
    pub position: crate::math_utils::Vec3,
    /// Current yaw angle (smoothed).
    pub yaw: f64,
    /// Current pitch angle (smoothed).
    pub pitch: f64,
}

/// The main bot controller.
pub struct BotController {
    state: BotState,
    config: Config,
    keyboard: KeyboardInput,
    nav_processor: NavigationProcessor,
    navmesh: Option<NavMesh>,
    current_path: Vec<crate::math_utils::Vec3>,
    current_waypoint_idx: usize,
    wind_params: WindMouseParams,
    one_euro_x: Option<OneEuroFilter>,
    one_euro_y: Option<OneEuroFilter>,
    tick_count: u64,
    dry_run: bool,
}

impl BotController {
    /// Create a new bot controller from configuration.
    pub fn new(config: Config) -> Result<Self> {
        let keyboard = KeyboardInput::new()?;
        let nav_processor = NavigationProcessor::new(config.general.tick_rate);

        let navmesh = if config.nav.nav_file.exists() {
            match NavMesh::load(&config.nav.nav_file) {
                Ok(mesh) => {
                    info!(areas = mesh.areas.len(), "Nav mesh loaded");
                    Some(mesh)
                }
                Err(e) => {
                    warn!("Failed to load nav mesh: {e}");
                    None
                }
            }
        } else {
            info!("No nav mesh file specified or found");
            None
        };

        let wind_params = WindMouseParams {
            gravity: config.aim.wind_gravity,
            wind: config.aim.wind_wind,
            max_step: config.aim.wind_max_step,
            min_dist: config.aim.wind_min_dist,
            ..Default::default()
        };

        let (one_euro_x, one_euro_y) = if config.aim.one_euro {
            (
                Some(OneEuroFilter::new(
                    config.aim.one_euro_min_cutoff,
                    config.aim.one_euro_beta,
                    1.0,
                )),
                Some(OneEuroFilter::new(
                    config.aim.one_euro_min_cutoff,
                    config.aim.one_euro_beta,
                    1.0,
                )),
            )
        } else {
            (None, None)
        };

        let dry_run = config.general.dry_run;

        Ok(Self {
            state: BotState::Idle,
            config,
            keyboard,
            nav_processor,
            navmesh,
            current_path: Vec::new(),
            current_waypoint_idx: 0,
            wind_params,
            one_euro_x,
            one_euro_y,
            tick_count: 0,
            dry_run,
        })
    }

    /// Get current bot state.
    pub fn state(&self) -> BotState {
        self.state
    }

    /// Process a single tick with perception data.
    pub fn tick(&mut self, data: &TickData) -> Result<()> {
        self.tick_count += 1;

        let screen_center = Vec2::new(
            self.config.screen.width as f64 / 2.0,
            self.config.screen.height as f64 / 2.0,
        );

        // Check for targets
        let target = detection::select_target(
            &data.detections,
            screen_center,
            self.config.aim.fov_degrees,
            self.config.screen.width,
        );

        match target {
            Some(det) => {
                // Enemy found — engage
                let delta = detection::target_to_mouse_delta(
                    det.center,
                    screen_center,
                    self.config.sensitivity.game_sensitivity,
                    self.config.sensitivity.fov_scale,
                );

                let dist_to_target = det.center.distance_to(&screen_center);

                if dist_to_target < 5.0 {
                    // On target — shoot
                    self.state = BotState::Engaging;
                    self.handle_engage()?;
                } else {
                    // Aim toward target with WindMouse
                    self.state = BotState::Acquiring;
                    self.handle_acquire(delta)?;
                }
            }
            None => {
                // No target — navigate
                if self.current_path.is_empty() || self.current_waypoint_idx >= self.current_path.len() {
                    self.state = BotState::Idle;
                    self.recalculate_path(&data.position);
                } else {
                    self.state = BotState::Navigating;
                    self.handle_navigate(data)?;
                }
            }
        }

        if self.tick_count.is_multiple_of(60) {
            debug!(
                state = ?self.state,
                tick = self.tick_count,
                waypoint = self.current_waypoint_idx,
                path_len = self.current_path.len(),
                "Tick status"
            );
        }

        Ok(())
    }

    /// Handle the Acquiring state: move mouse toward target using WindMouse.
    fn handle_acquire(&mut self, delta: Vec2) -> Result<()> {
        let timestamp = self.tick_count as f64 / self.config.general.tick_rate as f64;

        // Apply One-Euro filter if enabled
        let filtered_delta = match (&mut self.one_euro_x, &mut self.one_euro_y) {
            (Some(fx), Some(fy)) => Vec2::new(
                fx.filter(delta.x, timestamp),
                fy.filter(delta.y, timestamp),
            ),
            _ => delta,
        };

        // Generate WindMouse path
        let origin = Vec2::new(0.0, 0.0);
        let moves = wind_mouse(origin, filtered_delta, &self.wind_params);

        if self.dry_run {
            debug!(move_count = moves.len(), "WindMouse moves (dry run)");
            return Ok(());
        }

        // Apply mouse movements with micro-timing
        for (dx, dy) in moves {
            RawMouse::move_relative(dx, dy)?;
            std::thread::sleep(std::time::Duration::from_micros(self.wind_params.wait_us));
        }

        Ok(())
    }

    /// Handle the Engaging state: fire at target.
    fn handle_engage(&mut self) -> Result<()> {
        if self.dry_run {
            debug!("Firing (dry run)");
            return Ok(());
        }

        RawMouse::click(MouseButton::Left)?;
        Ok(())
    }

    /// Handle the Navigating state: move toward next waypoint.
    fn handle_navigate(&mut self, data: &TickData) -> Result<()> {
        if self.current_waypoint_idx >= self.current_path.len() {
            return Ok(());
        }

        let target = self.current_path[self.current_waypoint_idx];
        let dist = data.position.distance_to(&target);

        // Check if we've reached the current waypoint
        if dist < self.config.nav.waypoint_threshold {
            self.current_waypoint_idx += 1;
            debug!(
                idx = self.current_waypoint_idx,
                "Advanced to next waypoint"
            );
            return Ok(());
        }

        // Calculate direction to waypoint
        let dx = target.x - data.position.x;
        let dy = target.y - data.position.y;
        let angle_to_target = dy.atan2(dx).to_degrees();

        // Determine WASD direction based on relative angle
        let relative_angle = angle_to_target - data.yaw;
        let dir = angle_to_movement(relative_angle);

        if self.dry_run {
            debug!(?dir, dist, "Navigate (dry run)");
            return Ok(());
        }

        // Release previous keys and press new direction
        self.keyboard.release_all_movement()?;
        self.keyboard.press_direction(dir)?;

        Ok(())
    }

    /// Recalculate navigation path when idle.
    fn recalculate_path(&mut self, current_pos: &crate::math_utils::Vec3) {
        let navmesh = match &self.navmesh {
            Some(m) => m,
            None => return,
        };

        let start = match navmesh.find_nearest_area(*current_pos) {
            Some(id) => id,
            None => return,
        };

        // Pick a random goal area (for deathmatch, wander toward hot zones)
        let areas: Vec<_> = navmesh.areas.keys().copied().collect();
        if areas.is_empty() {
            return;
        }

        use rand::RngExt;
        let mut rng = rand::rng();
        let goal_idx = rng.random_range(0..areas.len());
        let goal = areas[goal_idx];

        if let Some(path) = navmesh.find_path(start, goal) {
            let waypoints = navmesh.path_to_waypoints(&path);

            self.current_path = match self.config.nav.smooth_method.as_str() {
                "funnel" => navmesh.smooth_funnel(&path),
                _ => navmesh.smooth_catmull_rom(&waypoints, 4),
            };

            self.current_waypoint_idx = 0;
            debug!(
                waypoint_count = self.current_path.len(),
                "Path recalculated"
            );
        }
    }

    /// Get a reference to the navigation processor.
    pub fn nav_processor_mut(&mut self) -> &mut NavigationProcessor {
        &mut self.nav_processor
    }
}

/// Convert a relative angle (degrees) to a movement direction.
fn angle_to_movement(angle_deg: f64) -> MoveDirection {
    // Normalize to -180..180
    let mut a = angle_deg % 360.0;
    if a > 180.0 {
        a -= 360.0;
    }
    if a < -180.0 {
        a += 360.0;
    }

    if a.abs() < 45.0 {
        MoveDirection::Forward
    } else if a.abs() > 135.0 {
        MoveDirection::Backward
    } else if a > 0.0 {
        MoveDirection::Left
    } else {
        MoveDirection::Right
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_angle_to_movement() {
        assert_eq!(angle_to_movement(0.0), MoveDirection::Forward);
        assert_eq!(angle_to_movement(30.0), MoveDirection::Forward);
        assert_eq!(angle_to_movement(90.0), MoveDirection::Left);
        assert_eq!(angle_to_movement(-90.0), MoveDirection::Right);
        assert_eq!(angle_to_movement(180.0), MoveDirection::Backward);
        assert_eq!(angle_to_movement(-180.0), MoveDirection::Backward);
    }
}
