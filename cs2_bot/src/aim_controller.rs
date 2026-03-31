//! Aim controller: WindMouse human-like mouse movement, One-Euro filter, target selection.
//!
//! WindMouse algorithm implemented exactly as described in:
//! https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/

use crate::math_utils::Vec2;
use rand::RngExt;
use std::f64::consts::SQRT_2;

/// WindMouse parameters for human-like mouse movement.
#[derive(Debug, Clone)]
pub struct WindMouseParams {
    /// Gravity force — pulls cursor toward target.
    pub gravity: f64,
    /// Wind force — random perturbation strength.
    pub wind: f64,
    /// Maximum step size per move.
    pub max_step: f64,
    /// Minimum distance where wind applies (below this, only gravity).
    pub min_dist: f64,
    /// Target wait time between moves in microseconds.
    pub wait_us: u64,
}

impl Default for WindMouseParams {
    fn default() -> Self {
        Self {
            gravity: 9.0,
            wind: 3.0,
            max_step: 10.0,
            min_dist: 12.0,
            wait_us: 500,
        }
    }
}

/// Generate a sequence of relative mouse movements using the WindMouse algorithm.
///
/// The algorithm simulates realistic human mouse movement by combining:
/// - Gravity: a constant pull toward the target
/// - Wind: random forces that create natural curve/wobble
///
/// Near the target, wind is suppressed and gravity dominates for precision.
///
/// Returns a list of (dx, dy) integer moves to apply sequentially.
pub fn wind_mouse(start: Vec2, end: Vec2, params: &WindMouseParams) -> Vec<(i32, i32)> {
    let mut rng = rand::rng();
    let mut moves = Vec::new();

    let mut current_x = start.x;
    let mut current_y = start.y;

    let mut wind_x = 0.0;
    let mut wind_y = 0.0;

    let sqrt2 = SQRT_2;
    let sqrt3 = 3.0f64.sqrt();

    loop {
        let dist = ((end.x - current_x).powi(2) + (end.y - current_y).powi(2)).sqrt();

        if dist < 1.0 {
            break;
        }

        // Wind component: random walk, magnitude decreases near target
        if dist >= params.min_dist {
            wind_x = wind_x / sqrt3 + (rng.random::<f64>() * (params.wind * 2.0 + 1.0) - params.wind) / sqrt2;
            wind_y = wind_y / sqrt3 + (rng.random::<f64>() * (params.wind * 2.0 + 1.0) - params.wind) / sqrt2;
        } else {
            wind_x /= sqrt2;
            wind_y /= sqrt2;
            if params.max_step < 3.0 {
                // Very close: just step directly
                let remaining_x = end.x - current_x;
                let remaining_y = end.y - current_y;
                moves.push((remaining_x.round() as i32, remaining_y.round() as i32));
                break;
            }
        }

        // Gravity component: pulls toward target
        let velo_x = wind_x + params.gravity * (end.x - current_x) / dist;
        let velo_y = wind_y + params.gravity * (end.y - current_y) / dist;

        // Clamp velocity to max step
        let velo_mag = (velo_x.powi(2) + velo_y.powi(2)).sqrt();
        let max_step = if dist < params.min_dist {
            // Near target: smaller steps for precision
            (params.max_step * (dist / params.min_dist)).max(1.0)
        } else {
            params.max_step
        };

        let (step_x, step_y) = if velo_mag > max_step {
            let scale = max_step / velo_mag;
            (velo_x * scale, velo_y * scale)
        } else {
            (velo_x, velo_y)
        };

        let dx = step_x.round() as i32;
        let dy = step_y.round() as i32;

        if dx != 0 || dy != 0 {
            moves.push((dx, dy));
        }

        current_x += step_x;
        current_y += step_y;
    }

    // Final correction: ensure we arrive exactly at the target
    let final_dx = (end.x - current_x).round() as i32;
    let final_dy = (end.y - current_y).round() as i32;
    if final_dx != 0 || final_dy != 0 {
        moves.push((final_dx, final_dy));
    }

    moves
}

/// One-Euro filter for adaptive low-pass filtering.
///
/// Based on: https://cristal.univ-lille.fr/~casiez/1euro/
///
/// The filter adapts its cutoff frequency based on the speed of the input signal:
/// - When the signal is slow-moving, use a low cutoff (heavy smoothing)
/// - When the signal is fast-moving, use a higher cutoff (less lag)
pub struct OneEuroFilter {
    /// Minimum cutoff frequency (Hz). Lower = more smoothing when stationary.
    min_cutoff: f64,
    /// Speed coefficient. Higher = less lag during fast movement.
    beta: f64,
    /// Derivative cutoff frequency (Hz).
    d_cutoff: f64,
    /// Previous filtered value.
    prev_value: Option<f64>,
    /// Previous filtered derivative.
    prev_derivative: f64,
    /// Previous timestamp.
    prev_timestamp: Option<f64>,
}

impl OneEuroFilter {
    pub fn new(min_cutoff: f64, beta: f64, d_cutoff: f64) -> Self {
        Self {
            min_cutoff,
            beta,
            d_cutoff,
            prev_value: None,
            prev_derivative: 0.0,
            prev_timestamp: None,
        }
    }

    /// Compute the smoothing factor alpha from cutoff frequency and sample rate.
    fn alpha(cutoff: f64, dt: f64) -> f64 {
        let tau = 1.0 / (2.0 * std::f64::consts::PI * cutoff);
        dt / (dt + tau)
    }

    /// Filter a new value at the given timestamp (in seconds).
    pub fn filter(&mut self, value: f64, timestamp: f64) -> f64 {
        match (self.prev_value, self.prev_timestamp) {
            (Some(prev_val), Some(prev_ts)) => {
                let dt = (timestamp - prev_ts).max(1e-6);

                // Estimate derivative
                let raw_derivative = (value - prev_val) / dt;
                let alpha_d = Self::alpha(self.d_cutoff, dt);
                let filtered_derivative =
                    alpha_d * raw_derivative + (1.0 - alpha_d) * self.prev_derivative;

                // Adaptive cutoff
                let cutoff = self.min_cutoff + self.beta * filtered_derivative.abs();

                // Filter the value
                let alpha = Self::alpha(cutoff, dt);
                let filtered = alpha * value + (1.0 - alpha) * prev_val;

                self.prev_value = Some(filtered);
                self.prev_derivative = filtered_derivative;
                self.prev_timestamp = Some(timestamp);

                filtered
            }
            _ => {
                self.prev_value = Some(value);
                self.prev_timestamp = Some(timestamp);
                value
            }
        }
    }

    /// Reset the filter state.
    pub fn reset(&mut self) {
        self.prev_value = None;
        self.prev_derivative = 0.0;
        self.prev_timestamp = None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wind_mouse_no_teleport() {
        let start = Vec2::new(0.0, 0.0);
        let end = Vec2::new(100.0, 80.0);
        let params = WindMouseParams {
            max_step: 10.0,
            ..Default::default()
        };

        let moves = wind_mouse(start, end, &params);

        // Verify no single move exceeds max_step by a large margin
        for (dx, dy) in &moves {
            let step_size = ((*dx as f64).powi(2) + (*dy as f64).powi(2)).sqrt();
            assert!(
                step_size <= params.max_step * 2.0 + 1.0,
                "Step {dx},{dy} (size {step_size}) exceeds max_step {} by too much",
                params.max_step
            );
        }
    }

    #[test]
    fn test_wind_mouse_reaches_target() {
        let start = Vec2::new(10.0, 20.0);
        let end = Vec2::new(150.0, 100.0);
        let params = WindMouseParams::default();

        let moves = wind_mouse(start, end, &params);

        // Sum up all moves to verify we approximately reach the target
        let total_dx: i32 = moves.iter().map(|(dx, _)| dx).sum();
        let total_dy: i32 = moves.iter().map(|(_, dy)| dy).sum();

        let final_x = start.x + total_dx as f64;
        let final_y = start.y + total_dy as f64;

        let error = ((final_x - end.x).powi(2) + (final_y - end.y).powi(2)).sqrt();
        assert!(
            error < 5.0,
            "WindMouse should reach target: error={error}, final=({final_x},{final_y}), target=({},{})",
            end.x,
            end.y
        );
    }

    #[test]
    fn test_wind_mouse_generates_moves() {
        let start = Vec2::new(0.0, 0.0);
        let end = Vec2::new(50.0, 50.0);
        let params = WindMouseParams::default();

        let moves = wind_mouse(start, end, &params);
        assert!(
            moves.len() > 1,
            "WindMouse should generate multiple moves, got {}",
            moves.len()
        );
    }

    #[test]
    fn test_wind_mouse_zero_distance() {
        let pos = Vec2::new(100.0, 100.0);
        let params = WindMouseParams::default();

        let moves = wind_mouse(pos, pos, &params);
        // Should produce zero or very few moves
        let total: i32 = moves.iter().map(|(dx, dy)| dx.abs() + dy.abs()).sum();
        assert!(total < 3, "Zero distance should produce minimal movement");
    }

    #[test]
    fn test_one_euro_filter_smoothing() {
        let mut filter = OneEuroFilter::new(1.0, 0.007, 1.0);

        // Feed a jittery signal
        let values = [10.0, 10.5, 9.5, 10.2, 9.8, 10.1, 9.9, 10.0];
        let mut results = Vec::new();

        for (i, &v) in values.iter().enumerate() {
            let t = i as f64 / 60.0;
            results.push(filter.filter(v, t));
        }

        // The filtered output should have less variance than input
        let input_var = variance(&values);
        let output_var = variance(&results);

        assert!(
            output_var <= input_var,
            "One-Euro filter should reduce variance: input={input_var}, output={output_var}"
        );
    }

    fn variance(data: &[f64]) -> f64 {
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64
    }
}
