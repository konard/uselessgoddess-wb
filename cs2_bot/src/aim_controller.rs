//! Aim controller: WindMouse human-like mouse movement, One-Euro filter, target selection.
//!
//! WindMouse algorithm implemented exactly as described in:
//! https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/

use crate::math_utils::Vec2;
use rand::RngExt;

/// WindMouse parameters for human-like mouse movement.
#[derive(Debug, Clone)]
pub struct WindMouseParams {
    /// G_0: Gravity constant — strength of pull toward destination.
    pub gravity: f64,
    /// W_0: Wind constant — maximum magnitude of random wind force.
    pub wind: f64,
    /// M_0: Maximum step size / velocity clamp.
    pub max_step: f64,
    /// D_0: Distance threshold — switches from "far" to "near" behavior.
    pub min_dist: f64,
    /// Target wait time between moves in microseconds.
    pub wait_us: u64,
}

impl Default for WindMouseParams {
    fn default() -> Self {
        Self {
            gravity: 9.0,
            wind: 3.0,
            max_step: 15.0,
            min_dist: 12.0,
            wait_us: 500,
        }
    }
}

/// Generate a sequence of mouse move points using the WindMouse algorithm.
///
/// Exact port of the reference Python implementation from:
/// https://ben.land/post/2021/04/25/windmouse-human-mouse-movement/
///
/// The algorithm models the cursor as a particle subject to:
/// - **Gravity**: deterministic pull toward destination (like a spring)
/// - **Wind**: stochastic force for natural wobble (decays near target)
///
/// Returns a list of (dx, dy) relative integer moves to apply sequentially.
pub fn wind_mouse(start: Vec2, end: Vec2, params: &WindMouseParams) -> Vec<(i32, i32)> {
    let mut rng = rand::rng();
    let mut moves = Vec::new();

    let sqrt3 = 3.0f64.sqrt();
    let sqrt5 = 5.0f64.sqrt();

    // Floating-point position accumulators
    let mut pos_x = start.x;
    let mut pos_y = start.y;

    // Last emitted integer position
    let mut current_ix = pos_x.round() as i32;
    let mut current_iy = pos_y.round() as i32;

    // Velocity (accumulated, not recomputed each step)
    let mut v_x = 0.0;
    let mut v_y = 0.0;

    // Wind vector (smoothed random walk)
    let mut w_x = 0.0;
    let mut w_y = 0.0;

    // M_0 is mutated during execution (shrinks near target)
    let mut m_0 = params.max_step;

    loop {
        let dist = (end.x - pos_x).hypot(end.y - pos_y);
        if dist < 1.0 {
            break;
        }

        // Wind magnitude capped by distance
        let w_mag = params.wind.min(dist);

        if dist >= params.min_dist {
            // Far from target: wind has random perturbation
            w_x = w_x / sqrt3 + (2.0 * rng.random::<f64>() - 1.0) * w_mag / sqrt5;
            w_y = w_y / sqrt3 + (2.0 * rng.random::<f64>() - 1.0) * w_mag / sqrt5;
        } else {
            // Near target: wind decays, step size shrinks
            w_x /= sqrt3;
            w_y /= sqrt3;
            if m_0 < 3.0 {
                m_0 = rng.random::<f64>() * 3.0 + 3.0;
            } else {
                m_0 /= sqrt5;
            }
        }

        // Accumulate velocity: wind + gravity (unit vector toward dest scaled by G_0)
        v_x += w_x + params.gravity * (end.x - pos_x) / dist;
        v_y += w_y + params.gravity * (end.y - pos_y) / dist;

        // Clamp velocity magnitude to M_0
        let v_mag = v_x.hypot(v_y);
        if v_mag > m_0 {
            let v_clip = m_0 / 2.0 + rng.random::<f64>() * m_0 / 2.0;
            v_x = (v_x / v_mag) * v_clip;
            v_y = (v_y / v_mag) * v_clip;
        }

        // Update floating-point position
        pos_x += v_x;
        pos_y += v_y;

        // Emit integer move only if the rounded position changed
        let move_ix = pos_x.round() as i32;
        let move_iy = pos_y.round() as i32;
        if current_ix != move_ix || current_iy != move_iy {
            let dx = move_ix - current_ix;
            let dy = move_iy - current_iy;
            moves.push((dx, dy));
            current_ix = move_ix;
            current_iy = move_iy;
        }
    }

    // Final correction to land exactly on target
    let final_ix = end.x.round() as i32;
    let final_iy = end.y.round() as i32;
    if current_ix != final_ix || current_iy != final_iy {
        moves.push((final_ix - current_ix, final_iy - current_iy));
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
