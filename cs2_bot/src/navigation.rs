//! Navigation module: Kalman filter for position smoothing, RepVIT output processing.
//!
//! Uses a 3D constant-velocity Kalman filter (6-state: x, y, z, vx, vy, vz)
//! implemented with nalgebra matrices.

use crate::math_utils::Vec3;
use nalgebra::{Matrix6, Vector6, Matrix3x6, Matrix6x3, Vector3};
use tracing::debug;

/// 3D constant-velocity Kalman filter.
///
/// State vector: [x, y, z, vx, vy, vz]
/// Measurement vector: [x, y, z]
pub struct KalmanFilter3D {
    /// State estimate.
    x: Vector6<f64>,
    /// Error covariance.
    p: Matrix6<f64>,
    /// Process noise covariance.
    q: Matrix6<f64>,
    /// Measurement noise covariance.
    r: nalgebra::Matrix3<f64>,
    /// Time step.
    dt: f64,
    /// Whether the filter has been initialized with a measurement.
    initialized: bool,
}

impl KalmanFilter3D {
    /// Create a new Kalman filter with the given time step and noise parameters.
    ///
    /// - `dt`: time step in seconds (e.g., 1/60 for 60Hz).
    /// - `process_noise`: process noise standard deviation.
    /// - `measurement_noise`: measurement noise standard deviation.
    pub fn new(dt: f64, process_noise: f64, measurement_noise: f64) -> Self {
        let q_scale = process_noise * process_noise;
        let r_scale = measurement_noise * measurement_noise;

        // Process noise: higher for velocity components
        let q = Matrix6::from_diagonal(&Vector6::new(
            q_scale * dt,
            q_scale * dt,
            q_scale * dt,
            q_scale,
            q_scale,
            q_scale,
        ));

        let r = nalgebra::Matrix3::from_diagonal(&Vector3::new(r_scale, r_scale, r_scale));

        Self {
            x: Vector6::zeros(),
            p: Matrix6::identity() * 1000.0,
            q,
            r,
            dt,
            initialized: false,
        }
    }

    /// State transition matrix F.
    fn transition_matrix(&self) -> Matrix6<f64> {
        let mut f = Matrix6::identity();
        f[(0, 3)] = self.dt;
        f[(1, 4)] = self.dt;
        f[(2, 5)] = self.dt;
        f
    }

    /// Observation matrix H: maps state to measurement.
    fn observation_matrix() -> Matrix3x6<f64> {
        let mut h = Matrix3x6::zeros();
        h[(0, 0)] = 1.0;
        h[(1, 1)] = 1.0;
        h[(2, 2)] = 1.0;
        h
    }

    /// Predict step.
    pub fn predict(&mut self) {
        let f = self.transition_matrix();
        let x = self.x;
        let p = self.p;
        self.x = f * x;
        self.p = f * p * f.transpose() + self.q;
    }

    /// Update step with a new measurement.
    pub fn update(&mut self, measurement: Vec3) {
        let z = Vector3::new(measurement.x, measurement.y, measurement.z);
        let h = Self::observation_matrix();

        if !self.initialized {
            self.x[0] = z[0];
            self.x[1] = z[1];
            self.x[2] = z[2];
            self.initialized = true;
            debug!("Kalman filter initialized with first measurement");
            return;
        }

        // Innovation
        let y = z - h * self.x;

        // Innovation covariance
        let s = h * self.p * h.transpose() + self.r;

        // Kalman gain
        let s_inv = s.try_inverse().unwrap_or_else(nalgebra::Matrix3::identity);
        let h_t: Matrix6x3<f64> = h.transpose();
        let k = self.p * h_t * s_inv;

        // State update
        self.x += k * y;

        // Covariance update (Joseph form for numerical stability)
        let i_kh = Matrix6::identity() - k * h;
        self.p = i_kh * self.p * i_kh.transpose() + k * self.r * k.transpose();
    }

    /// Run a full predict + update cycle.
    pub fn step(&mut self, measurement: Vec3) -> Vec3 {
        self.predict();
        self.update(measurement);
        self.position()
    }

    /// Get the current estimated position.
    pub fn position(&self) -> Vec3 {
        Vec3::new(self.x[0], self.x[1], self.x[2])
    }

    /// Get the current estimated velocity.
    pub fn velocity(&self) -> Vec3 {
        Vec3::new(self.x[3], self.x[4], self.x[5])
    }
}

/// 1D Kalman filter for angle smoothing (yaw or pitch).
pub struct KalmanFilter1D {
    /// State: [value, velocity].
    x: [f64; 2],
    /// Error covariance (2x2 flattened).
    p: [[f64; 2]; 2],
    /// Process noise variance.
    q: f64,
    /// Measurement noise variance.
    r: f64,
    /// Time step.
    dt: f64,
    initialized: bool,
}

impl KalmanFilter1D {
    pub fn new(dt: f64, process_noise: f64, measurement_noise: f64) -> Self {
        Self {
            x: [0.0, 0.0],
            p: [[1000.0, 0.0], [0.0, 1000.0]],
            q: process_noise * process_noise,
            r: measurement_noise * measurement_noise,
            dt,
            initialized: false,
        }
    }

    pub fn step(&mut self, measurement: f64) -> f64 {
        if !self.initialized {
            self.x[0] = measurement;
            self.initialized = true;
            return measurement;
        }

        // Predict
        let x_pred = self.x[0] + self.x[1] * self.dt;
        let v_pred = self.x[1];

        let p00 = self.p[0][0] + self.dt * (self.p[1][0] + self.p[0][1]) + self.dt * self.dt * self.p[1][1] + self.q * self.dt;
        let p01 = self.p[0][1] + self.dt * self.p[1][1];
        let p10 = self.p[1][0] + self.dt * self.p[1][1];
        let p11 = self.p[1][1] + self.q;

        // Update
        let y = measurement - x_pred;
        let s = p00 + self.r;
        let k0 = p00 / s;
        let k1 = p10 / s;

        self.x[0] = x_pred + k0 * y;
        self.x[1] = v_pred + k1 * y;

        self.p[0][0] = (1.0 - k0) * p00;
        self.p[0][1] = (1.0 - k0) * p01;
        self.p[1][0] = -k1 * p00 + p10;
        self.p[1][1] = -k1 * p01 + p11;

        self.x[0]
    }
}

/// Process RepVIT output into smoothed world-space position and angles.
pub struct NavigationProcessor {
    position_filter: KalmanFilter3D,
    yaw_filter: KalmanFilter1D,
    pitch_filter: KalmanFilter1D,
}

impl NavigationProcessor {
    pub fn new(tick_rate: u32) -> Self {
        let dt = 1.0 / tick_rate as f64;
        Self {
            position_filter: KalmanFilter3D::new(dt, 5.0, 10.0),
            yaw_filter: KalmanFilter1D::new(dt, 1.0, 2.0),
            pitch_filter: KalmanFilter1D::new(dt, 1.0, 2.0),
        }
    }

    /// Process raw RepVIT output and return smoothed position and yaw/pitch.
    pub fn process(
        &mut self,
        position: [f32; 3],
        angles: [f32; 4],
    ) -> (Vec3, f64, f64) {
        let raw_pos = Vec3::new(
            position[0] as f64,
            position[1] as f64,
            position[2] as f64,
        );

        let smoothed_pos = self.position_filter.step(raw_pos);

        // Convert yaw_x, yaw_y to yaw angle
        let yaw = (angles[1] as f64).atan2(angles[0] as f64).to_degrees();
        // Convert pitch_x, pitch_y to pitch angle
        let pitch = (angles[3] as f64).atan2(angles[2] as f64).to_degrees();

        let smoothed_yaw = self.yaw_filter.step(yaw);
        let smoothed_pitch = self.pitch_filter.step(pitch);

        (smoothed_pos, smoothed_yaw, smoothed_pitch)
    }

    pub fn position(&self) -> Vec3 {
        self.position_filter.position()
    }

    pub fn velocity(&self) -> Vec3 {
        self.position_filter.velocity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kalman_3d_reduces_noise() {
        let mut kf = KalmanFilter3D::new(1.0 / 60.0, 1.0, 5.0);

        // Feed a noisy signal around (100, 200, 300)
        let true_pos = Vec3::new(100.0, 200.0, 300.0);
        let noise_offsets = [2.0, -3.0, 1.5, -1.0, 4.0, -2.5, 0.5, 3.0, -1.5, 2.0];

        let mut raw_errors = Vec::new();
        let mut filtered_errors = Vec::new();

        for &noise in &noise_offsets {
            let noisy = Vec3::new(
                true_pos.x + noise,
                true_pos.y - noise * 0.5,
                true_pos.z + noise * 0.3,
            );
            raw_errors.push(noisy.distance_to(&true_pos));

            let filtered = kf.step(noisy);
            filtered_errors.push(filtered.distance_to(&true_pos));
        }

        // After convergence, filtered error should be lower than raw error (on average)
        let avg_raw: f64 = raw_errors.iter().sum::<f64>() / raw_errors.len() as f64;
        let avg_filtered_last5: f64 =
            filtered_errors[5..].iter().sum::<f64>() / 5.0;

        assert!(
            avg_filtered_last5 < avg_raw,
            "Kalman filter should reduce noise: avg_filtered={avg_filtered_last5}, avg_raw={avg_raw}"
        );
    }

    #[test]
    fn test_kalman_1d_smoothing() {
        let mut kf = KalmanFilter1D::new(1.0 / 60.0, 0.5, 2.0);

        // Feed constant value with noise
        let values = [10.0, 12.0, 9.0, 11.0, 10.5, 10.2, 9.8, 10.1, 10.3, 9.9];
        let mut results = Vec::new();
        for &v in &values {
            results.push(kf.step(v));
        }

        // Last few results should be close to 10.0
        let last = *results.last().unwrap();
        assert!(
            (last - 10.0).abs() < 2.0,
            "1D Kalman should converge near true value: got {last}"
        );
    }
}
