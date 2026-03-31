//! ONNX inference pipeline using the `ort` crate.
//!
//! Manages two models:
//! - YOLO: object detection (enemy heads/bodies)
//! - RepVIT: navigation (player position + orientation)

use anyhow::Result;
use ndarray::Array;
use ort::session::Session;
use ort::value::Tensor;
use std::path::Path;
use tracing::info;

/// YOLO detection model wrapper.
pub struct YoloModel {
    session: Session,
    input_width: u32,
    input_height: u32,
}

/// RepVIT navigation model wrapper.
pub struct RepVitModel {
    session: Session,
}

/// Raw YOLO inference output (before NMS).
#[derive(Debug, Clone)]
pub struct YoloRawOutput {
    /// Shape: [num_detections, 6] — (x_center, y_center, width, height, confidence, class_id)
    pub detections: Vec<[f32; 6]>,
}

/// Raw RepVIT inference output.
#[derive(Debug, Clone)]
pub struct RepVitOutput {
    /// World position [x, y, z].
    pub position: [f32; 3],
    /// Orientation angles [yaw_x, yaw_y, pitch_x, pitch_y].
    pub angles: [f32; 4],
}

impl YoloModel {
    /// Load a YOLO ONNX model.
    pub fn load(model_path: &Path, _use_cuda: bool) -> Result<Self> {
        info!(?model_path, "Loading YOLO model");

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load YOLO model: {e}"))?;

        Ok(Self {
            session,
            input_width: 384,
            input_height: 288,
        })
    }

    /// Run inference on a CHW f32 tensor.
    /// `input_data` shape: [1, 3, height, width] flattened.
    pub fn infer(&mut self, input_data: &[f32]) -> Result<YoloRawOutput> {
        let shape = [
            1,
            3,
            self.input_height as usize,
            self.input_width as usize,
        ];
        let input_array = Array::from_shape_vec(shape, input_data.to_vec())?;
        let input_tensor = Tensor::from_array(input_array)
            .map_err(|e| anyhow::anyhow!("Failed to create input tensor: {e}"))?;

        let outputs = self.session.run(
            ort::inputs![input_tensor]
        ).map_err(|e| anyhow::anyhow!("YOLO inference failed: {e}"))?;

        // YOLO output is typically [1, num_detections, 6]
        let (output_shape, output_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Output extraction failed: {e}"))?;

        let mut detections = Vec::new();

        if output_shape.len() == 3 {
            let num_dets = output_shape[1] as usize;
            let num_attrs = output_shape[2] as usize;
            for i in 0..num_dets {
                if num_attrs >= 6 {
                    let base = i * num_attrs;
                    let det = [
                        output_data[base],
                        output_data[base + 1],
                        output_data[base + 2],
                        output_data[base + 3],
                        output_data[base + 4],
                        output_data[base + 5],
                    ];
                    detections.push(det);
                }
            }
        }

        Ok(YoloRawOutput { detections })
    }

    pub fn input_width(&self) -> u32 {
        self.input_width
    }

    pub fn input_height(&self) -> u32 {
        self.input_height
    }
}

impl RepVitModel {
    /// Load a RepVIT ONNX model.
    pub fn load(model_path: &Path, _use_cuda: bool) -> Result<Self> {
        info!(?model_path, "Loading RepVIT model");

        let session = Session::builder()
            .map_err(|e| anyhow::anyhow!("Failed to create session builder: {e}"))?
            .with_intra_threads(4)
            .map_err(|e| anyhow::anyhow!("Failed to set intra threads: {e}"))?
            .commit_from_file(model_path)
            .map_err(|e| anyhow::anyhow!("Failed to load RepVIT model: {e}"))?;

        Ok(Self { session })
    }

    /// Run inference with screen frame, minimap crop, and map index.
    pub fn infer(
        &mut self,
        frame_data: &[f32],
        frame_shape: [usize; 4],
        minimap_data: &[f32],
        minimap_shape: [usize; 4],
        map_idx: i64,
    ) -> Result<RepVitOutput> {
        let frame_array = Array::from_shape_vec(frame_shape, frame_data.to_vec())?;
        let frame_tensor = Tensor::from_array(frame_array)
            .map_err(|e| anyhow::anyhow!("Failed to create frame tensor: {e}"))?;

        let minimap_array = Array::from_shape_vec(minimap_shape, minimap_data.to_vec())?;
        let minimap_tensor = Tensor::from_array(minimap_array)
            .map_err(|e| anyhow::anyhow!("Failed to create minimap tensor: {e}"))?;

        let map_array = Array::from_shape_vec([1], vec![map_idx])?;
        let map_tensor = Tensor::from_array(map_array)
            .map_err(|e| anyhow::anyhow!("Failed to create map tensor: {e}"))?;

        let outputs = self.session.run(
            ort::inputs![frame_tensor, minimap_tensor, map_tensor]
        ).map_err(|e| anyhow::anyhow!("RepVIT inference failed: {e}"))?;

        // Position output: [B, 2]
        let (_pos_shape, pos_data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Position extraction failed: {e}"))?;

        // Z output: [B, 1]
        let (_z_shape, z_data) = outputs[1]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Z extraction failed: {e}"))?;

        // Angle output: [B, 4]
        let (_angle_shape, angle_data) = outputs[2]
            .try_extract_tensor::<f32>()
            .map_err(|e| anyhow::anyhow!("Angle extraction failed: {e}"))?;

        let position = [pos_data[0], pos_data[1], z_data[0]];
        let angles = [angle_data[0], angle_data[1], angle_data[2], angle_data[3]];

        Ok(RepVitOutput { position, angles })
    }
}
