//! YOLO detection post-processing: NMS, class filtering, screen-to-delta conversion.

use crate::math_utils::Vec2;
use crate::onnx_inference::YoloRawOutput;

/// Detection class IDs matching the YOLO model output.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DetectionClass {
    THead = 0,
    CtHead = 1,
    TBody = 2,
    CtBody = 3,
}

impl DetectionClass {
    pub fn from_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(Self::THead),
            1 => Some(Self::CtHead),
            2 => Some(Self::TBody),
            3 => Some(Self::CtBody),
            _ => None,
        }
    }

    pub fn is_head(&self) -> bool {
        matches!(self, Self::THead | Self::CtHead)
    }
}

/// A processed detection with screen coordinates and class.
#[derive(Debug, Clone)]
pub struct Detection {
    /// Bounding box center in screen pixels.
    pub center: Vec2,
    /// Bounding box width in pixels.
    pub width: f64,
    /// Bounding box height in pixels.
    pub height: f64,
    /// Detection confidence.
    pub confidence: f64,
    /// Detection class.
    pub class: DetectionClass,
}

/// Intersection over Union between two bounding boxes (center format).
fn iou(a: &Detection, b: &Detection) -> f64 {
    let a_x1 = a.center.x - a.width / 2.0;
    let a_y1 = a.center.y - a.height / 2.0;
    let a_x2 = a.center.x + a.width / 2.0;
    let a_y2 = a.center.y + a.height / 2.0;

    let b_x1 = b.center.x - b.width / 2.0;
    let b_y1 = b.center.y - b.height / 2.0;
    let b_x2 = b.center.x + b.width / 2.0;
    let b_y2 = b.center.y + b.height / 2.0;

    let inter_x1 = a_x1.max(b_x1);
    let inter_y1 = a_y1.max(b_y1);
    let inter_x2 = a_x2.min(b_x2);
    let inter_y2 = a_y2.min(b_y2);

    let inter_w = (inter_x2 - inter_x1).max(0.0);
    let inter_h = (inter_y2 - inter_y1).max(0.0);
    let inter_area = inter_w * inter_h;

    let a_area = a.width * a.height;
    let b_area = b.width * b.height;
    let union_area = a_area + b_area - inter_area;

    if union_area > 0.0 {
        inter_area / union_area
    } else {
        0.0
    }
}

/// Apply Non-Maximum Suppression to a list of detections.
pub fn nms(detections: &mut Vec<Detection>, iou_threshold: f64, confidence_threshold: f64) {
    // Filter by confidence
    detections.retain(|d| d.confidence >= confidence_threshold);

    // Sort by confidence (highest first)
    detections.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal));

    let mut keep = vec![true; detections.len()];

    for i in 0..detections.len() {
        if !keep[i] {
            continue;
        }
        for j in (i + 1)..detections.len() {
            if !keep[j] {
                continue;
            }
            if iou(&detections[i], &detections[j]) > iou_threshold {
                keep[j] = false;
            }
        }
    }

    let mut idx = 0;
    detections.retain(|_| {
        let k = keep[idx];
        idx += 1;
        k
    });
}

/// Parse raw YOLO output into Detection structs.
pub fn parse_yolo_output(raw: &YoloRawOutput, screen_width: u32, screen_height: u32) -> Vec<Detection> {
    let mut detections = Vec::new();

    for det in &raw.detections {
        let x_center = det[0] as f64;
        let y_center = det[1] as f64;
        let width = det[2] as f64;
        let height = det[3] as f64;
        let confidence = det[4] as f64;
        let class_id = det[5] as u32;

        if let Some(class) = DetectionClass::from_id(class_id) {
            detections.push(Detection {
                center: Vec2::new(
                    x_center * screen_width as f64,
                    y_center * screen_height as f64,
                ),
                width: width * screen_width as f64,
                height: height * screen_height as f64,
                confidence,
                class,
            });
        }
    }

    detections
}

/// Select the best target from detections.
/// Priority: heads > bodies, nearest to crosshair wins.
/// Only targets within `fov_degrees` from screen center are considered.
pub fn select_target(
    detections: &[Detection],
    screen_center: Vec2,
    fov_degrees: f64,
    screen_width: u32,
) -> Option<&Detection> {
    // Convert FOV to pixel radius (approximate)
    let fov_radius_px = (fov_degrees / 90.0) * (screen_width as f64 / 2.0);

    // Separate heads and bodies, filter by FOV
    let in_fov: Vec<&Detection> = detections
        .iter()
        .filter(|d| d.center.distance_to(&screen_center) <= fov_radius_px)
        .collect();

    // Try heads first
    let heads: Vec<&&Detection> = in_fov.iter().filter(|d| d.class.is_head()).collect();

    let candidates = if heads.is_empty() { &in_fov } else {
        // Return the closest head
        return heads
            .into_iter()
            .min_by(|a, b| {
                let da = a.center.distance_to(&screen_center);
                let db = b.center.distance_to(&screen_center);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied();
    };

    // Closest body
    candidates
        .iter()
        .min_by(|a, b| {
            let da = a.center.distance_to(&screen_center);
            let db = b.center.distance_to(&screen_center);
            da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
        })
        .copied()
}

/// Compute mouse delta from screen target position.
pub fn target_to_mouse_delta(
    target_px: Vec2,
    screen_center: Vec2,
    sens_factor: f64,
    fov_scale: f64,
) -> Vec2 {
    let offset = target_px - screen_center;
    Vec2::new(
        offset.x * sens_factor / fov_scale,
        offset.y * sens_factor / fov_scale,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_detection(cx: f64, cy: f64, w: f64, h: f64, conf: f64, class: DetectionClass) -> Detection {
        Detection {
            center: Vec2::new(cx, cy),
            width: w,
            height: h,
            confidence: conf,
            class,
        }
    }

    #[test]
    fn test_nms_removes_overlapping() {
        let mut dets = vec![
            make_detection(100.0, 100.0, 50.0, 50.0, 0.9, DetectionClass::THead),
            make_detection(105.0, 105.0, 50.0, 50.0, 0.8, DetectionClass::THead),
            make_detection(300.0, 300.0, 50.0, 50.0, 0.7, DetectionClass::TBody),
        ];
        nms(&mut dets, 0.5, 0.5);
        // The overlapping detection (second) should be removed
        assert_eq!(dets.len(), 2);
        assert!((dets[0].confidence - 0.9).abs() < 1e-5);
        assert!((dets[1].confidence - 0.7).abs() < 1e-5);
    }

    #[test]
    fn test_nms_filters_low_confidence() {
        let mut dets = vec![
            make_detection(100.0, 100.0, 50.0, 50.0, 0.3, DetectionClass::THead),
            make_detection(200.0, 200.0, 50.0, 50.0, 0.1, DetectionClass::TBody),
        ];
        nms(&mut dets, 0.5, 0.5);
        assert!(dets.is_empty());
    }

    #[test]
    fn test_select_target_prefers_heads() {
        let dets = vec![
            make_detection(200.0, 150.0, 30.0, 30.0, 0.9, DetectionClass::TBody),
            make_detection(195.0, 145.0, 20.0, 20.0, 0.85, DetectionClass::THead),
        ];
        let center = Vec2::new(192.0, 144.0);
        let target = select_target(&dets, center, 45.0, 384);
        assert!(target.is_some());
        assert!(target.unwrap().class.is_head());
    }

    #[test]
    fn test_target_to_mouse_delta() {
        let target = Vec2::new(200.0, 150.0);
        let center = Vec2::new(192.0, 144.0);
        let delta = target_to_mouse_delta(target, center, 2.0, 90.0);
        assert!((delta.x - (8.0 * 2.0 / 90.0)).abs() < 1e-10);
        assert!((delta.y - (6.0 * 2.0 / 90.0)).abs() < 1e-10);
    }

    #[test]
    fn test_iou_no_overlap() {
        let a = make_detection(0.0, 0.0, 10.0, 10.0, 1.0, DetectionClass::THead);
        let b = make_detection(100.0, 100.0, 10.0, 10.0, 1.0, DetectionClass::THead);
        assert!((iou(&a, &b) - 0.0).abs() < 1e-10);
    }

    #[test]
    fn test_iou_full_overlap() {
        let a = make_detection(50.0, 50.0, 10.0, 10.0, 1.0, DetectionClass::THead);
        let b = make_detection(50.0, 50.0, 10.0, 10.0, 1.0, DetectionClass::THead);
        assert!((iou(&a, &b) - 1.0).abs() < 1e-10);
    }
}
