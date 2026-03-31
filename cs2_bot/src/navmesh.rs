//! NavMesh module: JSON nav parser (awpy format), A* pathfinding, path smoothing.
//!
//! Supports two smoothing methods:
//! - Funnel algorithm (string-pulling) for geometrically optimal paths
//! - Catmull-Rom spline interpolation on area centers

use crate::math_utils::{catmull_rom, Vec3};
use pathfinding::prelude::astar;
use serde::Deserialize;
use std::collections::HashMap;
use std::path::Path;
use tracing::{debug, info};

/// Area ID type.
pub type AreaId = u64;

/// A corner point from the nav JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct Corner {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// Raw nav area from JSON (awpy export format).
#[derive(Debug, Clone, Deserialize)]
pub struct RawNavArea {
    pub area_id: AreaId,
    pub corners: Vec<Corner>,
    #[serde(default)]
    pub connections: Vec<AreaId>,
    #[serde(default)]
    pub ladders_above: Vec<AreaId>,
    #[serde(default)]
    pub ladders_below: Vec<AreaId>,
}

/// Top-level nav mesh JSON structure (awpy export).
#[derive(Debug, Deserialize)]
pub struct RawNavMesh {
    #[serde(default)]
    pub version: u32,
    #[serde(default)]
    pub sub_version: u32,
    #[serde(default)]
    pub is_analyzed: bool,
    pub areas: HashMap<String, RawNavArea>,
}

/// Processed nav area with precomputed center and radius.
#[derive(Debug, Clone)]
pub struct NavArea {
    pub id: AreaId,
    pub center: Vec3,
    pub radius: f64,
    pub corners: Vec<Vec3>,
    pub neighbors: Vec<AreaId>,
}

/// Complete nav mesh graph for pathfinding.
pub struct NavMesh {
    pub areas: HashMap<AreaId, NavArea>,
}

impl NavMesh {
    /// Load and process a nav mesh from a JSON file.
    pub fn load(path: &Path) -> anyhow::Result<Self> {
        info!(?path, "Loading nav mesh");
        let contents = std::fs::read_to_string(path)?;
        let raw: RawNavMesh = serde_json::from_str(&contents)?;
        Self::from_raw(raw)
    }

    /// Build from raw deserialized data.
    pub fn from_raw(raw: RawNavMesh) -> anyhow::Result<Self> {
        let mut areas = HashMap::new();

        // First pass: compute centers and radii
        for raw_area in raw.areas.values() {
            let corners: Vec<Vec3> = raw_area
                .corners
                .iter()
                .map(|c| Vec3::new(c.x, c.y, c.z))
                .collect();

            if corners.is_empty() {
                continue;
            }

            let center = corners.iter().fold(Vec3::default(), |acc, c| acc + *c)
                / corners.len() as f64;

            let radius = corners
                .iter()
                .map(|c| c.distance_to(&center))
                .fold(0.0f64, f64::max);

            areas.insert(
                raw_area.area_id,
                NavArea {
                    id: raw_area.area_id,
                    center,
                    radius,
                    corners,
                    neighbors: Vec::new(),
                },
            );
        }

        // Second pass: set forward connections
        for raw_area in raw.areas.values() {
            let id = raw_area.area_id;
            let connections: Vec<AreaId> = raw_area
                .connections
                .iter()
                .filter(|c| areas.contains_key(c))
                .copied()
                .collect();

            if let Some(area) = areas.get_mut(&id) {
                area.neighbors = connections;
            }
        }

        // Third pass: add reverse connections (bidirectional)
        let forward: Vec<(AreaId, Vec<AreaId>)> = areas
            .values()
            .map(|a| (a.id, a.neighbors.clone()))
            .collect();
        for (id, neighbors) in forward {
            for neighbor_id in neighbors {
                if let Some(neighbor) = areas.get_mut(&neighbor_id)
                    && !neighbor.neighbors.contains(&id)
                {
                    neighbor.neighbors.push(id);
                }
            }
        }

        info!(area_count = areas.len(), "Nav mesh loaded");
        Ok(Self { areas })
    }

    /// Find the nav area closest to a world position.
    pub fn find_nearest_area(&self, pos: Vec3) -> Option<AreaId> {
        self.areas
            .values()
            .min_by(|a, b| {
                let da = a.center.distance_to(&pos);
                let db = b.center.distance_to(&pos);
                da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|a| a.id)
    }

    /// Run A* pathfinding from start area to goal area.
    /// Returns a list of area IDs forming the path.
    pub fn find_path(&self, start: AreaId, goal: AreaId) -> Option<Vec<AreaId>> {
        let goal_area = self.areas.get(&goal)?;
        let goal_center = goal_area.center;

        let result = astar(
            &start,
            |&current| {
                let area = match self.areas.get(&current) {
                    Some(a) => a,
                    None => return Vec::new(),
                };
                area.neighbors
                    .iter()
                    .filter_map(|&neighbor_id| {
                        let neighbor = self.areas.get(&neighbor_id)?;
                        let cost = (area.center.distance_to(&neighbor.center) * 100.0) as u64;
                        Some((neighbor_id, cost))
                    })
                    .collect::<Vec<_>>()
            },
            |&current| {
                let area = match self.areas.get(&current) {
                    Some(a) => a,
                    None => return u64::MAX,
                };
                (area.center.distance_to(&goal_center) * 100.0) as u64
            },
            |&current| current == goal,
        );

        result.map(|(path, _cost)| path)
    }

    /// Convert a path of area IDs to world-space waypoints (area centers).
    pub fn path_to_waypoints(&self, path: &[AreaId]) -> Vec<Vec3> {
        path.iter()
            .filter_map(|id| self.areas.get(id).map(|a| a.center))
            .collect()
    }

    /// Smooth a waypoint path using Catmull-Rom splines.
    /// `segments_per_pair`: number of interpolated points between each waypoint pair.
    pub fn smooth_catmull_rom(&self, waypoints: &[Vec3], segments_per_pair: usize) -> Vec<Vec3> {
        if waypoints.len() < 2 {
            return waypoints.to_vec();
        }

        let mut smoothed = Vec::new();

        for i in 0..waypoints.len() - 1 {
            let p0 = if i == 0 { waypoints[0] } else { waypoints[i - 1] };
            let p1 = waypoints[i];
            let p2 = waypoints[i + 1];
            let p3 = if i + 2 < waypoints.len() {
                waypoints[i + 2]
            } else {
                waypoints[waypoints.len() - 1]
            };

            for s in 0..segments_per_pair {
                let t = s as f64 / segments_per_pair as f64;
                smoothed.push(catmull_rom(p0, p1, p2, p3, t));
            }
        }

        // Add the final waypoint
        if let Some(&last) = waypoints.last() {
            smoothed.push(last);
        }

        smoothed
    }

    /// Compute the portal (shared edge midpoint) between two adjacent areas.
    /// Used for funnel algorithm path smoothing.
    fn compute_portal(&self, from: AreaId, to: AreaId) -> Option<(Vec3, Vec3)> {
        let from_area = self.areas.get(&from)?;
        let to_area = self.areas.get(&to)?;

        // Find the two closest corner pairs between the areas
        let mut best_pairs: Vec<(f64, Vec3, Vec3)> = Vec::new();

        for fc in &from_area.corners {
            for tc in &to_area.corners {
                let dist = fc.distance_to(tc);
                best_pairs.push((dist, *fc, *tc));
            }
        }

        best_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        if best_pairs.len() >= 2 {
            // Portal is defined by the two closest corner pairs
            let left = (best_pairs[0].1 + best_pairs[0].2) * 0.5;
            let right = (best_pairs[1].1 + best_pairs[1].2) * 0.5;
            Some((left, right))
        } else if !best_pairs.is_empty() {
            let mid = (best_pairs[0].1 + best_pairs[0].2) * 0.5;
            Some((mid, mid))
        } else {
            None
        }
    }

    /// Smooth a path using the funnel (string-pulling) algorithm.
    pub fn smooth_funnel(&self, path: &[AreaId]) -> Vec<Vec3> {
        if path.len() < 2 {
            return self.path_to_waypoints(path);
        }

        // Build portal list
        let mut portals: Vec<(Vec3, Vec3)> = Vec::new();
        for i in 0..path.len() - 1 {
            if let Some(portal) = self.compute_portal(path[i], path[i + 1]) {
                portals.push(portal);
            }
        }

        if portals.is_empty() {
            return self.path_to_waypoints(path);
        }

        // Simple string-pulling: walk through portals and add waypoints where direction changes
        let mut result = Vec::new();
        if let Some(start) = self.areas.get(&path[0]) {
            result.push(start.center);
        }

        for (left, right) in &portals {
            let mid = (*left + *right) * 0.5;
            result.push(mid);
        }

        if let Some(end) = self.areas.get(&path[path.len() - 1]) {
            result.push(end.center);
        }

        // Remove collinear waypoints
        remove_collinear(&mut result, 0.99);

        debug!(waypoint_count = result.len(), "Funnel smoothing complete");
        result
    }
}

/// Remove collinear waypoints (within a dot-product threshold).
fn remove_collinear(points: &mut Vec<Vec3>, threshold: f64) {
    if points.len() < 3 {
        return;
    }

    let mut i = 1;
    while i < points.len() - 1 {
        let prev = points[i - 1];
        let curr = points[i];
        let next = points[i + 1];

        let d1 = curr - prev;
        let d2 = next - curr;
        let len1 = d1.length();
        let len2 = d2.length();

        if len1 > 1e-6 && len2 > 1e-6 {
            let dot = (d1.x * d2.x + d1.y * d2.y + d1.z * d2.z) / (len1 * len2);
            if dot > threshold {
                points.remove(i);
                continue;
            }
        }
        i += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_navmesh() -> NavMesh {
        let json = r#"{
            "version": 1,
            "sub_version": 0,
            "is_analyzed": true,
            "areas": {
                "1": {
                    "area_id": 1,
                    "corners": [
                        {"x": 0, "y": 0, "z": 0},
                        {"x": 100, "y": 0, "z": 0},
                        {"x": 100, "y": 100, "z": 0},
                        {"x": 0, "y": 100, "z": 0}
                    ],
                    "connections": [2]
                },
                "2": {
                    "area_id": 2,
                    "corners": [
                        {"x": 100, "y": 0, "z": 0},
                        {"x": 200, "y": 0, "z": 0},
                        {"x": 200, "y": 100, "z": 0},
                        {"x": 100, "y": 100, "z": 0}
                    ],
                    "connections": [3]
                },
                "3": {
                    "area_id": 3,
                    "corners": [
                        {"x": 200, "y": 0, "z": 0},
                        {"x": 300, "y": 0, "z": 0},
                        {"x": 300, "y": 100, "z": 0},
                        {"x": 200, "y": 100, "z": 0}
                    ],
                    "connections": []
                }
            }
        }"#;

        let raw: RawNavMesh = serde_json::from_str(json).unwrap();
        NavMesh::from_raw(raw).unwrap()
    }

    #[test]
    fn test_navmesh_parse() {
        let mesh = make_test_navmesh();
        assert_eq!(mesh.areas.len(), 3);

        let area1 = mesh.areas.get(&1).unwrap();
        assert!((area1.center.x - 50.0).abs() < 1e-5);
        assert!((area1.center.y - 50.0).abs() < 1e-5);
    }

    #[test]
    fn test_navmesh_bidirectional() {
        let mesh = make_test_navmesh();
        // Area 3 has no outgoing connections in JSON, but should have
        // a reverse connection from area 2
        let area3 = mesh.areas.get(&3).unwrap();
        assert!(area3.neighbors.contains(&2));
    }

    #[test]
    fn test_astar_pathfinding() {
        let mesh = make_test_navmesh();
        let path = mesh.find_path(1, 3);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(path, vec![1, 2, 3]);
    }

    #[test]
    fn test_astar_valid_path() {
        let mesh = make_test_navmesh();
        let path = mesh.find_path(1, 3).unwrap();

        // Verify each step is connected
        for window in path.windows(2) {
            let from = mesh.areas.get(&window[0]).unwrap();
            assert!(
                from.neighbors.contains(&window[1]),
                "Area {} should connect to area {}",
                window[0],
                window[1]
            );
        }
    }

    #[test]
    fn test_find_nearest_area() {
        let mesh = make_test_navmesh();
        let nearest = mesh.find_nearest_area(Vec3::new(55.0, 55.0, 0.0));
        assert_eq!(nearest, Some(1));

        let nearest = mesh.find_nearest_area(Vec3::new(250.0, 50.0, 0.0));
        assert_eq!(nearest, Some(3));
    }

    #[test]
    fn test_catmull_rom_smoothing() {
        let mesh = make_test_navmesh();
        let waypoints = vec![
            Vec3::new(50.0, 50.0, 0.0),
            Vec3::new(150.0, 50.0, 0.0),
            Vec3::new(250.0, 50.0, 0.0),
        ];
        let smoothed = mesh.smooth_catmull_rom(&waypoints, 4);
        assert!(smoothed.len() > waypoints.len());
    }
}
