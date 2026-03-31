# CS2 Deathmatch Bot — Implementation Plan

## Module Checklist

- [ ] 1. screen_capture   — zero-copy frame grab, target < 2ms latency
- [ ] 2. onnx_inference   — dual-model pipeline with ort, batched if beneficial
- [ ] 3. detection        — YOLO output parsing, NMS, screen->world projection
- [ ] 4. navigation       — RepVIT output -> world XYZ, Kalman smoothing (port from Python)
- [ ] 5. navmesh          — CS2 nav JSON parser + A* pathfinding, waypoint smoothing
- [ ] 6. aim_controller   — WindMouse impl, FOV-gated target selection, speed curves
- [ ] 7. input_sim        — mouse (raw Win32 SendInput) + keyboard (enigo)
- [ ] 8. game_loop        — tick-based main loop, state machine (navigate->aim->shoot)
- [ ] 9. config           — TOML config for sensitivity, FOV, speeds, model paths
- [ ] 10. cli             — clap-based entry point, --map flag, --dry-run mode

## Architecture

```
main.rs (CLI + entry)
  |
  +-- config.rs        (TOML deserialization)
  +-- screen_capture.rs (scap wrapper)
  +-- onnx_inference.rs (ort session management)
  +-- detection.rs      (YOLO post-processing, NMS)
  +-- navigation.rs     (Kalman filter, RepVIT output processing)
  +-- navmesh.rs        (JSON nav parser, A*, path smoothing)
  +-- aim_controller.rs (WindMouse, One-Euro filter, target selection)
  +-- input_sim.rs      (RawMouse + keyboard abstraction)
  +-- game_loop.rs      (state machine, tick loop)
  +-- math_utils.rs     (Vec2, Vec3, shared math)
```

## Filter Decisions

- **Position smoothing**: 3D constant-velocity Kalman (6-state: x,y,z,vx,vy,vz)
- **Aim smoothing**: WindMouse only (no pre-filtering). One-Euro filter as optional layer.
- **Path smoothing**: Both funnel and Catmull-Rom available via config.

## Inference Pipeline

- YOLO and RepVIT run concurrently via tokio::spawn_blocking
- Pre-allocated tensors outside hot loop
- Target: < 10ms total per tick at 384x288
