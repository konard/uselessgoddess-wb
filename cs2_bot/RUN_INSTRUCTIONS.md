# CS2 Deathmatch Bot — Run Instructions

## Prerequisites

1. **CS2** running in windowed mode at 384×288 resolution
2. **Windows 10/11** (required for screen capture and input simulation)
3. **Rust toolchain** (1.75+ recommended)
4. **ONNX models** placed in a `models/` directory:
   - `yolo_detect.onnx` — YOLO detection model
   - `repvit_nav.onnx` — RepVIT navigation model
5. **Nav mesh JSON** (awpy export) for your map, placed in `maps/`

## Setup

```bash
# Clone and build
git clone <repo-url>
cd cs2_bot

# Copy and edit config
cp config.toml.example config.toml
# Edit config.toml with your model paths and sensitivity settings

# Build release
cargo build --release
```

## Running

```bash
# Basic run
./target/release/cs2_bot --config config.toml --map de_dust2

# With nav mesh override
./target/release/cs2_bot --config config.toml --map de_dust2 --nav-file maps/de_dust2.json

# Dry-run mode (logs actions without sending inputs)
./target/release/cs2_bot --config config.toml --map de_dust2 --dry-run
```

## Tuning Sensitivity

The mouse delta is calculated as:

```
mouse_delta = (target_px - screen_center) * game_sensitivity / fov_scale
```

- `game_sensitivity`: Must match your CS2 in-game sensitivity setting exactly
- `fov_scale`: Corresponds to your CS2 FOV. Default 90.0 = 90° horizontal FOV

If the bot overshoots targets: decrease `game_sensitivity` or increase `fov_scale`.
If the bot undershoots: increase `game_sensitivity` or decrease `fov_scale`.

## WindMouse Parameters

The bot uses the WindMouse algorithm for human-like mouse movement:

- `wind_gravity` (default 9.0): Pull strength toward target. Higher = faster aim
- `wind_wind` (default 3.0): Random perturbation. Higher = more natural curves
- `wind_max_step` (default 10.0): Max pixels per mouse event. Prevents teleporting
- `wind_min_dist` (default 12.0): Distance below which wind stops (precision zone)

## Expected Performance

- **CPU**: ~5-10% on modern quad-core (inference is the bottleneck)
- **GPU** (with CUDA): < 5% CPU, ~10-20% GPU depending on model size
- **Frame delivery**: < 2ms target for screen capture
- **Inference**: < 10ms total per tick at 384×288
- **Tick rate**: 60 Hz (16.6ms per tick budget)

## Nav Mesh Format

The bot accepts nav meshes in awpy JSON format with this structure:

```json
{
  "version": 1,
  "areas": {
    "1": {
      "area_id": 1,
      "corners": [{"x": 0, "y": 0, "z": 0}, ...],
      "connections": [2, 3],
      "ladders_above": [],
      "ladders_below": []
    }
  }
}
```

## Verbose Logging

```bash
RUST_LOG=cs2_bot=debug ./target/release/cs2_bot --config config.toml
```
