# Tennis Court Line Simulation

Lightweight synthetic data and training framework for low-mounted wide-angle tennis court line detection.

## Structure

```text
configs/                 Example generation settings
docs/                    Plans, research, and experiment notes
src/tenniscourt/         Python package
  camera.py              OpenCV camera projection helpers
  court.py               Metric tennis court geometry
  render.py              Synthetic RGB/mask renderer
  generate.py            Dataset generation CLI
  data.py                PyTorch dataset loader
  model.py               Small segmentation model
  train.py               GPU-first training CLI
```

Generated datasets and training runs are written under `outputs/` and `runs/`, both ignored by git.

## Setup

```bash
uv sync
```

For training:

```bash
uv sync --extra train
```

## Generate A Dataset

```bash
uv run tenniscourt-generate --config configs/sim.toml --count 1000 --out outputs/synth
```

Each sample contains:

- `images/*.png`: synthetic RGB image
- `masks/*.png`: binary court-line mask
- `labels/*.json`: camera pose, K/D, visible line polylines, keypoints, and metadata

Court lines are rendered as 8 cm metric strips on the ground plane, then projected through the OpenCV camera model. Pinhole/no-distortion strips are rendered as projected quadrilaterals, so their edges stay straight. RGB output uses supersampling to reduce jagged edges. The net is also projected by OpenCV and drawn as an image occluder; it is not labeled as a court line.

## Train

```bash
uv run tenniscourt-train --data outputs/synth --epochs 10 --batch-size 16 --device auto
```

Training is multi-task: the model predicts a court-line mask and 14 keypoint heatmaps. `--device auto` uses CUDA when PyTorch can see a GPU. Use `--require-cuda` when a CPU fallback should be treated as an error.

## Interactive Viewer

The viewer uses the same Python/OpenCV image renderer as dataset generation. It does not use Three.js, WebGL, Blender, or another 3D engine.

```bash
uv sync --extra viewer
uv run --extra viewer tenniscourt-viewer
```

Default start pose is a low robot-like camera at `(0, -13, 0.4)m`, looking toward the court with `pitch=-2°`.
Use `--supersample 1` for faster preview or `--supersample 3` for smoother edges.

To overlay live model keypoint predictions in the same viewer:

```bash
./start_gui.sh
```

This starts the viewer with `runs/v6/best.pt` by default. If the checkpoint is missing, the script fetches it from the configured remote path first.

Equivalent explicit command:

```bash
uv run --extra viewer --extra train tenniscourt-viewer \
  --width 640 \
  --height 360 \
  --supersample 1 \
  --checkpoint runs/heatmap-10000-v5-hard-keypoints/best.pt \
  --selection-score combined \
  --score-threshold 0.5 \
  --short-labels
```

Use `--reload-checkpoint` to hot-reload the checkpoint when `best.pt` is updated, and `--predict-every 2` or higher if the model inference is slower than the GUI frame rate.

Controls:

- `W/A/S/D`: move on the court plane
- `Space`: move up
- `Shift`: move down
- mouse: look around
- `Esc`: quit

## Smoke Test

```bash
uv run tenniscourt-generate --count 16 --width 320 --height 180 --out outputs/smoke
uv run --extra train tenniscourt-train --data outputs/smoke --epochs 1 --batch-size 4 --max-steps 2
uv run --extra viewer tenniscourt-viewer --headless --max-frames 2 --save-frame outputs/viewer-smoke.png
```
