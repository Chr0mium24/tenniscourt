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
- `labels/*.json`: camera pose, K/D, visible line polylines, and metadata

## Train

```bash
uv run tenniscourt-train --data outputs/synth --epochs 10 --batch-size 16 --device auto
```

`--device auto` uses CUDA when PyTorch can see a GPU. Use `--require-cuda` when a CPU fallback should be treated as an error.

## Smoke Test

```bash
uv run tenniscourt-generate --count 16 --width 320 --height 180 --out outputs/smoke
uv run --extra train tenniscourt-train --data outputs/smoke --epochs 1 --batch-size 4 --max-steps 2
```
