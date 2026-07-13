# GUI 模型预测点 Overlay 计划与验证

日期：2026-07-14

## 目标

在现有 `tenniscourt-viewer` 中实时加载 heatmap 模型 checkpoint，对当前 GUI 渲染帧做推理，并把模型预测到的 14 个 keypoint 点叠加显示。

约束：

- 继续使用现有 Python/OpenCV 渲染后端；
- 不引入 Three.js、Blender 或第二个 3D 渲染引擎；
- 普通 viewer 不带 `--checkpoint` 时不强制走 PyTorch 推理路径；
- 支持低频推理，以便在较慢设备上保持交互流畅。

## 实现

新增模块：

```text
src/tenniscourt/prediction.py
```

功能：

- 加载 `TinyUNet` checkpoint；
- 自动读取 checkpoint 中的 `base_channels`；
- 对 OpenCV BGR 图像做与训练一致的 RGB/0-1 预处理；
- 解码 14 通道 keypoint heatmap peak；
- 支持 subpixel 二次曲线 refinement；
- 支持 `peak`、`visibility`、`combined` 三种点选择分数；
- 支持 `reload_if_changed()`，用于 GUI 热重载 checkpoint。

`tenniscourt-viewer` 新增参数：

```bash
--checkpoint PATH
--device auto|cpu|cuda
--require-cuda
--base-channels INT
--predict-every INT
--score-threshold FLOAT
--selection-score peak|visibility|combined
--show-all-keypoints
--short-labels
--no-overlay-labels
--reload-checkpoint
--no-subpixel
```

默认 overlay 行为：

- 只显示 `selection_score >= --score-threshold` 的点；
- `--show-all-keypoints` 会把低分点也以灰色显示；
- 状态栏显示设备、选中的点数和推理耗时；
- 如果 checkpoint 是旧模型，缺少 visibility head，会在状态栏显示 partial-load warning。

## 推荐使用命令

用 v5/v6 这类带 visibility head 的 checkpoint：

```bash
uv run --extra viewer --extra train tenniscourt-viewer \
  --width 640 \
  --height 360 \
  --supersample 1 \
  --checkpoint runs/heatmap-10000-v5-hard-keypoints/best.pt \
  --selection-score combined \
  --score-threshold 0.5 \
  --device auto \
  --short-labels
```

如果想边训练边看 `best.pt` 更新：

```bash
uv run --extra viewer --extra train tenniscourt-viewer \
  --width 640 \
  --height 360 \
  --supersample 1 \
  --checkpoint runs/heatmap-10000-v5-hard-keypoints/best.pt \
  --selection-score combined \
  --score-threshold 0.5 \
  --reload-checkpoint
```

如果帧率不够：

```bash
--predict-every 2
```

表示每 2 帧推理一次，其余帧复用上一次预测结果。

## 本地 Smoke Test

编译检查：

```bash
uv run python -m compileall src/tenniscourt
```

结果：

```text
Compiling 'src/tenniscourt/prediction.py'...
Compiling 'src/tenniscourt/viewer.py'...
```

带模型 headless overlay：

```bash
uv run --extra train --extra viewer tenniscourt-viewer \
  --headless \
  --max-frames 1 \
  --width 320 \
  --height 180 \
  --supersample 1 \
  --checkpoint runs/smoke-hard-keypoints/best.pt \
  --device cpu \
  --selection-score combined \
  --score-threshold 0.05 \
  --show-all-keypoints \
  --short-labels \
  --save-frame outputs/viewer-prediction-smoke.png
```

结果：

```text
frames=1 visible_lines=10 position=[0.0, -12.9, 0.4] selected_keypoints=14 infer_ms=27.1
```

不带模型 headless viewer：

```bash
uv run --extra viewer tenniscourt-viewer \
  --headless \
  --max-frames 1 \
  --width 320 \
  --height 180 \
  --supersample 1 \
  --save-frame outputs/viewer-smoke-no-model.png
```

结果：

```text
frames=1 visible_lines=10 position=[0.0, -12.9, 0.4]
```

说明普通 viewer 路径仍然可用，模型 overlay 路径也能正常加载 checkpoint、推理并写出叠加图。
