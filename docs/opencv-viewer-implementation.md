# OpenCV 交互查看器实现记录

日期：2026-07-14

## 目标

提供一个本地 GUI，用于在网球场线仿真场景里移动相机：

- `W/A/S/D`：水平移动；
- `Space`：上移；
- `Shift`：下移；
- 鼠标：改变 yaw/pitch；
- `Esc`：退出。

## 设计约束

只保留一个图像后端：Python/OpenCV 渲染器。

本次没有使用：

- Three.js；
- WebGL；
- Blender；
- Unreal；
- 任何额外 3D 渲染引擎。

`pygame` 只负责窗口、键盘、鼠标和把 OpenCV 生成的 BGR 图像显示到屏幕上。网球场几何、相机投影和线条光栅化仍然由现有 Python/OpenCV 路径完成。

## 改动

- `src/tenniscourt/render.py`
  - 新增 `render_camera_view`，支持传入相机 `position` 和 `target` 后直接渲染一帧。
- `src/tenniscourt/viewer.py`
  - 新增 `tenniscourt-viewer` 命令；
  - 捕获键鼠输入；
  - 每帧调用 `render_camera_view`；
  - 支持 `--headless` 和 `--save-frame` 做 smoke test。
- `pyproject.toml`
  - 新增 `viewer` optional extra；
  - 新增 CLI 入口。

默认起始相机为低机位机器人视角：

```text
position = (0, -13, 0.4) m
yaw = 0 deg
pitch = -2 deg
```

也可以通过 `--start-x`、`--start-y`、`--start-z`、`--yaw-deg`、`--pitch-deg` 覆盖。

默认 `--supersample 3`，用于减少低分辨率线条锯齿；交互预览卡顿时可降到 `--supersample 1` 或 `2`。

## 运行

```bash
uv sync --extra viewer
uv run --extra viewer tenniscourt-viewer
```

## Headless Smoke Test

```bash
uv run --extra viewer tenniscourt-viewer \
  --headless \
  --max-frames 2 \
  --save-frame outputs/viewer-smoke.png
```

这个测试只验证渲染和命令入口，不验证真实键鼠交互。真实交互需要在本地窗口中手动确认。
