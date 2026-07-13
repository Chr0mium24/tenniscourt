# OpenCV 交互查看器 Smoke Test 结果

日期：2026-07-14

## 环境

- Python：`uv` 虚拟环境
- GUI 依赖：`pygame 2.6.1`
- 图像后端：现有 Python/OpenCV 渲染器
- 额外 3D 引擎：无

## 验证命令

编译检查：

```bash
uv run python -m compileall src
```

Headless 渲染检查：

```bash
uv run --extra viewer tenniscourt-viewer \
  --headless \
  --max-frames 2 \
  --save-frame outputs/viewer-smoke.png
```

Dummy 窗口主循环检查：

```bash
SDL_VIDEODRIVER=dummy uv run --extra viewer tenniscourt-viewer \
  --width 320 \
  --height 180 \
  --max-frames 2 \
  --no-grab
```

## 结果

- 编译检查：通过
- Headless 渲染：通过
- Dummy 窗口主循环：通过
- 保存帧：`outputs/viewer-smoke.png`
- 保存帧尺寸：`540 x 960 x 3`
- 可见线条数：`9`
- 最终相机位置：`[0.0, -12.8, 0.4]`
- court line 渲染：3D 地面 strip 投影
- court line 默认宽度：`0.08 m`
- 抗锯齿：`3x` supersampling 后 `INTER_AREA` 下采样
- 球网渲染：通过
- 边缘裁剪：使用 `cv2.clipLine`

## 说明

本测试验证了命令入口、OpenCV 渲染路径、pygame 窗口主循环和低机位默认视角。真实键鼠交互仍需要在本地窗口里手动确认。

`visible_lines=9` 来自当前低机位默认视角下的中心线可见性统计，不代表所有地面线带都不可见。后续如果把真实相机 K/D 接入，需要重新记录该指标。
