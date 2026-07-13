# 渲染稳定性修正记录

日期：2026-07-14

## 问题

低机位 viewer 中，场地线会随着视角变化出现抽搐。根因不是 supersampling，而是裁剪不连续：

- 地面线带四角直接投影；
- 部分角点接近或穿过相机 near plane；
- 超出图像边界的 polygon 坐标被简单 clamp；
- 视角轻微变化时，polygon 顶点会突然跳到很远的位置或突然被丢弃。

这种行为会导致线条边缘和整条线在画面中突变。

## 修正

新增 `src/tenniscourt/projection_clip.py`，专门处理投影前后的裁剪：

- `world_to_camera`：世界坐标转相机坐标；
- `clip_polygon_near`：在相机空间裁剪 `z >= 0.05m` 的 polygon；
- `clip_polyline_near`：在相机空间裁剪中心线；
- `project_pinhole_camera`：对 near-clipped 相机坐标做 pinhole 投影；
- `clip_polygon_to_image`：用 Sutherland-Hodgman 裁剪到图像矩形。

`render.py` 中 pinhole 且无畸变的 court line strip 现在流程为：

```text
3D 地面线带四角
        ↓
世界坐标转相机坐标
        ↓
near-plane polygon clipping
        ↓
pinhole projection
        ↓
image-rectangle polygon clipping
        ↓
OpenCV fillPoly
```

这替代了之前的坐标 clamp，避免视角变化时顶点突然跳动。

## 验证

命令：

```bash
uv run python -m compileall src
uv run --extra viewer tenniscourt-viewer --headless --max-frames 2 --save-frame outputs/viewer-smoke-clipped.png --supersample 3
uv run tenniscourt-generate --count 4 --width 320 --height 180 --supersample 3 --out outputs/smoke-clipped --seed 19
SDL_VIDEODRIVER=dummy uv run --extra viewer tenniscourt-viewer --width 320 --height 180 --max-frames 2 --no-grab --supersample 2
```

Yaw 扫描：

- 相机位置：`(0, -13, 0.4)m`
- pitch：`-2°`
- yaw：`-4°` 到 `4°`
- 图像：[`assets/yaw_scan_stability_contact.png`](assets/yaw_scan_stability_contact.png)

结果：

| yaw | mask area | visible lines | net segments |
|---:|---:|---:|---:|
| -4 | 4192 | 9 | 27 |
| -3 | 4293 | 9 | 27 |
| -2 | 4276 | 10 | 27 |
| -1 | 4339 | 10 | 27 |
| 0 | 3994 | 10 | 27 |
| 1 | 4342 | 10 | 27 |
| 2 | 4276 | 10 | 27 |
| 3 | 4264 | 9 | 27 |
| 4 | 4220 | 9 | 27 |

观察：线条随小幅 yaw 平滑移动，没有出现顶点炸开到画面边缘或整条线突然跳变。
