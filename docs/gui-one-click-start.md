# GUI 一键启动脚本

日期：2026-07-14

## 目标

提供一个默认入口，避免每次手写一长串 `uv run --extra viewer --extra train tenniscourt-viewer ...` 参数。

默认行为：

- 使用 `runs/v6/best.pt`；
- 如果本地 checkpoint 不存在，自动从远端拉取；
- 使用 `640x360`；
- `supersample=1`，保证交互速度；
- `selection-score=combined`；
- `score-threshold=0.5`；
- 显示短标签。

## 使用

```bash
./start_gui.sh
```

临时显示所有点：

```bash
SHOW_ALL=1 SCORE_THRESHOLD=0.2 ./start_gui.sh
```

边训练边热重载 checkpoint：

```bash
RELOAD_CHECKPOINT=1 ./start_gui.sh
```

指定其他 checkpoint：

```bash
CHECKPOINT=runs/v6/best.pt ./start_gui.sh
```

调整窗口大小：

```bash
WIDTH=960 HEIGHT=540 ./start_gui.sh
```

## 本地 Smoke Test

命令：

```bash
HEADLESS=1 \
SAVE_FRAME=outputs/start-gui-smoke.png \
SHOW_ALL=1 \
SCORE_THRESHOLD=0.2 \
./start_gui.sh
```

预期：

- 能找到或拉取 `runs/v6/best.pt`；
- 能加载模型；
- 能渲染一帧；
- 能保存带 keypoint overlay 的 smoke 图片。

结果：

```text
starting GUI with checkpoint: runs/v6/best.pt
frames=1 visible_lines=10 position=[0.0, -12.9, 0.4] selected_keypoints=10 infer_ms=148.9
```

输出图片：

```text
outputs/start-gui-smoke.png
```
