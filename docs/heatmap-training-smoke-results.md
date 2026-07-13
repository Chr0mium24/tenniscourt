# Keypoint Heatmap 训练 Smoke Test 结果

日期：2026-07-14

## 目标

将训练目标从纯 mask 分割扩展为多任务：

- court line mask；
- 14 个 court keypoint heatmap。

## 生成测试

命令：

```bash
uv run tenniscourt-generate \
  --count 16 \
  --width 320 \
  --height 180 \
  --supersample 3 \
  --out outputs/heatmap-smoke \
  --seed 31
```

结果：

- 样本数：16
- 每个 label 包含 keypoints：14
- 第一张可见 keypoints：9
- Dataset 输出：
  - image：`(3, 180, 320)`
  - mask：`(1, 180, 320)`
  - heatmaps：`(14, 180, 320)`
  - heatmap max：`0.998909592628479`

## 训练测试

命令：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/heatmap-smoke \
  --out runs/heatmap-smoke \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 2 \
  --require-cuda
```

结果：

- 设备：`cuda`
- epoch：`1`
- train loss：`2.3308`
- val IoU：`0.0222`
- val keypoint peak error：`160.21 px`

这个指标只验证链路。只训练 2 step 时，keypoint error 很大是正常现象，不代表最终模型效果。

## 结论

- JSON keypoints 已进入训练标签；
- Dataset 能生成 14 通道 heatmap；
- 模型能同时输出 mask 和 heatmap；
- CUDA 训练链路已跑通。
