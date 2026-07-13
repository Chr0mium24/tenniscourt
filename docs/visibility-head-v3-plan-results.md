# Visibility Head v3 计划与本地验证

日期：2026-07-14

## 背景

v2 训练后，验证集可见 keypoint 的 median 已经到 `0.90px`，`87.9%` 的可见点在 `5px` 内。但仍有 `6.8%` 的可见点大于 `50px`，把 mean 拉到 `13.24px`。

PnP 评估显示：

- `oracle_visible` 成功率：96.45%；
- `score_gated` 成功率：89.50%；
- 两者成功样本的位姿中位误差接近。

说明主要瓶颈不是 PnP 本身，而是部署时可见点/置信点选择不够可靠。

## 改动计划

1. `TinyUNet` 增加 14 通道 keypoint visibility head。
2. 训练时加入 visibility BCE loss。
3. 训练指标记录 `visibility_loss` 和 `val_visibility_acc`。
4. checkpoint 加载支持从 v2 权重部分加载：
   - 继承 mask/headmap/backbone；
   - 新增 visibility head 从随机初始化开始；
   - v3 从 v2 继续训练时使用 `--reset-optimizer`。
5. pose eval 支持三种点选择分数：
   - `peak`：只用 heatmap peak score；
   - `visibility`：只用 visibility probability；
   - `combined`：用 `peak_score * visibility_prob`。

## 本地 Smoke Test

从旧 smoke checkpoint 部分加载并训练：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/smoke-heatmap-v2 \
  --out runs/smoke-visibility-v3 \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 2 \
  --device cpu \
  --resume runs/smoke-heatmap-v2/last.pt \
  --reset-optimizer \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 5 \
  --heatmap-pos-weight 50 \
  --visibility-loss-weight 1 \
  --viz-count 1
```

结果：

```text
partial checkpoint load missing=['visibility_head.weight', 'visibility_head.bias'] unexpected=[]
resumed checkpoint=runs/smoke-heatmap-v2/last.pt start_epoch=2
epoch=2 train_loss=3.3417 mask_loss=1.5942 heatmap_loss=0.2194 visibility_loss=0.6506 val_iou=0.0267 val_kp_px=142.13 val_vis_acc=0.7679 device=cpu
```

评估 smoke：

```bash
uv run --extra train tenniscourt-eval-pose \
  --data outputs/smoke-heatmap-v2 \
  --checkpoint runs/smoke-visibility-v3/last.pt \
  --out runs/pose-smoke-visibility-v3-lowthr \
  --split val \
  --batch-size 4 \
  --workers 0 \
  --device cpu \
  --peak-threshold 0.05 \
  --selection-score combined \
  --pnp-solver ippe \
  --ransac-reproj-error 20 \
  --max-samples 2
```

命令可正常输出 summary。该 smoke checkpoint 只训练 2 step，指标不用于判断效果，只验证链路。

## 远端计划

使用远端已有 10000 张数据和 v2 best checkpoint 继续训练：

```bash
.venv/bin/python -m tenniscourt.train \
  --data outputs/synth-10000 \
  --out runs/heatmap-10000-v3-visible \
  --epochs 20 \
  --batch-size 16 \
  --workers 4 \
  --require-cuda \
  --resume runs/heatmap-10000-v2/best.pt \
  --reset-optimizer \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 10 \
  --heatmap-pos-weight 100 \
  --visibility-loss-weight 1 \
  --viz-count 8
```

训练完成后使用 `selection-score=combined` 扫阈值，并与 v2 的 `peak` gating 结果对比。
