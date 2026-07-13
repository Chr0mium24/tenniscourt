# 模型优化 v6 计划与小范围验证

日期：2026-07-14

## 当前基线

当前最好结果来自 v5：

- 训练输出：`runs/heatmap-10000-v5-hard-keypoints`
- checkpoint：`runs/heatmap-10000-v5-hard-keypoints/best.pt`
- best epoch：94
- 训练验证 keypoint error：`7.48px`
- pose eval visible-point mean：`7.41px`
- visible-point p90：`4.50px`
- visible-point p95：`10.61px`
- visible points `<=5px`：`91.03%`
- visible points `>50px`：`3.62%`
- score-gated PnP success：`95.50%`

关键判断：

- 不是典型过拟合。v5 最优点出现在最后一个 epoch，训练 loss 下降时验证 keypoint error 也继续下降。
- median 已经低于 `1px`，p90 已经低于 `5px`。
- mean 仍高于 `5px`，主要由少数 wrong peak 长尾拉高。

## 可选优化方向

### 1. 低学习率继续训练

v5 best 出现在最后一轮，说明当前结构和 loss 还没有完全收敛。

优点：

- 不改模型结构；
- 可以直接从 v5 best 继续；
- 风险最低；
- 预计能继续压低 long-tail。

风险：

- 如果继续训练开始震荡，需要降低学习率或早停；
- 对非常稀有的近端 corner 帮助有限。

### 2. 增强 remaining hard-keypoint 权重

v5 后剩余长尾主要来自：

| keypoint | v5 mean | v5 median | v5 p95 |
| --- | ---: | ---: | ---: |
| near_right_doubles_corner | 271.47 px | 330.14 px | 415.16 px |
| near_right_singles_corner | 82.87 px | 4.27 px | 277.56 px |
| right_near_service_corner | 39.03 px | 0.92 px | 252.24 px |
| center_near_service_t | 29.22 px | 0.73 px | 245.74 px |
| center_far_service_t | 7.10 px | 0.71 px | 61.59 px |

v5 权重为：

```text
1,1,1,1,1,1,2,1,2,4,1,1,4,1
```

v6 计划提高以下点：

- `near_right_doubles_corner`: `1 -> 2`
- `near_right_singles_corner`: `1 -> 4`
- `right_near_service_corner`: `4 -> 8`
- `center_near_service_t`: `4 -> 8`
- `center_far_service_t`: `1 -> 2`

v6 权重：

```text
1,2,1,1,1,4,2,1,2,8,1,1,8,2
```

### 3. Hard-case oversampling

通过 sampler 提高包含难点的样本比例。

这条线可能有效，但需要改 dataset/sampler，并先定义 hard-case 规则。当前先不做，避免把变量混在一起。

### 4. 更大模型或更高分辨率

例如：

- `base_channels=32`
- `960x540`

这条线可能更接近 `5px` mean，但成本更高，而且从 v5 checkpoint 迁移到更大模型需要额外实现 size-compatible partial load。当前先不做。

### 5. 几何后处理

对每个 channel 解 top-k 候选，再用球场几何/PnP 全局一致性选点。

这更偏部署后处理，不直接优化模型训练。当前先继续优化 heatmap 质量。

## v6 选择

先做：

1. 从 v5 best 继续训练；
2. 学习率降到 `1e-4`；
3. 使用更强 hard-keypoint 权重；
4. 继续训练 20 epoch；
5. 使用 v5 相同 pose eval 设置评估，重点看：
   - visible-point mean；
   - visible-point p95；
   - `>50px` 比例；
   - score-gated PnP success。

远端训练命令：

```bash
.venv/bin/python -m tenniscourt.train \
  --data outputs/synth-10000 \
  --out runs/heatmap-10000-v6-low-lr-hard \
  --epochs 20 \
  --batch-size 16 \
  --workers 4 \
  --require-cuda \
  --resume runs/heatmap-10000-v5-hard-keypoints/best.pt \
  --reset-optimizer \
  --lr 1e-4 \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 10 \
  --heatmap-pos-weight 100 \
  --visibility-loss-weight 1 \
  --keypoint-channel-weights 1,2,1,1,1,4,2,1,2,8,1,1,8,2 \
  --viz-count 8
```

评估命令：

```bash
.venv/bin/python -m tenniscourt.evaluate_pose \
  --data outputs/synth-10000 \
  --checkpoint runs/heatmap-10000-v6-low-lr-hard/best.pt \
  --out runs/pose-eval-10000-v6-low-lr-hard-final \
  --split val \
  --batch-size 32 \
  --workers 4 \
  --require-cuda \
  --selection-score combined \
  --peak-threshold 0.5 \
  --pnp-solver ippe \
  --ransac-reproj-error 12
```

## 本地小范围验证

目的：验证新超参组合可以从已有 checkpoint 正常 resume、训练、保存，不用 smoke 指标判断最终效果。

命令：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/smoke-heatmap-v2 \
  --out runs/smoke-v6-low-lr-hard-weights \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 3 \
  --device cpu \
  --resume runs/smoke-hard-keypoints/best.pt \
  --reset-optimizer \
  --lr 1e-4 \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 10 \
  --heatmap-pos-weight 100 \
  --visibility-loss-weight 1 \
  --keypoint-channel-weights 1,2,1,1,1,4,2,1,2,8,1,1,8,2 \
  --viz-count 1
```

结果：

```text
resumed checkpoint=runs/smoke-hard-keypoints/best.pt start_epoch=5
epoch=5 train_loss=4.2181 mask_loss=1.5246 heatmap_loss=0.2043 visibility_loss=0.6507 val_iou=0.0267 val_kp_px=139.18 val_vis_acc=0.7679 device=cpu
```

结论：

- 新权重字符串长度和 keypoint 顺序正确；
- 低学习率 resume 链路正常；
- checkpoint partial/full load 正常；
- 可以按 v6 方案启动远端训练。
