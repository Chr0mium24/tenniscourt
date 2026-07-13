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

## 远端 v6 训练结果

远端机器：`anilam@10.31.151.120`
远端路径：`/home/anilam/Codes/tenniscourt`
训练输出：`runs/heatmap-10000-v6-low-lr-hard`
评估输出：`runs/pose-eval-10000-v6-low-lr-hard-final`

训练正常结束：

```text
TRAIN_V6_LOW_LR_HARD_EXIT:0
best_epoch=114 best_val_kp_px=5.59
```

训练后期指标：

| epoch | train loss | val IoU | val keypoint px | val visibility acc |
| ---: | ---: | ---: | ---: | ---: |
| 107 | 0.0626 | 0.9634 | 5.87 | 0.9821 |
| 108 | 0.0621 | 0.9632 | 6.30 | 0.9818 |
| 109 | 0.0615 | 0.9633 | 5.62 | 0.9813 |
| 110 | 0.0611 | 0.9634 | 5.80 | 0.9819 |
| 111 | 0.0613 | 0.9634 | 5.72 | 0.9809 |
| 112 | 0.0608 | 0.9636 | 5.64 | 0.9818 |
| 113 | 0.0602 | 0.9634 | 5.93 | 0.9819 |
| 114 | 0.0600 | 0.9638 | 5.59 | 0.9812 |

best 出现在最后一轮，说明低学习率继续训练仍有效，没有明显过拟合迹象。

## 远端 v6 pose eval

使用 `selection-score=combined`、`peak-threshold=0.5`、`pnp-solver=ippe`。

| 指标 | v5 | v6 |
| --- | ---: | ---: |
| train best val keypoint error | 7.48 px | 5.59 px |
| sample mean keypoint error | 7.23 px | 5.49 px |
| sample median / p95 | 1.55 px / 32.91 px | 1.41 px / 28.24 px |
| visible-point mean | 7.41 px | 5.48 px |
| visible-point median | 0.89 px | 0.87 px |
| visible-point p90 | 4.50 px | 3.98 px |
| visible-point p95 | 10.61 px | 7.90 px |
| visible-point p99 | 189.64 px | 122.39 px |
| visible points <= 5px | 91.03% | 92.03% |
| visible points > 50px | 3.62% | 2.68% |
| score-gated PnP success | 95.50% | 96.35% |
| position error median / p95 | 0.176 m / 0.857 m | 0.158 m / 0.866 m |
| rotation error median / p95 | 0.418° / 2.269° | 0.409° / 2.263° |
| inlier reproj median / p95 | 0.949 px / 2.476 px | 0.924 px / 2.646 px |

v6 达到了接近 `5px` mean 的阶段：visible-point mean 为 `5.48px`，p90 已经低于 `4px`，但仍有 `2.68%` 的 `>50px` 长尾。

## v6 threshold scan

checkpoint：`runs/heatmap-10000-v6-low-lr-hard/best.pt`
`selection-score=combined`

| threshold | selected median | success | position median | position p95 | rotation median | rotation p95 | reproj median | reproj p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 8 | 96.05% | 0.158 m | 0.843 m | 0.412° | 2.254° | 0.939 px | 2.678 px |
| 0.3 | 8 | 96.05% | 0.160 m | 0.854 m | 0.412° | 2.256° | 0.936 px | 2.678 px |
| 0.4 | 8 | 96.05% | 0.159 m | 0.866 m | 0.411° | 2.270° | 0.931 px | 2.678 px |
| 0.5 | 8 | 96.35% | 0.158 m | 0.866 m | 0.409° | 2.263° | 0.924 px | 2.646 px |
| 0.6 | 7 | 95.70% | 0.157 m | 0.816 m | 0.403° | 2.110° | 0.902 px | 2.485 px |
| 0.7 | 7 | 93.90% | 0.154 m | 0.803 m | 0.400° | 2.081° | 0.868 px | 2.271 px |

部署默认仍建议 `threshold=0.5`，它给出最高成功率。若更重视 p95 稳定性而能接受更多失败，`0.7` 更保守。

## v6 剩余问题

v6 最差 keypoint：

| keypoint | visible count | v5 mean | v6 mean | v6 median | v6 p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| near_right_doubles_corner | 4 | 271.47 px | 326.53 px | 397.81 px | 495.27 px |
| near_left_doubles_corner | 2 | 14.15 px | 173.92 px | 173.92 px | 315.86 px |
| near_right_singles_corner | 12 | 82.87 px | 92.65 px | 5.74 px | 319.34 px |
| left_near_service_corner | 615 | 8.81 px | 17.81 px | 0.92 px | 117.41 px |
| right_near_service_corner | 647 | 39.03 px | 11.77 px | 0.77 px | 4.57 px |
| center_near_service_t | 821 | 29.22 px | 8.02 px | 0.63 px | 3.44 px |
| far_right_singles_corner | 1956 | 5.76 px | 5.32 px | 0.92 px | 30.66 px |

v6 明显修复了：

- `right_near_service_corner`: mean `39.03 -> 11.77`
- `center_near_service_t`: mean `29.22 -> 8.02`

但代价是：

- `left_near_service_corner` 长尾变差；
- 极少数近端 corner 仍然不稳定。

这说明继续靠静态 per-channel 权重会产生 tradeoff。下一步如果要稳定进入 `5px` mean 以下，优先不应继续单纯把某些点权重加大，而应加入 hard-case 采样或几何一致性约束。

## 下一步建议

优先级：

1. v7 使用更平衡的 hard weights，继续低学习率。
   建议从 v6 best 继续，`lr=5e-5`，把 `left_near_service_corner` 权重提高，同时不要继续提高已经修复的 `right_near_service_corner` 和 `center_near_service_t`。

   候选权重：

   ```text
   1,2,1,1,1,4,2,1,6,6,1,1,6,2
   ```

2. 实现 hard-case oversampling。
   根据 label 中的可见 keypoint 统计，提高这些样本采样概率：

   - near left/right service corners 可见；
   - near singles/doubles corners 可见；
   - center service T 可见；
   - 强透视、远端线角靠近的样本。

3. 几何后处理。
   对每个 keypoint channel 解 top-k 候选，然后用球场几何/PnP RANSAC 选择全局一致点。当前剩余问题是少数 wrong peak，后处理会比继续压 heatmap loss 更直接。

4. 再考虑更大模型或更高分辨率。
   如果 v7 + oversampling 仍卡在 `5px` 左右，再试 `base_channels=32` 或 `960x540`。
