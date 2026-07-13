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

## Visibility label refinement

v3 后续评估发现，部分 long-tail 错误来自标签可见性定义过宽：只要 keypoint 投影在画面内就算 visible，但实际渲染后可能被 occluder 覆盖、被清出 mask，或者太靠近图像边缘。

已增加 mask-based visibility refinement：

- 对每个 label keypoint，先保留原始投影可见性；
- 再检查 keypoint 周围 `5px` 半径内是否至少有 `2` 个 line-mask 像素；
- 如果没有 mask 支持，或者点距离边界小于 `2px`，则置为不可见；
- 训练和 pose eval 使用同一套 refined visibility。

本地 smoke 通过：

```text
epoch=3 train_loss=3.2934 mask_loss=1.5601 heatmap_loss=0.2159 visibility_loss=0.6539 val_iou=0.0267 val_kp_px=142.17 val_vis_acc=0.7679 device=cpu
```

下一步先在远端重跑 v2/v3 checkpoint 的 refined visibility 评估。如果仅因标签可见性过宽导致 mean 偏高，指标会直接下降；否则再训练 v4 refined-visible。

## Hard-keypoint loss weights

v4 refined-visible 训练后，剩余误差仍主要来自少数 keypoint 的长尾。训练脚本增加了：

```bash
--keypoint-channel-weights
```

格式为 14 个逗号分隔权重，顺序与 `keypoint_names()` 一致。默认全为 `1`，不影响已有命令。

本地 smoke 使用：

```bash
--keypoint-channel-weights 1,1,1,1,1,1,2,1,2,4,1,1,4,1
```

对应增强：

- `far_left_singles_corner`: 2
- `left_near_service_corner`: 2
- `right_near_service_corner`: 4
- `center_near_service_t`: 4

本地 smoke 通过：

```text
epoch=4 train_loss=3.2341 mask_loss=1.5418 heatmap_loss=0.2080 visibility_loss=0.6524 val_iou=0.0267 val_kp_px=139.43 val_vis_acc=0.7679 device=cpu
```

远端 v5 将从 v4 best checkpoint 继续训练，使用 hard-keypoint weights，目标是进一步压低 `>50px` 长尾比例。

## 远端 v4/v5 结果

远端机器：`anilam@10.31.151.120`
远端路径：`/home/anilam/Codes/tenniscourt`
数据：`outputs/synth-10000`
验证集：2000 张

### v4 refined-visible

训练输出：`runs/heatmap-10000-v4-refined-visible`
评估输出：`runs/pose-eval-10000-v4-refined-final`

训练从 v3 checkpoint 继续，使用 refined visibility label。最优 epoch：

```text
epoch=74 val_kp_px=11.41
```

使用 `selection-score=combined`、`peak-threshold=0.5` 评估：

| 指标 | 数值 |
| --- | ---: |
| sample mean keypoint error | 10.52 px |
| visible-point mean | 11.38 px |
| visible-point median | 0.92 px |
| visible-point p90 | 5.20 px |
| visible-point p95 | 67.09 px |
| visible points <= 5px | 89.74% |
| visible points > 50px | 5.53% |
| score-gated PnP success | 95.65% |
| position error median / p95 | 0.172 m / 0.920 m |
| rotation error median / p95 | 0.412° / 2.348° |
| inlier reproj median / p95 | 0.856 px / 2.417 px |

v4 的主要改善来自修正可见性标签：不再强迫模型预测画面中实际没有线支持的点。

### v5 hard-keypoint weights

训练输出：`runs/heatmap-10000-v5-hard-keypoints`
评估输出：`runs/pose-eval-10000-v5-hard-final`

训练命令核心参数：

```bash
.venv/bin/python -m tenniscourt.train \
  --data outputs/synth-10000 \
  --out runs/heatmap-10000-v5-hard-keypoints \
  --epochs 20 \
  --batch-size 16 \
  --workers 4 \
  --require-cuda \
  --resume runs/heatmap-10000-v4-refined-visible/best.pt \
  --reset-optimizer \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 10 \
  --heatmap-pos-weight 100 \
  --visibility-loss-weight 1 \
  --keypoint-channel-weights 1,1,1,1,1,1,2,1,2,4,1,1,4,1 \
  --viz-count 8
```

训练正常结束：

```text
TRAIN_V5_HARD_EXIT:0
best_epoch=94 best_val_kp_px=7.48
```

使用 `selection-score=combined`、`peak-threshold=0.5` 评估：

| 指标 | 数值 |
| --- | ---: |
| sample mean keypoint error | 7.23 px |
| sample median / p95 | 1.55 px / 32.91 px |
| visible-point mean | 7.41 px |
| visible-point median | 0.89 px |
| visible-point p90 | 4.50 px |
| visible-point p95 | 10.61 px |
| visible-point p99 | 189.64 px |
| visible points <= 5px | 91.03% |
| visible points > 50px | 3.62% |
| score-gated PnP success | 95.50% |
| position error median / p95 | 0.176 m / 0.857 m |
| rotation error median / p95 | 0.418° / 2.269° |
| inlier reproj median / p95 | 0.949 px / 2.476 px |

相对 v4：

| 指标 | v4 | v5 |
| --- | ---: | ---: |
| train best val keypoint error | 11.41 px | 7.48 px |
| visible-point mean | 11.38 px | 7.41 px |
| visible-point p95 | 67.09 px | 10.61 px |
| visible points <= 5px | 89.74% | 91.03% |
| visible points > 50px | 5.53% | 3.62% |
| score-gated PnP success | 95.65% | 95.50% |
| position error p95 | 0.920 m | 0.857 m |

hard-keypoint weighting 明显降低了 2D keypoint 长尾，PnP 成功率基本持平，位姿 p95 略有改善。

### v5 threshold scan

checkpoint：`runs/heatmap-10000-v5-hard-keypoints/best.pt`
`selection-score=combined`

| threshold | selected median | success | position median | position p95 | rotation median | rotation p95 | reproj median | reproj p95 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.2 | 8 | 95.40% | 0.173 m | 0.857 m | 0.415° | 2.265° | 0.962 px | 2.516 px |
| 0.3 | 8 | 95.35% | 0.174 m | 0.866 m | 0.417° | 2.291° | 0.959 px | 2.534 px |
| 0.4 | 8 | 95.25% | 0.174 m | 0.866 m | 0.417° | 2.307° | 0.958 px | 2.516 px |
| 0.5 | 7 | 95.50% | 0.176 m | 0.857 m | 0.418° | 2.269° | 0.949 px | 2.476 px |
| 0.6 | 7 | 94.85% | 0.177 m | 0.848 m | 0.421° | 2.266° | 0.937 px | 2.479 px |
| 0.7 | 7 | 93.50% | 0.173 m | 0.780 m | 0.419° | 2.020° | 0.882 px | 2.277 px |

默认部署建议先用 `threshold=0.5`。如果更重视成功率，0.2 到 0.5 差距很小；如果更重视 p95 稳定性，0.7 会更保守，但会损失成功率。

### 剩余长尾

v5 最差的 keypoint：

| keypoint | visible count | mean | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| near_right_doubles_corner | 4 | 271.47 px | 330.14 px | 415.16 px |
| near_right_singles_corner | 12 | 82.87 px | 4.27 px | 277.56 px |
| right_near_service_corner | 647 | 39.03 px | 0.92 px | 252.24 px |
| center_near_service_t | 821 | 29.22 px | 0.73 px | 245.74 px |
| near_left_singles_corner | 18 | 15.14 px | 3.21 px | 50.85 px |
| left_near_service_corner | 615 | 8.81 px | 0.89 px | 4.97 px |
| center_far_service_t | 1952 | 7.10 px | 0.71 px | 61.59 px |

结论：

- 这不是典型过拟合。v5 最优点在最后一个 epoch，训练 loss 下降时验证 keypoint error 也继续下降。
- 当前 median 已经低于 `1px`，p90 也低于 `5px`，mean 高主要仍来自少数 wrong peak。
- 要到 `5px` mean，核心是继续打掉 `>50px` 的 3.62% 长尾，尤其是 `right_near_service_corner`、`center_near_service_t`、近端右侧 corner，以及 `center_far_service_t` 的少数错峰。

## 下一步优化建议

优先级按投入产出排序：

1. 从 v5 best 继续训练，降低学习率。
   v5 best 出现在最后一轮，说明还没有收敛。建议先用同样 loss 继续 20 epoch，但把 `--lr` 从默认 `3e-4` 降到 `1e-4` 或 `5e-5`。

2. 对 remaining hard points 再加权。
   目前 `right_near_service_corner` 和 `center_near_service_t` 仍是主要长尾，可把权重从 `4` 提到 `6` 或 `8`，并给 `center_far_service_t` 加 `2`。

3. 做 hard-case oversampling。
   只加 loss 权重仍会被样本频率限制。应在 dataset/sampler 中提高“近端右侧 service/corner 可见”、“远端线角强透视压缩”样本出现频率。

4. 训练更高分辨率或升级 decoder。
   当前 `640x360` 下远端线角被压缩，部分点在图上接近重叠。下一步可试 `960x540`，或把 `base_channels` 从 `16` 提到 `32`。这会增加显存和训练时间，但更接近 `5px` mean 的目标。

5. 增加几何后处理。
   纯 heatmap 仍会偶发错峰。部署时可以先取每个点 top-k 候选，再用网球场几何/PnP RANSAC 选择全局一致的一组点，而不是每个 channel 只取单个 argmax。
