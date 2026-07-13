# Heatmap 解码与 PnP 位姿评估计划

日期：2026-07-14

## 目标

把模型输出的 14 通道 keypoint heatmap 转成可用于相机位姿估计的 2D-3D 对应点，并评估相机相对网球场的位姿误差。

## 评估链路

新增命令：

```bash
tenniscourt-eval-pose
```

流程：

1. 加载 `best.pt` 或 `last.pt`。
2. 对验证集图像推理 keypoint heatmap。
3. 每个 keypoint 通道取 heatmap peak，并做二次曲线 subpixel refinement。
4. 将 14 个球场关键点的预测 2D 点与固定 3D 球场点对应。
5. 使用 OpenCV `solvePnPRansac` 估计世界到相机的 `rvec/tvec`。
6. 输出：
   - keypoint 平均像素误差；
   - PnP 成功率；
   - 重投影误差；
   - 相机位置误差；
   - 旋转误差。

## 两种 PnP 模式

### `oracle_visible`

使用 label 中的可见点标记筛选 keypoint。

这个模式是诊断上限：它回答“如果可见性选择正确，仅靠当前点精度能得到多好位姿”。

### `score_gated`

只用模型 heatmap peak score 超过阈值的点。

这个模式更接近真实部署，但当前模型还没有单独训练 keypoint visibility/confidence，因此 score 未必可靠。

## 本地 sanity check

用 GT keypoint 直接跑 PnP，结果接近 0：

```text
reproj_error_px=0.00017
position_error_m=0.00006
rotation_error_deg=0.00016
```

说明 PnP 坐标系和网球场 3D 点对应关系是正确的。

## 本地 smoke

使用只训练 2 step 的 smoke checkpoint 跑命令：

```bash
uv run --extra train tenniscourt-eval-pose \
  --data outputs/smoke-heatmap-v2 \
  --checkpoint runs/smoke-heatmap-v2/last.pt \
  --out runs/pose-smoke-v2 \
  --split val \
  --batch-size 4 \
  --workers 0 \
  --device cpu \
  --peak-threshold 0.1 \
  --ransac-reproj-error 20 \
  --max-samples 2
```

命令能正常输出 summary 和 JSONL。该 checkpoint 没有实际收敛，所以位姿误差很大，只用于验证评估链路。

## 下一步

在远端使用 `runs/heatmap-10000-v2/best.pt` 对 `outputs/synth-10000` 的验证集跑完整评估。

## 远端 v2 位姿评估结果

远端机器：`anilam@10.31.151.120`
远端路径：`/home/anilam/Codes/tenniscourt`
数据：`outputs/synth-10000`
checkpoint：`runs/heatmap-10000-v2/best.pt`
评估输出：`runs/pose-eval-10000-v2-final`

最终评估命令：

```bash
.venv/bin/python -m tenniscourt.evaluate_pose \
  --data outputs/synth-10000 \
  --checkpoint runs/heatmap-10000-v2/best.pt \
  --out runs/pose-eval-10000-v2-final \
  --split val \
  --batch-size 32 \
  --workers 4 \
  --require-cuda \
  --pnp-solver ippe \
  --ransac-reproj-error 12
```

说明：

- 默认 `peak_threshold=0.7`；
- 默认 `pnp_solver=ippe`，更适合平面球场点；
- 对明显不合理的相机位置进行 reject；
- PnP 重投影误差按 RANSAC inlier 计算。

### Heatmap 解码误差

验证集样本数：2000
可见 keypoint 总数：16068

| 指标 | 数值 |
| --- | ---: |
| sample mean keypoint error | 12.41 px |
| visible-point mean keypoint error | 13.24 px |
| visible-point median | 0.90 px |
| visible-point p90 | 7.22 px |
| visible-point p95 | 98.68 px |
| visible points <= 2px | 76.1% |
| visible points <= 5px | 87.9% |
| visible points <= 10px | 91.1% |
| visible points > 50px | 6.8% |

结论：当前不是“整体都在 13px 附近”，而是大多数点已经很准，少数极端错峰把 mean 拉高。

### PnP 指标

`oracle_visible` 使用 GT 可见性，是当前 keypoint 准确度的 PnP 上限：

| 指标 | 数值 |
| --- | ---: |
| success rate | 96.45% |
| position error median | 0.164 m |
| position error p95 | 0.831 m |
| rotation error median | 0.381° |
| rotation error p95 | 2.197° |
| inlier reproj median | 0.840 px |
| inlier reproj p95 | 2.443 px |

`score_gated` 只按模型 peak score 选点，更接近部署：

| 指标 | 数值 |
| --- | ---: |
| peak threshold | 0.7 |
| success rate | 89.50% |
| position error median | 0.167 m |
| position error p95 | 0.762 m |
| rotation error median | 0.387° |
| rotation error p95 | 2.162° |
| inlier reproj median | 0.792 px |
| inlier reproj p95 | 2.323 px |

部署模式的主要损失来自可见性/置信度选择，而不是 PnP 精度本身。`score_gated` 的 reject 原因：

- `pnp_failed`: 95
- `pose_out_of_bounds`: 115

### Peak threshold 扫描

| threshold | selected median | success | position median | rotation median | reproj median |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 0.4 | 12 | 80.8% | 0.185 m | 0.429° | 0.869 px |
| 0.5 | 11 | 83.5% | 0.176 m | 0.414° | 0.847 px |
| 0.6 | 10 | 86.1% | 0.174 m | 0.397° | 0.841 px |
| 0.7 | 8 | 89.5% | 0.167 m | 0.387° | 0.792 px |
| 0.8 | 7 | 84.6% | 0.163 m | 0.377° | 0.665 px |

`0.7` 是当前较好的折中；`0.8` 点更准，但因为点数不足导致成功率下降。

### Per-keypoint 长尾

按 mean 排序，最明显的长尾来自：

| keypoint | visible count | mean | median | p95 |
| --- | ---: | ---: | ---: | ---: |
| center_near_service_t | 845 | 87.94 px | 1.43 px | 352.81 px |
| right_near_service_corner | 661 | 52.69 px | 1.08 px | 285.15 px |
| far_left_singles_corner | 1991 | 18.88 px | 0.91 px | 119.96 px |
| left_near_service_corner | 623 | 16.85 px | 0.95 px | 70.90 px |

注意这些点的 median 很低，说明模型在大多数情况下能识别，但有明显错峰长尾。近端 service 点可能被低机位、画面边缘、遮挡和标签可见性策略影响；远端 singles/doubles 点可能因为透视压缩后相邻线角很近而混淆。

## 是否过拟合

当前没有典型过拟合迹象：

- 训练后期验证集 keypoint error 仍持续下降；
- mask IoU 同时提升；
- 解码后 median 已经到 1px 以内；
- 问题集中在少数长尾错峰，而不是全局泛化失败。

更准确的判断：当前模型已经学会大多数点，但缺少可靠的可见性/置信度建模，也缺少对 hard case 的约束。

## 到 5px mean 的优化方向

当前 visible-point median 是 `0.90px`，`87.9%` 的可见点已经小于 `5px`。要把 mean 也压到 `5px`，重点是减少 `>50px` 的 6.8% 长尾。

优先级建议：

1. 增加 keypoint visibility/confidence head
   不再只用 heatmap peak score 判断点是否可用。每个 keypoint 单独输出 visible probability，用 label 的 `visible` 监督。PnP 只使用 `visible_prob * peak_score` 高的点。

2. 修正标签可见性
   当前 visible 只判断投影是否在画面内，没有严格判断是否被遮挡、是否被随机 occluder 覆盖、是否落在真实可见线段上。应基于渲染后的 mask/occluder 更新 keypoint visible，否则模型会被要求预测“画面里看不到但标签说可见”的点。

3. hard-example / per-keypoint 加权
   对 `center_near_service_t`、`right_near_service_corner`、`far_left_singles_corner` 等长尾点增加 loss 权重或 oversampling。目标不是提高所有点，而是专门打掉长尾。

4. 高分辨率/高保真训练
   远端线角在 640x360 下被强透视压缩，容易混淆。可以训练 960x540 或 1280x720，再评估下采样部署成本。

5. 模型结构升级
   当前是 `TinyUNet(base_channels=16)`，容量很小。下一步建议：
   - `base_channels=32`；
   - stride 更小的 high-resolution decoder；
   - 或 HRNet/FPN/UNet++ 风格保持高分辨率特征；
   - 仍保持单文件低复杂度实现，先做 base_channels=32 对照实验。

6. heatmap 后处理改进
   当前是 argmax + 二次曲线 subpixel。可测试局部 soft-argmax / integral regression，减少峰值相邻像素抖动。

7. PnP 后处理
   当前 IPPE + RANSAC 已可用。实际部署建议再加：
   - 最小 inlier 数；
   - 位姿边界；
   - 连续视频时序滤波；
   - 用线 mask 做最终 pose refinement，把球场线重投影到 mask 上优化。
