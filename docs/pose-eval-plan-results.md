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
