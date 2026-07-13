# Heatmap Loss v2 计划与本地验证

日期：2026-07-14

## 背景

远端 10000 张训练中，mask 分支收敛正常，最终 `val_iou=0.9518`。但 keypoint heatmap 分支最终 `val_keypoint_peak_error_px=112.64`，不满足相机位姿估计需求。

主要判断：当前问题不是数据生成链路失败，而是 heatmap 监督太稀疏，普通 BCE 容易被大量背景像素主导。

## 改动计划

1. Dataset 增加 keypoint visible tensor。
2. heatmap loss 只对可见 keypoint 通道计算。
3. 默认 heatmap loss 改为 `weighted-mse`：
   - 对 `sigmoid(logits)` 和 GT gaussian heatmap 做 MSE；
   - 按 `1 + (pos_weight - 1) * target` 加权；
   - GT 峰值附近权重大，背景权重低。
4. 保留 `weighted-bce` 作为可选项。
5. 增加 `--resume`，支持从已有 `last.pt` 继续训练。
6. 增加 keypoint 可视化：
   - 红点：GT keypoint；
   - 绿点：预测 heatmap peak；
   - 黄线：误差向量。

## 本地 Smoke Test

生成 24 张小图：

```bash
uv run tenniscourt-generate \
  --config configs/sim.toml \
  --count 24 \
  --out outputs/smoke-heatmap-v2 \
  --width 320 \
  --height 180 \
  --supersample 1
```

短训练：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/smoke-heatmap-v2 \
  --out runs/smoke-heatmap-v2 \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 2 \
  --device cpu \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 5 \
  --heatmap-pos-weight 50 \
  --viz-count 2
```

结果：

```text
epoch=1 train_loss=2.7699 mask_loss=1.6560 heatmap_loss=0.2228 val_iou=0.0267 val_kp_px=116.22 device=cpu
```

Resume 验证：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/smoke-heatmap-v2 \
  --out runs/smoke-heatmap-v2-resume \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 1 \
  --device cpu \
  --resume runs/smoke-heatmap-v2/last.pt \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 5 \
  --heatmap-pos-weight 50 \
  --viz-count 1
```

结果：

```text
resumed checkpoint=runs/smoke-heatmap-v2/last.pt start_epoch=2
epoch=2 train_loss=2.6592 mask_loss=1.5627 heatmap_loss=0.2193 val_iou=0.0267 val_kp_px=116.69 device=cpu
```

输出检查：

- `runs/smoke-heatmap-v2/best.pt`
- `runs/smoke-heatmap-v2/last.pt`
- `runs/smoke-heatmap-v2/metrics.jsonl`
- `runs/smoke-heatmap-v2/viz/epoch_0001/sample_000.png`
- `runs/smoke-heatmap-v2-resume/viz/epoch_0002/sample_000.png`

## 结论

本地 smoke 已确认：

- 新 Dataset 输出可见点 tensor；
- weighted MSE heatmap loss 可训练；
- checkpoint 可保存；
- `--resume` 可从旧 checkpoint 继续；
- keypoint overlay 可视化可生成。

下一步在远端使用已有 `outputs/synth-10000` 和旧 `runs/heatmap-10000/last.pt` 继续训练，输出到新目录 `runs/heatmap-10000-v2`，避免覆盖旧实验。

## 远端 10000 张续训结果

远端机器：`anilam@10.31.151.120`  
远端路径：`/home/anilam/Codes/tenniscourt`  
代码提交：`cb569fa`  
输入数据：`outputs/synth-10000`  
初始化 checkpoint：`runs/heatmap-10000/last.pt`  
输出目录：`runs/heatmap-10000-v2`  

续训命令：

```bash
.venv/bin/tenniscourt-train \
  --data outputs/synth-10000 \
  --out runs/heatmap-10000-v2 \
  --epochs 30 \
  --batch-size 16 \
  --workers 4 \
  --require-cuda \
  --resume runs/heatmap-10000/last.pt \
  --heatmap-loss weighted-mse \
  --heatmap-loss-weight 10 \
  --heatmap-pos-weight 100 \
  --viz-count 8
```

退出状态：`TRAIN_V2_EXIT:0`

输出文件：

- `runs/heatmap-10000-v2/best.pt`
- `runs/heatmap-10000-v2/last.pt`
- `runs/heatmap-10000-v2/metrics.jsonl`
- `runs/heatmap-10000-v2/viz/epoch_0040/sample_000.png`

关键指标：

| epoch | train_loss | mask_loss | heatmap_loss | val_iou | val_keypoint_peak_error_px |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 baseline | 0.0350 | - | - | 0.9518 | 112.64 |
| 11 | 0.0737 | 0.0348 | 0.0039 | 0.9523 | 78.90 |
| 20 | 0.0358 | 0.0255 | 0.0010 | 0.9573 | 31.79 |
| 27 | 0.0314 | 0.0234 | 0.0008 | 0.9591 | 19.10 |
| 35 | 0.0286 | 0.0218 | 0.0007 | 0.9606 | 14.97 |
| 40 | 0.0273 | 0.0210 | 0.0006 | 0.9638 | 13.27 |

最佳 checkpoint 为 epoch 40：

```text
val_iou=0.9638
val_keypoint_peak_error_px=13.27
```

相对 baseline，keypoint peak error 从 `112.64px` 降到 `13.27px`，约降低 `88.2%`。mask IoU 也从 `0.9518` 提升到 `0.9638`。

已抽查 `epoch_0040/sample_000.png` overlay，可见红色 GT 点和绿色预测点基本贴合。后续还需要把 heatmap peak 转成 2D-3D 对应点，再做 PnP/RANSAC 位姿估计验证。
