# 远端 10000 张合成图 heatmap 训练结果

日期：2026-07-14  
远端机器：`anilam@10.31.151.120`  
远端路径：`/home/anilam/Codes/tenniscourt`  
代码提交：`71492a0`  
GPU：NVIDIA GeForce RTX 5070 Ti, 16GB  

## 实验目标

在远端 GPU 机器上生成 10000 张网球场合成图，并训练当前的 mask + 14 点 heatmap 模型，验证当前数据和训练目标是否能学习场地线关键点。

## 数据生成

生成命令：

```bash
.venv/bin/tenniscourt-generate \
  --config configs/sim.toml \
  --count 10000 \
  --out outputs/synth-10000
```

生成结果：

- images: 10000
- masks: 10000
- labels: 10000
- 日志：`logs/generate-10000.log`
- 退出状态：`GENERATE_EXIT:0`

## 训练配置

训练命令：

```bash
.venv/bin/tenniscourt-train \
  --data outputs/synth-10000 \
  --out runs/heatmap-10000 \
  --epochs 10 \
  --batch-size 16 \
  --workers 4 \
  --require-cuda
```

输出文件：

- `runs/heatmap-10000/best.pt`
- `runs/heatmap-10000/last.pt`
- `runs/heatmap-10000/metrics.jsonl`
- 日志：`logs/train-10000.log`
- 退出状态：`TRAIN_EXIT:0`

## 指标

| epoch | train_loss | val_iou | val_keypoint_peak_error_px |
| ---: | ---: | ---: | ---: |
| 1 | 1.8277 | 0.8302 | 204.80 |
| 2 | 1.1139 | 0.8858 | 224.26 |
| 3 | 0.4726 | 0.9245 | 260.09 |
| 4 | 0.1616 | 0.9307 | 253.35 |
| 5 | 0.0859 | 0.9401 | 225.00 |
| 6 | 0.0596 | 0.9477 | 183.06 |
| 7 | 0.0483 | 0.9479 | 163.65 |
| 8 | 0.0417 | 0.9491 | 149.23 |
| 9 | 0.0378 | 0.9526 | 133.76 |
| 10 | 0.0350 | 0.9518 | 112.64 |

## 结论

训练流程和 CUDA 环境可用，10000 张数据生成与 10 epoch 训练均正常完成。

mask 分支学习正常，最终验证集 IoU 约 0.952，说明合成图像、线 mask 和基础网络训练链路可用。

keypoint heatmap 分支当前不合格，最终 `val_keypoint_peak_error_px` 仍为 112.64px。这个误差对于后续 PnP/相机位姿估计不可用。主要问题不是数据量不够，而是当前 heatmap 训练目标过于稀疏，普通 BCE 容易被大量背景像素主导，模型优先学 mask/背景分布，点峰值定位没有被充分约束。

## 后续建议

下一步应调整 keypoint 训练目标，而不是继续单纯增加数据量：

1. 使用按通道的可见点 mask，只对画面内关键点计算 heatmap loss。
2. 对 heatmap 正样本区域加权，或者改为 focal/MSE gaussian heatmap loss。
3. 输出验证可视化，把预测 peak 与 GT keypoint 叠到图上，避免只看一个平均像素误差。
4. 单独提高 keypoint loss 权重，降低 mask loss 对共享 backbone 的主导。
