# 训练标签目标说明

日期：2026-07-14

## 当前训练目标

当前 `tenniscourt-train` 训练的是多任务模型：

- court line 二值 mask；
- 14 个 court keypoint heatmap。
- 14 个 keypoint visibility/confidence logits。

训练数据读取路径：

```text
outputs/.../images/*.png
outputs/.../masks/*.png
outputs/.../labels/*.json
```

`src/tenniscourt/data.py` 中的 `LineMaskDataset` 读取：

- RGB 图像；
- 二值 court line mask。
- JSON 里的 keypoints，如果旧数据没有 keypoints，则根据相机参数动态投影生成；
- 每个 keypoint 的可见性标记，用于避免不可见点的全黑 heatmap 主导 loss。

训练脚本 `src/tenniscourt/train.py` 使用：

- mask：`BCEWithLogitsLoss + dice_loss`；
- keypoint heatmap：默认 `weighted-mse`，只对可见 keypoint 通道计算，并对 GT gaussian 峰值附近加权；
- 可选 `weighted-bce`，同样使用可见点 mask 和正样本加权；
- keypoint visibility：`BCEWithLogitsLoss`，监督每个 keypoint 是否可见；
- 默认 best checkpoint 按 `val_keypoint_peak_error_px` 最小值保存。

训练脚本还支持：

- `--resume`：从已有 checkpoint 继续训练；
- `--viz-count`：保存 keypoint 预测可视化，红点为 GT，绿点为预测 peak。

## JSON 标签的作用

`labels/*.json` 现在参与 heatmap 监督。

它里面的内容包括：

- 相机内参；
- 相机外参；
- 可见 court line 的 2D polyline；
- net segments；
- 位置和姿态信息。

这些 JSON 标签现在用于：

- 调试合成数据；
- 生成 14 通道 keypoint heatmap；
- 验证投影是否稳定；
- 后续从 mask/line 恢复相机位姿；

## 为什么保留 mask

只训练 keypoint heatmap 在低机位下仍有风险：

- 很多关键点不可见；
- 局部视野里可能只有线，没有交点；
- mask 可以提供密集监督。

因此当前保留 mask 分支，同时增加 heatmap 分支，后处理优先使用 heatmap 点，mask 作为辅助置信和线区域约束。
