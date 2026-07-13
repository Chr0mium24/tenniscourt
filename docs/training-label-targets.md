# 训练标签目标说明

日期：2026-07-14

## 当前训练目标

当前 `tenniscourt-train` 训练的是二值分割模型，不是点检测模型。

训练数据读取路径：

```text
outputs/.../images/*.png
outputs/.../masks/*.png
```

`src/tenniscourt/data.py` 中的 `LineMaskDataset` 只读取：

- RGB 图像；
- 二值 court line mask。

训练脚本 `src/tenniscourt/train.py` 使用 `BCEWithLogitsLoss + dice_loss`，目标是预测每个像素是否属于球场线。

## JSON 标签的作用

`labels/*.json` 目前不参与训练。

它里面的内容包括：

- 相机内参；
- 相机外参；
- 可见 court line 的 2D polyline；
- net segments；
- 位置和姿态信息。

这些 JSON 标签现在主要用于：

- 调试合成数据；
- 验证投影是否稳定；
- 后续从 mask/line 恢复相机位姿；
- 将来如果要改成关键点或 polyline 训练，可以作为标签来源。

## 如果要训练点

如果目标改成“检测图像中出现的球场关键点/交点”，需要新增一条训练路径：

- 从 JSON 或几何关系生成 keypoint heatmap；
- 数据集读取 heatmap；
- 模型输出 heatmap 而不是 mask；
- 损失函数改成 heatmap MSE、focal loss 或类似形式。

当前代码还没有做这条路径。
