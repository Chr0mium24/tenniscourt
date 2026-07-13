# 网球场线检测仿真方案建议

日期：2026-07-13

## 背景

当前目标不是完整机器人仿真，也不是高真实感宣传视频，而是验证：

- 低位约 40 cm 广角相机能否看到足够稳定的网球场线；
- 能否用合成数据训练一个场地线检测模型；
- 能否从检测到的场地线进一步估计相机相对球场的位置。

已有条件：

- 已完成 OpenCV 相机标定；
- 真实相机低位，只能看到半场或局部半场；
- 传统视觉受光照、泛化和视野不完整影响较大；
- 需要自动生成图像和线标注。

## 结论

第一阶段不建议直接上 Blender/BlenderProc。更合适的是先做一个专用的 **OpenCV 几何投影仿真器**：

```text
标准网球场线 3D 坐标
        ↓
采样低位相机外参
        ↓
使用真实 OpenCV K/D 投影到图像
        ↓
把投影后的线段光栅化成 RGB 图、line mask、线段/keypoint JSON
        ↓
加入背景、模糊、曝光、噪声、压缩、遮挡等随机化
        ↓
训练/验证场地线检测模型
```

原因：

- 这个任务的核心真值是场地线几何，OpenCV 投影比 3D 引擎更直接、更可控；
- 标注可以天然精确到像素级 line mask，不需要从渲染器里反推；
- 可以直接复用真实标定的 `cv2.projectPoints` 或 `cv2.fisheye.projectPoints`；
- 运行很轻，CPU 即可批量生成大量图片；
- 便于快速做可行性验证：模型到底能不能在半场低视角识别线。

Blender headless 不是不能用。Blender 官方支持 `--background` 无界面渲染，BlenderProc 也能设置 K 矩阵和 camera-to-world 位姿。但对当前第一阶段来说，它的安装、渲染、材质和畸变适配成本偏高。等几何检测路线跑通后，再引入 BlenderProc 做真实光照、阴影、球网、围栏和材质随机化更合理。

## 推荐分阶段

### 阶段 1：OpenCV 几何仿真器

目标：用最小成本验证线检测可行性。

功能：

- 用真实尺寸生成网球场线段坐标，单位米；
- 采样相机高度 `0.35 m ~ 0.45 m`，yaw/pitch/roll 覆盖实际安装误差；
- 用已有 OpenCV 标定投影场地线；
- 输出：
  - RGB 合成图；
  - 二值或多类别 line mask；
  - 每条线段的 2D polyline；
  - 可见关键点；
  - 相机外参；
  - K/D 和图像分辨率；
- 加入域随机化：
  - 场地颜色、线宽、线亮度、线磨损；
  - 背景纹理或真实场地图像背景；
  - 阴影、局部遮挡、运动模糊；
  - 曝光、白平衡、gamma、噪声、JPEG/H.264 压缩。

适合训练：

- segmentation 模型：预测场地线 mask；
- polyline/heatmap 模型：预测线段或线交点；
- 后处理：从 mask 拟合线段，再用 PnP/线约束估计相机位姿。

风险：

- 纯几何合成和真实图像存在 sim-to-real gap；
- 需要用真实采集图片做少量微调或验证；
- 线条纹理、阴影、污渍、场地反光要做足随机化，否则模型会学到合成图特征。

### 阶段 2：BlenderProc 增强真实感

目标：当阶段 1 证明几何路线可行后，再补真实感。

适合加入：

- 真实网球场材质；
- 光照和阴影；
- 球网、围栏、球员和遮挡物；
- 低机位运动轨迹；
- RGB、深度、分割等多模态真值。

注意：

- BlenderProc 可直接设置 3x3 K 矩阵；
- 相机外参使用 camera-to-world 矩阵；
- 如果标定来自 `cv2.fisheye.calibrate`，不要直接把鱼眼参数当成 Brown-Conrady 参数，需要单独验证畸变一致性；
- 建议保留逐帧 PNG/JSON，MP4 只作为预览。

### 阶段 3：真实数据闭环

目标：验证模型是否真正能用。

建议采集：

- 不同时间：晴天、阴天、傍晚、夜间灯光；
- 不同场地：硬地、草地或不同颜色硬地；
- 不同位置：边线附近、底线附近、网前、斜角；
- 不同状态：球网、球员、球、阴影、线条磨损。

训练策略：

- 先用合成图预训练；
- 用少量真实图人工标注微调；
- 在真实视频上评估线 mask IoU、线段角度误差、交点误差和最终相机位姿误差。

## 为什么不是直接 Blender

Blender headless 的问题不是“太重到不能用”，而是它解决的是更复杂的问题：

- 真实渲染；
- 材质；
- 光照；
- 阴影；
- 3D 遮挡；
- 视频帧输出。

而当前第一阶段最关键的是：

- 投影几何对不对；
- 低位半场视野里线条是否足够可观测；
- 标签是否精确；
- 生成速度是否够快；
- 能否快速迭代模型和后处理。

这些用 OpenCV 几何仿真器更直接。

## 第一版实验

建议先做 4 个小实验：

1. 生成 1,000 张纯几何图：不同相机位置、姿态、场地颜色、线宽和噪声。
2. 训练一个轻量 segmentation 模型检测线 mask。
3. 用真实相机采集 50-100 张图，人工标注线 mask 或线段。
4. 对比合成验证集和真实图效果，判断是否需要 BlenderProc 增强真实感。

关键指标：

- line mask IoU；
- 线段端点或交点平均像素误差；
- 能成功恢复位姿的帧比例；
- 位姿误差：平移误差、yaw/pitch/roll 误差；
- 低可见线数量下的失败模式。

## 参考资料

- [OpenCV Camera Calibration and 3D Reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- [OpenCV Fisheye camera model](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html)
- [Blender command line arguments](https://docs.blender.org/manual/en/latest/advanced/command_line/arguments.html)
- [BlenderProc camera configuration](https://dlr-rm.github.io/BlenderProc/docs/tutorials/camera.html)
