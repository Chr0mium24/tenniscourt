# 广角镜头网球场相机相对位姿调研结果

调研日期：2026-07-13

项目补充信息：用户已使用 OpenCV 完成相机标定，因此后续方案默认相机内参和畸变参数已知；仍需确认使用的是普通针孔模型还是 `cv::fisheye` 模型。

## 结论先行

有现成方案可以作为起点，但没有找到同时满足“网球场专用 + 超广角/鱼眼 + 单帧局部可见 + 直接输出相机 6DoF 位姿”的开箱即用仓库。

最值得先验证的是：

> 深度网络预测带可见性/置信度的网球场关键点和线段，利用网球场几何建立初始单应性，再在已知鱼眼相机模型下用点和线的重投影误差优化相机位姿；视频中用时序滤波和多帧累积补足单帧缺失。

这条路线能直接针对此前传统视觉失败的两个原因：光照泛化交给学习模型，画面不全交给“可见性建模 + 几何约束 + 跨帧累积”，而不是依赖一次完整的线段聚类。

## 问题拆解

需要区分两个输出层级：

1. **球场平面注册**：求球场平面与图像之间的单应矩阵 `H`，可以把球员脚点映射到球场平面坐标。
2. **相机相对位姿**：求球场坐标系下相机位置、朝向、相机高度，以及必要时的焦距/畸变参数。

只求 `H` 通常足以做二维球场定位，但不能可靠描述球场上方的三维目标。ProCC 论文指出，很多体育标定基准偏向单应性，忽略了非平面物体和镜头畸变；本项目应直接以 3D 点/线重投影误差评价最终位姿，而不是只看球场平面 IoU。

广角镜头还需要明确相机模型：

- 普通广角、畸变较小：使用已有 OpenCV 内参和畸变参数，可选择先去畸变再使用针孔模型。
- 鱼眼或超广角：确认已有标定是否使用 `cv::fisheye::calibrate`；建议保留鱼眼/等距投影模型，直接在畸变图像上做重投影优化。简单地把边缘裁掉或强行套普通单应性会损失有效视野。

## GitHub 现成实现

### 1. `yastrebksv/TennisCourtDetector`：最接近网球场需求

[GitHub 仓库](https://github.com/yastrebksv/TennisCourtDetector)

特点：

- 热力图网络预测 14 个网球场关键点，并增加球场中心点辅助收敛；
- 训练集有 8,841 张 1280×720 图像，覆盖 hard、clay、grass 三种场地；
- 提供预训练权重、训练脚本、单图/视频推理代码；
- 后处理包括局部线段精修和用参考球场单应性重建关键点；
- README 报告的关键点准确率为 0.961，组合后中位距离为 1.83 像素，阈值是 7 像素。

适配判断：这是首选 baseline，但不能直接认为解决了本项目问题。其数据来自网球赛事转播，视角和画面分布与固定广角相机可能不同；数据标注主要是完整可见的 14 点，代码虽然允许越界点不产生热力图，但没有成熟的可见性损失、部分球场专门训练或鱼眼模型。因此需要用本项目视频微调，并把每个关键点扩展为 `(x, y, visible, confidence)`。

许可证判断：仓库页面没有显示明确的开源许可证，预训练权重也通过网盘提供。商业集成前需向作者确认代码、数据和权重的使用权限。

### 2. `gchlebus/tennis-court-detection`：传统方法参考实现

[GitHub 仓库](https://github.com/gchlebus/tennis-court-detection)

这是基于 Farin 等人的 court-model 方法的 C++ 实现，流程是白线像素检测、候选线段提取、组合搜索和球场模型匹配，输出 16 个球场交点。许可证是 BSD-3-Clause。

它适合用来理解球场模型、关键点顺序和几何验证，不建议作为主方案：核心仍依赖颜色/纹理/线段检测，正好会受到光照、阴影、污渍和鱼眼曲线的影响。

### 3. `peterson-scbr/tennis-player-tracking`：工程流水线参考

[GitHub 仓库](https://github.com/peterson-scbr/tennis-player-tracking)

该仓库把 YOLO、ByteTrack、网球场关键点、单应性和球场坐标轨迹串成了一个工程流程，适合参考“检测球员脚点 → 映射到球场坐标”的接口组织方式。但仓库规模小、没有公开成熟的训练权重/评测基准，不能作为相机位姿算法的主要依据。

### 4. `mguti97/PnLCalib`：可移植的点线联合标定框架

[GitHub 仓库](https://github.com/mguti97/PnLCalib)

这是足球场的官方实现，但方法和网球场高度可迁移：

- 网络同时预测关键点和线段端点；
- 从线段交点扩充关键点，缓解可见点不足；
- 初始相机标定后，用检测到的点和线做非线性重投影优化；
- 仓库更新中加入了镜头畸变优化。

它能直接借鉴“点检测负责初始化，线检测负责精修”的结构，但不能直接使用足球权重。需要替换球场模型、重新生成网球场标注，并处理网球场网线/球网等与足球场不同的几何元素。

许可证是 GPL-2.0，若项目有闭源或商业分发要求，建议只借鉴论文和算法思想，重新实现模块或先做许可证评估。

### 5. `tobibaum/PartialSportsFieldReg_3DHPE`：部分场地可见的直接参考

[GitHub 仓库](https://github.com/tobibaum/PartialSportsFieldReg_3DHPE)  
[论文：Monocular 3D Human Pose Estimation for Sports Broadcasts using Partial Sports Field Registration](https://arxiv.org/abs/2304.04437)

论文专门讨论特写画面遮挡球场标记的情况：部分场地注册可以得到一组场景一致的相机标定，但存在一个自由度的不确定性，再通过时序/人体等上下文约束解决。仓库提供 MIT 许可证、代码和 Unreal Engine 生成的合成数据。

它不是网球场现成模型，但对“画面不全”最有启发：不要要求单帧恢复完整球场，而是输出带不确定性的候选位姿，再使用相邻帧或场景先验消除歧义。

### 6. 体育场地通用代码：`KpSFR` 和 SoccerNet

[KpSFR 项目页](https://ericsujw.github.io/KpSFR/)  
[SoccerNet 标定代码](https://github.com/SoccerNet/sn-calibration)  
[Sportlight 第一名方案](https://github.com/NikolasEnt/soccernet-calibration-sportlight)

这些项目不是网球场专用，但可复用的设计很明确：

- KpSFR 用规则网格关键点替代稀疏角点，提升视角变化、局部遮挡和场地纹理单一时的可用点数量；
- SoccerNet 官方 baseline 用语义线分割后再估计线段和相机参数，并明确把“利用 mask、RANSAC、线/椭圆精修”列为改进方向；
- Sportlight 方案把点模型扩充到 57 个几何点，同时使用关键点模型、线模型和启发式相机标定，说明“多类型几何观测 + 约束筛选”比只依赖少数交点更稳。

对网球场的启示是：除了 14/16 个经典交点，还应加入均匀采样的球场线点、边线/底线端点、发球线交点和网线两端，并让模型同时输出 line mask 或 line extremities。

## 论文与方法对比

| 方案 | 关键思想 | 对本项目的价值 | 主要限制 |
|---|---|---|---|
| [Camera calibration in sport event scenarios](https://doi.org/10.1016/j.patcog.2013.05.011) | 从单张网球/篮球/足球场图像中，用少量线和圆估计单应性、焦距、相机位置和姿态；讨论阴影与镜头畸变 | 证明“少量可见几何也能做单帧位姿”，可作为几何优化基线 | 特征提取仍偏手工，遇到强反光、复杂背景和极端鱼眼仍可能失败 |
| [Fast Camera Calibration for the Analysis of Sport Sequences](https://research.tue.nl/en/publications/fast-camera-calibration-for-the-analysis-of-sport-sequences/) | 专用线检测 + RANSAC + 球场模型组合搜索 + 时序跟踪 | 适合参考初始化、RANSAC 和视频跟踪结构 | 2005 年方法，学习泛化和广角模型不足 |
| [A Robust and Efficient Framework for Sports-Field Registration](https://openaccess.thecvf.com/content/WACV2021/html/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.html) | 均匀网格关键点 + 密集到线/关键区域的距离特征 + 多任务网络 | 直接针对纹理单一、窄视野和球员遮挡，最符合“画面不全无法聚类” | 主要输出平面注册/单应性；需要换成网球场数据和模型 |
| [Sports Field Registration via Keypoints-aware Label Condition](https://ericsujw.github.io/KpSFR/) | 关键点身份条件化的动态滤波和实例分割式关键点检测 | 适合在局部可见、特征稀疏时提高关键点预测鲁棒性 | 论文数据是足球；代码/数据迁移成本较高 |
| [TVCalib](https://mm4spa.github.io/tvcalib/) | 由语义线段/点云对应关系直接优化相机姿态、视场角和畸变 | 强调直接优化 3D 相机参数，而不是只做单应性 | 以足球场为对象，需要重新定义网球场段类别和训练数据 |
| [PnLCalib](https://arxiv.org/abs/2404.08401) | 点和线联合检测，点线非线性重投影精修，并支持畸变优化 | 最适合作为最终“点线联合 + 鱼眼模型优化”的算法模板 | 足球代码不能直接运行在网球场；GPL-2.0 |
| [A Universal Protocol to Benchmark Camera Calibration for Sports](https://arxiv.org/abs/2404.09807) | 用任意已知 3D 物体的重投影评估相机模型，不只评估平面单应性 | 提供本项目应采用的评测思想，特别适合广角/鱼眼 | 它是评测协议，不是直接推理模型 |

## 对“光照泛化”和“画面不全”的具体回答

### 光照泛化

单纯白色阈值、边缘和 Hough 线段会把曝光变化、阴影、球员衣服和场地纹理误当成几何特征。学习模型应负责从原始 RGB 或轻度归一化图像中预测：

- 关键点热力图；
- 每条球场线的语义 mask/端点；
- 每个点的可见性和置信度。

训练时应做亮度/对比度/gamma、色温、阴影、局部过曝、运动模糊、压缩噪声和不同场地颜色增强。几何模块只使用高置信度观测，并通过 RANSAC 或鲁棒损失抵抗错误预测。

### 画面不全

完整 14 点并不是必要条件。可采用分层策略：

1. 关键点模型在画面外点上输出“不可见”，而不是把画面边界当作球场边界。
2. 只要有足够的、几何上不退化的点/线，就估计候选单应性或位姿。
3. 用球场模型预测未观测点的位置，但把它们标记为推断值，不能当作真实观测参与同等权重的优化。
4. 在视频上对连续帧的观测做滑动窗口优化；相机固定时，可将多帧关键点/线段并入同一个位姿估计。
5. 当观测退化到无法区分左右半场、尺度或深度时，输出 `invalid/uncertain`，不要硬聚类出一个看似稳定的结果。

这里需要注意可观测性：只看到球场局部时，平面映射可能仍然可行，但完整 6DoF 位姿可能有多个等价解。部分场地论文也明确指出会留下自由度，因此必须靠已知内参、相机安装先验、时序连续性或额外的非平面点来消除歧义。

## 推荐落地架构

```text
广角/鱼眼视频
    ↓
相机内参与畸变模型（离线标定，或固定设备一次标定）
    ↓
学习模型：关键点热力图 + 球场线 mask/端点 + visibility/confidence
    ↓
网球场模板：真实尺寸、线段、网线两端、可选非平面参考物
    ↓
RANSAC 初始 H / 位姿候选
    ↓
鱼眼投影模型下的点线联合重投影优化
    ↓
跨帧滤波/滑动窗口优化 + 退化检测
    ↓
相机位置 (x,y,z)、姿态 (yaw,pitch,roll)、质量分数和有效性
```

### 推荐的第一版实现顺序

1. 先复现 `TennisCourtDetector`，只验证网球场关键点能否在项目视频上稳定预测。
2. 不做聚类，直接按固定关键点语义编号与网球场模板对应；用 RANSAC 求 H。
3. 采集覆盖白天/阴天/逆光/室内、不同场地颜色、不同相机高度和视角的数据；标注关键点可见性，额外标注线 mask。
4. 读取已有 OpenCV 标定结果，确认相机矩阵、畸变系数、图像分辨率和标定模型；不再重复做内参标定。
5. 在 H 稳定后，直接做已知内参下的位姿分解和对应投影模型的重投影优化。
6. 最后加入滑动窗口和质量门限，验证相机轻微移动、遮挡和局部可见时的恢复能力。

## 建议的最小验证实验

本轮只做了论文和仓库静态调研，没有在本地运行模型；当前工作区没有视频、标注或依赖环境，因此不虚构精度结果。下一轮应保存每组实验的配置和结果。

### 数据划分

- 至少 5 个不同光照/天气条件；
- 至少 3 个相机位置或安装高度；
- 每个条件包含完整球场、半场、边缘裁切、球员遮挡和强阴影；
- 按“视频/场次”划分训练和测试，不能随机打散相邻帧。

### 指标

- 关键点：可见点 PCK@5/10 px、可见性 F1、每类点的误差；
- 球场注册：线段重投影误差、球场模板 IoU、平面坐标误差；
- 相机位姿：位置误差、姿态角误差、相机高度误差；
- 广角模型：边缘区域和中心区域分别统计误差；
- 时序：有效帧比例、跳变次数、连续遮挡后的恢复时间；
- 失败检测：退化场景是否正确输出 invalid，而不是输出错误位姿。

### 需要对比的实验组

| 组别 | 检测器 | 几何模型 | 目的 |
|---|---|---|---|
| A | 传统白线/Hough | 针孔单应性 | 复现此前失败基线 |
| B | TennisCourtDetector | 针孔单应性 | 验证学习关键点的收益 |
| C | B + 可见性训练 | RANSAC + 时序 | 验证部分画面和遮挡 |
| D | C | 鱼眼投影 + 点线优化 | 验证广角边缘误差和真实位姿 |
| E | D | 多帧滑动窗口 | 验证稳定性与相机轻微位移 |

## 风险与决策

- **最大风险不是模型结构，而是域差异**：公开数据是赛事转播，项目视频可能是固定监控式广角，必须尽早用自采数据微调。
- **超广角不能只靠普通 homography**：球场平面仍可用单应性描述，但边缘畸变和球场上方目标需要鱼眼投影模型。
- **网球场几何信息比足球稀疏**：可以加入规则网格采样点和线 mask，但这些点不是视觉上天然的角点，训练标注和损失设计比直接套 YOLO pose 更重要。
- **许可证要前置确认**：`TennisCourtDetector` 页面未显示明确许可证；`PnLCalib` 是 GPL-2.0；`PartialSportsFieldReg_3DHPE` 是 MIT。生产集成前不能只看代码能否运行。

## 参考资料

### GitHub

- [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- [gchlebus/tennis-court-detection](https://github.com/gchlebus/tennis-court-detection)
- [peterson-scbr/tennis-player-tracking](https://github.com/peterson-scbr/tennis-player-tracking)
- [mguti97/PnLCalib](https://github.com/mguti97/PnLCalib)
- [SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration)
- [NikolasEnt/soccernet-calibration-sportlight](https://github.com/NikolasEnt/soccernet-calibration-sportlight)
- [tobibaum/PartialSportsFieldReg_3DHPE](https://github.com/tobibaum/PartialSportsFieldReg_3DHPE)

### 论文与技术资料

- [Camera calibration in sport event scenarios](https://doi.org/10.1016/j.patcog.2013.05.011)
- [Fast Camera Calibration for the Analysis of Sport Sequences](https://research.tue.nl/en/publications/fast-camera-calibration-for-the-analysis-of-sport-sequences/)
- [A Robust and Efficient Framework for Sports-Field Registration](https://openaccess.thecvf.com/content/WACV2021/html/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.html)
- [Sports Field Registration via Keypoints-aware Label Condition](https://ericsujw.github.io/KpSFR/)
- [TVCalib: Camera Calibration for Sports Field Registration in Soccer](https://mm4spa.github.io/tvcalib/)
- [PnLCalib: Sports Field Registration via Points and Lines Optimization](https://arxiv.org/abs/2404.08401)
- [Monocular 3D Human Pose Estimation for Sports Broadcasts using Partial Sports Field Registration](https://arxiv.org/abs/2304.04437)
- [A Universal Protocol to Benchmark Camera Calibration for Sports](https://arxiv.org/abs/2404.09807)
- [Improving Tennis Court Line Detection with Machine Learning](https://www.ml6.eu/en/blog/improving-tennis-court-line-detection-with-machine-learning)
