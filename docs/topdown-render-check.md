# 俯视全场渲染检查

日期：2026-07-13

## 目的

在继续修改网球场几何或低机位相机采样前，先生成全场俯视基准图，确认：

- 当前标准球场线拓扑是否能画出完整网球场；
- 同一套几何经过相机投影后是否能完整渲染；
- 问题是否更可能来自低机位相机采样/透视，而不是全场拓扑定义。

## 输出

- 正交俯视几何图：[`assets/court_topdown_orthographic.png`](assets/court_topdown_orthographic.png)
- 高空相机投影图：[`assets/court_topdown_camera.png`](assets/court_topdown_camera.png)
- 侧向斜视全场图：[`assets/court_side_oblique_camera.png`](assets/court_side_oblique_camera.png)

## 生成方式

使用现有代码中的：

- `tenniscourt.court.tennis_court_lines`
- `tenniscourt.court.sample_line_points`
- `tenniscourt.camera.default_intrinsics`
- `tenniscourt.camera.look_at_rvec_tvec`
- `tenniscourt.camera.project_points`

其中：

- 正交俯视图直接把球场 `x/y` 米制坐标映射到像素；
- 高空相机投影图使用针孔相机，从 `(0, 0, 22m)` 看向球场中心 `(0, 0, 0)`；
- 高空相机图中可见线条数为 `11`。
- 侧向斜视图使用针孔相机，从 `(-8, -22, 10m)` 看向 `(0.5, 0, 0)`，FOV 为 `95°`；
- 侧向斜视图中可见线条数为 `11`。

## 观察

正交俯视图中，双打边线、单打边线、底线、发球线、中心发球线和中心标记的拓扑关系看起来正常。

高空相机投影图中，完整球场能通过当前相机投影路径渲染出来，说明基础 `project_points` 链路没有明显全局错误。

侧向斜视图不沿球场中轴线正对球场，能用于检查非正对视角下的透视关系。图中完整球场仍能被投影出来，说明基础几何在侧向斜视相机下也能保持连通和可见。

这次检查只能说明全场俯视几何和高空投影正常。低位 40cm 相机视角下如果出现“看起来几何不对”，下一步应重点检查：

- 低机位相机采样位置和朝向；
- 近距离透视导致的线条比例；
- 是否需要显示相机在俯视图中的位置、朝向和视锥；
- 是否需要使用真实 OpenCV 标定 K/D 替换默认 FOV 模型。
