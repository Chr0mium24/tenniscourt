# 场地线合成生成与训练实现计划

日期：2026-07-13

## 目标

搭建一个最小可跑框架，用于：

- 生成低位广角相机视角下的网球场线合成图；
- 同步导出二值 line mask 和几何标签；
- 用 GPU 优先训练一个轻量线分割模型；
- 保持根目录清晰、代码分层明确，单个代码文件不超过 500 行。

## 工程结构

```text
configs/                 生成配置
src/tenniscourt/
  camera.py              OpenCV 投影、相机内参/外参
  court.py               标准网球场线几何
  render.py              合成 RGB/mask
  generate.py            数据生成命令
  data.py                PyTorch Dataset
  model.py               轻量 U-Net
  train.py               训练命令
docs/                    计划、调研、实验结果
outputs/                 生成数据，git 忽略
runs/                    训练输出，git 忽略
```

## 第一阶段范围

- 默认用程序化 OpenCV 几何投影生成线条，不依赖 Blender；
- 默认相机高度范围 `0.35 m ~ 0.45 m`；
- 默认视场角 `105°`，模拟低位广角；
- 默认输出 RGB、mask、JSON label；
- 训练入口默认 `--device auto`，PyTorch 可见 CUDA 时使用 GPU。

## 验证步骤

1. `uv sync` 安装生成依赖。
2. 生成 16 张 smoke test 图片。
3. `uv sync --extra train` 安装训练依赖。
4. 用 `--max-steps 2` 跑一次训练 smoke test。
5. 保存结果到 `docs/line-simulation-smoke-results.md`。
