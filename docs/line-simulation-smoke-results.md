# 场地线生成与训练 Smoke Test 结果

日期：2026-07-13

## 环境

- Python：通过 `uv` 创建 `.venv`
- PyTorch：`2.13.0+cu130`
- CUDA 可用：`True`
- GPU：`NVIDIA GeForce RTX 4060 Ti`

## 生成测试

命令：

```bash
uv run tenniscourt-generate --count 16 --width 320 --height 180 --out outputs/smoke --seed 11
```

结果：

- 生成样本数：16
- 输出文件数：49
  - `images/*.png`：16
  - `masks/*.png`：16
  - `labels/*.json`：16
  - `metadata.json`：1
- 第一张图像尺寸：`180 x 320 x 3`
- 第一张 mask 尺寸：`180 x 320`
- 第一张 mask 非零像素：`3115`
- 第一张可见线条数：`8`
- 相机模型：`pinhole`

## 训练测试

命令：

```bash
uv run --extra train tenniscourt-train \
  --data outputs/smoke \
  --out runs/smoke-line-seg-v2 \
  --epochs 1 \
  --batch-size 4 \
  --workers 0 \
  --max-steps 2 \
  --require-cuda
```

结果：

- 设备：`cuda`
- epoch：`1`
- train loss：`1.6054003834724426`
- val IoU：`0.052650462836027145`
- 训练完成：是
- `--require-cuda` 通过：是

这个 IoU 只代表 16 张样本、2 个训练 step 的链路验证，不代表模型效果。

## 代码约束检查

当前提交候选中，代码和文档文件均低于 500 行：

- 最大代码文件：`src/tenniscourt/render.py`，203 行
- 最大训练文件：`src/tenniscourt/train.py`，153 行
- `uv.lock` 由 uv 自动生成，超过 500 行，已加入 `.gitignore`，不纳入提交

## 结论

- OpenCV 几何投影生成链路已跑通；
- RGB 图、二值 line mask 和 JSON label 已同步输出；
- PyTorch 训练入口已跑通，并确认使用 CUDA；
- 下一步可以把生成数量提高到 1,000-10,000，并补少量真实图验证 sim-to-real 差距。
