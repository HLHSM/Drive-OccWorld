# R101 教师到 R50 学生的占用蒸馏

运行 `bash tools/train_farmsim_front3_s.sh` 默认依次训练四组，每组追加 5 epoch：

| 任务选择器 | 实验 | 占用 KL 权重 | FPN CWD 权重 |
|---|---|---:|---:|
| `kd_control` | A：仅继续训练 | 0 | 0 |
| `kd_logits` | B：仅输出蒸馏 | 1 | 0 |
| `kd_fpn` | C：仅图像特征蒸馏 | 0 | 1 |
| `kd_both` | D：两者联合 | 1 | 1 |

例如只运行 D：

```bash
FARMSIM_TASKS=kd_both bash tools/train_farmsim_front3_s.sh
```

四组均使用 R50 + TSA + Agri-AMoE96，输入为三相机 512×288 图像，
预测 75×75×25 网格，作物抽样比例 0.1、阈值 500。有效 batch 默认为 24。
旧的 `r50_amoe_cropsample` 任务仍可显式选择，会重新训练原来的 15/20 epoch 两组。

## 检查点与优化器

- 学生：`work_dirs/front3_s_base_r50_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261014_200858/epoch_15.pth`。
- 教师：`work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth`，使用同目录保存的配置。
- 使用 `--load-from` 加载学生权重，新建优化器和 5 epoch 学习率计划；不恢复旧优化器及已衰减的旧学习率。
- 初始学习率 `5e-5`，骨干倍率沿用 0.1，学习率 warm-up 300 iteration；
  蒸馏权重在前 0.5 epoch 线性增长，每个 epoch 进行验证并保存指标和检查点。
- 四组统一将新建动态 AMP scaler 的初始值设为 128（仍会自动调整），
  避免用默认 65536 加载成熟学生权重后，起始阶段反复溢出、跳过更新。

可用环境变量调整：`KD_STUDENT_CHECKPOINT`、`KD_TEACHER_CHECKPOINT`、
`KD_TEACHER_CONFIG`、`KD_EPOCHS`、`KD_LR`、`KD_LR_WARMUP_ITERS`、
`KD_WARMUP_EPOCHS`、`KD_LOGIT_WEIGHT`、`KD_FPN_WEIGHT`、
`KD_LOGIT_TEMPERATURE`（默认 2）和 `KD_FPN_TEMPERATURE`（默认 1）。
`KD_FP16_INITIAL_SCALE` 可调整上述 AMP 初始值。

## 实现约定

`FarmSimOccupancyDistiller` 仅在训练前向中惰性构建冻结的 R101 教师，教师使用
`eval()` 和 `no_grad()`，不进入优化器、DDP 同步或学生检查点。
教师加载不改变学生的随机数状态；图像 GridMask 在原图像增强位置只执行一次，
两者使用完全相同的增强图像和标定。

- 输出蒸馏仅使用最终 Agri-AMoE logits：排除忽略标签与教师预测错误的体素，
  每个 GT 类别先求平均 KL，再对本 batch 中存在有效体素的类别平均。
- 图像蒸馏使用第 0、1 层 FPN。学生通过恒等初始化的 1×1 适配层对齐通道，
  每个相机、每个通道分别做空间均值/方差标准化及 softmax，计算教师到学生的 KL。
  适配层也接收标准化特征并以 FP32 计算，防止不同骨干的激活幅度差异导致 AMP 梯度溢出。
- 原始占用监督及辅助头损失不缩减，不按新增损失项数量重新平均。
- 日志新增 `loss_kd_logits`、`loss_kd_fpn`（对应启用项）、
  `kd_reliable_fraction` 和 `kd_warmup_scale`。
- 评估与推理走原学生路径，不调用教师或适配层；只做推理时也不需要读取教师文件。
  训练检查点保留少量适配层参数供断点续训，完全不包含教师权重。

## 验证

CPU 损失与梯度检查：

```bash
PYTHONPATH="$(pwd):${PYTHONPATH:-}" /home/HL/.conda/envs/dow2/bin/python tools/tests/test_farmsim_distillation.py
```

两卡、两轮、包含 epoch 间评估的 A–D 冒烟测试：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash tools/train_farmsim_r50_distillation_smoke.sh
```

该脚本使用已有 `smoke48` manifest；零历史、零未来设置下实际读取 50 个训练帧、
18 个验证帧。测试输出目录带 `_smoke48` 标记，便于与正式完整数据实验区分。

2026-10-15 已在 GPU 0、1 完成上述四组两轮测试：每组均写入第 1、2 轮
验证指标及检查点，50 条训练日志中的损失、梯度范数均为有限值。
最终测试目录的时间后缀依次为 `093559`、`093645`、`093736`、`093823`。
另已通过 5 项 CPU 单元测试，并用联合蒸馏检查点验证：即使将教师路径改成
不存在的文件，学生仍可独立推理。冒烟测试用于验证训练路径，不用于判断精度收益。
