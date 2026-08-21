# UE5 农业机器人 3D 语义占用训练

本适配以 `/mnt/g/SimData` 的 FarmSim UE5 数据为输入，不再依赖 nuScenes、CAN bus 或轨迹标签。模型训练目标是当前时刻的 3D 语义占用；配置中的 `predict_trajectory=False` 会屏蔽规划/未来轨迹头，`future_pred_frame_num=0` 表示不进行未来占用预测。

## 已生成的数据划分

划分文件位于 `data/farmsim/splits/`：`train.json`、`val.json` 和 `split_report.json`。以完整 sequence 为最小单元，绝不将同一 sequence 的不同帧拆到训练和验证中。使用固定 seed `20260821`，按作物种类、生长阶段、时段、天气以及帧数进行约束分层，验证集目标比例为 1/6：

| 划分 | sequence | 完整帧数 | 占比 |
| --- | ---: | ---: | ---: |
| 训练 | 273 | 34,571 | 83.33% |
| 验证 | 55 | 6,913 | 16.67% |

验证集包含全部 11 种作物，且生长阶段为 early/mature/mid = 22/21/22；时段和天气的比例也与全集接近。三个不完整 sequence 被自动排除，详见 `split_report.json`。

若 `/mnt/g/SimData` 有新增数据，重新生成可复现划分：

```bash
/home/hl/miniconda3/envs/py311/bin/python tools/create_farmsim_split.py \
  --data-root /mnt/g/SimData --output-dir data/farmsim/splits \
  --val-ratio 0.16666666666666666 --seed 20260821
```

## 坐标、标签和相机

- FarmSim 标注体素存储格式为 `[z,y,x]`，适配器转换为模型使用的 `[x,y,z]`。
- 坐标为 `x` 前、`y` 右、`z` 上；体素大小 0.2 m，语义 ID 保持原始 0--11，`occupancy_valid=0` 写为 ignore ID 255。
- 六相机模式输入 `front_left/front/front_right/rear_left/rear/rear_right`，预测 `x∈[-20,20]m, y∈[-10,10]m, z∈[-2,3]m` 的 200×100×25 体素。
- 前视三相机模式输入前三路相机，仅监督/预测 `x∈[0,20]m` 前半区的 100×100×25 体素；后半区不参与网络输出和损失。
- 每个样本使用前两帧加当前帧的时序 BEV 输入。UE 相机的 forward/right/up 轴已转换为针孔相机的 right/down/forward 轴，并使用每帧 JSON 中的内外参。

## 启动训练

### `dow2` 最小环境

已创建 `/home/hl/miniconda3/envs/dow2`，并验证 `torch 2.7.1+cu128` 可在
RTX 5090 上完成 CUDA 矩阵运算。PyTorch wheel 自带 CUDA 12.8 用户态运行库；
没有重装系统 CUDA、没有修改 `~/.bashrc`。新版 NVIDIA 驱动可运行该 wheel，
但系统 CUDA 13.2 工具链不能直接拿来编译旧项目的扩展，因此此 FarmSim 路径
改为不依赖这些扩展。

实际安装的核心包为 `Python 3.10`、`torch==2.7.1+cu128`、
`torchvision==0.22.1+cu128`、`mmcv==1.4.0`、`mmdet==2.14.0`、
`mmdet3d==0.17.1`、`mmsegmentation==0.14.1`、`numpy==1.23.5`、
`opencv-python-headless==4.8.1.78`、`einops==0.8.1` 与
`yapf==0.40.1`。`mmcv==1.4.0` 依赖 YAPF 的旧接口，因此不要升级 YAPF 至
`0.40.2` 或更高版本。`mmdet3d` 在该环境
中是仅保留模型注册器和图像分支的轻量安装，不包含点云算子。

以下库/组件不需要安装：nuScenes devkit、lyft SDK、DD3D/Detectron2、
InternImage/DCNv3、spconv、MMCV full 的 CUDA 算子、项目的 `dvxlr` 未来渲染
扩展、SciPy 轨迹采样、IPython 调试工具和 TensorBoard（FarmSim 默认只记录文本
日志）。FarmSim 配置也显式关闭了 ResNet
的 DCNv2。没有这些组件时，BEV 的多尺度可变形注意力使用 PyTorch 回退实现；
结果路径正确，但通常会比原生 CUDA 算子慢。

如需进入环境：

```bash
conda activate dow2
cd ~/Drive-OccWorld
```

### 训练命令

然后直接执行：

```bash
bash tools/train_farmsim_6cam.sh
# 或
bash tools/train_farmsim_front3.sh
```

两个脚本顶部都有 `CUDA_VISIBLE_DEVICES`、`NUM_GPUS`、`BATCH_SIZE` 和 `EPOCHS`，可直接修改。例如 `CUDA_VISIBLE_DEVICES="2,3,5"` 时必须写 `NUM_GPUS=3`。`BATCH_SIZE` 是每张卡的 batch size，总 batch size 为 `BATCH_SIZE × NUM_GPUS`；`EPOCHS` 是完整遍历训练集的次数。脚本会检查其为正整数及 GPU 配置一致性，并默认使用 `dow2` 的 Python；若 conda 位于别处，可通过 `PYTHON_BIN=/实际路径/python bash ...` 覆盖。`NUM_GPUS=1` 时直接单进程训练；多卡时才使用 PyTorch 推荐的 `torchrun --standalone`，并自动选择空闲本地端口。

主要实现位于 `FarmSimWorldDataset`：它直接读取 split JSON、RGB、元数据和占用二进制文件。模型改动还包括：动态支持 3/6 路相机注意力、去除轨迹头开关，以及根据真实目标尺寸计算占用损失，不再固定为 nuScenes 的 256×256×20 体素。
