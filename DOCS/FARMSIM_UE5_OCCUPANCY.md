# UE5 农业机器人 3D 语义占用训练

本适配以 FarmSim UE5 数据为输入，不再依赖 nuScenes、CAN bus 或轨迹标签。模型训练目标是当前时刻的 3D 语义占用；配置中的 `predict_trajectory=False` 会屏蔽规划/未来轨迹头，`future_pred_frame_num=0` 表示不进行未来占用预测。

## 已生成的数据划分

划分文件位于 `data/farmsim/splits/`：`train.json`、`val.json` 和 `split_report.json`。JSON 中的 `sequences[*].path` 是相对于数据集根目录的路径，`source_root` 为 `.`；实际根目录由训练脚本中的 `DATA_ROOT` 指定。以完整 sequence 为最小单元，绝不将同一 sequence 的不同帧拆到训练和验证中。使用固定 seed `20260821`，按作物种类、生长阶段、时段、天气以及帧数进行约束分层，验证集目标比例为 1/6：

| 划分 | sequence | 完整帧数 | 占比 |
| --- | ---: | ---: | ---: |
| 训练 | 248 | 32,369 | 83.29% |
| 验证 | 50 | 6,493 | 16.71% |

用户已从数据根目录删除全部 garlic 序列；现有划分同步删除训练集 25 个、验证集 5 个 garlic 序列，其他 sequence 的训练/验证归属保持不变。当前共 10 种作物，训练/验证帧数比例为 4.99:1；生长阶段、时段和天气分布仍按原划分保持接近。三个不完整 sequence 被自动排除，详见 `split_report.json`。

若 `/mnt/g/SimData` 有新增数据，重新生成可复现划分：

```bash
/home/hl/miniconda3/envs/py311/bin/python tools/create_farmsim_split.py \
  --data-root /mnt/g/SimData --output-dir data/farmsim/splits \
  --val-ratio 0.16666666666666666 --seed 20260821
```

如果只是删除某一作物、希望保留其余 sequence 的原有划分，可使用：

```bash
/home/hl/miniconda3/envs/py311/bin/python tools/prune_farmsim_split.py \
  --split-dir data/farmsim/splits --crop-type garlic
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

然后直接执行（先在脚本顶部修改 `DATA_ROOT`，或在命令前通过环境变量覆盖）：

```bash
bash tools/train_farmsim_6cam.sh
# 或
DATA_ROOT=/data/FarmSim bash tools/train_farmsim_front3.sh
```

两个脚本顶部都有可直接修改的训练开关：

- `DATA_ROOT`：数据集根目录，目录下应直接包含各个作物目录。例如 `DATA_ROOT=/data/FarmSim bash tools/train_farmsim_front3.sh`。
- `CUDA_VISIBLE_DEVICES`、`NUM_GPUS`：可见显卡及并行进程数。例如 `CUDA_VISIBLE_DEVICES="2,3,5"` 时必须写 `NUM_GPUS=3`。
- `BATCH_SIZE`、`EPOCHS`：每张卡的 batch size 和完整遍历训练集的次数；总 batch size 为 `BATCH_SIZE × NUM_GPUS`。该 batch size 同时写入验证/测试数据加载器（六相机脚本默认 `--no-validate`，需要每 epoch 验证时可删除该参数）。
- `HISTORY_FRAMES`：历史图像帧数，当前帧会额外输入，因此实际时序长度为 `HISTORY_FRAMES + 1`。
- `PREDICT_FUTURE_OCC`、`FUTURE_OCC_STEPS`：是否训练/评估未来占用，以及未来占用步数。开关为 `0` 时步数自动按 0 处理；开关为 `1` 时步数必须至少为 1。
- `PREDICT_FUTURE_TRAJ`、`FUTURE_TRAJ_STEPS`：是否启用未来轨迹头，以及轨迹预测步数。开关为 `0` 时步数忽略；开关为 `1` 时步数必须至少为 1。FarmSim 轨迹由 UE5 位姿计算得到；数据没有 3D 目标框，因此轨迹回归可用，碰撞框损失不提供监督。两个预测步数可以不同；占用预测需要更长轨迹时，超出轨迹标签范围的条件位移按 0 补齐。

脚本会将这些变量同步传给数据集、未来占用头和轨迹头，并检查数值及 GPU 配置一致性。每次运行默认写入带时间戳的新 `work_dirs` 子目录，不覆盖之前的实验；若 conda 位于别处，可通过 `PYTHON_BIN=/实际路径/python bash ...` 覆盖。`NUM_GPUS=1` 时直接单进程训练；多卡时使用 `torchrun --standalone`。

主要实现位于 `FarmSimWorldDataset`：它直接读取 split JSON、RGB、元数据和占用二进制文件。相机帧支持 `.jpg`、`.jpeg` 和 `.png` 扩展名，划分脚本也会统一识别这三种格式。模型改动还包括：动态支持 3/6 路相机注意力、去除轨迹头开关，以及根据真实目标尺寸计算占用损失，不再固定为 nuScenes 的 256×256×20 体素。

### 单独评估 checkpoint

单卡评估已有 checkpoint 时，不需要重新训练。以前视三相机为例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/hl/miniconda3/envs/dow2/bin/python tools/test.py \
  projects/configs/farmsim/farmsim_occ_front3.py \
  work_dirs/farmsim_occ_front3/epoch_1.pth \
  --launcher none --eval occ --deterministic \
  --cfg-options data.test.data_root=/mnt/g/SimData
```

六相机 checkpoint 将配置替换为 `projects/configs/farmsim/farmsim_occ_6cam.py`，并将 checkpoint 路径替换为对应文件。评估脚本会直接输出混淆矩阵、各类别 IoU、仅统计出现类别的 mIoU、全类别 mIoU 和 voxel accuracy；不会启动训练流程。

如果只想快速验证 checkpoint 和验证流程是否正常，可限制验证样本数，例如只跑 2 个样本：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/hl/miniconda3/envs/dow2/bin/python tools/test.py \
  projects/configs/farmsim/farmsim_occ_front3.py \
  work_dirs/front3_YYYYMMDD_HHMMSS/epoch_1.pth \
  --launcher none --eval occ --deterministic \
  --cfg-options data.test.data_root=/mnt/g/SimData \
                data.test.max_samples=2 data.test.samples_per_gpu=1
```

这会执行完整的模型 `forward_test`、占用统计和自定义单卡评估路径，但不会遍历全部 6,393 个验证样本。若 checkpoint 使用了未来占用或轨迹开关，还需把训练时的 `model.test_future_frame_num`、`model.turn_on_plan`、`model.predict_trajectory` 等配置通过 `--cfg-options` 原样补上。

训练过程中的每 epoch 验证会在 `Epoch(val)` 日志中先输出各类 mIoU 和 voxel accuracy，再输出原始混淆矩阵；这些内容同时写入对应的 `.log` 和 `.log.json` 文件。
