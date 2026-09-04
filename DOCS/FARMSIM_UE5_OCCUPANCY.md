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
/home/HL/.conda/envs/dow2/bin/python tools/create_farmsim_split.py \
  --data-root /mnt/g/SimData --output-dir data/farmsim/splits \
  --val-ratio 0.16666666666666666 --seed 20260821
```

如果只是删除某一作物、希望保留其余 sequence 的原有划分，可使用：

```bash
/home/HL/.conda/envs/dow2/bin/python tools/prune_farmsim_split.py \
  --split-dir data/farmsim/splits --crop-type garlic
```

## 坐标、标签和相机

- FarmSim 标注体素存储格式为 `[z,y,x]`，适配器转换为模型使用的 `[x,y,z]`。训练/评估使用聚合后的 6 类 taxonomy：`free`、`crop`、`soil_ground`、`drivable`、`other_vegetation`、`other_obstacle`；原始 `tree_foliage` 合并至 `other_vegetation`，原始 `building`、`fence_barrier`、`vehicle`、`tree_trunk` 合并至 `other_obstacle`，原始 `person_animal` 写为 ignore ID `255`，不参与损失或 IoU。
- 该聚合将 semantic occupancy head 从 12 类改为 6 类，旧 12 类 checkpoint 的输出层维度不兼容，必须以新的 `WORK_DIR` 从头训练；新 checkpoint 可用更新后的 `tools/test.py` 正常评估和可视化。
- 坐标为 `x` 前、`y` 右、`z` 上；体素大小 0.2 m。源语义 ID 为 0--11，读取时重映射为上述 0--5 训练 ID；`occupancy_valid=0`、`person_animal` 和未知源 ID 均写为 ignore ID 255。
- 六相机模式输入 `front_left/front/front_right/rear_left/rear/rear_right`，预测 `x∈[-20,20]m, y∈[-10,10]m, z∈[-2,3]m` 的 200×100×25 体素。
- 前视三相机模式输入前三路相机，仅监督/预测 `x∈[0,20]m` 前半区的 100×100×25 体素；后半区不参与网络输出和损失。
- 每个样本使用前两帧加当前帧的时序 BEV 输入。UE 相机的 forward/right/up 轴已转换为针孔相机的 right/down/forward 轴，并使用每帧 JSON 中的内外参。

## 启动训练

### `dow2` 最小环境

已创建 `/home/HL/.conda/envs/dow2`，并验证 `torch 2.7.1+cu128` 可在
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
cd /data/HL/Drive-OccWorld
```

### 训练命令

然后直接执行（先在脚本顶部修改参数）：

```bash
bash tools/train_farmsim_6cam.sh
# 或
bash tools/train_farmsim_front3.sh
```

若进程在保存 checkpoint 后中断，可在 `tools/train_farmsim_front3.sh` 顶部将 `WORK_DIR` 设为原工作目录、`RESUME_FROM` 设为对应 checkpoint（例如 `epoch_1.pth`），然后重新执行脚本。

两个脚本顶部都有可直接修改的训练开关：

- `DATA_ROOT`：数据集根目录，目录下应直接包含各个作物目录。
- `CUDA_VISIBLE_DEVICES`、`NUM_GPUS`：可见显卡及并行进程数。例如 `CUDA_VISIBLE_DEVICES="2,3,5"` 时必须写 `NUM_GPUS=3`。
- `BATCH_SIZE`、`EPOCHS`：每张卡的微 batch size 和完整遍历训练集的次数。`BATCH_SIZE` 决定峰值激活显存，且同时写入训练、验证和测试数据加载器（六相机脚本默认 `--no-validate`，需要每 epoch 验证时可删除该参数）。
- `IMAGE_WIDTH`、`IMAGE_HEIGHT`：每路 RGB 图像的输入分辨率，格式为宽×高，前视三相机脚本默认 `512×288`。该值会同步写入 train/val/test；减小分辨率可显著降低显存，但会改变输入规格，应作为独立实验记录。
- `USE_FP16`：`1` 时启用 PyTorch AMP FP16（动态 loss scale），可降低激活显存；`0` 时使用 FP32。当前 FarmSim 路径已对变参 BEV encoder 和概率 BCE 尺度损失作 FP16 兼容处理。
- `USE_CROP_GAP_REFINEMENT`：`1` 时启用 Crop-Gap-Aware Boundary Refinement。模型从现有 occupancy 标签在线生成 crop/free 3D 边界，并对距离 crop 较近的 free voxel 加权；最终 decoder 的边界置信度仅门控一个零初始化的占用残差。`CROP_GAP_BOUNDARY_LOSS_WEIGHT`、`CROP_GAP_FREE_LOSS_WEIGHT` 控制两项附加损失，`CROP_GAP_ALPHA`、`CROP_GAP_SIGMA`、`CROP_GAP_RADIUS` 控制 free-gap 距离权重。
- `USE_SELECTIVE_C2F`：`1` 时启用 Selective Coarse-to-Fine Plant Occupancy Refinement。它按 coarse logits 的 crop/free 不确定性挑选 `C2F_ACTIVE_RATIO` 的 BEV cell，仅为这些 cell 解码四个 2×2 子查询，再将子查询残差聚合回原 `100×100` 输出；因此不改变数据标签、评估接口或全局 BEV query 数。`C2F_CHANNELS` 控制子查询解码宽度。
- `--use-dual-hardness-refinement 1`：启用 ADHR（Agricultural Dual-Hardness Refinement）。训练期从最终 occupancy logits 选取类别间隔最小的困难 voxel，并为 crop/free 边界和近 crop free gap 保留配额；BEV 特征加高度嵌入后由轻量 MLP 细化，同时用 EMA teacher 蒸馏。它仅增加 `loss_adhr_refine`、`loss_adhr_distill`，不改变推理输出图。
- `--use-gvad-attention 1`：在 `HISTORY_FRAMES=0` 下用 Geometry-Visible Anchor Deformable Attention（GVAD）替换 TSA。它保留 query-adaptive deformable 局部采样，并从已有相机投影 `bev_mask` 中构造几何可见性加权的 BEV anchors，将可靠可见区域的上下文传播给弱可见区域。`--gvad-anchor-grid-height/width` 默认生成 `4×8=32` 个 anchors。
- `--gvad-use-visibility 0`：GVAD 的去可见性消融，anchor 改为普通平均池化；`--gvad-use-local-deformable 0`：去局部可变形路径，仅保留 anchor 上下文。三种 GVAD 变体均保持原始 occupancy 输出接口。
- `--use-directional-decay-retention 1`：在 `HISTORY_FRAMES=0` 下用 Directional-Decay Selective Retention 替换 TSA。模块以四个带可学习指数衰减的方向性 depthwise 核建模长垄行依赖，并以短/长感受野分支和局部方差门控保留株间细节；`--ddsr-retention-radius`、`--ddsr-local-dilation` 控制长程尺度。前视训练脚本用 `RUN_DIRECTIONAL_DECAY_RETENTION=1` 启动该任务。
- GVAD、DDSR 与“移除 TSA”开关彼此互斥，且不能与 `USE_NEARFAR_BEV=1` 联用；历史帧大于 0 时仍使用原始 TSA。
- `USE_GAP_RESIDUAL_REFINER`：`1` 时启用端到端 Gap-Aware BEV Residual Refiner。最终 decoder 的完整 logits、压缩后的 `ref_bev`、crop/free 歧义和熵进入轻量各向异性 depthwise Conv3D；网络预测边界门控和零初始化残差，得到 `L_refined=L_coarse+M×ΔL`。除低权重的 coarse 深监督外，训练同时使用边界门控、近 crop free 的 `loss_gap_refiner_free` 和近 free crop 的对称 `loss_gap_refiner_crop`，以免仅保留空隙时侵蚀小作物。默认权重依次为 coarse `0.15`、boundary `0.25`、free gap `0.5`、crop preservation `0.5`；`GAP_REFINER_*` 可覆盖。
- `tools/train_farmsim_gapref_diagnostics.sh`：新版图像 GapRef 的二阶段消融脚本。三组实验都从同一 H=0/current-occupancy base checkpoint 开始，冻结 R101、FPN、BEV encoder、world decoder 与 coarse occupancy head，只训练 GapRef 和其图像证据分支：A 所有 GapRef 附加损失均为 0；B 使用 boundary=`0.10`、free-gap=`0.05`；C 在 B 的基础上增加 crop=`0.05`。每个 epoch 都保留训练框架的常规验证，无须先重复评估 base checkpoint。
- `GAP_REFINER_USE_IMAGE_FEATURES`：`1` 时，新版 GapRef 会按 crop/free 歧义和 crop 覆盖率选取 `GAP_REFINER_IMAGE_ACTIVE_RATIO` 的当前 BEV cell，把其 25 个高度查询利用已有 `lidar2img` 标定投影到当前相机的最高 `GAP_REFINER_IMAGE_LEVELS` 个 FPN 层。可见视角由 BEV-query 条件的 attention 融合，并以零初始化残差注入原 3D GapRef；因此初始时与旧 GapRef 完全一致。`GAP_REFINER_IMAGE_CROP_RATIO` 控制所选 cell 中 crop 区域的比例，余下为边界歧义区域，输出仍是 `100×100×25×6`。
- `USE_EFFICIENT_BASELINE`：`1` 时启用 Efficient baseline；默认 `0`，保留原版 R101。该开关将 backbone 切为 R50，FPN 从 4 个输出改为高分辨率 C3/C4 两个输出，将 BEV encoder 从 6 层改为 4 层，并把 `MSDeformableAttention3D.num_points` 从 8 改为 4；对应的 transformer `num_feature_levels` 与 deformable attention `num_levels` 都同步为 2。BEV 网格、occupancy decoder、损失和数据划分保持不变。Efficient 模式会强制启用项目现有的 AMP FP16；当前 MMCV 1.x 训练入口未配置 BF16 optimizer hook，因此这里的实际精度为 FP16，而非 BF16。
- `TOTAL_BATCH_SIZE`：目标有效全局 batch，包含所有 GPU 和梯度累积。`tools/train.py` 自动计算 `GRAD_ACCUM_STEPS = TOTAL_BATCH_SIZE / (BATCH_SIZE × NUM_GPUS)`；必须整除，否则会在启动前报错，避免实际 batch 与设定不一致。例如两卡、每卡 2、总 batch 为 8 时自动累积 2 步；总 batch 为 4 时自动累积 1 步。训练脚本不再定义 `GRAD_ACCUM_STEPS`。
- `HISTORY_FRAMES`：历史图像帧数，当前帧会额外输入，因此实际时序长度为 `HISTORY_FRAMES + 1`。
- `PREDICT_FUTURE_OCC`、`FUTURE_OCC_STEPS`：是否训练/评估未来占用，以及未来占用步数。开关为 `0` 时步数自动按 0 处理；开关为 `1` 时步数必须至少为 1。
- `PREDICT_FUTURE_TRAJ`、`FUTURE_TRAJ_STEPS`：是否启用未来轨迹头，以及轨迹预测步数。开关为 `0` 时步数忽略；开关为 `1` 时步数必须至少为 1。FarmSim 轨迹由 UE5 位姿计算得到；数据没有 3D 目标框，因此轨迹回归可用，碰撞框损失不提供监督。两个预测步数可以不同；占用预测需要更长轨迹时，超出轨迹标签范围的条件位移按 0 补齐。

训练脚本只保留上述参数赋值和一条固定的 `torchrun tools/train.py` 启动命令；不再在 shell 中进行条件判断、数组拼装或配置覆盖。`tools/train.py` 负责校验数据根目录、数值范围、`NUM_GPUS` 与 `CUDA_VISIBLE_DEVICES` 一致性、checkpoint 路径，并将参数同步写入数据集、模型和优化器配置。每次运行默认写入带时间戳的新 `work_dirs` 子目录，不覆盖之前的实验；若 conda 位于别处，直接编辑脚本中的 `PYTHON_BIN`。单卡也统一使用 `torchrun --nproc_per_node=1`，多卡时使用相同入口。

启用/关闭 FP16 或改变 `TOTAL_BATCH_SIZE`（从而改变自动推导的梯度累积）会改变优化器状态和有效 batch；建议新建 `WORK_DIR` 进行对照，不要直接续接原 FP32 实验的 checkpoint。

每个 epoch 的验证会在 `.log` 和 `.log.json` 中写入紧凑的 occupancy 指标，而不再
写入完整混淆矩阵。字段包括 `occ_current_mIoU`、`occ_future_mIoU`、
`occ_all_mIoU`、各自的 `*_mIoU_all`、`*_voxel_acc`，以及逐类
`*_IoU_free`、`*_IoU_crop` 等。`mIoU` 仅平均验证集中 union 非零的类别，
`mIoU_all` 将所有 6 个聚合 FarmSim 类别均纳入平均（不存在的类别 IoU 记为 0）。
当 `PREDICT_FUTURE_OCC=0` 时，`occ_future_available=0`，不会伪造未来 mIoU；
开启未来占用后则会记录当前、未来、合并和时间加权未来四组指标。

### 48 样本端到端 Smoke Test

`tools/create_farmsim_smoke_split.py` 会从原始 train/val 划分各取一个
独立 sequence 的连续帧，生成 `48` 个训练样本和 `16` 个验证样本；训练和
验证 sequence 仍彼此独立。运行：

```bash
bash tools/train_farmsim_front3_smoke48.sh
```

该脚本默认使用 CUDA GPU 2、3、每卡 batch size 1、FP16、1 epoch 和 2 个
DataLoader worker，并在训练后执行完整分布式验证。它适用于修改 dataset、
训练入口或评估收集逻辑后的回归检查，不应用于正式指标。`train.py` 同时支持
`--train-ann-file`、`--val-ann-file` 和 `--workers-per-gpu`，以便其他小规模
调试划分复用同一训练入口。

主要实现位于 `FarmSimWorldDataset`：它直接读取 split JSON、RGB、元数据和占用二进制文件。相机帧支持 `.jpg`、`.jpeg` 和 `.png` 扩展名，划分脚本也会统一识别这三种格式。模型改动还包括：动态支持 3/6 路相机注意力、去除轨迹头开关，以及根据真实目标尺寸计算占用损失，不再固定为 nuScenes 的 256×256×20 体素。

### Efficient baseline

在 `tools/train_farmsim_front3.sh` 中设置：

```bash
USE_EFFICIENT_BASELINE=1
```

该开关等价于如下结构变化，且不改变输入分辨率、BEV 分辨率、occupancy head、
训练划分或损失函数：

| 模块 | 原版 | Efficient baseline |
| --- | ---: | ---: |
| Backbone | R101 | R50 |
| FPN 输出层 | 4 | 2（C3/C4） |
| BEV encoder | 6 layers | 4 layers |
| `MSDeformableAttention3D.num_points` | 8 | 4 |
| AMP 精度 | 按 `USE_FP16` | FP16（强制） |

以 `farmsim_occ_front3.py`、使用聚合后 6 类 taxonomy 时实际构建的可训练参数计：
原版 R101 为 `53,500,101`，Efficient R50 为 `30,615,877`，减少 `22,884,224` 个参数
（`42.77%`）。FP16 不改变参数量；参数减少比例也不等同于速度或显存减少比例，
因为 attention 的激活和采样计算占比更高。应在相同 batch、图像分辨率、GPU 与
数据划分下报告吞吐、峰值显存和 mIoU。

### 单独评估 checkpoint

单卡评估已有 checkpoint 时，不需要重新训练。以前视三相机为例：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/HL/.conda/envs/dow2/bin/python tools/test.py \
  projects/configs/farmsim/farmsim_occ_front3.py \
  work_dirs/farmsim_occ_front3/epoch_1.pth \
  --launcher none --eval occ --deterministic --batch-size 2 \
  --cfg-options data.test.data_root=/mnt/g/SimData
```

六相机 checkpoint 将配置替换为 `projects/configs/farmsim/farmsim_occ_6cam.py`，并将 checkpoint 路径替换为对应文件。评估脚本会直接输出混淆矩阵、各类别 IoU、仅统计出现类别的 mIoU、全类别 mIoU 和 voxel accuracy；不会启动训练流程。

如果只想快速验证 checkpoint 和验证流程是否正常，可限制验证样本数，例如只跑 2 个样本：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/HL/.conda/envs/dow2/bin/python tools/test.py \
  projects/configs/farmsim/farmsim_occ_front3.py \
  work_dirs/front3_YYYYMMDD_HHMMSS/epoch_1.pth \
  --launcher none --eval occ --deterministic \
  --cfg-options data.test.data_root=/mnt/g/SimData \
                data.test.max_samples=2
```

这会执行完整的模型 `forward_test`、占用统计和自定义单卡评估路径，但不会遍历全部 6,393 个验证样本。若 checkpoint 使用了未来占用或轨迹开关，还需把训练时的 `model.test_future_frame_num`、`model.turn_on_plan`、`model.predict_trajectory` 等配置通过 `--cfg-options` 原样补上。

训练过程中的每 epoch 验证会在 `Epoch(val)` 日志中先输出各类 mIoU 和 voxel accuracy，再输出原始混淆矩阵；这些内容同时写入对应的 `.log` 和 `.log.json` 文件。

### 保存推理结果并在浏览器中逐条预览

`tools/test.py` 的测试 batch 默认每卡为 1（配置没有设置 `data.test.samples_per_gpu` 时）；可通过 `--batch-size N` 覆盖。保存选项会对每个样本输出压缩 `.npz`：包含当前占用预测与标签；模型实际输出未来占用时也保存 `future_pred/future_gt`，输出轨迹时也保存预测/GT 轨迹。

下面的示例按 `front3_base_nohis_ep8_20260828_043722/epoch_8.pth`
训练时的无历史帧、`512×288` 输入和每卡 6 个样本设置评估，并从每个验证序列等间隔保存，总计 1000 个结果：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH="$PWD" \
  /home/HL/.conda/envs/dow2/bin/python tools/test.py \
  projects/configs/farmsim/farmsim_occ_front3.py \
  work_dirs/front3_base_nohis_ep8_20260828_043722/epoch_8.pth \
  --launcher none --eval occ --deterministic --batch-size 6 \
  --save-predictions work_dirs/front3_base_nohis_ep8_20260828_043722/predictions \
  --save-prediction-count 100 \
  --save-prediction-sampling per-sequence \
  --cfg-options \
    data.test.data_root=/data/HL/SimData-Occ/SimData \
    data.test.queue_length=0 \
    data.test.image_size='[512,288]' \
    model.future_pred_head.history_queue_length=0
```

`--save-prediction-sampling per-sequence` 会在每个验证序列中等间隔抽取，序列间保存数量最多相差 1，总数为 `--save-prediction-count` 指定值；默认 `leading` 保持旧行为，即保存验证集前 N 个样本。保存全部结果时传入 `--save-prediction-count -1`。多卡评估同样支持这两种策略；每个输出文件以稳定的全局样本索引命名，因此不会因进程分片而重复或超过指定数量。请使用空目录（示例中的 `predictions_per_sequence_1000`），以免与旧的保存结果混合。

启动本地预览服务：

```bash
/home/HL/.conda/envs/dow2/bin/python \
  tools/visualize_farmsim_predictions.py \
  work_dirs/front3_base_nohis_ep8_20260828_043722/predictions
```

命令会先预加载全部 `.npz` 并显示进度，随后输出类似 `http://127.0.0.1:8000/` 的地址；在浏览器打开它后，可通过样本下拉框或 `Previous`/`Next` 逐条查看保存的结果。浏览器也会在后台预取全部样本，因此预加载完成后切换样本无需再等待 NPZ 读取或体素转换；代价是启动会变慢且服务端、浏览器会占用更多内存。页面左侧固定显示标签（GT），右侧固定显示预测结果；预测会使用同一帧 GT 的有效区域（非 ignore）掩码，避免未标注区域占用绘制点数，状态栏会显示两侧有效非 free 体素数。底部图例标明 6 类 FarmSim taxonomy 的颜色，其中 crop 为黄绿色、drivable 为天蓝色。存在未来占用时可切换未来时间步，存在轨迹时会自动叠加 GT 与预测轨迹。为控制浏览器性能，每侧默认最多显示 30,000 个有效非 free 体素；可通过 `--max-points 50000` 调大。需要自动打开浏览器时添加 `--open-browser`；默认仅本机可访问，需让局域网其他机器访问时可添加 `--host 0.0.0.0`。按 `Ctrl-C` 停止服务。页面仍使用 Plotly CDN，浏览器首次打开时需要可访问该 CDN。
