# UE5 农业机器人 3D 语义占用训练

本适配以 `/mnt/g/SimData` 的 FarmSim UE5 数据为输入，不再依赖 nuScenes、CAN bus 或轨迹标签。模型训练目标是当前时刻的 3D 语义占用；配置中的 `predict_trajectory=False` 会屏蔽规划/未来轨迹头，`future_pred_frame_num=0` 表示不进行未来占用预测。

## 已生成的数据划分

划分文件位于 `data/farmsim/splits/`：`train.json`、`val.json` 和 `split_report.json`。以完整 sequence 为最小单元，绝不将同一 sequence 的不同帧拆到训练和验证中。使用固定 seed `20260821`，按作物种类、生长阶段、时段、天气以及帧数进行约束分层：

| 划分 | sequence | 完整帧数 | 占比 |
| --- | ---: | ---: | ---: |
| 训练 | 263 | 33,193 | 80.01% |
| 验证 | 65 | 8,291 | 19.99% |

验证集包含全部 11 种作物，且生长阶段为 early/mature/mid = 22/21/22；时段和天气的比例也与全集接近。三个不完整 sequence 被自动排除，详见 `split_report.json`。

若 `/mnt/g/SimData` 有新增数据，重新生成可复现划分：

```bash
/home/hl/miniconda3/envs/py311/bin/python tools/create_farmsim_split.py \
  --data-root /mnt/g/SimData --output-dir data/farmsim/splits \
  --val-ratio 0.2 --seed 20260821
```

## 坐标、标签和相机

- FarmSim 标注体素存储格式为 `[z,y,x]`，适配器转换为模型使用的 `[x,y,z]`。
- 坐标为 `x` 前、`y` 右、`z` 上；体素大小 0.2 m，语义 ID 保持原始 0--11，`occupancy_valid=0` 写为 ignore ID 255。
- 六相机模式输入 `front_left/front/front_right/rear_left/rear/rear_right`，预测 `x∈[-20,20]m, y∈[-10,10]m, z∈[-2,3]m` 的 200×100×25 体素。
- 前视三相机模式输入前三路相机，仅监督/预测 `x∈[0,20]m` 前半区的 100×100×25 体素；后半区不参与网络输出和损失。
- 每个样本使用前两帧加当前帧的时序 BEV 输入。UE 相机的 forward/right/up 轴已转换为针孔相机的 right/down/forward 轴，并使用每帧 JSON 中的内外参。

## 启动训练

先按项目原有说明安装依赖与 CUDA 算子。然后直接执行：

```bash
bash tools/train_farmsim_6cam.sh
# 或
bash tools/train_farmsim_front3.sh
```

两个脚本顶部都有 `CUDA_VISIBLE_DEVICES` 与 `NUM_GPUS`，直接修改即可。例如 `CUDA_VISIBLE_DEVICES="2,3,5"` 时必须写 `NUM_GPUS=3`。脚本会在两者不一致时退出，避免启动错误。

主要实现位于 `FarmSimWorldDataset`：它直接读取 split JSON、RGB、元数据和占用二进制文件。模型改动还包括：动态支持 3/6 路相机注意力、去除轨迹头开关，以及根据真实目标尺寸计算占用损失，不再固定为 nuScenes 的 256×256×20 体素。
