# FarmSim 占用地图的作物行提取

从 `tools/test.py --save-predictions` 保存的完整 NPZ 中提取作物行和局部行间参考路径。脚本只需要 NumPy；可直接使用项目的 `dow2` 环境，无须加载训练模型或 GPU。生成 PNG 预览时还需要 Matplotlib。

## 生成与查看

在项目根目录执行，参数可以是训练任务目录，也可以是该任务的 `prediction` 目录：

```bash
/home/HL/.conda/envs/dow2/bin/python row_det/generate_row_predictions.py \
  work_dirs/front3_s_base_nohis_ep15_20261009_131823 \
  --preview-count 6

/home/HL/.conda/envs/dow2/bin/python tools/visualize_farmsim_predictions.py \
  work_dirs/front3_s_base_nohis_ep15_20261009_131823/prediction
```

输出到 `work_dirs/<训练任务>/row_prediction/`：

- `<占用文件同名>.json`：当前帧及可选未来帧的预测行线、GT 行线和各自的导航参考线。
- `summary.json`：本次处理文件列表、行数量、路径生成状态及逐帧比较指标。
- `previews/*.png`：`--preview-count N` 指定的前 N 个样本的 BEV 对比图。

服务自动查找与 `prediction` 同级的 `row_prediction`。网页勾选 **Crop row lines** 显示粉色作物行线，勾选 **Navigation reference** 显示蓝色参考线；两项默认关闭。左侧叠加从语义 GT 提取的线，右侧叠加从模型预测提取的线。状态栏说明当前帧提取了几条行线，以及为什么没有参考路径。切换显示时保持相机视角。

查看器将行线和参考线抬升到各面板可见作物冠层上方 0.2 m，并加粗、添加白色描边，避免被作物占用点遮挡。冠层高度从点数采样之前的完整可见占用计算。抬升仅用于显示，JSON 中的原始坐标保持不变；两侧坐标范围一致，默认视角适度缩小以显示完整地图。

如果占用文件已被重新生成，网页会提示行线结果过期。重新执行生成脚本并重启服务即可；服务启动时将所有结果加载到内存。

其他预测目录和输出位置可以显式指定：

```bash
python row_det/generate_row_predictions.py work_dirs/<训练任务> \
  --prediction-dir work_dirs/<训练任务>/predictions_per_sequence_1000 \
  --output-dir work_dirs/<训练任务>/row_prediction

python tools/visualize_farmsim_predictions.py \
  work_dirs/<训练任务>/predictions_per_sequence_1000 \
  --row-prediction-dir work_dirs/<训练任务>/row_prediction
```

## 提取方法与参数

输入采用六类 FarmSim taxonomy 和 `[x,y,z]` 体素顺序，输出坐标单位为米，x 向前、y 向右、z 向上。根据每个文件的 `point_cloud_range` 和数组形状计算体素尺寸，不将尺寸固定为 `75×75×25`。

1. 将指定高度范围内的 `crop` 体素投影为 XY 密度，使用密度平方根作为权重。
2. 搜索作物行方向，以平滑后的横向密度集中程度评分。默认加入前向先验，评分乘以 `1 - 0.5 × sin²(方向角)`，降低规则株距产生的对角线歧义。
3. 从横向密度峰分配作物点，在沿行分段内求加权中心，使用确定性 RANSAC 和加权最小二乘拟合直线或二次曲线。
4. 从地面 `soil_ground/drivable` 估计线的绘制高度。行线只绘制在有拟合支持的范围内，不外推整幅地图。
5. 选择近端切线投影到车辆原点后、能包围车辆的相邻两行，在两行共同支持范围内取中线。检查估计行宽、地面支持和栅格障碍物间距；障碍处截断连续路径，不跨过障碍接续。

常用参数：

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `--z-min / --z-max` | −2 / 3 m | 作物提取的绝对 z 范围，车辆坐标系；不是离地高度 |
| `--max-angle-deg` | 60° | 相对车辆前向的方向搜索范围 |
| `--heading-prior-strength` | 0.5 | 前向先验强度；0 关闭，1 最强 |
| `--min-row-spacing` | 0.45 m | 允许的横向密度峰最小间距 |
| `--row-band` | 0.35 m | 密度峰周围的作物点分配半宽 |
| `--min-row-length` | 2 m | 行线最小沿行支持长度 |
| `--curve-degree` | 1 | 1 拟合直线，2 拟合二次曲线 |
| `--min-path-length` | 2 m | 有效连续参考路径的最小前向长度 |
| `--vehicle-width` | 0 m | 0 表示未配置车宽，仅输出几何参考；正值启用车宽检查 |
| `--vehicle-height` | 1.5 m | 未知体素检查的离地高度范围 |
| `--clearance-margin` | 0.1 m | 每侧净空余量；另计半个体素对角线的离散误差 |
| `--count` | −1 | 全部样本；正值仅处理排序后的前 N 个 |
| `--overwrite` | 关闭 | 强制重算；输入、算法版本或参数变化时自动重算 |

例如配置 0.6 m 车宽、每侧 0.15 m 余量，并拟合缓弯行：

```bash
python row_det/generate_row_predictions.py work_dirs/<训练任务> \
  --vehicle-width 0.6 --clearance-margin 0.15 --curve-degree 2
```

行距与作物冠幅不同的田块需要调整 `min-row-spacing`、`row-band` 和作物高度范围。规则点阵的多个方向可能同时具有几何支持，当前先验假设车辆大致沿行；侧向观察、冠层粘连和极少作物点可能导致错误或缺失的行线。

## 结果格式与限制

每个 JSON 的 `current.pred` 和 `current.gt` 均包含：

- `rows`：每条行的 `id`、三维 `points`、旋转坐标系的多项式系数、支持数量、拟合 RMSE、内点比例和估计冠幅半宽。
- `reference_path`：三维 `points`、所依赖的两个 `row_ids`、状态、车宽检查标记和最小净空。
- `diagnostics`：作物体素/BEV cell 数量、方向和提取状态。

`future` 是相同结构的未来帧数组；没有未来占用时为空。系数按升幂排列，`u=x cosθ+y sinθ`、`v=−x sinθ+y cosθ`，行模型为 `v=c0+c1*u(+c2*u²)`。

参考路径状态：

| 状态 | 含义 |
| --- | --- |
| `geometry_only` | 有几何参考，尚未配置车宽；不是可执行驾驶轨迹 |
| `clearance_checked` | 已按配置车宽检查局部栅格净空 |
| `insufficient_crop` | 作物 BEV 点不足 |
| `insufficient_rows` | 有效作物行少于两条 |
| `no_bracketing_rows` | 没有能包围车辆且具有足够共同长度的相邻行 |
| `blocked_or_unknown_corridor` | 通道受阻、过窄或缺乏地面/空间标注支持 |

预测分支只读取模型预测，不依赖 GT 有效掩码或 GT 行线；查看器仍保留原有占用点的 GT 有效区域显示约定。GT 地下的 ignore 不作为障碍，地表以上车辆高度范围内的 ignore 和缺失地面会阻止参考路径。作物、其他植被和障碍物采用全高度投影，因此冠层遮盖通道时检查较保守。

`comparison` 通过共同 x 支持范围上的横向距离和航向差做一对一贪心匹配（横向 ≤0.4 m、航向 ≤10°、重叠 ≥1 m）。它比较的是**同一算法从预测/语义 GT 提取的行**，不能替代真实种植中心线标注或闭环驾驶评价。

本版为逐帧离线提取，没有里程计对齐、跨帧行 ID 跟踪、行末换行或车辆控制。`clearance_checked` 只表示当前语义栅格和配置宽度下的检查，未考虑车辆转弯扫掠、动力学和地面坡度。当前 FarmSim 模型不监督人员/动物，需要额外感知才能覆盖这些障碍。

验证命令：

```bash
/home/HL/.conda/envs/dow2/bin/python -m unittest row_det.test_row_detection -v
```

## 使用世界坐标作物行真值评估

对于带有 `evaluation_ground_truth.grid_frame_world_ue` 和 `crop_rows_file` 的新采集序列，可以用独立几何真值评估标签占用和模型预测占用的行拟合质量。脚本仍放在 `row_det/`，该次新数据评估结果放到训练任务下的独立目录，避免与旧验证集混合。

已执行的示例：

```bash
TASK_DIR=work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738
EVAL_DIR="$TASK_DIR/croprow_evaluation_20261020"
PYTHON_BIN=/home/HL/.conda/envs/dow2/bin/python

"$PYTHON_BIN" row_det/prepare_croprow_evaluation.py \
  /data/HL/SimData-croprow "$EVAL_DIR"

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. "$PYTHON_BIN" tools/test.py \
  "$TASK_DIR/farmsim_occ_front3_s.py" "$TASK_DIR/epoch_15.pth" \
  --out "$EVAL_DIR/occupancy_metrics.pkl" \
  --save-predictions "$EVAL_DIR/prediction" --save-prediction-count -1 \
  --cfg-options data.test.ann_file="$EVAL_DIR/evaluation_manifest.json" \
  data.test.data_root=/data/HL/SimData-croprow \
  data.test.samples_per_gpu=2 data.workers_per_gpu=2

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 "$PYTHON_BIN" row_det/evaluate_croprow.py \
  "$EVAL_DIR" --data-root /data/HL/SimData-croprow \
  --checkpoint "$TASK_DIR/epoch_15.pth" --workers 4
```

评估协议：

- 用该帧 `grid_frame_world_ue` 的平移（cm 转 m）和逆 yaw 将世界 XY 行段变换到占用网格 XY；不使用车辆中心位姿代替网格位姿。若网格非水平而只有世界 XY 标注，直接报错，避免假造世界 Z。
- 过滤 `fit_valid=false`，将几何真值和拟合结果都裁剪到模型评价范围。几何真值不按拟合器的最小长度过滤，不根据模型预测或 GT 可见掩码删除真值行。
- 在裁剪后的**完整参考行段**上等距采样（包含两端点，间距不超过 0.05 m），计算采样点到估计有限行段/折线的欧氏距离。禁止仅截取两行重叠部分比较，也不外推拟合线端点。
- 至少 85% 的采样点距离严格小于阈值，才允许匹配；用最大匹配数的一对一二分图匹配统计 TP、FP、FN。每帧没有 GT 时仍保留其 FP；全空帧的单帧 F1 为 null。
- 在全部帧汇总 TP、FP、FN 后计算 micro F1，报告 F1@0.10/0.20/0.30 m。该数据集上不事后选择主阈值。可视化仅用 0.20 m 指示匹配颜色，不代表已选定主阈值。
- 标签拟合和预测拟合使用完全相同的固定 `RowConfig`，模型拟合不读取几何真值、GT 掩码或行方向。可用 `--row-config` 显式指定事先固定的配置 JSON；重新评估应使用单独目录保存不同配置的结果。

指标借鉴 [RowDetr 的覆盖率匹配思想](https://arxiv.org/html/2412.10525v2#S3.SS1)，距离改为米制网格 XY 到有限行段的距离；数值不能与其图像坐标结果直接比较。

输出包括：

- `dataset_audit.json`、`evaluation_manifest.json`、`frame_index.json`：数据完整性、完整评估集和全局样本索引。
- `prediction/`：全部帧占用 NPZ；`row_prediction/`：每帧几何真值、标签拟合、模型拟合、三阈值匹配明细和坐标变换信息。可直接将 `prediction/` 传给已有查看器，叠加标签/预测拟合线。
- `row_metrics.json`、`per_frame_metrics.csv`、`grouped_metrics.csv`：总体、作物、生长阶段、序列级结果。主要比较为 `label_fit_vs_geometry`、`model_fit_vs_geometry`；`model_fit_vs_label_fit` 仅是附加重建诊断。
- `visualization/`：100 张三列 BEV 对比图（几何真值 / 标签占用拟合 / 模型占用拟合）。沿用占用预览的按序列均衡、序列内居中连续窗口规则，以及 `<全局样本索引>_<scenario>_<frame>_prediction.png` 文件命名；选择索引固定保存到 `visualization_indices.txt`。
- `REPORT.md`、`run_config.json`、`inference.log`、`occupancy_metrics.pkl`：结果说明、固定拟合参数、协议、checkpoint SHA256 和原模型推理日志。推理保留原模型的相机适配方式，不在评估中调整标定。

指标测试：

```bash
python -m unittest row_det.test_geometric_evaluation row_det.test_row_detection -v
```

### 独立复核匹配并定位低分来源

`audit_croprow_matching.py` 使用 GEOS/Shapely 独立裁剪和计算有限行段距离，
用 SciPy 最大二分图匹配复核每帧三个阈值的 TP/FP/FN、已保存匹配的覆盖率、
平均距离和采样数量；另用 SciPy 逆旋转复核世界行段到网格的变换。
它只读取 `row_prediction`，不修改占用、拟合线或正式指标。
审计依赖 NumPy、SciPy、Shapely >= 2；当前系统 `python` 可运行，`dow2` 不包含后两者。

```bash
TASK_DIR=work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python row_det/audit_croprow_matching.py \
  --evaluation-dir "$TASK_DIR/croprow_evaluation_20261020" \
  --evaluation-dir "$TASK_DIR/croprow_evaluation_20261020_grid_fixed" \
  --output-dir "$TASK_DIR/croprow_matching_audit_20261020" --workers 8
```

输出 `matching_audit.json`、`REPORT.md` 和逐真值行的 `reference_diagnostics_0p20m.csv`。
CSV 包含完整真值覆盖率、达到 85% 覆盖所需的距离次序统计量 `d85`、最佳候选线方向差和中点横向误差。
这些最佳候选由几何距离选出，不代表已确认物理行身份；没有候选的真值不进入误差分位数统计。
诊断把未匹配真值分为：无预测行、有合格候选但一对一竞争、仅无限延长后存在合格候选、位置/方向偏差或缺行。
无限延长只用于区分端点支持不足，不作为正式分数或导航路径。正式评估仍覆盖完整有限真值行段，保持 85% 门限。

注意前文旧 `comparison` 字段的贪心重建诊断，与几何真值评估是两套独立用途；
`evaluate_croprow.py` 的正式 F1 调用 `geometric_evaluation.py` 的最大匹配。

### 修复地面网格与相机投影不一致

旧适配器把相机 `relative_transform_ue` 直接当成相对于占用网格的外参。但采集数据的占用网格是地面原点、只含 yaw 的坐标系，相机挂点还可能与车辆中心存在根组件偏移。这会让倾斜、起伏地形上的图像特征采样偏移；原训练数据也存在此问题。

`FarmSimWorldDataset(camera_geometry='grid_world')` 使用逐帧 `semantic_occupancy.grid_frame_world_ue` 和各相机的 `world_transform_ue`：

```text
optical_from_grid = UE_to_optical @ inverse(world_from_camera) @ world_from_grid
lidar2img = K @ optical_from_grid
```

世界姿态采用 [UE FRotator 约定](https://dev.epicgames.com/documentation/en-us/unreal-engine/API/Runtime/Core/TRotator)，即列向量形式 `Rz(yaw) @ Ry(-pitch) @ Rx(-roll)`。图像缩放继续同步缩放投影矩阵。该路径不读取作物行真值或占用类别；时序运动也改用同一个网格位姿。缺失必需世界位姿时直接报错。

默认仍为 `legacy`，以便复现旧 checkpoint；旧权重是在旧采样方式下学习的，直接换外参不能保证各项指标改善。新训练应在 `data.train/val/test` 三处统一设置 `camera_geometry='grid_world'`。

本次修复另外进行固定两轮的校准微调：每个原训练序列均匀抽取 8 帧，共 1,984 帧，原验证序列各取 4 帧，共 200 帧；冻结图像 backbone，学习率 `5e-5`，有效 batch 12，seed 0。按原验证集作物体素 IoU 选择 checkpoint，不用 SimData-croprow 的行线分数选择模型。原 epoch_15.pth 保留，行线拟合配置、XY 范围和完整真值覆盖指标保持一致。

可复现整个流程（默认 GPU 0；日志写入各输出目录）：

```bash
GPU_ID=0 bash row_det/repair_croprow_calibration.sh \
  work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738 \
  /data/HL/SimData-croprow
```

微调产物保存在 `<task>/grid_calibration_finetune_20261020/`，新评估保存在 `<task>/croprow_evaluation_20261020_grid_fixed/`。其中 `prediction/`、`row_prediction/` 和 `visualization/` 与原评估分开；100 张图仍按原序列均衡、居中连续窗口规则生成。`selected_checkpoint.json` 记录两轮原验证集结果和选择依据。

本次全量复测（1,540 帧）的行线 F1@0.20m 为 `0.1573 → 0.1735`，F1@0.30m 为 `0.2851 → 0.3075`；F1@0.10m 为 `0.0441 → 0.0421`，作物体素 IoU 为 `0.2003 → 0.1850`。这是有限的校准适配收益，不能作为整体优于原模型的结论；原模型未替换。详细对照和限制见新评估目录的 `REPAIR_REPORT.md`。后续训练应保持标定一致，并在独立验证集上验证足够训练后的收益。
