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
