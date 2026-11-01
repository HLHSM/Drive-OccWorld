# AgriOcc 中文论文：小范围版本

入口为 `main.tex`，编译产物为 `main.pdf`。原 `paper_cn` 完全保留；本目录独立使用 Elsevier CAS 单栏模板和 XeLaTeX。

## 本版研究对象

- 前向 ROI：`[0,-7.5,-2,15,7.5,3] m`，坐标依次为前、右、上。
- 体素：`0.2 × 0.2 × 0.2 m`，输出 `75 × 75 × 25`，六类且忽略标签 255。
- 模型：R101/FPN + 保留 TSA 的 BEVFormer + Agri-AMoE（96 通道）。当前帧三相机，无历史与未来预测。
- CropSample：替换配额 0.10、当前 ROI 内作物体素阈值 500，不增加 epoch 长度。
- 训练 15 epoch，主检查点：
  `work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth`。
- 主结果：mIoU 64.079%、Crop IoU 39.382%。作物行试验与主实验使用同一检查点。

## 实测结果和临时数值的界限

`data/results.json`、`data/results.csv` 记录日志来源、最终 epoch、各类 IoU；表格取最后一轮，不是每类最优值拼接。

**模型主要组件消融表中，无 TSA 的 Base 行（mIoU 62.98%、Crop IoU 36.54%）以及 Agri-AMoE 内部消融表前九个配置的十八个 IoU 均为暂填值，尚未实测。** 无 TSA 的 50.99 M 参数量由原 52.37 M 基线扣除六层 TSA 及对应 LayerNorm（共 1,383,552 个参数）估算，需由完整模型统计核实；完整 Agri-AMoE 沿用已有结果 64.20% / 38.93%。正文按当前表值撰写分析，不另加蓝色或 `†` 标记；暂填范围在本节记录，实际实验完成后须同步替换表值、增益和分析。

内部消融包括独立去除通道注意力、三个同构平面专家、分别去除三个专家、平均融合、全局固定融合，以及分别去除空间显著性和梯度能量。各配置均保留 TSA、不使用 CropSample，固定 96 通道和 15 个 epoch。模型已支持 `agri_amoe_use_channel_attention` 和 `agri_amoe_use_spatial_saliency` 两个独立配置；旧 `use_saliency` 联合开关仍兼容，但不能用来代表任一单项消融。

对应训练入口为 `tools/train_farmsim_front3_s.sh`：默认串行执行无 TSA Base、完整 Agri-AMoE 及九项内部消融（共 11 组），`FARMSIM_TASKS=all` 额外加入含 TSA 的 MLP 基线。所有任务使用相同图像预训练权重重新训练，不加载已有占用模型。用 `DRY_RUN=1` 仅预览命令；可通过 `FARMSIM_TASKS` 传入逗号分隔的任务名选择部分配置，具体名称及对应表格行见脚本顶部注释。

```bash
# 预览全部 12 组；匹配本文已有实验的投影方式。
CAMERA_GEOMETRY=legacy FARMSIM_TASKS=all DRY_RUN=1 bash tools/train_farmsim_front3_s.sh

# 训练无 TSA Base，以及通道增强和两种路由信号的独立消融。
CUDA_VISIBLE_DEVICES=1,2,3 CAMERA_GEOMETRY=legacy \
  FARMSIM_TASKS=base_no_tsa,amoe,amoe_no_channel,amoe_no_saliency,amoe_no_energy \
  bash tools/train_farmsim_front3_s.sh
```

脚本默认保留当前的 `grid_world` 投影设置；复现本文旧主模型时必须统一指定 `CAMERA_GEOMETRY=legacy`。两种投影的实验应分别比较，均需使用同投影的完整 Agri-AMoE 对照。默认有效 batch 固定为 24，修改 GPU 数量或单卡 batch 时通过梯度累积保持该值，要求 `TOTAL_BATCH_SIZE` 可被 `BATCH_SIZE * NUM_GPUS` 整除。

主要协议修正：

1. 新版主模型不含 GVAD、NearFar 或旧 ADHR；不可沿用旧版模块消融与效率结论。
2. Base 表示现有世界模型骨架的当前帧适配，未冒充独立重训的 Drive-OccWorld 对比结果。
3. COTR 日志的评估钩子评估 EMA，因此主表和图库统一采用最终 `epoch_15_ema.pth`。普通权重的探索性导出仅存档于 `artifacts/cotr_normal_unreported`，不进入论文。
4. SparseOcc 使用几何适配修复后的 `farmsim_front3_s_geomfix_ep15`，不使用先前 16.8% 结果。
5. ORAD-3D 的 Test 与 Farm 均固定九类分母；不混用两个模型类别集合不同的 present-class 平均。新版迁移是 2 轮对比 6 轮，不支持“同预算精度提高”；时间只算目标域训练与逐轮验证。
6. 分辨率迁移比较三组小范围 15 轮实验：高分辨率从头训练、冻结查询适配、冻结细化头适配。输出均为 `150 × 150 × 25`。
7. 所有参数表注明可训练参数。它与网络总参数及推理实际执行分支参数不是同一口径，不能据此推断延迟。

主性能表的前向耗时在单张 RTX 4090、batch size 1 下测量，每个模型预热 10 帧并统计 50 帧；各条记录的 `wall_mean_ms` 和具体配置、检查点见以下文件。计时包括 `MMDataParallel` 调用和 GPU 同步，不含数据加载、输出解码、GT 指标计算与结果保存；不同方法的计算精度和预测任务并不完全相同。

- AgriOcc：`profile_output/paper_v2_agriocc_gpu3.json`，91.85 ms，混合精度。
- SurroundOcc：`profile_output/paper_v2_surroundocc_gpu3.json`，83.73 ms，混合精度。
- SparseOcc：`profile_output/paper_v2_sparseocc_raw_gpu3.json`，28.88 ms，FP32。
- COTR：`profile_output/paper_v2_cotr_gpu3.json`，32.33 ms，FP32。
- IR-WM：`profile_output/paper_v2_irwm_no_gt_gpu3.json`，173.83 ms，混合精度，保留历史输入、未来占用和轨迹预测。
- Drive-OccWorld：未提供可用检查点，耗时留空。

旧的 AgriOcc 测试记录 `profile_output/farmsim_front3_s_amoe_crop01_batch1_gpu0_10.json` 使用不同 GPU 和较短采样，平均 63.44 ms，不与本次表格混用。文中 TensorRT FP16 的 6.82 ms 为作者提供的既往 RTX 4090 测试结果，尚无原始报告；模型版本、测试范围和软件环境待核验，不属于本次 PyTorch 测试。

## 图像来源及绘制

`artifacts` 保存本次批量为 1 的预测 NPZ、调用命令与日志。`data/figure_sources.json` 将每个定性面板关联至帧、配置、检查点、NPZ 和 GT 来源。
论文使用的 PNG、PDF 和 SVG 图像均直接保存在本目录；架构图的 FigureSpec 源文件保存在 `figures/specs/`。

- 定性占用均从对应小范围模型输出直接重绘，不是把旧大范围截图裁剪。
- 视角固定为 `elevation=55° / azimuth=180°`：居中的正后上方，沿前向 x 观察，不引入横向偏角。
- 机器人坐标为前 x、右 y、上 z 的左手系；显示时对应调整 y 轴方向，使正 y（机器人右方）位于图像右侧。
- 按真实体素中心定位，无坐标抖动、无点数截断；统一采用 `GT != 255` 有效掩码，各数据集均隐藏 Free。
- 圆形标记面积按 XY 网格数量缩放，避免高分辨率图使用过大点径。
- AgriOcc 的三帧直接复用既有标定审计中的 `legacy_prediction`。虽然该缓存所在父目录名称含 grid_world，其实际检查点是上文的 **20261012 主检查点**，详见 `artifacts/agriocc/provenance.json`，不是 grid-world 模型。
- COTR、SurroundOcc、SparseOcc、IR-WM、ORAD 和分辨率迁移图使用本次补导出的新权重预测，GT 在配对后逐元素核对。
- 真实农田四例来自实拍前视照片构造的重建/生成三视角，借用仿真标定，**不是同步真实多相机采集，无体素 GT**；只做定性展示。本次重新推理，无手工修改类别。
- 不随模型变化的数据集组成、外观和采集流程图保留自原稿；模型总图和专家图已按新模型重绘为低饱和度、可编辑 SVG/PDF。

## 重新生成

新增作物行拟合试验见 `main.tex` 的 `sec:crop_row_fitting`。原二维 BEV 图已退出正文，改为在三维占用网页中挑选作物行线、导航参考线示例后再排入论文。
当前正文采用蓝色行间几何中心线的横向 RMSE，以相邻种植行几何中线为参考，不再使用逐条作物行 F1 作为主指标。
标签与模型误差使用共同的 1,222 帧和完全相同的纵向采样点，不外推，不按距离或角度排除大误差；缺失中心线通过生成率单独报告。
作物行表格和三维网页预测均使用与主实验一致的 `epoch_15.pth`。
标签与模型横向 RMSE 为 0.401 m / 0.609 m，生成率按全部 1,540 帧计算，为 87.5% / 83.0%。
来源和逐作物统计见 `data/crop_centerline_rmse.json`。表格与数值段落直接写在 `main.tex` 中，评价脚本只更新相应注释标记内的内容，不生成外部表格 TEX。没有中心线的帧不计入 RMSE/MAE 的分子或分母。旧 F1 文件保留为历史诊断，不与中心线 RMSE 直接比较。
网页命令与含中心线的选图目录见 `../DOCS/FARMSIM_UE5_OCCUPANCY.md`。

重新计算和更新论文表格（CPU，不重新运行模型推理）：

```bash
python row_det/evaluate_centerline_rmse.py \
  work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/croprow_evaluation_20261020 \
  --paper-dir paper_cn_v2
```

仅编译，不调用 GPU：

```bash
bash paper_cn_v2/build.sh
```

命令行编译使用当前 `PATH` 中的 TeX Live（需要 `latexmk`、`xelatex` 和 `bibtex`）；可设置 `TEXLIVE_BIN=/data/HL/texlive/2026/bin/x86_64-linux`。VS Code LaTeX Workshop 的工作区配方已明确指定该 TeX Live 路径。

作物行三行三列可视化采用用户选定的胡萝卜 #000255、玉米 #000288、番茄 #001241，每行依次为完整正前方 RGB、标签占用拟合、预测占用拟合。复现命令：

```bash
python3 row_det/make_paper_row_3d_figure.py
```

需要 NumPy、Matplotlib、Pillow、Plotly、websocket-client 和本机 Chrome/Chromium。输出为 `figures/crop_row_3d_rgb.pdf` 和 `.png`；原始 RGB、占用、行线文件的路径与 SHA-256 记录在 `data/crop_row_3d_rgb.json`。六个占用面板沿用网页的有效区域掩码、语义配色和冠层上方行线显示，使用同一视角与坐标范围。`--reuse-panels` 可复用面板，仅重新拼图。

重新提取日志、绘制已有预测（CPU）：

```bash
/home/HL/.conda/envs/dow2/bin/python paper_cn_v2/scripts/collect_results.py
/home/HL/.conda/envs/dow2/bin/python paper_cn_v2/scripts/generate_figures.py
/home/HL/.conda/envs/dow2/bin/python paper_cn_v2/scripts/generate_diagrams.py
/home/HL/.conda/envs/dow2/bin/python paper_cn_v2/scripts/check_artifacts.py
bash paper_cn_v2/build.sh
```

绘图依赖 NumPy、Matplotlib、Pillow；矢量图转换额外依赖 CairoSVG。当前临时安装在 `/tmp/agriocc_v2_svg_dependencies`，环境中已有 CairoSVG 时无需该目录。FigureSpec 渲染器使用本项目安装的 ARIS 版本。

缺少预测时可用以下命令串行补导出（已有三帧缓存会跳过；需先确认相应 GPU 可共享）：

```bash
PAPER_GPU=2 /home/HL/.conda/envs/dow2/bin/python paper_cn_v2/scripts/export_predictions.py
```

导出器使用各对比仓库的既有 conda 环境和保存配置，不修改训练权重。`check_artifacts.py` 只读核对来源与图表，并将检查结果写入本目录 `data/artifact_check.json`。
