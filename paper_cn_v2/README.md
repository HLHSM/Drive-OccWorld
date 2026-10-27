# AgriOcc 中文论文：小范围版本

入口为 `main.tex`，编译产物为 `main.pdf`。原 `paper_cn` 完全保留；本目录独立使用 Elsevier CAS 单栏模板和 XeLaTeX。

## 本版研究对象

- 前向 ROI：`[0,-7.5,-2,15,7.5,3] m`，坐标依次为前、右、上。
- 体素：`0.2 × 0.2 × 0.2 m`，输出 `75 × 75 × 25`，六类且忽略标签 255。
- 模型：R101/FPN + 保留 TSA 的 BEVFormer + Agri-AMoE（96 通道）。当前帧三相机，无历史与未来预测。
- CropSample：替换配额 0.10、当前 ROI 内作物体素阈值 500，不增加 epoch 长度。
- 训练 15 epoch，主检查点：
  `work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth`。
- 主结果：mIoU 64.079%、Crop IoU 39.382%。grid-world 标定的另一组实验与本主模型分开呈现，未跨检查点拼接结果。

## 实测结果和临时数值的界限

`data/results.json`、`data/results.csv` 记录日志来源、最终 epoch、各类 IoU；表格取最后一轮，不是每类最优值拼接。

**Agri-AMoE 内部消融表的七个配置、共十四个蓝色 `†` 数值为临时示意。** 完整配置沿用已有实测结果，示意数值需在实验完成后替换。其余定量表均来自现有运行。

主要协议修正：

1. 新版主模型不含 GVAD、NearFar 或旧 ADHR；不可沿用旧版模块消融与效率结论。
2. Base 表示现有世界模型骨架的当前帧适配，未冒充独立重训的 Drive-OccWorld 对比结果。
3. COTR 日志的评估钩子评估 EMA，因此主表和图库统一采用最终 `epoch_15_ema.pth`。普通权重的探索性导出仅存档于 `artifacts/cotr_normal_unreported`，不进入论文。
4. SparseOcc 使用几何适配修复后的 `farmsim_front3_s_geomfix_ep15`，不使用先前 16.8% 结果。
5. ORAD-3D 的 Test 与 Farm 均固定九类分母；不混用两个模型类别集合不同的 present-class 平均。新版迁移是 2 轮对比 6 轮，不支持“同预算精度提高”；时间只算目标域训练与逐轮验证。
6. 分辨率迁移比较三组小范围 15 轮实验：高分辨率从头训练、冻结查询适配、冻结细化头适配。输出均为 `150 × 150 × 25`。
7. 所有参数表注明可训练参数。它与网络总参数及推理实际执行分支参数不是同一口径，不能据此推断延迟。

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

仅编译，不调用 GPU：

```bash
bash paper_cn_v2/build.sh
```

编译默认使用当前 WSL `PATH` 中的 TeX Live（需要 `latexmk`、`xelatex` 和 `bibtex`）；如需指定其他安装，可设置 `TEXLIVE_BIN`。VS Code LaTeX Workshop 的默认配方同样使用 WSL 中的 `latexmk`。

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
