# AgriOcc v2 写作和图表计划

## 核心论点

农业前向三维感知可由当前帧三视角 BEV、各向异性专家解码与温和作物样本曝光构建；结构解码和采样对 Crop IoU 的作用需分开解释，细化头可复用粗分辨率先验。

## 论点—证据

| 论点 | 证据 | 适用界限 |
|---|---|---|
| 前向农业占用统一几何语义 | 298 序列、38,862 帧、10 种作物 | UE5 仿真，非闭环导航验证 |
| Agri-AMoE 改善基础网络 | Base / Base+AMoE 的最终 epoch | 固定种子，差异不作显著性结论 |
| CropSample 改变作物曝光 | AMoE 无采样及 0.10/0.15/0.20 配额 | Crop 上升不等价于整体 mIoU 上升 |
| 受控农业协议下的方法比较 | SurroundOcc / COTR EMA / IR-WM / 几何修复 SparseOcc | 输入和 ROI 适配，非城市原生基准排名 |
| 短预算真实域适配 | ORAD 测试与 Farm 固定九类指标 | 2 轮 vs 6 轮，精度稍低且时间更短 |
| 粗分辨率模型可迁移 | 三组 0.1m 目标分辨率训练 | 仅细化头优于从头训练，查询适配没有 |

## 章节

1. 引言：农业空间接口、数据与结构难点、方法概览及贡献。
2. 相关工作：BEV、占用/世界模型、农业越野与迁移。
3. 方法：任务、数据集、整体网络、当前 BEV、Agri-AMoE、CropSample、损失。
4. 实验：协议、对比、结构与采样消融、真实域、分辨率、边界局限。
5. 结论：农业表示与采样折中、粗细分辨率适配。

## 图表计划

| 文件 | 类型 | 来源 | 用途 |
|---|---|---|---|
| overview.pdf | 可编辑架构图 | 当前代码结构 | TSA + AMoE + 训练采样 |
| amoe.pdf | 可编辑模块图 | `_AgriAMoE3D` | 三专家、结构路由及残差 |
| dataset.png / synthetic.png / crop_distribution.pdf | 原稿数据图 | 固定数据采集与划分 | 不因 ROI 改变的数据说明 |
| qualitative_comparison.png | 3×5 真预测网格 | 同三帧 GT/Surr/COTR EMA/AgriOcc | 当前占用对比 |
| qualitative_sparse_irwm.png | 3×5 真预测网格 | 同三帧 GT/Sparse/IR-WM/AgriOcc | 补充方法比较 |
| training_curves.pdf | 数据曲线 | 原始逐 epoch 日志 | Base、AMoE 与采样折中 |
| orad_transfer.png | 真实域配对预测 | 新小范围源模型迁移权重 | 目标域两组道路结构 |
| realfield.png | 无 GT 定性图 | 实拍照片重建三视角、新主模型 | 定性纹理响应 |
| resolution_transfer.png | 4 列预测对比 | 高分辨率同三帧、三个新权重 | 分辨率迁移 |
| limitations.png | GT/预测配对 | 当前主检查点 | 作物边界与连接问题 |

内部机制消融尚无对应实测文件，蓝色 dagger 数值仅供占位，不承担上述任何论点。
