# AgriOcc 论文计划与证据矩阵

## 一句话贡献

AgriOcc 将三路前向相机图像直接编码为农业机器人坐标系下的当前帧三维语义占用，并通过几何可见锚点注意力、农业结构自适应混合专家三维占用解码器（Agri-AMoE）和近远场查询布局，在 FarmSim 上以 55.74M 参数获得 56.99% mIoU。

## 主张—证据

| 主张 | 当前证据 | 边界 |
| --- | --- | --- |
| 三前视相机足以支持前方 20m 的当前语义占用 | FarmSim 32,385/6,477 train/val 划分与 AgriOcc 最终验证结果 | 仅限仿真数据、当前帧任务 |
| GVAD、Agri-AMoE、NearFar 的完整组合提升基线 | TSA、GVAD、Agri-AMoE 单模块及完整组合的 FarmSim 运行 | 缺少仅 NearFar 和 GVAD+Agri-AMoE（无 NearFar）运行；单种子，尚无统计显著性 |
| FarmSim 预训练能形成可迁移初始化 | ORAD-3D 10/25/50/100% 微调和 farm_all 子集评估 | 仍需补零样本与严格同 epoch 对照 |

## 待完成实验

外部方法的 FarmSim 完整对照、ORAD-3D 零样本、所有方法的速度与峰值显存、跨作物/天气泛化和真实农业机器人实测。
