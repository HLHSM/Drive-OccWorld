# 0.2 m 作物抽样实验分析（2026-10-08）

## 已完成实验

以下均为 front3、无历史帧、只预测当前占用、base + Agri-AMoE96、15 epoch，数值取各运行 JSON 日志最后一轮验证结果，单位为百分比。

| 策略 | mIoU | crop IoU | 相对原基线的 crop 变化（百分点） | free IoU | soil IoU | 有效 batch |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 原基线 | 60.754 | 32.222 | — | 92.306 | 78.081 | 24 |
| cropsample，ratio=0.1，min=1000 | 60.365 | 32.851 | +0.629 | 92.394 | 77.408 | 27 |
| hardsoil，weight=0.15 | 60.322 | 31.569 | -0.653 | 92.259 | 76.126 | 27 |
| focalfree，gamma=2 | 60.715 | 32.208 | -0.014 | 92.227 | 77.809 | 27 |

日志目录：

- `work_dirs/front3_base_agri_amoe96_nohis_ep15_20261002_224554`
- `work_dirs/front3_base_agri_amoe96_cropsample_nohis_ep15_20261006_094650`
- `work_dirs/front3_base_agri_amoe96_hardsoil_nohis_ep15_20261006_221137`
- `work_dirs/front3_base_agri_amoe96_focalfree_nohis_ep15_20261007_104739`

原基线每卡 batch=2、3 卡、累积 4 次；新实验每卡 batch=3、3 卡、累积 3 次。
数据划分、图像尺寸 512×288、优化器及学习率调度一致，但有效 batch 和优化器更新次数不同。
因此 +0.629 是已观察到的单次实验收益，还不能单独归因于抽样；下一次应补一组有效 batch=27 的无抽样 Agri-AMoE 基线。

`cropsample0.2_nohis_ep15_20261008_102056` 尚未完成，本次不把它列为已完成结果。
`cropfocus` 仅找到迷你数据集冒烟日志，没有完整 15 epoch 对照结果。

## 为什么清理另外两种策略

### hardsoil

旧实现只选择 GT=crop 的同高度 XY 邻域内的 GT=soil，再按预测 crop 概率取 top 50%，最多每帧 2048 个体素，增加权重 0.15 的土壤分类损失。

原基线的全验证集统计来自
`work_dirs/front3_base_agri_amoe96_nohis_ep15_20261002_224554/distance_crop_metrics.json`：
crop precision=39.575%，recall=63.436%；crop 误检中 90.599% 位于 GT=free，只有 5.410% 位于 GT=soil。
所以该实现没有针对主要误检来源；同高度限制还可能遗漏作物上方/下方的错误。
此外，固定 top-k 比例会在候选土壤已经容易分类时继续选点，增加局部土壤监督但未必增加有效的新信息。
实验最终连 soil IoU 也低于原基线，当前配置没有显示值得保留的收益。
这不意味着所有困难负样本方法无效，而是本次的候选标签、空间范围和选点方式不合适。

### focalfree

旧实现只把 GT=free 的 CE 乘以 `(1-p_free)^gamma`，其余类别 CE 不变；语义比例、几何比例和 Lovasz 损失不变。
当 p_free=0.95、gamma=2 时，该体素的 CE 乘数为 0.0025。
现有 CE 已把全部非自由类别权重设为 5、free 权重设为 1，并已有多种辅助损失。
进一步削弱易分类 free 并不能直接增强模糊 crop/free 边界的区分，也可能削弱自由空间的约束。
这属于根据实现与错误分布提出的解释；目前没有梯度贡献或逐类混淆统计能证明它是唯一原因。
crop IoU 只差 -0.014 个百分点，应视为基本持平，不能认为已证明 Focal Loss 本身无效。

已清理这两种具体实现的 head 代码、CLI 参数及 `hardsoil/cropfocus/focal` 任务入口。
保留 cropsample、ADHR、GapRef、NearFar 及历史实验日志和权重。

## cropsample 下一轮怎么调

当前抽样是固定 epoch 长度的替换：从低于阈值的帧中移除一小部分，再从达到阈值的帧中有放回抽取替代帧。
ratio=0.2 表示替换整个 epoch 的 20%，并不是把含作物样本比例设为 20%，也不是追加 20% 的训练时长。

训练集共有 32385 帧，约 90.008% 的帧含有效 crop，但只有 63.727% 达到 1000 个有效 crop 体素。

| ratio（min=1000） | 达标帧占比 | 低作物帧保留比例 | 每帧平均 crop 体素数（期望） |
| --- | ---: | ---: | ---: |
| 0，无替换 | 63.73% | 100% | 4130 |
| 0.1 | 73.73% | 72.43% | 4740 |
| 0.2 | 83.73% | 44.86% | 5350 |
| 0.3 | 93.73% | 17.29% | 5960 |

优先顺序：

1. 补同配置无抽样基线，同时等待 0.2 的完整结果。
2. 在阈值 1000 不变的条件下，比较 0.1、0.15、0.2；暂不推荐直接升到 0.3。
3. 选定 ratio 后，再比较阈值 500、1000、2000，避免一次改变两个变量。阈值 500、ratio=0.1 的达标帧占比约为 81.27%；阈值 2000、ratio=0.1 约为 63.51%。不同阈值的达标帧定义不同，这些比例不能作为效果排名。
4. 同时关注 mIoU、crop precision/recall 和各生长阶段指标。单独提高 recall、扩大作物预测区域不一定提高 IoU。

当前脚本保留用户设置的 ratio=0.2、min=1000；运行目录自动记录 ratio、min 和 seed。
可执行的下一轮命令：

```bash
# 同配置、无抽样的 Agri-AMoE 对照，使用脚本当前 batch 等设置。
FARMSIM_TASKS=amoe bash tools/train_farmsim_front3.sh

# 两个单变量候选；0.2 任务已经启动时不必重复运行。
FARMSIM_TASKS=cropsample CROP_OVERSAMPLE_RATIO=0.15 CROP_MIN_VOXELS=1000 bash tools/train_farmsim_front3.sh
FARMSIM_TASKS=cropsample CROP_OVERSAMPLE_RATIO=0.1 CROP_MIN_VOXELS=500 bash tools/train_farmsim_front3.sh
```

## 更值得引入的机制

### 首选：作物与生长阶段分层的作物抽样

固定 1000 阈值存在系统性偏差：早期玉米的达标帧只有 0.46%，成熟向日葵为 91.07%。
ratio=0.2 时，早期玉米占训练样本的期望份额会从 2.026% 降至 0.917%，成熟向日葵会从 2.835% 升至 3.506%。
这说明全局“体素多=值得多训练”的规则会削弱小株/早期作物覆盖，不能把所有低体素帧都视为无信息背景。

建议保持各 `crop_type × growth_stage` 的总抽样配额，在每组内部使用作物体素数分位数定义 donor，保留至少一半低体素帧。
早期小株因此也能进入 donor 集合；必要时再限制来自同一路线的重复帧，减少相邻视频帧的信息重复。
这是待实现、待验证的下一版机制，本次没有修改已有 sampler 行为。

### 次选：有负样本保留的在线帧级困难抽样

模型预训练后，使用**训练集**上带 GT 的 crop FP/FN 统计选择真正困难的帧。
同时保留 crop-rich 正样本、crop 被漏检的帧、以及无/少 crop 但 crop 误检严重的帧，仍通过小比例替换保持 epoch 长度不变。
这比只依据作物体素数选点更贴近当前 precision 偏低的问题，且不改变占用 head 或推理成本。
应先在训练集缓存统计，不能使用验证集标签驱动训练抽样。

### 最低成本诊断：crop logit 校准

冻结检查点，在独立校准子集上给 crop logit 施加小幅负偏置（例如 0、-0.1、-0.2、-0.4），重新计算 precision、recall 和 IoU。
若小偏置就能提高 IoU，说明一部分错误来自 crop 判决偏宽；若无改善，再优先考虑空间细节与监督不足。
这是诊断/校准实验，不应当成新架构贡献。最终报告前需固定偏置，并在未参与调参的测试集上评价。

参考原始论文：[Focal Loss](https://arxiv.org/abs/1708.02002)针对大量易负样本主导训练的问题；
[Class-Balanced Loss](https://openaccess.thecvf.com/content_CVPR_2019/html/Cui_Class-Balanced_Loss_Based_on_Effective_Number_of_Samples_CVPR_2019_paper.html)提供按有效样本数重平衡的思路。
它们不能直接保证本任务提升，以上建议首先依据本项目的误检分布和分层抽样统计。
