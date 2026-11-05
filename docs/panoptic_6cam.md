# 六相机全景占用训练

主模型：ResNet101/FPN → BEVFormer + GVAD → lift → Agri-AMoE → 11 类语义头 + 256 个实例 queries / 三层 mask decoder。输入为当前帧六个 RGB 相机，512×288；没有历史、未来预测或规划监督。

输出范围：x=[-20,20] m、y=[-10,10] m、z=[-2,3] m，分辨率0.2 m，200×100×25。BEV编码器H=100（Y）、W=200（X）。标签与保存结果布局为 [X,Y,Z]，模型内部卷积布局为 [Z,Y,X]。

该范围完全位于源标注范围内；配置设置 `allow_unknown_padding=False`，不再添加左右未知区域。地形下方或未知几何仍按原始valid mask忽略。

## 启动

从项目根目录：

```bash
bash tools/train_panoptic_6cam.sh
```

默认GPU 0,1,2,3，每卡batch=1，有效总batch=16（累积4步），15个epoch，seed=0。直接使用 `/data/HL/FlashOCC/data/simdata/splits/{train,val,test}.json`，不重新生成划分。初始化仅加载图像检测预训练权重 `pretrained/r101_dcn_fcos3d_pretrain.pth`；新增的占用/实例参数正常随机初始化。

两卡例子：

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 TOTAL_BATCH_SIZE=16 bash tools/train_panoptic_6cam.sh
```

可设置 `PYTHON_BIN`、`DATA_ROOT`、`SPLIT_DIR`、`WORK_DIR`、`BATCH_SIZE`、`TOTAL_BATCH_SIZE`、`EPOCHS`、`NUM_QUERIES`、`SAMPLE_STRIDE`。GPU个数须与NUM_GPUS一致，总batch须整除每卡batch×GPU数。

训练采样步长 `SAMPLE_STRIDE` 默认2，从每个序列的第0帧开始隔帧采样，与FlashOCC的sample_interval含义一致。训练54,872帧 → 27,574帧（奇数长度序列逐个向上取整，约减半）；验证7,153帧、测试7,147帧保持完整。配置单独加载时也默认train stride=2、val/test stride=1。

```bash
SAMPLE_STRIDE=1 bash tools/train_panoptic_6cam.sh  # 使用完整训练帧
```

脚本首先检查实际采样训练帧的原始实例标签以及最大实例数，结果缓存于 `work_dirs/panoptic_6cam_flashocc_stride2_training_data.json`（按步长分别缓存）。首次检查需要扫描NPZ；后续根据manifest、采样步长、schema及文件大小/修改时间验证缓存。若超过NUM_QUERIES，启动前直接报错，不删除GT。

本版本采用均匀帧采样；没有沿用只针对crop的过采样。类别频率和实例支持数会写入检查报告，后续可据此设计独立的thing-aware采样实验。

## 监督与融合

保留原始11类（含free）。things为building=4、fence_barrier=5、vehicle=7、person_animal=8、other_obstacle=9、tree=10；crop等stuff的实例为0。本数据没有作物单株实例监督。

实例ID是每帧用于集合匹配的身份，不作为固定类别预测。Hungarian成本为class + 2×BCE + 2×Dice，匹配后训练class/BCE/Dice；no-object权重0.1。每帧随机采样最多4096个有效体素，再为每个实例额外采样16个体素，保证小实例进入监督；mask负样本包括有效stuff/free和其它实例。空实例帧仍训练no-object分类。

实例decoder只对池化后的5×25×25三维token做cross attention，保留三维位置编码；最终mask使用query embedding与原分辨率体素特征的点积。训练只计算采样点mask，推理分块计算，避免创建完整Q×200×100×25 mask张量。BEV编码器与共享三维解码使用非重入激活检查点，FP16主干、FP32实例匹配/损失。

推理先取语义预测，仅在同类thing体素中按query置信度×mask概率分配实例，query分类阈值0.3，mask概率阈值0.5；未接受的thing保留ID0，在PQ中作为该类误检segment计入且不允许匹配GT，不静默删除。

## 评测

每epoch验证会记录11类语义IoU与 `panoptic_PQ/SQ/RQ`、`panoptic_PQ_th/PQ_st`、各类PQ及GT实例支持数。PQ在GT有效域计算，按类别一对一IoU>0.5匹配；stuff每帧每类合并，free排除。报告present-class均值，空支持类别同时给出support=0；这是有效域体素PQ协议，没有实现RayPQ或时序跟踪评价。

完成训练后，使用保存的实际配置（包含脚本施加的GVAD等覆盖参数）测试：

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. /home/HL/.conda/envs/dow2/bin/python tools/test.py \
  /absolute/work_dir/farmsim_panoptic_6cam.py /absolute/work_dir/epoch_15.pth \
  --launcher none --eval occ --batch-size 1 \
  --save-predictions /absolute/work_dir/predictions \
  --save-prediction-count 100 --save-prediction-sampling per-sequence \
  --out /absolute/work_dir/test_metrics.pkl
```

NPZ包含 `current_pred/current_gt`（语义）、`current_instance_pred/current_instance_gt`、`valid_mask`、`point_cloud_range`。原语义viewer尚未新增实例着色模式；实例数组可直接供后续可视化读取。

## 验证

```bash
PYTHONPATH=. /home/HL/.conda/envs/dow2/bin/python -m unittest \
  tools.tests.test_panoptic_model tools.tests.test_simdata_panoptic
bash -n tools/train_panoptic_6cam.sh
```

测试覆盖原生类别、轴序、超出标注区域、实例ID置换不变性、矩形Hungarian最优性、小实例与空帧、query容量溢出、语义约束融合、PQ合并/误检/ignore、batch反向及预测产物。

当前划分包含504/63/63个训练/验证/测试序列。加载器保留FlashOCC manifest中的序列顺序，按其相同规则排序meta帧；没有过滤或重新分配序列。兼容原有带frame_ids的Drive-OccWorld manifest。

验证包含FlashOCC格式读取、逐序列步长、非正方形网格轴序及模型训练/推理回归。旧40×40m检查产物仍保留在原work_dirs目录，其显存和模型指标不代表当前配置。完整训练尚未启动。

当前40×20m配置已通过15项回归测试，以及真实六相机双卡检查（4个训练样本、累积2步完成参数更新、3个完整验证样本）。日志位于 `work_dirs/panoptic_6cam_flashocc_stride2_smoke/`；这些是执行检查，不是完整训练结果。

高度已更新至−2～3m（25层）。上述smoke记录对应先前−2～2m配置；当前高度范围已单独核对实际标签形状。
