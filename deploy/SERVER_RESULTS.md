# 4090 部署验证记录

日期：2026-10-13。设备：RTX 4090，batch=1（三相机折叠到图像编码器 batch 维）。
软件：PyTorch 2.7.1/cu128、CUDA 12.8、TensorRT 10.3.0、ONNX opset 13。
检查点：`front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_20261012_214738/epoch_15.pth`。
输入三相机 512×288，输出 75×75×25，0.2 m 分辨率。

## 同样本、同计时范围对比

验证清单共有 6477 帧，本次均匀选择 128 帧。各运行预热 10 次，GPU 0 上串行测试。
从 GPU 常驻标准化图像到最终 logits，排除读取、解码、预处理、H2D/D2H 和评价指标计算。
这些是部署转换检查的子集结果，不是全验证集正式论文指标。

| 后端 | 平均耗时 ms | mIoU % | crop IoU % |
|---|---:|---:|---:|
| 原模型 PyTorch / 原 AMP 设置 | 57.946 | 64.1227 | 40.8259 |
| 纯推理图 / 专用 CUDA / FP32 | 25.690 | 64.1280 | 40.8412 |
| TensorRT FP16，归约/归一化 FP32 | 6.817 | 64.1225 | 40.8236 |
| TensorRT 混合 INT8 | 6.849 | 63.8281 | 40.7663 |

FP16 相对本次同口径 PyTorch 测试约 8.50 倍加速，mIoU 差约 -0.00019 个百分点。
CUDA FP32 相对原 AMP 的微小精度变化来自数值精度/运算顺序，不代表重新训练获得了提升。
此前 MMDataParallel 的约 63 ms 计时包含包装及传输开销，因此不用于这里的严格加速比。

INT8 校准：32 个训练样本、seed=0、Entropy。卷积权重 per-channel INT8、激活对称 INT8；
保留首个 stem 卷积、bias、BEV 编码器、Agri-AMoE 为较高精度。
engine inspector 显示实际 INT8 层/张量格式，文件为 `artifacts/int8/backbone_layers.json`。
当前 INT8 没有带来同样本平均耗时收益，mIoU 下降约 0.295 个百分点。优先推荐 FP16；
Orin 是否获益需要设备实测。默认量化脚本使用内存较省的 MinMax，也可指定 Entropy。

## 独立 TensorRT 运行与显存

单个标准化输入，预热 20 次，重复 100 次 CUDA Graph 推理（区别于上面 128 个不同帧）。
当前保存的测量 JSON 位于 `artifacts/{fp16,int8}/graph_prediction.json`。
FP16 进程显存约 918 MiB，INT8 约 870 MiB；这是 nvidia-smi 的进程统计，包含 CUDA context。
FP16 引擎/context/缓冲区及首次运行分配相对 CUDA 初始化基线的增量约 534 MiB，INT8 约 486 MiB。
不能将这些数字直接与 PyTorch 的 allocated/reserved 指标混作同一种显存统计。

## 已完成的检查

- 随机、含越界坐标的 MSDA FP32/FP16 对照通过。
- 单帧完整 FP32 图：CUDA logits 与原模型最大绝对差约 9.5e-6。
- 128 帧 FP16/INT8：无 NaN，真实 GT 指标与原模型对照完成。
- 融合 CUDA 缩放/BGR 归一化与 OpenCV float 预处理对照通过。
- 20 帧便携 NPZ 验证包的两种 TensorRT 后端运行通过。
- 采样/预处理 CUDA 源码已生成 SM87 目标文件；插件已通过 TensorRT 8.4 头文件语法编译。
- 实际 JetPack 5.0.2 / ARM64 / TensorRT 8.4.1 的 engine 构建和性能尚需在 Orin 上验证。

FP16 最初发生过 GroupNorm 半精度归约溢出，已通过 FP32 归约/归一化解决；
工具对非有限 logits 直接报错，避免将异常结果计为正常精度或速度。
