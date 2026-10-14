# AgriOcc：CUDA / TensorRT / INT8 部署

目标模型为 **base + Agri-AMoE + cropsample**，单帧、三前向相机、当前占用。
cropsample 是训练抽样策略，不进入推理图。保留原 TSA、六层 BEV 编码器、
Agri-AMoE 三个专家。默认检查点是 0.1/min500、15 epoch、75×75×25 输出的已完成实验。

目标设备：Jetson AGX Orin，JetPack 5.0.2 / TensorRT 8.4.1 / CUDA 11.4。
服务器：RTX 4090 / CUDA 12.8 / TensorRT 10.3。服务器 engine、x86 插件不能直接给 Orin 使用。
源码、ONNX opset 13、Q/DQ 量化尺度可迁移；Orin 必须重编译 SM87/ARM64 插件并重建 engine。
插件使用两端支持的 `IPluginV2DynamicExt`；已在服务器验证，并对 TRT 8.4 头文件做编译检查。
实际 ARM64/TRT 8.4.1 运行结果需要在设备上验证。

## 实现

- `model.py`：纯张量模型，固定几何索引，无 NumPy/元数据/GT 的网络前向。
- `cuda_ops/`：多尺度双线性采样与加权归约融合 kernel；FP16 存储、FP32 累加。
- `trt_plugins/`：同一 CUDA kernel 的 TensorRT 插件，TSA 和 SCA 共用。
- `export.py`：导出 R101/FPN、BEV、AMoE 三个连接的 ONNX 图。三段 engine 共享 GPU 指针，
  中间特征不经过 CPU，运行时不需要 PyTorch/MMCV。
- `quantize.py`：独立训练集代表性样本，seed=0，不重复作物过采样，INT8 对称 per-channel
  权重量化及 per-tensor 激活量化，输出显式 Q/DQ ONNX。量化 backbone/FPN 的 Conv，
  默认保留输入 stem 卷积为浮点；浮点 bias 避免 TensorRT 不支持的 INT32 DQ。
- `build_engine.py`：TRT 8.4/10 API 兼容。归一化及归约保留 FP32，避免半精度溢出。
- `evaluate.py`：同一验证样本的 mIoU、每类 IoU、crop precision/recall、原模型一致性与计时。
- `infer.py` / `runtime.py`：无 PyTorch 的运行时、固定缓冲区与可选 CUDA Graph。
  实际标签输出在 GPU 上 argmax，返回 uint8 占用图，减少 D2H 数据量；同时检查 NaN/Inf。
- `prepare_validation.py` / `validate.py`：导出验证包，到 Orin 用真实 GT 复核精度。

输出 logits 形状 `[1,5625,25,6]`；转换语义标签时严格复用原评价路径的 X/Y 约定，
不能直接把 token 维 reshape 后忽略转置。类别按 model.json 顺序。

## 服务器准备

从仓库根目录运行。环境是独立 venv，复用已有 dow2 的 PyTorch/MMCV，不修改 dow2。

```bash
bash deploy/scripts/setup_server.sh
export PATH="$PWD/deploy/.venv/bin:/usr/local/cuda/bin:$PATH"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=0
bash deploy/scripts/build_plugins.sh
python -m deploy.verify
python -m deploy.export
```

若机器的代理配置失效，可仅对安装命令清空 `HTTP_PROXY/HTTPS_PROXY/ALL_PROXY`；不必改全局设置。
更换模型请同时传 `--config` 和 `--checkpoint`，使用该实验保存的配置。
导出不支持 GVAD、NearFar、历史帧、未来任务或额外 refiner；遇到不匹配配置会报错。

## 构建、量化与评估

```bash
python -m deploy.build_engine --precision fp16
python -m deploy.quantize --samples 128 --method minmax
python -m deploy.build_engine --precision int8 --output deploy/artifacts/int8

python -m deploy.evaluate --backend original --samples 128 --output deploy/artifacts/original.json
python -m deploy.evaluate --backend cuda --samples 128 --compare-original --output deploy/artifacts/cuda.json
python -m deploy.evaluate --backend trt --samples 128 --compare-original --output deploy/artifacts/fp16.json
python -m deploy.evaluate --backend trt --engine-dir deploy/artifacts/int8 \
  --samples 128 --compare-original --output deploy/artifacts/int8.json
```

`--samples 0` 为全验证集；正数默认均匀覆盖整个验证清单，而不是仅取第一个序列。
`--sampling leading` 可复现顺序开头样本。mIoU 与当前项目 eval_hooks 相同：对存在的全部类别
（含 free）求均值，忽略 GT=255。比较模型产生的标签仅用于一致性检查，指标仍使用真实 GT。
PTQ MinMax 避免收集大量特征占用主机 RAM；Entropy 可选，但 R101 多层输出的内存开销较大。
真实无标签图像可按同样预处理保存为含 `images: [3,3,H,W]` 的 NPZ，传入
`quantize --calibration-dir <目录>`；只使用 images，不使用 GT/伪标签。
全流程脚本为 `bash deploy/scripts/run_server.sh`，可设置 `SAMPLES=0`、`CALIBRATION_SAMPLES=128`。

所有计时均 CUDA 同步、预热；报告 GPU 常驻的标准化图像到最终 logits 的耗时，排除
读取/解码/预处理、H2D、D2H、GT 指标计算。此前约 63 ms 的 MMDataParallel 测试包含包装及
传输开销，不能把它与这里的纯 GPU 计时直接作为严格加速比。
性能测试应串行运行；勿与 engine 构建或其他 GPU 工作并行。
`--compare-original` 的 PyTorch 显存统计包含对照模型，并不代表 TRT 独立运行的显存。
TRT 单独测速与进程显存统计：

```bash
python -m deploy.infer --engine-dir deploy/artifacts/fp16 \
  --plugin deploy/build/plugins/libagriocc_plugins.so --input deploy/artifacts/model/example.npz \
  --cuda-graph --output deploy/artifacts/fp16/prediction.npz
```

## 打包并转移到 Orin

```bash
python -m deploy.prepare_validation --samples 20
python -m deploy.package --validation-dir deploy/artifacts/validation
```

将 `deploy/artifacts/agriocc_orin_bundle.tar.gz` 传到 Orin，在一个空的工程目录解压。
包中不含服务器 engine、插件二进制、venv 或训练数据集，只含源码、模型和所选验证样本。

Orin 使用 JetPack 自带 TensorRT 8.4.1、CUDA 和开发头文件，系统需有 CMake/G++/python3-venv。
原始相机图片输入需要 OpenCV（JetPack 系统 python3-opencv 或与其兼容的安装）。

```bash
tar -xzf agriocc_orin_bundle.tar.gz
bash deploy/scripts/setup_orin.sh
PRECISION=fp16 bash deploy/scripts/run_orin.sh
PRECISION=int8 bash deploy/scripts/run_orin.sh

export PYTHONPATH="$PWD:${PYTHONPATH:-}"
deploy/.venv/bin/python -m deploy.validate --engine-dir deploy/artifacts/fp16
deploy/.venv/bin/python -m deploy.validate --engine-dir deploy/artifacts/int8 \
  --output deploy/artifacts/orin_int8_validation.json
```

设备测速记录 `nvpmodel -q` 的功耗模式、温度和频率；`tegrastats` 记录共享内存及功耗。
脚本不自动更改系统功耗模式或时钟。需持续运行测试实际部署延迟。

## 真实相机标定

固定几何缓存绑定 **相机顺序、标定、缩放方式、图像尺寸、预测 ROI**。
实际机器人若与 FarmSim 的相机不同，必须重新导出 BEV 图，而不是沿用仿真投影矩阵。
`model.json` 记录 camera_order。标定 JSON 需要原始分辨率的三个 4×4 `lidar2img`
投影矩阵（机器人 x 前/y 右/z 上坐标约定）及 `image_size: [width,height]`。

```bash
python -m deploy.export --calibration-json robot_calibration.json --output deploy/artifacts/robot
```

导出时矩阵按模型输入分辨率缩放。使用对应 bundle 构建 engine。原始图片按 model.json 的
相机顺序输入 `infer --images front.jpg left.jpg right.jpg`。预处理保持 BGR、先转 float32 再
双线性缩放、随后减 Caffe mean，不采用 RGB/ImageNet normalization。
CUDA Graph 固定输入缓冲区，可更新图像内容；尺寸或标定变化需要新的 bundle/capture。
`infer --images ... --gpu-preprocess` 可使用融合 CUDA 缩放/BGR 归一化；JPEG 解码仍由 CPU
完成。三个原始图像应同尺寸。使用 `python -m deploy.check_preprocess` 检查预处理数值一致性。

## 参考

- [BEVFormer TensorRT 插件部署](https://github.com/DerryHub/BEVFormer_tensorrt)
- [CUDA-FastBEV 部署流程](https://github.com/Mandylove1993/CUDA-FastBEV)
- [TensorRT 引擎兼容范围](https://docs.nvidia.com/deeplearning/tensorrt/latest/getting-started/support-matrix.html)

本目录 CUDA kernel 为本项目的独立实现，第三方 TensorRT 头文件只用于服务器编译，
不进入边缘部署包。实际使用的版本、精度和样本数量均记录在生成的 JSON 中。
