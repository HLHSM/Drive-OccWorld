# 安装

安装步骤与 [BEVFormer](https://github.com/fundamentalvision/BEVFormer/blob/master/docs/install.md) 类似。
为方便使用，具体步骤如下：

**1. 创建并激活 conda 虚拟环境。**
```shell
conda create -n driveoccworld python=3.8 -y
conda activate driveoccworld
```

**2. 安装 PyTorch 和 torchvision（mmdet3d 要求 torch<1.11.0）。**
```shell
pip install torch==1.10.1+cu111 torchvision==0.11.2+cu111 torchaudio==0.10.1 -f https://download.pytorch.org/whl/cu111/torch_stable.html
```

**3. 在 conda 环境中安装 gcc>=5（可选）。**
```shell
conda install -c omgarcia gcc-6 # 确保 GCC 版本与 CUDA 兼容
```

**4. 安装其他依赖包和 Detectron2。**
```bash
pip install setuptools einops fvcore seaborn ninja iopath==0.1.9 timm==0.6.13  typing-extensions==4.5.0 pylint ipython==8.12  numpy==1.19.5 matplotlib==3.5.2 numba==0.48.0 pandas==1.4.4 scikit-image==0.19.3 casadi==3.5.5 pytorch-lightning==1.2.5 lyft_dataset_sdk nuscenes-devkit plyfile networkx==2.2 trimesh==2.35.39 yapf==0.40.1

# 安装 Detectron2
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
# 或从本地克隆的代码安装：
git clone https://github.com/facebookresearch/detectron2.git
python -m pip install -e detectron2
```

**5. 安装 mm 系列软件包。**
```bash
pip install mmcv-full==1.4.0
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1

# 从源代码安装 mmdetection3d。
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v0.17.1 # 其他版本可能不兼容。
python setup.py install
```

**6. 克隆仓库并完成配置。**
```bash
git clone https://github.com/yuyang-cloud/Drive-OccWorld.git
cd Drive-OccWorld
git checkout ir-wm

mkdir -p pretrained
wget -c https://github.com/zhiqi-li/storage/releases/download/v1.0/r101_dcn_fcos3d_pretrain.pth \
  -O pretrained/r101_dcn_fcos3d_pretrain.pth

# 仅使用 InternImage 主干时才需要安装 DCNv3。
# FarmSim 默认的 ResNet-101 主干不使用 DCNv3，无需执行下面两行。
cd projects/mmdet3d_plugin/bevformer/backbones/ops_dcnv3
python setup.py install
```

`r101_dcn_fcos3d_pretrain.pth` 是训练时加载的初始化权重，不是需要通过
`setup.py` 安装的软件包。FarmSim 训练脚本通过 `PRETRAINED_FROM` 指向该文件；
从已有训练断点继续时，应清空 `PRETRAINED_FROM` 并设置 `RESUME_FROM`。
