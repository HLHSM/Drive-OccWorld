# 准备数据集

**目录结构**
```
Drive-OccWorld
├── projects/
├── tools/
├── pretrained/
│   ├── r101_dcn_fcos3d_pretrain.pth
├── data/
│   ├── can_bus/
│   ├── nuscenes/
│   │   ├── maps/
│   │   ├── samples/
│   │   ├── sweeps/
│   │   ├── lidarseg/
│   │   ├── v1.0-test/
|   |   ├── v1.0-trainval/
|   |   ├── nuscenes_infos_temporal_train_new.pkl
|   |   ├── nuscenes_infos_temporal_val_new.pkl
│   ├── cam4docc
│   │   ├── GMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   │   ├── MMO/
│   │   │   ├── segmentation/
│   │   │   ├── instance/
│   │   │   ├── flow/
│   ├── nuScenes-Occupancy/
```

**1. 下载 nuScenes V1.0 完整数据集和 CAN 总线扩展数据（[下载地址](https://www.nuscenes.org/download)），然后运行以下命令准备 nuScenes 数据：**
```bash
cd Drive-OccWorld
mkdir data
ln -s /path/to/nuscenes data/nuscenes
ln -s /path/to/nuscnens/can_bus data/can_bus
```

**2. 准备 nuscene.pkl 文件**
1) 下载预处理后的 [nuscenes_infos_temporal_train_new.pkl](https://drive.google.com/file/d/1rjyqgJAnjg4Iq1MSussm5yhKDPVIlIxe/view?usp=sharing) 和 [nuscenes_infos_temporal_val_new.pkl](https://drive.google.com/file/d/1UsIaoX6uw2am9FWJBRA5cR40hl8AcL7p/view?usp=sharing)。
2) 将 ```.pkl``` 文件移动到 nuscenes 文件夹中：
```bash
mv nuscenes_infos_temporal_train_new.pkl /path/to/nuscenes
mv nuscenes_infos_temporal_val_new.pkl /path/to/nuscenes
```

**3. 按照 [Cam4DOcc](https://github.com/haomo-ai/Cam4DOcc) 的定义生成膨胀式占用标注：**

创建数据集文件夹。训练和测试的第一个 epoch 期间会自动生成数据：
```bash
mkdir /path/to/save_data/cam4docc   # ~650G
cd /path/to/save_data/cam4docc
mkdir GMO GMO_lyft MMO MMO_lyft
ln -s /path/to/save_data/cam4docc Drive-OccWorld/data
```

**4. 下载 [nuScenes-Occupancy](https://github.com/JeffWang987/OpenOccupancy/blob/main/docs/prepare_data.md) 定义的细粒度占用标注，并按如下方式组织数据集：**
```bash
mv nuScenes-Occupancy-v0.1.7z /path/to/save_data
cd /path/to/save_data
7za x nuScenes-Occupancy-v0.1.7z
mv nuScenes-Occupancy-v0.1 nuScenes-Occupancy
ln -s /path/to/save_data/nuScenes-Occupancy Drive-OccWorld/data/
```
