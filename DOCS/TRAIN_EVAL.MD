# 训练与评估

**1. 按照[此处](https://github.com/yuyang-cloud/Drive-OccWorld/blob/c508199c69d6d14e4618f22ad16e2ddb91606b8a/DOCS/DATASET.MD?plain=1#L49)的说明生成膨胀式占用标注：**

1) 在 [projects/configs/inflated/action_condition_GMO.py](../projects/configs/inflated/action_condition_GMO.py) 和 [projects/configs/inflated/action_condition_MMO.py](../projects/configs/inflated/action_condition_MMO.py) 中都设置 ```only_generate_dataset=True```。
2) 使用这些配置运行训练和评估脚本。由于不会执行模型的前向过程，因此不会占用 GPU 显存：
```bash
CONFIG=projects/configs/inflated/action_condition_GMO.py
GPU_NUM=8   # 8 GPUs(Processes) consume ~20 hours

# 生成训练数据
./tools/dist_train.sh ${CONFIG} ${GPU_NUM}

# 生成验证数据
CKPT=work_dirs/action_condition_GMO/epoch_1.pth
./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```


**2. 训练**

```bash
CONFIG=path/to/config.py
GPU_NUM=8

./tools/dist_train.sh ${CONFIG} ${GPU_NUM}
```

**配置文件：**
1) [inflated/action_condition_GMO.py](../projects/configs/inflated/action_condition_GMO.py)：使用 Cam4DOcc 定义的膨胀式占用标注预测通用可移动物体（GMO）。
2) [inflated/action_condition_MMO.py](../projects/configs/inflated/action_condition_MMO.py)：使用 Cam4DOcc 定义的膨胀式占用标注预测多类别可移动物体（MMO）。
3) [fine_grained/action_condition_GMO.py](../projects/configs/fine_grained/action_condition_GMO.py)：使用 nuScenes-Occupancy 定义的细粒度占用标注预测通用可移动物体（GMO）。
4) [fine_grained/action_condition_GMO_GSO.py](../projects/configs/fine_grained/action_condition_GMO_GSO.py)：使用 nuScenes-Occupancy 定义的细粒度占用标注预测通用可移动物体（GMO）和通用静态物体（GSO）。
5) [fine_grained/action_condition_MMO_MSO.py](../projects/configs/fine_grained/action_condition_MMO_MSO.py)：使用 nuScenes-Occupancy 定义的细粒度占用标注预测多类别可移动物体（MMO）和多类别静态物体（MSO）。


**3. 评估**

下载提供的预训练[权重](https://huggingface.co/Jianbiao/IR-WM)，或使用自行训练得到的权重。

```bash
CONFIG=path/to/config.py
CKPT=work_dirs/config_file_name/epoch_24.pth
GPU_NUM=8

./tools/dist_test.sh ${CONFIG} ${CKPT} ${GPU_NUM}
```
