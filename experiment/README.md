# FarmSim external occupancy baselines

This directory reproduces the controlled **current-frame, front-three-camera**
FarmSim comparison:

| Method | Location | Status |
| --- | --- | --- |
| Drive-OccWorld | repository root | native implementation |
| IR-WM | repository root | existing result is reused; no duplicate run |
| SparseOcc | `official/SparseOcc` | official code with FarmSim adapter |
| SurroundOcc | `official/SurroundOcc` | official code with FarmSim adapter |
| COTR | `official/COTR` | official code with FarmSim adapter |

All external methods read the immutable sequence-level split in
`data/farmsim/splits/{train,val}.json`.  `common/infos/` is generated from
those manifests and only references the source images/occupancy files; it does
not copy the dataset.  The checked-in manifests produce 32,385 train and
6,477 validation current-frame samples.

## Common contract

- Cameras: `front_left_rgb`, `front_rgb`, `front_right_rgb` in that order.
- Range: `[0,-10,-2,20,10,3]` metres; grid `[100,100,25]`; voxel size `0.2m`.
- Reporting labels: `free, crop, soil_ground, drivable, other_vegetation,
  other_obstacle`, with `255` ignored.
- SparseOcc and COTR internally use `crop..obstacle=0..4, free=5`; their
  evaluators map predictions back before reporting FarmSim metrics.

The pinned upstream revisions are SparseOcc `af4d9df`, SurroundOcc `419bf5b`,
and COTR `4328e0a`.  SurroundOcc and COTR are partial Git checkouts containing
the upstream source, configuration, extension, tool, and documentation files
needed for this reproduction; the excluded demo assets are not needed.
SurroundOcc additionally keeps the official MMDetection3D `v0.17.1` source at
`official/SurroundOcc/third_party/mmdetection3d` for reference.  It is not put
on `PYTHONPATH`: its THC-era extensions cannot compile with PyTorch 2.0.  The
Torch-2-compatible MMDet3D API installed in `surroundocc20` is used instead.

## Environments

Keep the official environments isolated.  Do not install their requirements
into `dow2`, which is reserved for the native Drive-OccWorld/IR-WM stack.

| Environment variable | Suggested environment | Official stack |
| --- | --- | --- |
| `DRIVE_ENV` | `dow2` | existing project environment |
| `SURROUNDOCC_ENV` | `surroundocc20` | Python 3.9, PyTorch 2.0/CUDA 11.8, MMCV 1.6/MMDet 2.28/MMDet3D 1.0rc6 compatibility API; `yapf==0.40.1`, TensorBoard; Chamfer extension builds against CUDA 11.8 |
| `COTR_ENV` | `cotr118` | Python 3.9, PyTorch 2.0/CUDA 11.8, MMCV 1.6/MMDet 2.28; custom BEV pooling builds against CUDA 11.8 |
| `SPARSE_ENV` | `sparseocc` | Python 3.9, PyTorch 2.0/CUDA 11.8, MMCV 1.6, MMDet 2.28.2, MMDet3D 1.0rc6, TensorBoard and W&B Python package |

The external baselines use the user-local CUDA Toolkit at
`/home/HL/.local/cuda-11.8` only when compiling/running their extensions.  The
system CUDA 12.8 driver is left unchanged.  Override this location with
`CUDA118_HOME=/path/to/cuda-11.8` if necessary.

## Shared training controls

`experiment/train_farmsim_all.sh` accepts the following environment variables
for every method.  `BATCH_SIZE` means *per GPU*; `TOTAL_BATCH_SIZE` means the
effective batch across all GPUs.  For COTR, SurroundOcc, and SparseOcc the
launcher derives `TOTAL_BATCH_SIZE / (BATCH_SIZE * NUM_GPUS)` and configures
MMCV gradient accumulation.  It must be an integer.

| Variable | Native Drive-OccWorld / IR-WM default | External-baseline default |
| --- | --- | --- |
| `BATCH_SIZE` | `3` per GPU | `1` per GPU |
| `TOTAL_BATCH_SIZE` | `24` | `BATCH_SIZE * NUM_GPUS` |
| `IMAGE_WIDTH`, `IMAGE_HEIGHT` | `512`, `288` | `512`, `288` |
| `USE_FP16` | `1` | `0` |
| `WORKERS_PER_GPU` | `4` | `4` |

`USE_FP16=1` activates MMCV dynamic loss scaling for the external methods.
MMCV 1.6 exposes separate FP16 and gradient-cumulative hooks, so this version
intentionally rejects a request that combines FP16 with gradient accumulation.
Use `TOTAL_BATCH_SIZE=BATCH_SIZE*NUM_GPUS` for FP16, or use `USE_FP16=0` when
you need accumulation.  This fails before GPUs are allocated rather than
silently changing the requested effective batch.

Install each repository's official requirements and build its custom CUDA
extensions before training.  SparseOcc needs `models/csrc`; SurroundOcc needs
`extensions/chamfer_dist`; COTR needs its BEV pooling extension. COTR,
SurroundOcc, and SparseOcc intentionally do
not download pretrained weights in the launcher; set them in the respective
config only after obtaining the desired official image-backbone checkpoint.

## Ignore handling

FarmSim's `occupancy_valid` is converted to label `255` in the common loader.
SurroundOcc uses `255` in its CE and scale-loss masks; COTR supplies the same
valid mask to all occupancy, matching, and mask losses; SparseOcc supplies it
to matching/mask losses and filters `255` for CE, Lovasz, and scale losses.
All three FarmSim evaluators filter with this validity mask before their
six-class confusion matrices are accumulated.

## Run

Use the one launcher from repository root:

```bash
bash experiment/train_farmsim_all.sh surroundocc
bash experiment/train_farmsim_all.sh cotr
bash experiment/train_farmsim_all.sh sparseocc
```

It also contains the native Drive-OccWorld and optional IR-WM commands.  Set
`CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `EPOCHS`, and the environment variables
before use.  `EPOCHS` is forwarded to each selected method (default: `8`).

For example, a two-GPU, full-precision COTR run with an effective batch of 8
uses four accumulation steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 BATCH_SIZE=1 TOTAL_BATCH_SIZE=8 \
  IMAGE_WIDTH=512 IMAGE_HEIGHT=288 USE_FP16=0 \
  bash experiment/train_farmsim_all.sh cotr
```
