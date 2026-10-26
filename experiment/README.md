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
- Range: `[0,-7.5,-2,15,7.5,3]` metres; grid `[75,75,25]`; voxel size `0.2m`.
- All shared loaders sample the original occupancy at the new target voxel
  centers, exactly matching native `farmsim_occ_front3_s.py`. This matters
  because the lateral -7.5 m boundary is halfway through an original voxel.
- The launcher builds/reuses matching indexes in `common/infos_front3_s/`;
  old `common/infos/` indexes and old 20 m result directories are untouched.
  New work directories/run names contain `front3_s`.
- This ROI update leaves existing camera calibration implementations intact.
  The external shared camera loader still uses the legacy mount-only geometry;
  comparing against native `CAMERA_GEOMETRY=grid_world` results also requires
  a separate calibration-parity update.
- SurroundOcc uses 75/38/19 feature grids and aligns odd-sized skip features;
  COTR uses a 19-cell coarse grid spanning the same 15 m ROI and aligns its
  deconvolution output back to 75 cells. Both retain 0.2 m final predictions.
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

### SparseOcc geometry-corrected retraining

The corrected FarmSim port keeps metric query coordinates in FP32, normalizes
camera indices by `num_cams - 1`, and expands coarse supervision by its exact
integer scale before cropping odd grid boundaries. Its cosine LR schedule
tracks the actual runner length rather than inheriting nuScenes' epoch-22/24
decay milestones. Online evaluation restores the training precision flag;
the custom float-only CUDA sampler explicitly promotes its inputs to FP32.

Retrain from the configured nuImages image-backbone weights, **not** from the
old geometry-broken occupancy checkpoint:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 NUM_GPUS=3 BATCH_SIZE=4 TOTAL_BATCH_SIZE=24 \
  EPOCHS=15 USE_FP16=1 \
  bash experiment/train_farmsim_all.sh sparseocc
```

The new default output is
`official/SparseOcc/outputs/SparseOcc/farmsim_front3_s_geomfix_ep15`; the old
`farmsim_front3_s_ep15` logs/checkpoint remain untouched. An existing output
directory is rejected; set a new `SPARSE_RUN_NAME` to run another experiment.
The user's editable `all` queue is not changed by this fix; select `sparseocc`
explicitly to retrain only this method.

For a bounded integration check (8 training / 4 validation samples, two epochs,
FP16 plus two-step gradient accumulation, evaluation after each epoch):

```bash
CUDA_VISIBLE_DEVICES=0 bash tools/train_sparseocc_farmsim_smoke.sh
```

This uses the existing full split indexes and a dataset length override; it
does not modify the original split manifests. Each smoke run gets a unique
output directory. Smoke metrics check pipeline operation, not model quality.

It also contains the native Drive-OccWorld and optional IR-WM commands. Their
configs now use `farmsim_occ_front3_s.py`; history/future/planning flags are
otherwise unchanged. An old 20 m IR-WM result cannot replace the new ROI run. Set
`CUDA_VISIBLE_DEVICES`, `NUM_GPUS`, `EPOCHS`, and the environment variables
before use. `NUM_GPUS` defaults to the visible GPU count; `EPOCHS` is forwarded
to each selected method (default: `15`). Preview launch commands without
starting training or regenerating info files with:

```bash
DRY_RUN=1 bash experiment/train_farmsim_all.sh all
```

Resume COTR from its latest complete epoch and then run the remaining jobs
(SparseOcc, SurroundOcc, IR-WM) in the launcher's current `all` order:

```bash
CUDA_VISIBLE_DEVICES=1,2,3 NUM_GPUS=3 BATCH_SIZE=4 TOTAL_BATCH_SIZE=24 \
  EPOCHS=15 COTR_RESUME_FROM=auto \
  bash experiment/train_farmsim_all.sh all
```

Resume reuses `work_dirs/cotr_farmsim_front3_s_ep15/cotr_farmsim_front3.py`,
the normal checkpoint's optimizer/runner state, and the matching
`epoch_N_ema.pth` through `MEGVIIEMAHook.resume`. An interrupted epoch is
repeated; `EPOCHS=15` is the total target, not 15 additional epochs. Supply an
absolute checkpoint path as `COTR_RESUME_FROM` to select a specific epoch;
`COTR_WORK_DIR` optionally selects the directory for `auto`. Missing saved
config or matching EMA checkpoint stops execution instead of silently
reinitializing them. With no `COTR_RESUME_FROM`, the fresh-run behavior is
unchanged. Add `DRY_RUN=1` to preview the complete queue without launching it.

`PREPARE_INFOS=0` skips automatic index preparation when matching indexes
already exist. To use custom split manifests, pass absolute `TRAIN_ANN_FILE`,
`VAL_ANN_FILE`, and a separate `FARMSIM_INFOS_ROOT` directory. The index cache
checks the manifest hash, data root, and ROI before reuse.

For example, a two-GPU, full-precision COTR run with an effective batch of 8
uses four accumulation steps:

```bash
CUDA_VISIBLE_DEVICES=0,1 NUM_GPUS=2 BATCH_SIZE=1 TOTAL_BATCH_SIZE=8 \
  IMAGE_WIDTH=512 IMAGE_HEIGHT=288 USE_FP16=0 \
  bash experiment/train_farmsim_all.sh cotr
```
