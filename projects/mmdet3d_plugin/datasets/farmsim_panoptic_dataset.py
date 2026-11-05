"""Native eleven-class panoptic labels, with explicit unknown-domain padding."""
import json
from pathlib import Path
import numpy as np
import torch
from mmdet.datasets import DATASETS
from .farmsim_world_dataset import FarmSimWorldDataset

PANOPTIC_CLASSES = ('free', 'crop', 'soil_ground', 'drivable', 'building',
                    'fence_barrier', 'other_vegetation', 'vehicle',
                    'person_animal', 'other_obstacle', 'tree')
THING_IDS = (4, 5, 7, 8, 9, 10)
INSTANCE_IGNORE = 4294967295


@DATASETS.register_module()
class FarmSimPanopticDataset(FarmSimWorldDataset):
    CLASSES = PANOPTIC_CLASSES
    PALETTE = [(0, 0, 0), (154, 205, 50), (120, 72, 30), (135, 206, 235),
               (180, 120, 120), (190, 153, 153), (55, 150, 80), (0, 0, 142),
               (220, 20, 60), (160, 80, 190), (20, 100, 35)]

    def __init__(self, *args, allow_unknown_padding=False, **kwargs):
        super().__init__(*args, **kwargs)
        if self.queue_length or self.future_pred_frame_num or self.predict_trajectory:
            raise ValueError('Panoptic training currently requires current-frame occupancy only')
        if self.occupancy_target_range is None:
            raise ValueError('Panoptic dataset requires an explicit target range')
        self.allow_unknown_padding = allow_unknown_padding
        self._panoptic_grid_cache = {}

    def _read_manifest(self, ann_file, data_root=None):
        manifest = super()._read_manifest(ann_file, data_root)
        sequences = manifest['sequences']
        if not sequences or not isinstance(sequences[0], str):
            return manifest
        # FlashOCC manifests list sequence paths. Keep their order and enumerate
        # sorted metadata exactly like SimDataPanopticDataset in that project.
        root = Path(data_root if data_root is not None else manifest['data_root']).expanduser()
        rows = []
        for relative in sequences:
            seq = root / relative
            frames = [p.stem for p in sorted((seq / 'meta').glob('*.json'))]
            if not frames:
                raise ValueError(f'{seq}: split sequence has no metadata frames')
            rows.append(dict(path=relative, frame_ids=frames, scenario_id=seq.parent.name))
        return dict(manifest, source_root=str(root), sequences=rows)

    def _grid(self, seq_path):
        if seq_path not in self._panoptic_grid_cache:
            schema = json.loads((seq_path / 'occupancy_schema.json').read_text())
            if not schema.get('panoptic') or schema.get('semantic_taxonomy') != 'farmsim_tree_v2':
                raise ValueError(f'{seq_path}: expected native farmsim_tree_v2 panoptic schema')
            region = next(r for r in schema['regions'] if r['id'] == schema['region_mode'])
            lo = np.asarray(region['min_m_xyz'], dtype=np.float64)
            hi = np.asarray(region['max_m_xyz'], dtype=np.float64)
            shape = np.asarray(region['dimensions_xyz'], dtype=np.int64)
            pitch = (hi - lo) / shape
            target = np.asarray(self.occupancy_target_range)
            sizes = (target[3:] - target[:3]) / pitch
            if not np.allclose(sizes, np.rint(sizes)):
                raise ValueError('Target dimensions must be whole numbers of source voxels')
            indices, inside = [], []
            for axis, n in enumerate(np.rint(sizes).astype(int)):
                centers = target[axis] + (np.arange(n) + .5) * pitch[axis]
                idx = np.floor((centers - lo[axis]) / pitch[axis] + 1e-6).astype(int)
                inside.append((idx >= 0) & (idx < shape[axis]))
                indices.append(idx.clip(0, shape[axis] - 1))
            domain = (inside[0][:, None, None] & inside[1][None, :, None]
                      & inside[2][None, None, :])
            if not domain.all() and not self.allow_unknown_padding:
                raise ValueError('Target exceeds source labels; explicitly enable allow_unknown_padding')
            self._panoptic_grid_cache[seq_path] = (tuple(shape[::-1]), indices, domain)
        return self._panoptic_grid_cache[seq_path]

    def _load_panoptic(self, seq, frame_id):
        path = self._sequence_path(seq)
        shape, indices, domain = self._grid(path)
        with np.load(path / 'occupancy' / f'{frame_id}.npz', allow_pickle=False) as data:
            sem, inst, valid = (data[k] for k in
                                ('occupancy', 'occupancy_instances', 'occupancy_valid'))
            if (sem.shape != shape or inst.shape != shape or valid.shape != shape or
                    sem.dtype != np.uint8 or inst.dtype != np.uint32 or valid.dtype != np.uint8):
                raise ValueError(f'{path}/{frame_id}: invalid panoptic array shape/dtype')
            sem = sem.transpose(2, 1, 0)[np.ix_(*indices)].copy()
            inst = inst.transpose(2, 1, 0)[np.ix_(*indices)].astype(np.int64)
            valid = valid.transpose(2, 1, 0)[np.ix_(*indices)].astype(bool) & domain
        if np.any(valid & (sem != 255) & (sem > 10)):
            raise ValueError('Unexpected semantic ID in native eleven-class labels')
        valid &= sem != 255
        things = np.isin(sem, THING_IDS) & valid
        if np.any(things & ((inst == 0) | (inst == INSTANCE_IGNORE))):
            raise ValueError('Valid thing voxels must have positive instance identities')
        if np.any(valid & ~things & (inst != 0)):
            raise ValueError('Stuff/free must have instance ID zero')
        sem[~valid] = 255
        inst[~valid] = INSTANCE_IGNORE
        return torch.from_numpy(sem.astype(np.int64)), torch.from_numpy(inst)

    def _load_occupancy(self, seq, frame_id):
        # Parent __getitem__ loads the current semantic target through this hook.
        return self._load_panoptic(seq, frame_id)[0]

    def __getitem__(self, index):
        result = super().__getitem__(index)
        seq_idx, frame_idx = self.samples[index]
        seq = self.sequences[seq_idx]
        result['instance'] = self._load_panoptic(seq, seq['frame_ids'][frame_idx])[1].unsqueeze(0)
        return result
