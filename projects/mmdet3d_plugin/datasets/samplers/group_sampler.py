
# Copyright (c) OpenMMLab. All rights reserved.
import math
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from mmcv.runner import get_dist_info
from torch.utils.data import Sampler
from .sampler import SAMPLER
import random


@SAMPLER.register_module()
class DistributedGroupSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self,
                 dataset,
                 samples_per_gpu=1,
                 num_replicas=None,
                 rank=None,
                 seed=0):
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        self.dataset = dataset
        self.samples_per_gpu = samples_per_gpu
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.seed = seed if seed is not None else 0

        assert hasattr(self.dataset, 'flag')
        self.flag = self.dataset.flag
        self.group_sizes = np.bincount(self.flag)

        self.num_samples = 0
        for i, j in enumerate(self.group_sizes):
            self.num_samples += int(
                math.ceil(self.group_sizes[i] * 1.0 / self.samples_per_gpu /
                          self.num_replicas)) * self.samples_per_gpu
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch + self.seed)

        indices = []
        for i, size in enumerate(self.group_sizes):
            if size > 0:
                indice = np.where(self.flag == i)[0]
                assert len(indice) == size
                # add .numpy() to avoid bug when selecting indice in parrots.
                # TODO: check whether torch.randperm() can be replaced by
                # numpy.random.permutation().
                indice = indice[list(
                    torch.randperm(int(size), generator=g).numpy())].tolist()
                extra = int(
                    math.ceil(
                        size * 1.0 / self.samples_per_gpu / self.num_replicas)
                ) * self.samples_per_gpu * self.num_replicas - len(indice)
                # pad indice
                tmp = indice.copy()
                for _ in range(extra // size):
                    indice.extend(tmp)
                indice.extend(tmp[:extra % size])
                indices.extend(indice)

        assert len(indices) == self.total_size

        indices = [
            indices[j] for i in list(
                torch.randperm(
                    len(indices) // self.samples_per_gpu, generator=g))
            for j in range(i * self.samples_per_gpu, (i + 1) *
                           self.samples_per_gpu)
        ]

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset:offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


@SAMPLER.register_module()
class CropAwareDistributedGroupSampler(DistributedGroupSampler):
    """Replace a small quota of low-crop samples with crop-rich frames.

    The epoch length and per-GPU batch size stay identical to the standard
    FarmSim sampler.  The replacement set changes deterministically by epoch,
    and all ranks construct the same global order before taking disjoint parts.
    """

    def __init__(self, dataset, samples_per_gpu=1, num_replicas=None,
                 rank=None, seed=0, crop_stats_file=None,
                 crop_oversample_ratio=0.1, crop_min_voxels=1000):
        super().__init__(dataset, samples_per_gpu, num_replicas, rank, seed)
        if not 0 < crop_oversample_ratio < 1:
            raise ValueError('crop_oversample_ratio must be in (0, 1)')
        if crop_min_voxels < 1:
            raise ValueError('crop_min_voxels must be positive')
        if not crop_stats_file:
            raise ValueError('crop_stats_file is required for crop-aware sampling')
        if not hasattr(dataset, 'samples') or not hasattr(dataset, 'sequences'):
            raise TypeError('Crop-aware sampling requires FarmSimWorldDataset')
        if len(self.group_sizes) != 1:
            raise ValueError('Crop-aware sampling expects one FarmSim image group')
        stats = json.loads(Path(crop_stats_file).read_text(encoding='utf-8'))
        digest = hashlib.sha256(dataset.ann_file.read_bytes()).hexdigest()
        target_range = getattr(dataset, 'occupancy_target_range', None)
        expected_version = 2 if target_range is not None else 1
        if (stats.get('version') != expected_version or
                stats.get('target_point_cloud_range') !=
                (list(target_range) if target_range is not None else None) or
                stats.get('manifest_sha256') != digest or
                stats.get('data_root') != str(dataset.data_root) or
                stats.get('sequence_paths') !=
                [seq['path'] for seq in dataset.sequences]):
            raise ValueError('Crop stats do not match the training manifest/root')
        counts_by_sequence = stats['crop_voxel_counts']
        if (len(counts_by_sequence) != len(dataset.sequences) or
                any(len(counts) != len(seq['frame_ids']) for counts, seq in
                    zip(counts_by_sequence, dataset.sequences))):
            raise ValueError('Crop stats have the wrong sequence/frame counts')
        counts = [counts_by_sequence[seq_idx][frame_index]
                  for seq_idx, frame_index in dataset.samples]
        self.crop_counts = torch.as_tensor(counts, dtype=torch.long)
        self.positive_indices = torch.nonzero(
            self.crop_counts >= crop_min_voxels).flatten()
        if not len(self.positive_indices):
            raise ValueError('No crop-rich samples found at crop_min_voxels')
        self.crop_oversample_ratio = float(crop_oversample_ratio)
        self.crop_min_voxels = int(crop_min_voxels)

    def __iter__(self):
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        size = len(self.dataset)
        indices = torch.randperm(size, generator=generator)
        extra_count = min(size, round(size * self.crop_oversample_ratio))
        low_positions = torch.nonzero(
            self.crop_counts[indices] < self.crop_min_voxels).flatten()
        if len(low_positions) < extra_count:
            # Retain the requested quota even in crop-dense small smoke sets.
            other_positions = torch.nonzero(
                self.crop_counts[indices] >= self.crop_min_voxels).flatten()
            low_positions = torch.cat((low_positions, other_positions))
        selected = low_positions[
            torch.randperm(len(low_positions), generator=generator)[:extra_count]]
        donors = self.positive_indices[torch.randint(
            len(self.positive_indices), (extra_count,), generator=generator)]
        indices[selected] = donors

        # Match DistributedGroupSampler's full-batch DDP padding and rank split.
        indices = indices.tolist()
        padding = self.total_size - size
        if padding:
            indices.extend((indices * math.ceil(padding / size))[:padding])
        batch_order = torch.randperm(
            self.total_size // self.samples_per_gpu,
            generator=generator).tolist()
        indices = [indices[j] for batch in batch_order for j in range(
            batch * self.samples_per_gpu,
            (batch + 1) * self.samples_per_gpu)]
        offset = self.num_samples * self.rank
        return iter(indices[offset:offset + self.num_samples])

