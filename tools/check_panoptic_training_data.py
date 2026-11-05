#!/usr/bin/env python3
"""Check native targets and query capacity over all training frames; cache by file stats."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sitecustomize  # noqa: F401
import numpy as np
from projects.mmdet3d_plugin.datasets.farmsim_panoptic_dataset import FarmSimPanopticDataset, THING_IDS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--data-root', required=True)
    parser.add_argument('--num-queries', type=int, default=256)
    parser.add_argument('--sample-stride', type=int, default=2)
    parser.add_argument('--output', default='work_dirs/panoptic_6cam_training_data.json')
    args = parser.parse_args()
    ds = FarmSimPanopticDataset(args.manifest, data_root=args.data_root, queue_length=0,
        camera_mode='surround', camera_geometry='grid_world',
        occupancy_target_range=[-20, -10, -2, 20, 10, 3],
        allow_unknown_padding=False, sample_stride=args.sample_stride)
    digest = hashlib.sha256(Path(args.manifest).read_bytes() + Path(__file__).read_bytes())
    digest.update(str(args.sample_stride).encode())
    for seq in ds.sequences:
        path = ds._sequence_path(seq)
        digest.update((path/'occupancy_schema.json').read_bytes())
        for frame in seq['frame_ids']:
            p = path/'occupancy'/f'{frame}.npz'
            st = p.stat()
            digest.update(f'{p}:{st.st_size}:{st.st_mtime_ns}'.encode())
    signature = digest.hexdigest()
    output = Path(args.output)
    if output.exists():
        report = json.loads(output.read_text())
        if report.get('signature') == signature:
            if report['max_instances'] > args.num_queries:
                raise ValueError(f"Need >= {report['max_instances']} queries")
            print(f'Cached panoptic check: {report["frames"]} frames, max instances={report["max_instances"]}', flush=True)
            return
    distribution, counts = [], {str(c): dict(frames=0, instances=0, voxels=0) for c in THING_IDS}
    max_frame = None
    max_count = -1
    def check_sample(sample):
        seq_idx, frame_idx = sample
        seq = ds.sequences[seq_idx]
        frame = seq['frame_ids'][frame_idx]
        path = ds._sequence_path(seq)
        shape, _, _ = ds._grid(path)
        ix = source_indices[path]
        with np.load(path/'occupancy'/f'{frame}.npz', allow_pickle=False) as archive:
            sem, inst, valid = (archive[k] for k in ('occupancy', 'occupancy_instances', 'occupancy_valid'))
        if (sem.shape != shape or inst.shape != shape or valid.shape != shape or
                sem.dtype != np.uint8 or inst.dtype != np.uint32 or valid.dtype != np.uint8):
            raise ValueError(f'{path}/{frame}: malformed panoptic arrays')
        # Count source voxels inside the target, without allocating unknown padding.
        sem, inst, valid = sem[ix], inst[ix], valid[ix].astype(bool)
        valid &= sem != 255
        if np.any(valid & (sem > 10)):
            raise ValueError(f'{path}/{frame}: invalid semantic ID')
        things = np.isin(sem, THING_IDS) & valid
        if np.any(things & ((inst == 0) | (inst == 4294967295))):
            raise ValueError(f'{path}/{frame}: missing thing identity')
        if np.any(valid & ~things & (inst != 0)):
            raise ValueError(f'{path}/{frame}: nonzero stuff identity')
        labs = sem[things]
        keys = np.unique(inst[things].astype(np.int64) * 11 + labs)
        objects = np.bincount(keys % 11, minlength=11)
        voxels = np.bincount(labs, minlength=11)
        result = {str(c): (int(objects[c]), int(voxels[c])) for c in THING_IDS}
        return result, f'{seq["path"]}/{frame}'

    # Threads overlap archive inflation and array operations; keep memory bounded.
    source_indices = {}
    for seq in ds.sequences:
        path = ds._sequence_path(seq)
        _, indices, domain = ds._grid(path)
        axes = [np.unique(indices[a][domain.any(axis=tuple(i for i in range(3) if i != a))]) for a in range(3)]
        source_indices[path] = np.ix_(axes[2], axes[1], axes[0])
    with ThreadPoolExecutor(max_workers=8) as executor:
        # Submit bounded waves: no 50k-future queue and no unbounded completed arrays.
        for start in range(0, len(ds.samples), 128):
            for result, frame in executor.map(check_sample, ds.samples[start:start+128]):
                count = sum(n for n, _ in result.values())
                for c, (n, voxels) in result.items():
                    stats = counts[c]
                    stats['frames'] += int(n > 0)
                    stats['instances'] += n
                    stats['voxels'] += voxels
                distribution.append(count)
                if count > max_count:
                    max_count, max_frame = count, frame
            if start % 2048 == 0:
                print(f'Checked {len(distribution)}/{len(ds)} frames; max instances={max_count}', flush=True)
    report = dict(signature=signature, frames=len(ds), max_instances=max_count,
                  max_frame=max_frame, per_class=counts,
                  instance_count_percentiles=np.percentile(distribution, [0, 50, 90, 95, 99, 100]).tolist(),
                  target_range=list(ds.occupancy_target_range), sample_stride=args.sample_stride,
                  full_frames=sum(len(s['frame_ids']) for s in ds.sequences), unknown_padding_y=[])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if max_count > args.num_queries:
        raise ValueError(f'Need >= {max_count} queries; update config and NUM_QUERIES')


if __name__ == '__main__':
    main()
