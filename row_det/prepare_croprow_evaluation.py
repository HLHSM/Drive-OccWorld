#!/usr/bin/env python3
"""Validate a new crop-row export and prepare the full inference manifest."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.create_farmsim_split import load_sequences
from row_det.geometric_evaluation import geometric_rows


def prepare(data_root, output_dir, pc_range, visualization_count=100):
    data_root, output_dir = Path(data_root).resolve(), Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sequences, excluded = load_sequences(data_root)
    if excluded:
        raise ValueError(f'Incomplete capture sequences: {excluded}')
    frames, geometry_counts = [], []
    for seq_index, sequence in enumerate(sequences):
        seq_path = data_root / sequence['path']
        for frame_index, frame_id in enumerate(sequence['frame_ids']):
            meta = json.loads((seq_path / 'meta' / (frame_id + '.json')).read_text())
            rows, diagnostics = geometric_rows(meta, seq_path, pc_range)
            index = len(frames)
            frames.append(dict(sample_index=index, sequence_index=seq_index,
                sequence_path=sequence['path'], scenario_id=sequence['scenario_id'],
                frame_id=frame_id, crop_type=sequence['crop_type'],
                growth_stage=sequence['growth_stage'], time_of_day=sequence['time_of_day'],
                weather=sequence['weather'], geometry_row_count=len(rows)))
            geometry_counts.append(diagnostics['in_roi_rows'])
    # Same algorithm as tools/test.py: balanced round-robin quotas followed
    # by a centred contiguous window in each sequence. Save the selection.
    grouped = [[frame['sample_index'] for frame in frames if frame['sequence_index'] == i]
               for i in range(len(sequences))]
    quotas, remaining = [0] * len(sequences), min(visualization_count, len(frames))
    while remaining:
        for i, indices in enumerate(grouped):
            if remaining and quotas[i] < len(indices):
                quotas[i] += 1
                remaining -= 1
    selected = []
    for indices, quota in zip(grouped, quotas):
        start = (len(indices) - quota) // 2
        selected.extend(indices[start:start + quota])
    manifest = dict(source_root=str(data_root), sequences=sequences)
    (output_dir / 'evaluation_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    (output_dir / 'frame_index.json').write_text(json.dumps(frames, indent=2) + '\n')
    (output_dir / 'visualization_indices.txt').write_text(''.join(f'{i}\n' for i in selected))
    audit = dict(sequences=len(sequences), frames=len(frames), excluded=excluded,
                 crop_counts=dict(Counter(f['crop_type'] for f in frames)),
                 frames_without_geometric_rows=sum(count == 0 for count in geometry_counts),
                 geometric_rows_total=sum(geometry_counts), point_cloud_range=pc_range,
                 visualization_count=len(selected), visualization_policy='balanced_centred_contiguous_per_sequence')
    (output_dir / 'dataset_audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps(audit, indent=2), flush=True)
    return audit


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--point-cloud-range', type=float, nargs=6,
                        default=[0, -7.5, -2, 15, 7.5, 3])
    parser.add_argument('--visualization-count', type=int, default=100)
    args = parser.parse_args()
    if args.visualization_count < 0:
        parser.error('--visualization-count must be nonnegative')
    prepare(args.data_root, args.output_dir, args.point_cloud_range, args.visualization_count)
