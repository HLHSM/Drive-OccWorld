#!/usr/bin/env python3
"""Create a traceable FarmSim qualitative-comparison sample manifest.

The reference AgriOcc NPZ artifacts define the exact validation frames.  The
manifest then drives both Drive-OccWorld and official SurroundOcc exports, so
each method is rendered from the same 3-D ground-truth scene.
"""

import argparse
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference_prediction_dir', type=Path,
                        help='AgriOcc directory containing saved NPZ artifacts')
    parser.add_argument('farmsim_info_pkl', type=Path,
                        help='FarmSim front3 validation-info pickle')
    parser.add_argument('output_dir', type=Path,
                        help='directory for the manifest and filtered info pickle')
    parser.add_argument('--irwm-split', type=Path,
                        default=Path('data/farmsim/splits/val.json'),
                        help='FarmSim split used by IR-WM (default: data/farmsim/splits/val.json)')
    parser.add_argument('--irwm-queue-length', type=int, default=2,
                        help='IR-WM history queue length (default: 2)')
    parser.add_argument('--irwm-future-length', type=int, default=6,
                        help='maximum future frames reserved by IR-WM test data (default: 6)')
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(args.reference_prediction_dir.glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'No NPZ artifacts in {args.reference_prediction_dir}')

    selections = []
    for path in paths:
        with np.load(path, allow_pickle=False) as artifact:
            if 'sample_index' not in artifact:
                raise KeyError(f'{path} does not contain sample_index')
            selections.append((int(artifact['sample_index']), path.name))
    selections.sort()
    indices = [index for index, _ in selections]
    if len(indices) != len(set(indices)):
        raise ValueError('Reference artifacts contain duplicate sample_index values')

    with args.farmsim_info_pkl.open('rb') as handle:
        source = pickle.load(handle)
    infos = source.get('infos')
    if infos is None:
        raise KeyError(f'{args.farmsim_info_pkl} does not contain an infos list')
    if indices[0] < 0 or indices[-1] >= len(infos):
        raise IndexError('Reference sample_index is outside the FarmSim info list')

    selected_infos = [infos[index] for index in indices]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index_file = args.output_dir / 'selected_indices.txt'
    index_file.write_text('\n'.join(map(str, indices)) + '\n', encoding='utf-8')

    selected_info_pkl = args.output_dir / 'surroundocc_val_selected.pkl'
    filtered = dict(source)
    filtered['infos'] = selected_infos
    with selected_info_pkl.open('wb') as handle:
        pickle.dump(filtered, handle, protocol=pickle.HIGHEST_PROTOCOL)

    rows = []
    for (index, filename), info in zip(selections, selected_infos):
        rows.append(dict(sample_index=index, reference_npz=filename,
                         token=info['token'], sequence_path=info['sequence_path'],
                         frame_id=str(info['frame_id']), occ_path=info['occ_path']))
    manifest_path = args.output_dir / 'manifest.json'
    manifest_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n',
                             encoding='utf-8')

    # IR-WM is evaluated with history and future queues, so its dataset index
    # differs from the current-frame AgriOcc index.  Derive the index from the
    # same sequence/frame identity rather than assuming a fixed offset.
    world_manifest = json.loads(args.irwm_split.read_text(encoding='utf-8'))
    source_root = Path(world_manifest.get('source_root', '.'))
    if not source_root.is_absolute():
        source_root = (args.irwm_split.parent.parent.parent / source_root).resolve()
    def sequence_key(path):
        """Use the stable scene/sequence suffix across absolute and relative roots."""
        parts = Path(path).parts
        if len(parts) < 2:
            raise ValueError(f'Expected scene/sequence path, received {path}')
        return tuple(parts[-2:])

    target_keys = {
        (sequence_key(row['sequence_path']), row['frame_id']): row['sample_index']
        for row in rows
    }
    irwm_by_agri_index = {}
    irwm_index = 0
    for sequence in world_manifest['sequences']:
        sequence_path = Path(sequence['path'])
        if not sequence_path.is_absolute():
            sequence_path = (source_root / sequence_path).resolve()
        frame_ids = sequence['frame_ids']
        for frame_pos in range(args.irwm_queue_length,
                               len(frame_ids) - args.irwm_future_length):
            key = (sequence_key(sequence_path), str(frame_ids[frame_pos]))
            if key in target_keys:
                irwm_by_agri_index[target_keys[key]] = irwm_index
            irwm_index += 1
    missing = sorted(set(indices).difference(irwm_by_agri_index))
    if missing:
        raise ValueError('IR-WM cannot evaluate selected AgriOcc samples: '
                         f'{missing[:10]}{"..." if len(missing) > 10 else ""}')
    irwm_indices = [irwm_by_agri_index[index] for index in indices]
    irwm_index_file = args.output_dir / 'irwm_selected_indices.txt'
    irwm_index_file.write_text('\n'.join(map(str, irwm_indices)) + '\n',
                               encoding='utf-8')
    print(f'Prepared {len(rows)} matched FarmSim samples.')
    print(f'  indices: {index_file}')
    print(f'  SurroundOcc infos: {selected_info_pkl}')
    print(f'  manifest: {manifest_path}')
    print(f'  IR-WM indices: {irwm_index_file}')


if __name__ == '__main__':
    main()
