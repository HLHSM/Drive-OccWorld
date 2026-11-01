#!/usr/bin/env python3
"""Index saved row/path results and link a navigation-only 3D preview gallery."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.generate_row_predictions import source_signature, write_json


def link_file(source, destination):
    if destination.is_symlink():
        if destination.resolve() == source.resolve():
            return
        destination.unlink()
    elif destination.exists():
        raise FileExistsError(f'Refusing to replace a regular file: {destination}')
    destination.symlink_to(source.resolve())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('--row-prediction-dir', type=Path,
                        help='overlay directory, defaults to evaluation_dir/row_prediction')
    parser.add_argument('--reference-kind', choices=['centerline', 'checked'], default='centerline')
    args = parser.parse_args()
    base = args.evaluation_dir.resolve()
    paths = sorted((base / 'prediction').glob('*.npz'))
    if not paths:
        raise FileNotFoundError('No prediction NPZs')
    run = json.loads((base / 'run_config.json').read_text())
    row_dir = (args.row_prediction_dir or base / 'row_prediction').resolve()
    centerlines = args.reference_kind == 'centerline'
    field = 'geometric_centerline' if centerlines else 'reference_path'
    gallery = base / ('centerline_preview' if centerlines else 'navigation_preview')
    for subdir in ('prediction', 'row_prediction'):
        (gallery / subdir).mkdir(parents=True, exist_ok=True)
    records, selected, both = [], [], []
    counts = {side: Counter() for side in ('gt', 'pred')}
    expected_names = set()
    for path in paths:
        row_path = row_dir / (path.stem + '.json')
        artifact = json.loads(row_path.read_text())
        if artifact.get('source') != source_signature(path):
            raise ValueError(f'Stale row overlay: {row_path}')
        record = dict(name=path.stem)
        for side in counts:
            result = artifact['current'][side]
            if field not in result:
                raise ValueError(f'{field} missing in {row_path}; regenerate row results')
            ref = result[field]
            available = len(ref.get('points', [])) >= 2
            if available and ref['status'] not in ('geometric_centerline', 'geometry_only', 'clearance_checked'):
                raise ValueError(f'Unexpected path status: {row_path}')
            record.update({f'{side}_rows': len(result['rows']), f'{side}_reference_status': ref['status'],
                           f'{side}_reference_available': available})
            counts[side][ref['status']] += 1
        records.append(record)
        if record['pred_reference_available']:
            link_file(path, gallery / 'prediction' / path.name)
            link_file(row_path, gallery / 'row_prediction' / row_path.name)
            selected.append(path.stem)
            expected_names.add(path.stem)
            if record['gt_reference_available']:
                both.append(path.stem)
    # Only remove stale links created for this derived gallery, never source data.
    for subdir, suffix in [('prediction', '.npz'), ('row_prediction', '.json')]:
        for path in (gallery / subdir).glob('*' + suffix):
            if path.is_symlink() and path.stem not in expected_names:
                path.unlink()
    prefix = 'centerline' if centerlines else 'navigation'
    with (base / (prefix + '_reference_index.csv')).open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    report = dict(evaluation_dir=str(base), checkpoint=run['checkpoint'], frames=len(paths),
        gt_reference_status_counts=dict(counts['gt']), pred_reference_status_counts=dict(counts['pred']),
        navigation_preview_frames=len(selected), both_sides_have_reference_frames=len(both),
        navigation_preview_dir=str(gallery), selected_names=selected, both_sides_names=both,
        row_prediction_dir=str(row_dir), reference_field=field,
        description='Derived links for choosing 3D views, not an evaluation subset.',
        reference_type='row geometry only; no traversability checks' if centerlines else 'reference with corridor checks')
    write_json(base / ('centerline_browser_manifest.json' if centerlines else 'browser_manifest.json'), report)
    write_json(gallery / 'manifest.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('selected_names', 'both_sides_names')}, indent=2))


if __name__ == '__main__':
    main()
