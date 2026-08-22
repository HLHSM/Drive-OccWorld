#!/usr/bin/env python3
"""Remove one crop type from existing FarmSim train/validation manifests.

Unlike recreating a split, this keeps every remaining sequence in its current
train or validation partition and only refreshes the aggregate report.
"""
import argparse
import json
from collections import Counter
from pathlib import Path


FIELDS = ('crop_type', 'growth_stage', 'time_of_day', 'weather')


def summary(rows):
    return dict(
        sequences=len(rows),
        frames=sum(row['frame_count'] for row in rows),
        **{field: dict(sorted(Counter(row[field] for row in rows).items()))
           for field in FIELDS},
    )


def main():
    parser = argparse.ArgumentParser(
        description='Remove a crop from existing FarmSim split manifests.')
    parser.add_argument('--split-dir', type=Path,
                        default=Path('data/farmsim/splits'))
    parser.add_argument('--crop-type', required=True,
                        help='Crop type to remove, for example garlic.')
    args = parser.parse_args()

    crop_type = args.crop_type.casefold()
    manifests = {}
    removed = {}
    for split in ('train', 'val'):
        path = args.split_dir / f'{split}.json'
        with path.open(encoding='utf-8') as f:
            manifest = json.load(f)
        kept = [row for row in manifest['sequences']
                if row['crop_type'].casefold() != crop_type]
        removed[split] = len(manifest['sequences']) - len(kept)
        manifest['sequences'] = kept
        manifests[split] = manifest

    if not any(removed.values()):
        print(f'No {args.crop_type!r} sequences found; manifests unchanged.')
        return

    for split, manifest in manifests.items():
        with (args.split_dir / f'{split}.json').open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write('\n')

    report_path = args.split_dir / 'split_report.json'
    with report_path.open(encoding='utf-8') as f:
        report = json.load(f)
    all_rows = manifests['train']['sequences'] + manifests['val']['sequences']
    report['valid'] = summary(all_rows)
    report['train'] = summary(manifests['train']['sequences'])
    report['val'] = summary(manifests['val']['sequences'])
    report['removed_crop'] = dict(crop_type=args.crop_type, **removed)
    with report_path.open('w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write('\n')

    print(json.dumps(dict(
        removed=removed,
        train=report['train'],
        val=report['val'],
    ), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
