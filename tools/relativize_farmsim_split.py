#!/usr/bin/env python3
"""Convert existing FarmSim split manifests to portable relative paths."""

import argparse
import json
from pathlib import Path


def relative_path(value, data_root):
    path = Path(value).expanduser()
    if path.is_absolute():
        try:
            return str(path.resolve().relative_to(data_root))
        except ValueError:
            raise ValueError(f'{path} is outside --data-root {data_root}')
    return str(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-dir', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    for name in ('train.json', 'val.json', 'split_report.json'):
        path = args.split_dir / name
        if not path.is_file():
            continue
        with path.open(encoding='utf-8') as f:
            manifest = json.load(f)
        manifest['source_root'] = '.'
        for row in manifest.get('sequences', []):
            row['path'] = relative_path(row['path'], data_root)
        for row in manifest.get('excluded', []):
            if 'path' in row:
                row['path'] = relative_path(row['path'], data_root)
        with path.open('w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
            f.write('\n')
        print(f'updated {path}')


if __name__ == '__main__':
    main()
