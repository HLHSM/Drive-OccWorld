#!/usr/bin/env python3
"""Select calibration adaptation by original validation crop IoU, never row-test F1."""
import argparse
import json
from pathlib import Path


def select(root):
    root = Path(root).resolve()
    logs = list(root.glob('*.log.json'))
    if len(logs) != 1:
        raise ValueError(f'expected one training metric log, got {logs}')
    candidates = []
    for line in logs[0].read_text().splitlines():
        record = json.loads(line)
        if record.get('mode') == 'val' and 'occ_current_IoU_crop' in record:
            epoch = int(record['epoch'])
            checkpoint = root / f'epoch_{epoch}.pth'
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            candidates.append(dict(epoch=epoch, path=str(checkpoint),
                val_crop_iou=record['occ_current_IoU_crop'],
                val_miou=record['occ_current_mIoU']))
    if len(candidates) != 2:
        raise ValueError('expected both predeclared adaptation epochs to finish validation')
    best = max(candidates, key=lambda x: (x['val_crop_iou'], -x['epoch']))
    result = dict(selection='maximum original-validation crop voxel IoU; earliest epoch on ties',
                  evaluation_dataset_used_for_selection=False, candidates=candidates, selected=best)
    (root / 'selected_checkpoint.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('adaptation_dir', type=Path)
    print(json.dumps(select(parser.parse_args().adaptation_dir), indent=2))
