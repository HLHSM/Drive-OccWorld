#!/usr/bin/env python3
"""Save the exact official-test/test-farm columns used by the paper table."""

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.evaluate_orad3d_experiments import metrics_from_pickle


def summarize(work_dir, training_seconds):
    test = metrics_from_pickle(work_dir / 'test_metrics.pkl')
    farm = metrics_from_pickle(work_dir / 'test_farm_metrics.pkl')
    return dict(
        experiment=work_dir.name,
        training_seconds=training_seconds,
        training_minutes=training_seconds / 60.0,
        timing_scope='target training including epoch validation; excluding '
                     'source pretraining, checkpoint conversion and final tests',
        paper_columns_percent={
            'Test (w/o Free)': 100.0 * test['semantic_mIoU_without_free'],
            'Test': 100.0 * test['semantic_mIoU_all_classes'],
            'Test_farm': 100.0 * farm['semantic_mIoU'],
        },
        test_metrics_fraction=test,
        test_farm_metrics_fraction=farm,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--training-seconds', type=int, required=True)
    args = parser.parse_args()
    if args.training_seconds < 0:
        parser.error('--training-seconds must be non-negative')
    result = summarize(args.work_dir, args.training_seconds)
    path = args.work_dir / 'paper_transfer_metrics.json'
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    print(f'Wrote {path}')


if __name__ == '__main__':
    main()
