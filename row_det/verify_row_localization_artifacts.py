#!/usr/bin/env python3
"""Recheck metric sources, live geometry, checkpoint hashes and paired cohorts."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.geometric_evaluation import geometric_rows


def signature(path):
    stat = path.stat()
    return dict(name=path.name, size_bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)


def sha256(path):
    result = hashlib.sha256()
    with path.open('rb') as file:
        for block in iter(lambda: file.read(4 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results_root', type=Path)
    parser.add_argument('--paper-dir', type=Path)
    parser.add_argument('--cohorts', nargs='+', choices=['original', 'grid_fixed'], default=['original'])
    args = parser.parse_args()
    report = dict(cohorts={}, world_to_grid_and_ROI_rechecked=True, stale_sources=0)
    label_cache, all_ids = {}, []
    for name in args.cohorts:
        summary = json.loads((args.results_root / name / 'metrics.json').read_text())
        base = Path(summary['evaluation_dir'])
        run = json.loads((base / 'run_config.json').read_text())
        if sha256(Path(summary['checkpoint']['path'])) != summary['checkpoint']['sha256']:
            raise ValueError('Checkpoint hash changed')
        digest, ids, annotation_signatures = hashlib.sha256(), [], 0
        counts = {side: {t: np.zeros(3, dtype=np.int64) for t in ('0.10', '0.20', '0.30')}
                  for side in ('gt', 'pred')}
        for path in sorted((base / 'row_prediction').glob('*.json')):
            if not path.stem[:6].isdigit():
                continue
            raw = path.read_bytes()
            digest.update(path.name.encode() + b'\0' + raw + b'\0')
            item = json.loads(raw)
            frame = item['frame']
            ids.append((frame['sample_index'], frame['sequence_path'], frame['frame_id']))
            assert signature(base / 'prediction' / item['source']['name']) == item['source'], path
            annotations = item.get('annotation_source')
            sequence = Path(run['data_root']) / frame['sequence_path']
            meta_path = sequence / 'meta' / (frame['frame_id'] + '.json')
            metadata = json.loads(meta_path.read_text())
            layout = sequence / metadata['evaluation_ground_truth']['crop_rows_file']
            if annotations is not None:
                assert signature(meta_path) == annotations['metadata']
                assert signature(layout) == annotations['layout']
                annotation_signatures += 1
            # Even legacy artifacts without dependency signatures are checked
            # against live metadata/world annotations, rather than trusted.
            rows, _ = geometric_rows(metadata, sequence, item['point_cloud_range'])
            saved = item['current']['geometry']['rows']
            assert len(rows) == len(saved)
            for a, b in zip(rows, saved):
                assert a['id'] == b['id']
                np.testing.assert_allclose(a['points'], b['points'], atol=1e-9, rtol=0)
            paired = json.loads((args.results_root / name / 'per_frame' / path.name).read_text())
            assert paired['frame'] == frame
            for side in counts:
                for threshold, m in paired['localization'][side].items():
                    assert m['tp'] + m['fn'] == len(saved)
                    assert m['tp'] + m['fp'] == len(item['current'][side]['rows'])
                    counts[side][threshold] += [m['tp'], m['fp'], m['fn']]
            if path.name in label_cache:
                assert label_cache[path.name] == item['current']['gt']['rows']
            else:
                label_cache[path.name] = item['current']['gt']['rows']
        assert digest.hexdigest() == summary['source_artifacts_sha256']
        assert len(ids) == summary['frames']
        for side, thresholds in counts.items():
            for threshold, count in thresholds.items():
                m = summary['results']['overall']['all']['localization'][side][threshold]
                np.testing.assert_array_equal(count, [m['tp'], m['fp'], m['fn']])
        all_ids.append(ids)
        report['cohorts'][name] = dict(frames=len(ids), checkpoint_sha256=summary['checkpoint']['sha256'],
            row_artifacts_sha256=digest.hexdigest(), annotations_with_dependency_signatures=annotation_signatures,
            source_signature_geometry_and_aggregation_checks='pass')
    if len(all_ids) > 1:
        assert all(ids == all_ids[0] for ids in all_ids[1:])
        report.update(paired_cohort_identical=True, label_fits_identical=True)
    output = json.dumps(report, indent=2) + '\n'
    (args.results_root / 'artifact_verification.json').write_text(output)
    if args.paper_dir:
        (args.paper_dir / 'data' / 'crop_row_artifact_verification.json').write_text(output)
    print(output)


if __name__ == '__main__':
    main()
