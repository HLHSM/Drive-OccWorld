#!/usr/bin/env python3
"""Diagnostic cross-section row localization, without changing official F1.

These task-oriented exploratory metrics assess finite row support at fixed
lookahead sections. They do not measure full-map reconstruction or driving
success. Defaults are engineering examples, not validation-selected gates.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from row_det.geometric_evaluation import counts_metrics, maximum_matching


def section_rows(rows, x):
    result = []
    for index, row in enumerate(rows):
        points = np.asarray(row['points'], dtype=float)[:, :2]
        # No extension into unsupported space. A row that does not cross this
        # physical section cannot provide a localization observation there.
        crossings = []
        for a, b in zip(points[:-1], points[1:]):
            dx = b[0] - a[0]
            if abs(dx) < 1e-10:
                continue
            fraction = (x - a[0]) / dx
            if -1e-10 <= fraction <= 1 + 1e-10:
                y = float(a[1] + fraction * (b[1] - a[1]))
                angle = float(np.arctan2(b[1] - a[1], dx))
                crossings.append((y, angle))
        if crossings:
            # Degree-1 fits and straight geometric truth cross once. Shared
            # polyline vertices can produce identical duplicate crossings.
            if np.ptp([item[0] for item in crossings]) > 1e-3:
                raise ValueError('A row crosses the same x section repeatedly')
            result.append((index, *crossings[0]))
    return result


def score_section(truth, estimate, x, delta, angle_limit_deg):
    gt, pred = section_rows(truth, x), section_rows(estimate, x)
    distances = np.array([[abs(a[1] - b[1]) for b in pred] for a in gt]).reshape(len(gt), len(pred))
    angular = np.array([[np.rad2deg(np.arccos(np.clip(abs(np.cos(a[2] - b[2])), 0, 1)))
                         for b in pred] for a in gt]).reshape(distances.shape)
    pairs = maximum_matching((distances < delta) & (angular <= angle_limit_deg), distances)
    metrics = counts_metrics(len(pairs), len(pred) - len(pairs), len(gt) - len(pairs))
    metrics.update(gt_crossings=len(gt), pred_crossings=len(pred),
        # Rows not crossing this section are not declared detected at it;
        # full-range omissions remain visible in the unchanged official F1.
        matched_lateral_errors_m=[float(distances[i, j]) for i, j in pairs],
        matched_heading_errors_deg=[float(angular[i, j]) for i, j in pairs])
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('evaluation_dir', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--sections', type=float, nargs='+', default=[3., 5., 8.])
    parser.add_argument('--delta', type=float, default=.2)
    parser.add_argument('--angle-limit-deg', type=float, default=3.)
    args = parser.parse_args()
    if args.delta <= 0 or not 0 < args.angle_limit_deg < 90:
        parser.error('invalid distance/angle gate')
    paths = sorted(p for p in (args.evaluation_dir / 'row_prediction').glob('*.json') if p.stem[:6].isdigit())
    if not paths:
        raise FileNotFoundError('No saved row artifacts')
    accumulators = {side: {str(x): np.zeros(3, dtype=np.int64) for x in args.sections}
                    for side in ('gt', 'pred')}
    samples, digest = {}, hashlib.sha256()
    for path in paths:
        raw = path.read_bytes()
        digest.update(path.name.encode() + b'\0' + raw + b'\0')
        item = json.loads(raw)
        if item['config']['curve_degree'] != 1:
            raise ValueError('This diagnostic currently supports straight fits only')
        frame = {}
        for side in accumulators:
            frame[side] = {}
            for x in args.sections:
                result = score_section(item['current']['geometry']['rows'],
                                       item['current'][side]['rows'], x,
                                       args.delta, args.angle_limit_deg)
                accumulators[side][str(x)] += [result['tp'], result['fp'], result['fn']]
                frame[side][str(x)] = result
        if item['frame']['sample_index'] == 262:
            samples[path.name] = frame
    summary = {side: {x: counts_metrics(*counts) for x, counts in sections.items()}
               for side, sections in accumulators.items()}
    for side, sections in accumulators.items():
        summary[side]['pooled_sections'] = counts_metrics(*sum(sections.values(), np.zeros(3, dtype=np.int64)))
    result = dict(evaluation=str(args.evaluation_dir.resolve()), frames=len(paths),
        source_artifacts_sha256=digest.hexdigest(), sections_x_m=args.sections,
        delta_m=args.delta, angular_gate_deg=args.angle_limit_deg,
        metric='finite_support_cross_section_row_localization',
        exploratory_not_validation_selected=True,
        full_range_scores_unchanged=True, summary=summary, screenshot_sample=samples,
        limitations=['Rows outside a section do not contribute to its denominator.',
                     'Scores at distinct sections do not represent unique physical row counts.',
                     'Cross-section scores do not measure length completeness or closed-loop driving.',
                     'Do not compare the numerical score directly with full-GT-extent F1.'])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'navigation_sections.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(summary=summary, screenshot_sample=samples), indent=2))


if __name__ == '__main__':
    main()
