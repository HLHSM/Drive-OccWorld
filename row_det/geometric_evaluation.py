"""Geometry-grounded crop-row evaluation in a frame's occupancy-grid XY.

The matching direction is reference -> estimate. Reference samples cover the
entire ROI-clipped finite segment, not the intersection with an estimate.
"""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np


def world_xy_to_grid(points, grid_frame_world_ue):
    """Invert a level UE grid transform, converting location cm to metres.

    The export supplies world XY only. Nonlevel grids cannot be unambiguously
    transformed without world Z, so fail rather than inventing an elevation.
    """
    rpy = np.asarray(grid_frame_world_ue['rotation_deg_rpy'], dtype=float)
    origin = np.asarray(grid_frame_world_ue['location_cm'], dtype=float) / 100
    points = np.asarray(points, dtype=float)
    if rpy.shape != (3,) or origin.shape != (3,) or not np.isfinite(np.r_[rpy, origin]).all():
        raise ValueError('invalid grid_frame_world_ue transform')
    if not np.allclose(rpy[:2], 0, atol=1e-6):
        raise ValueError('world XY rows require a yaw-aligned grid (roll=pitch=0)')
    if points.ndim != 2 or points.shape[1] != 2 or not np.isfinite(points).all():
        raise ValueError('world row coordinates must be finite Nx2 values')
    angle = np.deg2rad(rpy[2])
    c, s = np.cos(angle), np.sin(angle)
    # Row vectors multiply R, equivalent to R^T times a column vector.
    return (points - origin[:2]) @ np.array([[c, -s], [s, c]])


def clip_segment(points, xy_range, epsilon=1e-10):
    """Liang-Barsky clip of a finite segment against [xmin,ymin,xmax,ymax]."""
    points = np.asarray(points, dtype=float)
    bounds = np.asarray(xy_range, dtype=float)
    if points.shape != (2, 2) or bounds.shape != (4,) or not np.isfinite(np.r_[points.ravel(), bounds]).all():
        raise ValueError('segment/ROI coordinates must be finite')
    if np.any(bounds[:2] >= bounds[2:]):
        raise ValueError('XY ROI must have positive extent')
    start, end = points
    vector = end - start
    lower, upper = 0.0, 1.0
    for axis in range(2):
        if abs(vector[axis]) < epsilon:
            if start[axis] < bounds[axis] or start[axis] > bounds[axis + 2]:
                return None
        else:
            limits = sorted(((bounds[axis] - start[axis]) / vector[axis],
                             (bounds[axis + 2] - start[axis]) / vector[axis]))
            lower, upper = max(lower, limits[0]), min(upper, limits[1])
            if lower > upper:
                return None
    result = np.array([start + lower * vector, start + upper * vector])
    return result if np.linalg.norm(result[1] - result[0]) > epsilon else None


def clip_polyline(points, xy_range):
    """Keep clipped finite primitives from one detected row as one detection.

    This does not bridge disconnected pieces or count each primitive as an
    independent predicted row. Endpoint-only contact has zero support.
    """
    points = np.asarray(points, dtype=float)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        raise ValueError('row points must be finite Nx2/Nx3 values')
    segments = [clip_segment(points[i:i + 2, :2], xy_range) for i in range(len(points) - 1)]
    return [segment for segment in segments if segment is not None]


@lru_cache(maxsize=8)
def read_layout(path):
    layout = json.loads(Path(path).read_text())
    if layout.get('coordinate_frame') != 'world_ue_xy_m':
        raise ValueError(f'{path}: unsupported coordinate_frame')
    if not layout.get('complete', False):
        raise ValueError(f'{path}: row layout is incomplete')
    return layout


def geometric_rows(metadata, sequence_dir, pc_range):
    annotation = metadata['evaluation_ground_truth']
    if not annotation.get('crop_rows_complete', False):
        raise ValueError('frame crop_rows_complete is false')
    grid = annotation['grid_frame_world_ue']
    # Both metadata blocks, when present, must identify the same label grid.
    occ_grid = metadata.get('semantic_occupancy', {}).get('grid_frame_world_ue')
    if occ_grid is not None:
        for key in ('location_cm', 'rotation_deg_rpy'):
            if not np.allclose(grid[key], occ_grid[key], atol=1e-6):
                raise ValueError('evaluation and semantic occupancy grid frames disagree')
    pc = np.asarray(pc_range, dtype=float)
    bounds = pc[[0, 1, 3, 4]]
    rows, invalid, outside = [], 0, 0
    layout = read_layout(str((Path(sequence_dir) / annotation['crop_rows_file']).resolve()))
    for row in layout['rows']:
        if row.get('fit_valid') is not True:
            invalid += 1
            continue
        world = np.array([row['start_world_xy_m'], row['end_world_xy_m']], dtype=float)
        grid_points = world_xy_to_grid(world, grid)
        clipped = clip_segment(grid_points, bounds)
        if clipped is None:
            outside += 1
            continue
        rows.append(dict(id=row['row_id'], field_id=row.get('field_id'),
                         points=clipped.tolist(), world_points=world.tolist(),
                         length_m=float(np.linalg.norm(clipped[1] - clipped[0]))))
    return rows, dict(layout_rows=len(layout['rows']), fit_invalid_rows=invalid,
                      outside_or_zero_length_rows=outside, in_roi_rows=len(rows),
                      grid_frame_world_ue=grid,
                      layout_file=annotation['crop_rows_file'])


def sample_segments(segments, sample_step):
    """Equidistant arc-length samples, including both finite endpoints."""
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    lengths = np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1)
    keep = lengths > 1e-10
    segments, lengths = segments[keep], lengths[keep]
    if not len(segments):
        return np.empty((0, 2))
    cumulative = np.r_[0, np.cumsum(lengths)]
    distances = np.linspace(0, cumulative[-1], max(2, int(np.ceil(cumulative[-1] / sample_step)) + 1))
    indices = np.clip(np.searchsorted(cumulative, distances, side='right') - 1, 0, len(segments) - 1)
    fraction = ((distances - cumulative[indices]) / lengths[indices])[:, None]
    return segments[indices, 0] + fraction * (segments[indices, 1] - segments[indices, 0])


def point_to_finite_segments(points, segments):
    segments = np.asarray(segments, dtype=float).reshape(-1, 2, 2)
    points = np.asarray(points, dtype=float)
    if not len(segments):
        return np.full(len(points), np.inf)
    vector = segments[:, 1] - segments[:, 0]
    squared = np.sum(vector ** 2, axis=1)
    delta = points[:, None] - segments[None, :, 0]
    t = np.clip(np.sum(delta * vector[None], axis=2) / np.maximum(squared, 1e-20), 0, 1)
    closest = segments[None, :, 0] + t[:, :, None] * vector[None]
    return np.linalg.norm(points[:, None] - closest, axis=2).min(axis=1)


def maximum_matching(eligible, costs=None):
    """Maximum-cardinality bipartite matching, deterministic augmenting paths.

    Shorter-distance edges are tried first; no greedy conflict can remove a
    possible true positive. Matching cardinality defines TP, not a cost gate.
    """
    eligible = np.asarray(eligible, dtype=bool)
    n_ref, n_est = eligible.shape
    costs = np.zeros_like(eligible, dtype=float) if costs is None else np.asarray(costs)
    candidates = [sorted(np.flatnonzero(eligible[i]), key=lambda j: (costs[i, j], j)) for i in range(n_ref)]
    owner = [-1] * n_est

    def augment(i, visited):
        for j in candidates[i]:
            if j in visited:
                continue
            visited.add(j)
            if owner[j] < 0 or augment(owner[j], visited):
                owner[j] = i
                return True
        return False

    for i in range(n_ref):
        augment(i, set())
    return sorted((i, j) for j, i in enumerate(owner) if i >= 0)


def counts_metrics(tp, fp, fn):
    return dict(tp=int(tp), fp=int(fp), fn=int(fn),
                precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None)


def evaluate_rows(reference_rows, estimated_rows, xy_range, thresholds=(.1, .2, .3),
                  sample_step=.05, required_coverage=.85):
    if sample_step <= 0 or not 0 < required_coverage <= 1 or any(t <= 0 for t in thresholds):
        raise ValueError('invalid sampling/matching settings')
    reference = [(i, clip_polyline(row['points'], xy_range)) for i, row in enumerate(reference_rows)]
    estimates = [(i, clip_polyline(row['points'], xy_range)) for i, row in enumerate(estimated_rows)]
    reference = [(i, pieces) for i, pieces in reference if pieces]
    estimates = [(i, pieces) for i, pieces in estimates if pieces]
    samples = [sample_segments(pieces, sample_step) for _, pieces in reference]
    distances = [[point_to_finite_segments(points, pieces) for _, pieces in estimates] for points in samples]
    costs = np.array([[float(d.mean()) for d in row] for row in distances]).reshape(len(reference), len(estimates))
    metrics = {}
    for threshold in thresholds:
        coverage = np.array([[float(np.mean(d < threshold)) for d in row] for row in distances]).reshape(len(reference), len(estimates))
        pairs = maximum_matching(coverage >= required_coverage, costs)
        matches = [dict(reference_index=reference[i][0], estimate_index=estimates[j][0],
                        coverage=float(coverage[i, j]), mean_distance_m=float(costs[i, j]),
                        sample_count=len(samples[i])) for i, j in pairs]
        result = counts_metrics(len(pairs), len(estimates) - len(pairs), len(reference) - len(pairs))
        result.update(matches=matches, reference_count=len(reference), estimate_count=len(estimates))
        metrics[f'{threshold:.2f}'] = result
    return metrics
