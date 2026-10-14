"""Metric crop-row models and local inter-row references (no GT dependency).

Arrays follow FarmSim's [x, y, z] convention; output points are in metres in
the same ego frame. This is a per-frame geometric baseline, not a controller.
"""

from dataclasses import asdict, dataclass

import numpy as np

ALGORITHM_VERSION = 'projection-ransac-v1'


def nearest_cells(mask, pitch=(1.0, 1.0)):
    """Exact separable 2D Euclidean distance and nearest true-cell indices.

    NumPy only, with bounded temporary batches for the small occupancy BEV.
    Unlike a cell-count distance, this respects unequal metric voxel pitches.
    """
    nx, ny = mask.shape
    if not mask.any():
        raise ValueError('nearest_cells needs at least one true cell')
    initial = np.where(mask, 0.0, np.inf)
    vertical, source_x = np.empty_like(initial), np.empty(mask.shape, dtype=int)
    xs, ys = np.arange(nx), np.arange(ny)
    for start in range(0, nx, 32):
        cost = ((xs[start:start + 32, None, None] - xs[None, :, None]) * pitch[0]) ** 2 + initial[None, :, :]
        vertical[start:start + 32] = cost.min(axis=1)
        source_x[start:start + 32] = cost.argmin(axis=1)
    horizontal = ((ys[:, None] - ys[None, :]) * pitch[1]) ** 2
    distances, source_y = np.empty_like(initial), np.empty(mask.shape, dtype=int)
    for start in range(0, nx, 32):
        cost = vertical[start:start + 32, None, :] + horizontal[None, :, :]
        distances[start:start + 32] = np.sqrt(cost.min(axis=2))
        source_y[start:start + 32] = cost.argmin(axis=2)
    source_x = source_x[np.arange(nx)[:, None], source_y]
    return distances, np.stack((source_x, source_y))


def _smooth(values, sigma):
    radius = max(1, int(np.ceil(4 * sigma)))
    kernel = np.exp(-.5 * (np.arange(-radius, radius + 1) / sigma) ** 2)
    kernel /= kernel.sum()
    return np.convolve(np.pad(values, radius, mode='edge'), kernel, mode='valid')


def _peaks(values, spacing, height, prominence):
    """Prominent local maxima followed by strongest-first peak suppression."""
    local = np.flatnonzero((values[1:-1] > values[:-2]) &
                           (values[1:-1] >= values[2:])) + 1
    selected = []
    for index in sorted(local, key=lambda i: (-values[i], i)):
        if values[index] < height or any(abs(index - p) < spacing for p in selected):
            continue
        left, right = index, index
        while left > 0 and values[left - 1] <= values[index]:
            left -= 1
        while right < len(values) - 1 and values[right + 1] <= values[index]:
            right += 1
        base = max(values[left:index + 1].min(), values[index:right + 1].min())
        if values[index] - base >= prominence:
            selected.append(index)
    return np.array(sorted(selected), dtype=int)


@dataclass
class RowConfig:
    crop_id: int = 1
    z_min: float = -2.0
    z_max: float = 3.0
    max_angle_deg: float = 60.0
    angle_step_deg: float = 1.0
    heading_prior_strength: float = 0.5
    min_row_spacing: float = 0.45
    row_band: float = 0.35
    min_crop_cells: int = 12
    min_row_points: int = 8
    min_row_length: float = 2.0
    slice_length: float = 0.5
    residual_threshold: float = 0.2
    curve_degree: int = 1
    sample_step: float = 0.2
    min_path_length: float = 2.0
    vehicle_width: float = 0.0
    vehicle_height: float = 1.5
    clearance_margin: float = 0.1
    seed: int = 0

    def validate(self):
        positive = ('angle_step_deg', 'min_row_spacing', 'row_band',
                    'min_row_length', 'slice_length', 'residual_threshold',
                    'sample_step', 'min_path_length', 'vehicle_height')
        for name in positive:
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if not 0 < self.max_angle_deg < 80:
            raise ValueError('max_angle_deg must be in (0, 80)')
        if not 0 <= self.heading_prior_strength <= 1:
            raise ValueError('heading_prior_strength must be in [0, 1]')
        if not np.isfinite([self.z_min, self.z_max]).all() or self.z_min >= self.z_max:
            raise ValueError('z_min must be smaller than z_max')
        if self.curve_degree not in (1, 2):
            raise ValueError('curve_degree must be 1 or 2')
        if self.min_crop_cells < 1 or self.min_row_points < 2:
            raise ValueError('crop/row point counts must be positive (row >= 2)')
        if not np.isfinite([self.vehicle_width, self.clearance_margin]).all():
            raise ValueError('vehicle dimensions must be finite')
        if self.vehicle_width < 0 or self.clearance_margin < 0:
            raise ValueError('vehicle width and margin must be nonnegative')


def grid_geometry(labels, pc_range):
    pc_range = np.asarray(pc_range, dtype=float)
    if labels.ndim != 3 or min(labels.shape) < 1:
        raise ValueError('occupancy must be a nonempty [x, y, z] array')
    if pc_range.shape != (6,) or not np.isfinite(pc_range).all():
        raise ValueError('point_cloud_range must contain six finite numbers')
    pitch = (pc_range[3:] - pc_range[:3]) / np.asarray(labels.shape)
    if (pitch <= 0).any():
        raise ValueError('point_cloud_range must have positive extents')
    axes = [pc_range[i] + (np.arange(labels.shape[i]) + .5) * pitch[i]
            for i in range(3)]
    return pc_range, pitch, axes


def _profile(points, weights, angle, bin_size):
    lateral = -points[:, 0] * np.sin(angle) + points[:, 1] * np.cos(angle)
    lo = np.floor(lateral.min() / bin_size) * bin_size - 3 * bin_size
    hi = np.ceil(lateral.max() / bin_size) * bin_size + 4 * bin_size
    edges = np.arange(lo, hi + bin_size / 2, bin_size)
    values, _ = np.histogram(lateral, bins=edges, weights=weights)
    values = _smooth(values.astype(float), sigma=max(1.0, .1 / bin_size))
    return (edges[:-1] + edges[1:]) / 2, values


def _robust_fit(u, v, weights, cfg, rng):
    """RANSAC over slice centres, followed by weighted least squares."""
    degree = cfg.curve_degree
    if len(u) < degree + 2:
        return None
    best_mask, best_score = None, -1.0
    for _ in range(128):
        indices = rng.choice(len(u), degree + 1, replace=False)
        if np.ptp(u[indices]) < cfg.slice_length:
            continue
        coefficients = np.polynomial.polynomial.polyfit(u[indices], v[indices], degree)
        residual = np.abs(v - np.polynomial.polynomial.polyval(u, coefficients))
        mask = residual <= cfg.residual_threshold
        score = weights[mask].sum()
        if mask.sum() >= degree + 2 and score > best_score:
            best_mask, best_score = mask, score
    if best_mask is None or best_mask.mean() < .5:
        return None
    coefficients = np.polynomial.polynomial.polyfit(
        u[best_mask], v[best_mask], degree, w=np.sqrt(weights[best_mask]))
    residual = v - np.polynomial.polynomial.polyval(u, coefficients)
    return coefficients, best_mask, float(np.sqrt(np.average(
        residual[best_mask] ** 2, weights=weights[best_mask])))


def _indices(xy, pc_range, pitch, shape):
    indices = np.floor((xy - pc_range[:2]) / pitch[:2]).astype(int)
    inside = ((indices >= 0) & (indices < np.asarray(shape[:2]))).all(axis=1)
    indices = np.clip(indices, 0, np.asarray(shape[:2]) - 1)
    return indices, inside


def _ground(labels, axes, pitch):
    ground = (labels == 2) | (labels == 3)
    heights = np.max(np.where(ground, axes[2][None, None, :], -np.inf), axis=2)
    known = np.isfinite(heights)
    if known.all():
        filled = heights
    elif known.any():
        _, nearest = nearest_cells(known, pitch[:2])
        filled = heights[tuple(nearest)]
    else:
        filled = np.zeros(labels.shape[:2], dtype=float)
    return filled, known


def _xyz(xy, labels, pc_range, pitch, ground_height):
    indices, _ = _indices(xy, pc_range, pitch, labels.shape)
    # Ground labels identify voxel centres, so lift lines to the top surface.
    z = ground_height[tuple(indices.T)] + pitch[2] / 2 + .05
    return np.column_stack((xy, z)).round(4).tolist()


def _reference(rows, labels, pc_range, pitch, ground_height, ground_known, cfg):
    result = dict(status='no_bracketing_rows', points=[], row_ids=[],
                  vehicle_clearance_checked=cfg.vehicle_width > 0,
                  vehicle_width_m=cfg.vehicle_width,
                  clearance_margin_m=cfg.clearance_margin)
    if len(rows) < 2:
        result['status'] = 'insufficient_rows'
        return result
    # Ground clearance is conservative: all predicted crop/vegetation/obstacle
    # heights block the column; ground and air alone do not block it.
    z = pc_range[2] + (np.arange(labels.shape[2]) + .5) * pitch[2]
    surface = ground_height + pitch[2] / 2
    vehicle_volume = ((z[None, None, :] >= surface[:, :, None]) &
                      (z[None, None, :] <= surface[:, :, None] + cfg.vehicle_height))
    # FarmSim deliberately ignores underground voxels. Only unknown voxels in
    # the above-ground vehicle volume block navigation; unknown ground still
    # blocks the whole column. Crop/obstacle checks retain all canopy heights.
    blocked = (np.isin(labels, [cfg.crop_id, 4, 5]).any(axis=2) |
               ((labels == 255) & vehicle_volume).any(axis=2) | ~ground_known)
    # Padding prevents a vehicle footprint from running outside the ROI.
    clearance = nearest_cells(np.pad(blocked, 1, constant_values=True), pitch[:2])[0][1:-1, 1:-1]
    candidates, rejected = [], []
    for left, right in zip(rows, rows[1:]):
        lp, rp = np.asarray(left['points']), np.asarray(right['points'])
        x_start = max(0.0, pc_range[0], lp[0, 0], rp[0, 0])
        x_end = min(pc_range[3], lp[-1, 0], rp[-1, 0])
        if x_end - x_start < cfg.min_path_length:
            continue
        x = np.linspace(x_start, x_end, max(2, int(np.ceil((x_end - x_start) / cfg.sample_step)) + 1))
        yl, yr = np.interp(x, lp[:, 0], lp[:, 1]), np.interp(x, rp[:, 0], rp[:, 1])
        # Project the supported curves' near-end tangent to the ego origin;
        # choose the corridor containing the vehicle, not an arbitrary row pair.
        y0_left = yl[0] - (yl[1] - yl[0]) / (x[1] - x[0]) * x[0]
        y0_right = yr[0] - (yr[1] - yr[0]) / (x[1] - x[0]) * x[0]
        if not y0_left <= 0 <= y0_right:
            continue
        xy = np.column_stack((x, (yl + yr) / 2))
        indices, inside = _indices(xy, pc_range, pitch, labels.shape)
        ix, iy = indices.T
        required = cfg.vehicle_width / 2 + cfg.clearance_margin
        # Grid-centre distance can overestimate free space by half a cell's
        # diagonal. Include that discretisation allowance in the clearance.
        allowance = np.linalg.norm(pitch[:2]) / 2
        corridor_width = (yr - yl) - (left['half_width_m'] + right['half_width_m'])
        valid = (inside & (yr > yl) &
                 (corridor_width >= 2 * required) &
                 (clearance[ix, iy] >= required + allowance) &
                 ground_known[ix, iy])
        # Retain only the continuous prefix; never jump over an obstacle.
        end = int(np.flatnonzero(~valid)[0]) if (~valid).any() else len(x)
        if end < 2 or x[end - 1] - x[0] < cfg.min_path_length:
            rejected.append('blocked_or_unknown_corridor')
            continue
        candidates.append((abs((y0_left + y0_right) / 2), dict(
            status='clearance_checked' if cfg.vehicle_width > 0 else 'geometry_only',
            points=_xyz(xy[:end], labels, pc_range, pitch, ground_height),
            row_ids=[left['id'], right['id']],
            vehicle_clearance_checked=cfg.vehicle_width > 0,
            vehicle_width_m=cfg.vehicle_width, clearance_margin_m=cfg.clearance_margin,
            min_center_clearance_m=round(float(clearance[ix[:end], iy[:end]].min() - allowance), 4),
            min_corridor_width_m=round(float(corridor_width[:end].min()), 4),
            truncated=end < len(x), ground_coverage=1.0)))
    if candidates:
        return min(candidates, key=lambda candidate: candidate[0])[1]
    if rejected:
        result['status'] = rejected[0]
    return result


def detect_rows(labels, point_cloud_range, config=None):
    """Return JSON-safe rows, quality diagnostics, and an inter-row reference.

    Prediction must be passed directly; no GT validity mask is used here.
    Missing ground, unknown columns, or an obstruction suppress/truncate the
    reference, but never hide successfully fitted crop rows.
    """
    cfg = config or RowConfig()
    cfg.validate()
    labels = np.asarray(labels)
    pc_range, pitch, axes = grid_geometry(labels, point_cloud_range)
    ground_height, ground_known = _ground(labels, axes, pitch)
    crop = (labels == cfg.crop_id) & ((axes[2] >= cfg.z_min) & (axes[2] <= cfg.z_max))[None, None, :]
    density = crop.sum(axis=2)
    indices = np.argwhere(density > 0)
    result = dict(rows=[], reference_path=dict(status='insufficient_crop', points=[], row_ids=[],
                  vehicle_clearance_checked=cfg.vehicle_width > 0),
                  diagnostics=dict(crop_voxels=int(crop.sum()), crop_cells=len(indices),
                                   direction_deg=None, status='insufficient_crop'))
    if len(indices) < cfg.min_crop_cells:
        return result
    points = pc_range[:2] + (indices + .5) * pitch[:2]
    # Square-root weights preserve height evidence without letting one tall
    # plant dominate all short crops. Each XY cell is still used only once.
    weights = np.sqrt(density[tuple(indices.T)])
    bin_size = min(pitch[:2]) / 2
    angles = np.deg2rad(np.arange(-cfg.max_angle_deg, cfg.max_angle_deg + cfg.angle_step_deg / 2, cfg.angle_step_deg))
    scores = []
    for angle in angles:
        _, profile = _profile(points, weights, angle, bin_size)
        # A planting lattice can produce diagonal aliases. A configurable ego
        # heading prior resolves weak ties towards forward rows, without using
        # GT or forcing genuinely rotated high-support rows to angle zero.
        prior = 1.0 - cfg.heading_prior_strength * np.sin(angle) ** 2
        scores.append(float(np.dot(profile, profile)) * prior)
    angle = float(angles[int(np.argmax(scores))])
    offsets, profile = _profile(points, weights, angle, bin_size)
    peaks = _peaks(profile, spacing=max(1, int(np.ceil(cfg.min_row_spacing / bin_size))),
                   height=profile.max() * .12, prominence=profile.max() * .08)
    lateral_peaks = offsets[peaks]
    c, s = np.cos(angle), np.sin(angle)
    u = points[:, 0] * c + points[:, 1] * s
    v = -points[:, 0] * s + points[:, 1] * c
    rng = np.random.default_rng(cfg.seed)
    rows = []
    if len(peaks):
        assignments = np.argmin(np.abs(v[:, None] - lateral_peaks), axis=1)
        for peak_index, peak in enumerate(lateral_peaks):
            selected = (assignments == peak_index) & (np.abs(v - peak) <= cfg.row_band)
            if selected.sum() < cfg.min_row_points:
                continue
            su, sv, sw = u[selected], v[selected], weights[selected]
            slices = np.floor((su - su.min()) / cfg.slice_length).astype(int)
            centres = []
            for slice_id in np.unique(slices):
                mask = slices == slice_id
                centres.append((np.average(su[mask], weights=sw[mask]),
                                np.average(sv[mask], weights=sw[mask]), sw[mask].sum()))
            centres = np.asarray(centres)
            fit = _robust_fit(*centres.T, cfg, rng)
            if fit is None:
                continue
            coefficients, inliers, rms = fit
            u_min, u_max = centres[inliers, 0].min(), centres[inliers, 0].max()
            if u_max - u_min < cfg.min_row_length:
                continue
            samples = np.linspace(u_min, u_max, max(2, int(np.ceil((u_max - u_min) / cfg.sample_step)) + 1))
            lateral = np.polynomial.polynomial.polyval(samples, coefficients)
            xy = np.column_stack((samples * c - lateral * s, samples * s + lateral * c))
            _, inside = _indices(xy, pc_range, pitch, labels.shape)
            xy = xy[inside]
            if len(xy) < 2 or np.any(np.diff(xy[:, 0]) <= 0):
                continue
            residuals = np.abs(sv - np.polynomial.polynomial.polyval(su, coefficients))
            rows.append(dict(id=0, points=_xyz(xy, labels, pc_range, pitch, ground_height),
                direction_deg=round(float(np.rad2deg(angle)), 3),
                coefficients_uv=coefficients.round(6).tolist(), u_range_m=[float(u_min), float(u_max)],
                half_width_m=round(float(np.quantile(residuals, .9) + min(pitch[:2]) / 2), 4),
                support_cells=int(selected.sum()), inlier_fraction=round(float(inliers.mean()), 4),
                fit_rmse_m=round(rms, 4), length_m=round(float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()), 4)))
    rows.sort(key=lambda row: row['coefficients_uv'][0])
    for row_id, row in enumerate(rows):
        row['id'] = row_id
    result['rows'] = rows
    result['diagnostics'].update(direction_deg=round(float(np.rad2deg(angle)), 3),
                                 status='ok' if rows else 'no_supported_rows')
    result['reference_path'] = _reference(rows, labels, pc_range, pitch, ground_height, ground_known, cfg)
    return result


def config_dict(config):
    return asdict(config)
