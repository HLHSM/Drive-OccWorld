#!/usr/bin/env python3
"""Render static prediction occupancy previews from saved NPZ files.

``tools/test.py --save-predictions`` writes one compressed NPZ per sample.
This utility turns those artifacts into side-view PNGs without needing a GPU.
Ground-truth panels are cached once per dataset/split. Each experiment receives
only its own prediction-side images, so GT and predictions can be combined or
laid out differently in a later post-processing step.
"""

import argparse
import io
from pathlib import Path

import matplotlib

# This tool is normally launched on compute nodes without a display server.
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from PIL import Image, ImageChops, ImageDraw, ImageFont
import numpy as np


FARMSIM_PALETTE = [
    (0, 0, 0), (154, 205, 50), (120, 72, 30), (135, 206, 235),
    (55, 150, 80), (160, 80, 190),
]
FARMSIM_CLASSES = [
    'free (not rendered)', 'crop', 'soil_ground', 'drivable',
    'other_vegetation', 'other_obstacle',
]
ORAD3D_PALETTE = [
    (0, 0, 0), (128, 128, 128), (0, 200, 0), (255, 0, 0),
    (255, 128, 0), (0, 128, 255), (230, 230, 255), (80, 180, 80),
    (150, 100, 60),
]
ORAD3D_CLASSES = [
    'free (not rendered)', 'road', 'safe-road', 'car', 'people',
    'water', 'snow', 'grass-on-road', 'rock',
]
# The ORAD-3D paper renders all annotated terrain, including label 0, as a
# grey surface and draws the rutted road (label 1) in purple.  In contrast,
# the generic occupancy convention treats 0 as free and intentionally hides
# it.  Keep this palette separate so FarmSim continues to use that convention.
ORAD3D_PAPER_PALETTE = [
    (105, 105, 105), (125, 45, 135), (132, 180, 45), (220, 55, 45),
    (255, 145, 35), (35, 120, 230), (225, 230, 245), (70, 175, 75),
    (145, 95, 55),
]
# ORAD's label 0 remains visible as the accumulated terrain surface.  The
# remaining semantic colours intentionally reuse the FarmSim vocabulary:
# road/crop lime, safe-road/soil brown, car/drivable blue, vegetation green,
# and obstacles purple.
ORAD_FARMSIM_PALETTE = [
    (105, 105, 105), (154, 205, 50), (120, 72, 30), (135, 206, 235),
    (160, 80, 190), (55, 150, 80), (235, 235, 245), (55, 150, 80),
    (120, 72, 30),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Render static occupancy prediction PNGs and a shared GT cache.')
    parser.add_argument('prediction_dir', type=Path,
                        help='directory created by tools/test.py --save-predictions')
    parser.add_argument('output_dir', type=Path,
                        help='per-experiment prediction visualization output directory')
    parser.add_argument('--shared-gt-dir', type=Path, required=True,
                        help='shared cache for ground-truth panels')
    parser.add_argument('--count', type=int, default=100,
                        help='number of NPZ files to render (default: 100)')
    parser.add_argument('--max-points', type=int, default=30000,
                        help='maximum valid non-free voxels per panel (default: 30000)')
    parser.add_argument('--elev', type=float, default=26.0,
                        help='3D camera elevation in degrees (default: 26)')
    parser.add_argument('--azim', type=float, default=-135.0,
                        help='3D camera azimuth in degrees (default: -135)')
    parser.add_argument('--style',
                        choices=('auto', 'generic', 'farmsim-like', 'orad-paper'),
                        default='auto',
                        help=('render style: auto uses the FarmSim-compatible ORAD view for '
                              '9-class artifacts and generic rendering otherwise'))
    parser.add_argument('--with-legend', action='store_true',
                        help='append the semantic legend below each standalone image')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-render existing prediction and GT cache images')
    parser.add_argument('--skip-gt', action='store_true',
                        help='leave shared GT cache unchanged and render prediction images only')
    return parser.parse_args()


def style_for_classes(num_classes):
    if num_classes == len(FARMSIM_CLASSES):
        return FARMSIM_PALETTE, FARMSIM_CLASSES
    if num_classes == len(ORAD3D_CLASSES):
        return ORAD3D_PALETTE, ORAD3D_CLASSES
    cmap = plt.get_cmap('tab20', max(num_classes, 1))
    palette = [(0, 0, 0)] + [
        tuple(int(channel * 255) for channel in cmap(index)[:3])
        for index in range(1, num_classes)
    ]
    return palette, ['free (not rendered)'] + [
        f'class_{index}' for index in range(1, num_classes)
    ]


def occupancy_points(labels, point_cloud_range, max_points, valid_mask=None,
                     include_zero=False):
    """Return world-space occupancy points, with GT validity for predictions."""
    mask = labels != 255
    if not include_zero:
        mask &= labels != 0
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError('prediction and GT occupancy shapes do not match')
        mask &= valid_mask
    indices = np.argwhere(mask)
    total = len(indices)
    if total > max_points:
        indices = indices[np.linspace(0, total - 1, max_points, dtype=np.int64)]
    if not len(indices):
        return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=labels.dtype), total

    x0, y0, z0, x1, y1, z1 = np.asarray(point_cloud_range, dtype=np.float32)
    shape = np.asarray(labels.shape, dtype=np.float32)
    xyz = indices.astype(np.float32) + 0.5
    xyz[:, 0] = x0 + xyz[:, 0] * (x1 - x0) / shape[0]
    xyz[:, 1] = y0 + xyz[:, 1] * (y1 - y0) / shape[1]
    xyz[:, 2] = z0 + xyz[:, 2] * (z1 - z0) / shape[2]
    return xyz, labels[tuple(indices.T)], total


def render_panel(labels, point_cloud_range, palette, title, max_points, elev, azim,
                 valid_mask=None, include_zero=False, z_tick_step=None):
    xyz, classes, total = occupancy_points(
        labels, point_cloud_range, max_points, valid_mask=valid_mask,
        include_zero=include_zero)
    fig = plt.figure(figsize=(5.6, 4.2), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    if len(xyz):
        colors = np.asarray([palette[int(label) % len(palette)] for label in classes],
                            dtype=np.float32) / 255.0
        # A voxel is 20 cm wide in the FarmSim grids.  Large opaque square
        # markers preserve connected ground/vegetation surfaces in a static
        # raster preview instead of reducing them to sparse-looking speckles.
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=5.0,
                   marker='s', alpha=0.98, depthshade=False, linewidths=0,
                   rasterized=True)
    x0, y0, z0, x1, y1, z1 = point_cloud_range
    # ORAD's voxel range begins in front of the vehicle.  Including x=0 makes
    # the ego heading marker physically meaningful for both datasets.
    axis_x0 = min(0.0, x0)
    ax.set(xlim=(axis_x0, x1), ylim=(y0, y1), zlim=(z0, z1),
           xlabel='forward x', ylabel='right y', zlabel='up z')
    if title:
        ax.set_title(title)
    if z_tick_step is not None:
        ax.set_zticks(np.arange(np.ceil(z0), np.floor(z1) + 0.1,
                                z_tick_step))
    ax.set_box_aspect((x1 - axis_x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    # Keep the z-axis label fully visible and settle the axes before projecting
    # the vehicle heading into the final 2D output canvas.
    fig.subplots_adjust(left=0.115, right=0.99, bottom=0.035, top=0.99)
    fig.canvas.draw()
    arrow_z = float(np.clip(0.0, z0, z1))
    arrow_length = min(6.0, (x1 - axis_x0) * 0.20)
    projected = []
    for arrow_x in (0.0, arrow_length):
        u, v, _ = proj3d.proj_transform(arrow_x, 0.0, arrow_z,
                                        ax.get_proj())
        pixel = ax.transData.transform((u, v))
        projected.append(fig.transFigure.inverted().transform(pixel))
    fig.add_artist(FancyArrowPatch(
        projected[0], projected[1], transform=fig.transFigure,
        arrowstyle='-|>', color='#e31a1c', lw=2.35, mutation_scale=10,
        zorder=100))
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert('RGB')


def render_orad_paper_panel(labels, point_cloud_range, palette, max_points,
                            valid_mask=None):
    """Render ORAD terrain from a high, rear-facing camera looking forward.

    Unlike the generic renderer, label 0 remains visible as the broad grey
    terrain base.  The camera sits behind and above the vehicle (negative
    forward axis) and looks along the positive forward axis, retaining both
    the overhead view and forward depth used by the paper's point-cloud panels.
    """
    if labels.ndim != 3:
        raise ValueError('ORAD paper-style rendering expects a 3D occupancy grid')
    known = labels != 255
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError('prediction and GT occupancy shapes do not match')
        known &= valid_mask
    indices = np.argwhere(known)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)]

    x0, y0, z0, x1, y1, z1 = np.asarray(point_cloud_range, dtype=np.float32)
    shape = np.asarray(labels.shape, dtype=np.float32)
    # The source labels are on a perfectly regular 20 cm grid.  Drawing the
    # centres unchanged creates artificial parallel bars in a perspective
    # raster.  Deterministic sub-voxel jitter retains each voxel's position
    # and class while recovering the unstructured point-cloud appearance used
    # by the ORAD paper.
    seed = int((len(indices) * 1000003 + int(indices.sum())) % (2 ** 32))
    rng = np.random.default_rng(seed)
    xyz = indices.astype(np.float32) + 0.5
    xyz += rng.uniform(-0.42, 0.42, size=xyz.shape).astype(np.float32)
    xyz[:, 0] = x0 + xyz[:, 0] * (x1 - x0) / shape[0]
    xyz[:, 1] = y0 + xyz[:, 1] * (y1 - y0) / shape[1]
    xyz[:, 2] = z0 + xyz[:, 2] * (z1 - z0) / shape[2]
    classes = labels[tuple(indices.T)]

    fig = plt.figure(figsize=(5.2, 3.6), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    if len(xyz):
        colors = np.asarray([palette[int(label)] for label in classes],
                            dtype=np.float32) / 255.0
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=3.0,
                   marker='o', alpha=0.96, depthshade=False, linewidths=0,
                   rasterized=True)
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1))
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    # azim=180 observes from negative x, so the camera looks forward (+x).
    # A high oblique view preserves forward depth without exposing vertical
    # voxel stacks as bars.  azim=180 observes from negative x, looking +x.
    ax.view_init(elev=58, azim=180)
    ax.set_axis_off()
    # Matplotlib reserves generous margins around 3D axes.  Expand the axes
    # itself so the terrain fills the standalone image like the paper panels.
    ax.set_position((-0.08, -0.14, 1.16, 1.28))
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor='white', pad_inches=0)
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        image = image.convert('RGB')
    # Retain a small white frame but eliminate unused Matplotlib canvas space.
    bbox = ImageChops.difference(image, Image.new('RGB', image.size, 'white')).getbbox()
    if bbox is None:
        return image
    left, top, right, bottom = bbox
    margin = max(8, int(max(right - left, bottom - top) * 0.04))
    return image.crop((max(0, left - margin), max(0, top - margin),
                       min(image.width, right + margin), min(image.height, bottom + margin)))


def append_legend(image, palette, class_names):
    """Keep every standalone side-view image interpretable after export."""
    font = ImageFont.load_default()
    legend_height = 54
    output = Image.new('RGB', (image.width, image.height + legend_height), 'white')
    output.paste(image, (0, 0))
    draw = ImageDraw.Draw(output)
    draw.text((12, image.height + 8), 'Semantic colours:', fill='black', font=font)
    x, y = 140, image.height + 7
    for color, name in zip(palette[1:], class_names[1:]):
        item_width = 25 + len(name) * 7
        if x + item_width > output.width - 10:
            x, y = 12, y + 22
        draw.rectangle((x, y + 2, x + 14, y + 16), fill=color, outline='#555555')
        draw.text((x + 20, y + 2), name, fill='black', font=font)
        x += item_width
    return output


def main():
    args = parse_args()
    if args.count < 1 or args.max_points < 1:
        raise ValueError('--count and --max-points must be positive')
    paths = sorted(args.prediction_dir.glob('*.npz'))[:args.count]
    if len(paths) < args.count:
        raise FileNotFoundError(
            f'{args.prediction_dir} has {len(paths)} prediction files; expected at least {args.count}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.shared_gt_dir.mkdir(parents=True, exist_ok=True)

    for position, path in enumerate(paths, 1):
        output_path = args.output_dir / f'{path.stem}_prediction.png'
        gt_path = args.shared_gt_dir / f'{path.stem}_gt.png'
        if (output_path.exists() and gt_path.exists() and not args.overwrite):
            continue
        with np.load(path, allow_pickle=False) as data:
            required = {'current_pred', 'current_gt', 'point_cloud_range', 'num_classes'}
            missing = required.difference(data.files)
            if missing:
                raise KeyError(f'{path} is not an occupancy artifact; missing {sorted(missing)}')
            prediction = data['current_pred']
            gt = data['current_gt']
            pc_range = data['point_cloud_range'].astype(float)
            num_classes = int(data['num_classes'])
            palette, class_names = style_for_classes(num_classes)
            style = args.style
            if style == 'auto':
                style = ('farmsim-like' if num_classes == len(ORAD3D_CLASSES)
                         else 'generic')
            if style == 'farmsim-like':
                if num_classes != len(ORAD3D_CLASSES):
                    raise ValueError('FarmSim-compatible ORAD rendering requires 9 classes')
                palette = ORAD_FARMSIM_PALETTE
            if style == 'orad-paper':
                if num_classes != len(ORAD3D_CLASSES):
                    raise ValueError('ORAD paper-style rendering requires 9 classes')
                palette = ORAD3D_PAPER_PALETTE

        if args.skip_gt or (gt_path.exists() and not args.overwrite):
            pass
        else:
            if style == 'orad-paper':
                gt_image = render_orad_paper_panel(
                    gt, pc_range, palette, args.max_points)
            else:
                gt_image = render_panel(gt, pc_range, palette, None,
                                        args.max_points, args.elev, args.azim,
                                        include_zero=(style == 'farmsim-like'),
                                        z_tick_step=(2 if style == 'farmsim-like'
                                                     else None))
            if args.with_legend:
                gt_image = append_legend(gt_image, palette, class_names)
            gt_image.save(gt_path, optimize=True)
        if not output_path.exists() or args.overwrite:
            if style == 'orad-paper':
                prediction_image = render_orad_paper_panel(
                    prediction, pc_range, palette, args.max_points,
                    valid_mask=(gt != 255))
            else:
                prediction_image = render_panel(
                    prediction, pc_range, palette, None, args.max_points,
                    args.elev, args.azim, valid_mask=(gt != 255),
                    include_zero=(style == 'farmsim-like'),
                    z_tick_step=(2 if style == 'farmsim-like' else None))
            if args.with_legend:
                prediction_image = append_legend(
                    prediction_image, palette, class_names)
            prediction_image.save(output_path, optimize=True)
            print(f'[{position}/{len(paths)}] wrote {output_path}', flush=True)


if __name__ == '__main__':
    main()
