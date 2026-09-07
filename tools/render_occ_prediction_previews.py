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
from PIL import Image, ImageDraw, ImageFont
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
    parser.add_argument('--max-points', type=int, default=16000,
                        help='maximum valid non-free voxels per panel (default: 16000)')
    parser.add_argument('--elev', type=float, default=22.0,
                        help='3D camera elevation in degrees (default: 22)')
    parser.add_argument('--azim', type=float, default=-58.0,
                        help='3D camera azimuth in degrees (default: -58)')
    parser.add_argument('--overwrite', action='store_true',
                        help='re-render existing prediction and GT cache images')
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


def occupancy_points(labels, point_cloud_range, max_points, valid_mask=None):
    """Return world-space non-free voxels, with GT validity applied to predictions."""
    mask = (labels != 0) & (labels != 255)
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
                 valid_mask=None):
    xyz, classes, total = occupancy_points(
        labels, point_cloud_range, max_points, valid_mask=valid_mask)
    fig = plt.figure(figsize=(7.2, 5.4), dpi=130)
    ax = fig.add_subplot(111, projection='3d')
    if len(xyz):
        colors = np.asarray([palette[int(label) % len(palette)] for label in classes],
                            dtype=np.float32) / 255.0
        ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], c=colors, s=0.55,
                   marker='s', alpha=0.86, depthshade=False, linewidths=0,
                   rasterized=True)
    x0, y0, z0, x1, y1, z1 = point_cloud_range
    ax.set(xlim=(x0, x1), ylim=(y0, y1), zlim=(z0, z1),
           xlabel='forward x', ylabel='right y', zlabel='up z', title=title)
    ax.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    ax.view_init(elev=elev, azim=azim)
    ax.grid(True, alpha=0.25)
    ax.text2D(0.02, 0.02, f'valid non-free: {total:,}', transform=ax.transAxes,
              fontsize=8, color='#555555')
    fig.tight_layout(pad=0.35)
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', facecolor='white')
    plt.close(fig)
    buffer.seek(0)
    with Image.open(buffer) as image:
        return image.convert('RGB')


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
            palette, class_names = style_for_classes(int(data['num_classes']))

        if gt_path.exists() and not args.overwrite:
            pass
        else:
            gt_image = render_panel(gt, pc_range, palette, 'Label (GT)',
                                    args.max_points, args.elev, args.azim)
            append_legend(gt_image, palette, class_names).save(gt_path, optimize=True)
        if not output_path.exists() or args.overwrite:
            prediction_image = render_panel(
                prediction, pc_range, palette, 'Prediction', args.max_points,
                args.elev, args.azim, valid_mask=(gt != 255))
            append_legend(prediction_image, palette, class_names).save(
                output_path, optimize=True)
            print(f'[{position}/{len(paths)}] wrote {output_path}', flush=True)


if __name__ == '__main__':
    main()
