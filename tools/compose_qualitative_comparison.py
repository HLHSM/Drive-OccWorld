#!/usr/bin/env python3
"""Compose a 3x5 FarmSim qualitative comparison figure reproducibly."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SAMPLES = (
    '000143_cabbage_early_night_sunny_route_20260727_111821_seed1744107009_000064',
    '002077_potato_mature_evening_sunny_route_20260727_152645_seed1723141633_000042',
    '003569_redbeet_mature_night_cloudy_route_20260727_194708_seed646833409_000047',
)
COLUMNS = (
    ('Multi-view RGB', None, None),
    ('Ground Truth', 'gt_visualization', '_gt.png'),
    ('SurroundOcc', 'surroundocc_visualization_matched', '_prediction.png'),
    ('IR-WM', 'irwm_visualization_matched', '_prediction.png'),
    ('AgriOcc', 'agriocc_visualization', '_prediction.png'),
)
LEGEND = (
    ('Crop', (154, 205, 50)),
    ('Soil', (120, 72, 30)),
    ('Drivable', (135, 206, 235)),
    ('Other vegetation', (55, 150, 80)),
    ('Other obstacle', (160, 80, 190)),
)
# A 5:3 frame removes the substantial top/bottom whitespace in the original
# 4:3 occupancy renders while keeping their horizontal fields of view intact.
CELL_WIDTH, CELL_HEIGHT = 600, 360
# Compact canvas spacing while retaining enough separation for print readability.
MARGIN_X, MARGIN_Y, GAP_X, GAP_Y = 14, 10, 9, 10
HEADER_HEIGHT, LEGEND_HEIGHT = 72, 84


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--comparison-root', type=Path,
                        default=Path('work_dirs/qualitative_comparison'))
    parser.add_argument('--output', type=Path,
                        default=Path('paper_cn/figures/qualitative_comparison_3x5.png'))
    return parser.parse_args()


def load_font(size, bold=False):
    candidates = (
        # Nimbus Roman is the installed, metrically compatible Times family.
        '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf' if bold
        else '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf',
        '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit(image, width, height):
    return ImageOps.fit(image.convert('RGB'), (width, height),
                        method=Image.Resampling.LANCZOS)


def compose_rgb_panel(raw_dir):
    """Arrange front / left-front / right-front views in the requested triangle."""
    panel = Image.new('RGB', (CELL_WIDTH, CELL_HEIGHT), 'white')
    draw = ImageDraw.Draw(panel)
    # File order follows FarmSimSurroundOccDataset: left-front, front, right-front.
    front = fit(Image.open(raw_dir / '1.jpg'), 320, 180)
    left = fit(Image.open(raw_dir / '0.jpg'), 240, 135)
    right = fit(Image.open(raw_dir / '2.jpg'), 240, 135)

    panel.paste(front, ((CELL_WIDTH - front.width) // 2, 7))
    panel.paste(left, (34, 200))
    panel.paste(right, (CELL_WIDTH - right.width - 34, 200))
    draw.rectangle((0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1), outline=(190, 190, 190), width=2)
    return panel


def compose_panel(path):
    panel = fit(Image.open(path), CELL_WIDTH, CELL_HEIGHT)
    ImageDraw.Draw(panel).rectangle((0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1),
                                    outline=(190, 190, 190), width=2)
    return panel


def draw_legend(draw, canvas_width, y):
    """Draw a shared semantic legend using the dataset's authoritative palette."""
    title_font = load_font(36, bold=True)
    item_font = load_font(36)
    marker = 28
    item_gap = 34
    marker_gap = 11
    title = 'Semantic classes:'
    title_width = draw.textbbox((0, 0), title, font=title_font)[2]
    item_widths = [marker + marker_gap + draw.textbbox((0, 0), name, font=item_font)[2]
                   for name, _ in LEGEND]
    total_width = title_width + item_gap + sum(item_widths) + item_gap * (len(LEGEND) - 1)
    x = (canvas_width - total_width) // 2
    draw.text((x, y), title, fill=(20, 20, 20), font=title_font)
    x += title_width + item_gap
    for (name, color), item_width in zip(LEGEND, item_widths):
        marker_y = y + 9
        draw.rectangle((x, marker_y, x + marker, marker_y + marker),
                       fill=color, outline=(85, 85, 85), width=1)
        draw.text((x + marker + marker_gap, y), name, fill=(20, 20, 20), font=item_font)
        x += item_width + item_gap


def main():
    args = parse_args()
    rows = json.loads((args.comparison_root / 'manifest.json').read_text(encoding='utf-8'))
    row_by_sample = {Path(row['reference_npz']).stem: row for row in rows}
    missing = [sample for sample in SAMPLES if sample not in row_by_sample]
    if missing:
        raise KeyError(f'Samples missing from manifest: {missing}')

    width = 2 * MARGIN_X + len(COLUMNS) * CELL_WIDTH + (len(COLUMNS) - 1) * GAP_X
    height = (HEADER_HEIGHT + 2 * MARGIN_Y + len(SAMPLES) * CELL_HEIGHT +
              (len(SAMPLES) - 1) * GAP_Y + LEGEND_HEIGHT)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    font = load_font(46, bold=True)

    for column_index, (title, _, _) in enumerate(COLUMNS):
        x = MARGIN_X + column_index * (CELL_WIDTH + GAP_X)
        box = draw.textbbox((0, 0), title, font=font)
        draw.text((x + (CELL_WIDTH - (box[2] - box[0])) / 2, 8), title,
                  fill=(20, 20, 20), font=font)

    for row_index, sample in enumerate(SAMPLES):
        y = HEADER_HEIGHT + MARGIN_Y + row_index * (CELL_HEIGHT + GAP_Y)
        row = row_by_sample[sample]
        raw_dir = args.comparison_root / 'surroundocc_raw' / (
            f"{Path(row['sequence_path']).name}_{row['frame_id']}")
        panels = [compose_rgb_panel(raw_dir)]
        for _, directory, suffix in COLUMNS[1:]:
            panels.append(compose_panel(args.comparison_root / directory / f'{sample}{suffix}'))
        for column_index, panel in enumerate(panels):
            x = MARGIN_X + column_index * (CELL_WIDTH + GAP_X)
            canvas.paste(panel, (x, y))

    legend_y = HEADER_HEIGHT + MARGIN_Y + len(SAMPLES) * CELL_HEIGHT + (
        len(SAMPLES) - 1) * GAP_Y + 23
    draw_legend(draw, width, legend_y)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True, dpi=(300, 300))
    print(f'Wrote {args.output} ({width}x{height})')


if __name__ == '__main__':
    main()
