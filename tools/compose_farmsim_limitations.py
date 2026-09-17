#!/usr/bin/env python3
"""Compose the two-row qualitative figure used in the limitations discussion."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


PANELS = (
    (
        'Cabbage',
        Path('work_dirs/qualitative_comparison/gt_visualization/'
             '000363_cabbage_mid_morning_cloudy_route_20260727_133508_seed1779612929_000075_gt.png'),
        Path('work_dirs/qualitative_comparison/agriocc_visualization/'
             '000363_cabbage_mid_morning_cloudy_route_20260727_133508_seed1779612929_000075_prediction.png'),
    ),
    (
        'Corn',
        Path('work_dirs/_shared_gt_previews/farmsim/'
             '001126_corn_early_night_sunny_route_20260727_154656_seed292694529_000053_gt.png'),
        Path('work_dirs/limitations_qualitative/agriocc_visualization/'
             '001208_corn_early_night_sunny_route_20260727_154656_seed292694529_000053_prediction.png'),
    ),
    (
        'Potato',
        Path('work_dirs/qualitative_comparison/gt_visualization/'
             '001707_potato_early_dawn_sunny_route_20260727_185003_seed1234537729_000093_gt.png'),
        Path('work_dirs/qualitative_comparison/agriocc_visualization/'
             '001707_potato_early_dawn_sunny_route_20260727_185003_seed1234537729_000093_prediction.png'),
    ),
)
LEGEND = (
    ('Crop', (154, 205, 50)),
    ('Soil', (120, 72, 30)),
    ('Drivable', (135, 206, 235)),
    ('Other vegetation', (55, 150, 80)),
    ('Other obstacle', (160, 80, 190)),
)

# Preserve the horizontal field of view while removing the empty top/bottom
# margins from the original 4:3 matplotlib renders.
CELL_WIDTH, CELL_HEIGHT = 600, 360
ROW_LABEL_WIDTH = 320
MARGIN_X, MARGIN_Y, GAP_X, GAP_Y = 14, 10, 10, 10
HEADER_HEIGHT, LEGEND_HEIGHT = 66, 82


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path,
                        default=Path('paper_cn/figures/farmsim_limitations_2x3.png'))
    return parser.parse_args()


def load_font(size, bold=False):
    """Use the installed Times-compatible Nimbus Roman family."""
    candidates = (
        '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Bold.otf' if bold
        else '/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf',
        '/usr/share/fonts/truetype/liberation2/LiberationSerif-Bold.ttf' if bold
        else '/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf',
    )
    for candidate in candidates:
        if Path(candidate).is_file():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def fit_panel(path):
    panel = ImageOps.fit(Image.open(path).convert('RGB'), (CELL_WIDTH, CELL_HEIGHT),
                         method=Image.Resampling.LANCZOS)
    ImageDraw.Draw(panel).rectangle((0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1),
                                    outline=(185, 185, 185), width=2)
    return panel


def centered_text(draw, bounds, text, font):
    left, top, right, bottom = bounds
    box = draw.textbbox((0, 0), text, font=font)
    x = left + (right - left - (box[2] - box[0])) / 2
    y = top + (bottom - top - (box[3] - box[1])) / 2 - box[1]
    draw.text((x, y), text, fill=(20, 20, 20), font=font)


def draw_legend(draw, width, y):
    title_font = load_font(34, bold=True)
    item_font = load_font(34)
    marker, marker_gap, item_gap = 26, 10, 28
    title = 'Semantic classes:'
    title_width = draw.textbbox((0, 0), title, font=title_font)[2]
    item_widths = [marker + marker_gap + draw.textbbox((0, 0), name, font=item_font)[2]
                   for name, _ in LEGEND]
    total = title_width + item_gap + sum(item_widths) + item_gap * (len(LEGEND) - 1)
    x = (width - total) // 2
    draw.text((x, y), title, fill=(20, 20, 20), font=title_font)
    x += title_width + item_gap
    for (name, color), item_width in zip(LEGEND, item_widths):
        marker_y = y + 7
        draw.rectangle((x, marker_y, x + marker, marker_y + marker),
                       fill=color, outline=(85, 85, 85), width=1)
        draw.text((x + marker + marker_gap, y), name, fill=(20, 20, 20), font=item_font)
        x += item_width + item_gap


def main():
    args = parse_args()
    for _, gt_path, pred_path in PANELS:
        for path in (gt_path, pred_path):
            if not path.is_file():
                raise FileNotFoundError(f'Missing qualitative panel: {path}')

    width = (2 * MARGIN_X + ROW_LABEL_WIDTH + len(PANELS) * CELL_WIDTH +
             (len(PANELS) - 1) * GAP_X)
    height = (HEADER_HEIGHT + 2 * MARGIN_Y + 2 * CELL_HEIGHT + GAP_Y + LEGEND_HEIGHT)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(42, bold=True)
    row_font = load_font(36, bold=True)

    image_x0 = MARGIN_X + ROW_LABEL_WIDTH
    for column, (name, _, _) in enumerate(PANELS):
        x = image_x0 + column * (CELL_WIDTH + GAP_X)
        centered_text(draw, (x, 0, x + CELL_WIDTH, HEADER_HEIGHT), name, header_font)

    row_labels = ('Ground Truth', 'AgriOcc Prediction')
    row_y0 = HEADER_HEIGHT + MARGIN_Y
    for row, label in enumerate(row_labels):
        y = row_y0 + row * (CELL_HEIGHT + GAP_Y)
        centered_text(draw, (MARGIN_X, y, image_x0 - 12, y + CELL_HEIGHT), label, row_font)

    for column, (_, gt_path, pred_path) in enumerate(PANELS):
        x = image_x0 + column * (CELL_WIDTH + GAP_X)
        canvas.paste(fit_panel(gt_path), (x, row_y0))
        canvas.paste(fit_panel(pred_path), (x, row_y0 + CELL_HEIGHT + GAP_Y))

    legend_y = row_y0 + 2 * CELL_HEIGHT + GAP_Y + 21
    draw_legend(draw, width, legend_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True, dpi=(300, 300))
    print(f'Wrote {args.output} ({width}x{height})')


if __name__ == '__main__':
    main()
