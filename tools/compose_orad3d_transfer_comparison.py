#!/usr/bin/env python3
"""Compose the ORAD-3D scratch-versus-transfer qualitative comparison."""

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


SAMPLES = (
    '000004_testing_y0613_1242_1623721558899',
    '000012_testing_y2021_0223_1448_1619787768301',
    '000013_testing_y2021_0223_1604_1619791712965',
)
COLUMNS = (
    ('RGB Image', 'rgb'),
    ('Ground Truth', 'gt'),
    ('Scratch Training', 'scratch'),
    ('100% Transfer', 'transfer'),
)
# Matches the ORAD ``farmsim-like`` rendering palette used by the source PNGs.
LEGEND = (
    ('Terrain surface', (105, 105, 105)),
    ('Road', (154, 205, 50)),
    ('Safe road', (120, 72, 30)),
    ('Car', (135, 206, 235)),
    ('Person', (160, 80, 190)),
    ('Water', (55, 150, 80)),
    ('Snow', (235, 235, 245)),
    ('Grass on road', (55, 150, 80)),
    ('Rock', (120, 72, 30)),
)

CELL_WIDTH, CELL_HEIGHT = 600, 340
MARGIN_X, MARGIN_Y, GAP_X, GAP_Y = 14, 10, 10, 10
HEADER_HEIGHT, LEGEND_HEIGHT = 72, 84


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path,
                        default=Path('/data/HL/ORAD-3D/extracted'))
    parser.add_argument('--gt-dir', type=Path,
                        default=Path('work_dirs/orad3d_farm_qualitative/gt_visualization'))
    parser.add_argument('--scratch-dir', type=Path,
                        default=Path('work_dirs/orad3d_farm_qualitative/scratch_visualization'))
    parser.add_argument('--transfer-dir', type=Path,
                        default=Path('work_dirs/orad3d_farm_qualitative/transfer_visualization'))
    parser.add_argument('--output', type=Path,
                        default=Path('paper_cn/figures/orad3d_transfer_qualitative.png'))
    return parser.parse_args()


def load_font(size, bold=False):
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


def fit(image):
    return ImageOps.fit(image.convert('RGB'), (CELL_WIDTH, CELL_HEIGHT),
                        method=Image.Resampling.LANCZOS)


def compose_panel(path):
    panel = fit(Image.open(path))
    ImageDraw.Draw(panel).rectangle((0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1),
                                    outline=(190, 190, 190), width=2)
    return panel


def draw_legend(draw, canvas_width, y):
    title_font = load_font(32, bold=True)
    item_font = load_font(29)
    marker, marker_gap, item_gap = 24, 10, 25
    title = 'Semantic classes:'
    title_width = draw.textbbox((0, 0), title, font=title_font)[2]
    item_widths = [marker + marker_gap + draw.textbbox((0, 0), name, font=item_font)[2]
                   for name, _ in LEGEND]
    total_width = title_width + item_gap + sum(item_widths) + item_gap * (len(LEGEND) - 1)
    x = (canvas_width - total_width) // 2
    draw.text((x, y), title, fill=(20, 20, 20), font=title_font)
    x += title_width + item_gap
    for (name, color), item_width in zip(LEGEND, item_widths):
        marker_y = y + 8
        draw.rectangle((x, marker_y, x + marker, marker_y + marker),
                       fill=color, outline=(85, 85, 85), width=1)
        draw.text((x + marker + marker_gap, y), name, fill=(20, 20, 20), font=item_font)
        x += item_width + item_gap


def paths_for_sample(args, sample):
    _, split, route_and_frame = sample.split('_', maxsplit=2)
    route, frame_id = route_and_frame.rsplit('_', maxsplit=1)
    return (
        args.data_root / split / route / 'image_data' / f'{frame_id}.png',
        args.gt_dir / f'{sample}_gt.png',
        args.scratch_dir / f'{sample}_prediction.png',
        args.transfer_dir / f'{sample}_prediction.png',
    )


def main():
    args = parse_args()
    for sample in SAMPLES:
        missing = [str(path) for path in paths_for_sample(args, sample) if not path.is_file()]
        if missing:
            raise FileNotFoundError('Missing source panel(s):\n' + '\n'.join(missing))

    width = 2 * MARGIN_X + len(COLUMNS) * CELL_WIDTH + (len(COLUMNS) - 1) * GAP_X
    height = (HEADER_HEIGHT + 2 * MARGIN_Y + len(SAMPLES) * CELL_HEIGHT +
              (len(SAMPLES) - 1) * GAP_Y + LEGEND_HEIGHT)
    canvas = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(canvas)
    header_font = load_font(46, bold=True)

    for index, (title, _) in enumerate(COLUMNS):
        x = MARGIN_X + index * (CELL_WIDTH + GAP_X)
        bbox = draw.textbbox((0, 0), title, font=header_font)
        draw.text((x + (CELL_WIDTH - (bbox[2] - bbox[0])) / 2, 8), title,
                  fill=(20, 20, 20), font=header_font)

    for row, sample in enumerate(SAMPLES):
        y = HEADER_HEIGHT + MARGIN_Y + row * (CELL_HEIGHT + GAP_Y)
        for column, path in enumerate(paths_for_sample(args, sample)):
            x = MARGIN_X + column * (CELL_WIDTH + GAP_X)
            canvas.paste(compose_panel(path), (x, y))

    legend_y = HEADER_HEIGHT + MARGIN_Y + len(SAMPLES) * CELL_HEIGHT + (
        len(SAMPLES) - 1) * GAP_Y + 22
    draw_legend(draw, width, legend_y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(args.output, optimize=True, dpi=(300, 300))
    print(f'Wrote {args.output} ({width}x{height})')


if __name__ == '__main__':
    main()
