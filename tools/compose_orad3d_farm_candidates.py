#!/usr/bin/env python3
"""Compose one four-column qualitative panel for every available ORAD farm frame."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


COLUMNS = ('RGB Image', 'Ground Truth', 'Scratch Training', '100% Transfer')
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
MARGIN_X, MARGIN_Y, GAP_X = 14, 10, 10
HEADER_HEIGHT, LEGEND_HEIGHT = 72, 78


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
    parser.add_argument('--output-dir', type=Path,
                        default=Path('work_dirs/orad3d_farm_qualitative/candidates'))
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


def parse_stem(stem):
    _, sample = stem.split('_', maxsplit=1)
    split, route_and_frame = sample.split('_', maxsplit=1)
    route, frame_id = route_and_frame.rsplit('_', maxsplit=1)
    return split, route, frame_id


def sources_for_stem(args, stem):
    split, route, frame_id = parse_stem(stem)
    return (
        args.data_root / split / route / 'image_data' / f'{frame_id}.png',
        args.gt_dir / f'{stem}_gt.png',
        args.scratch_dir / f'{stem}_prediction.png',
        args.transfer_dir / f'{stem}_prediction.png',
    )


def compose_panel(path):
    image = ImageOps.fit(Image.open(path).convert('RGB'), (CELL_WIDTH, CELL_HEIGHT),
                         method=Image.Resampling.LANCZOS)
    ImageDraw.Draw(image).rectangle((0, 0, CELL_WIDTH - 1, CELL_HEIGHT - 1),
                                    outline=(190, 190, 190), width=2)
    return image


def draw_legend(draw, width, y):
    title_font, item_font = load_font(32, bold=True), load_font(29)
    marker, marker_gap, item_gap = 24, 10, 25
    title = 'Semantic classes:'
    title_width = draw.textbbox((0, 0), title, font=title_font)[2]
    item_widths = [marker + marker_gap + draw.textbbox((0, 0), name, font=item_font)[2]
                   for name, _ in LEGEND]
    total = title_width + item_gap + sum(item_widths) + item_gap * (len(LEGEND) - 1)
    x = (width - total) // 2
    draw.text((x, y), title, fill=(20, 20, 20), font=title_font)
    x += title_width + item_gap
    for (name, color), item_width in zip(LEGEND, item_widths):
        draw.rectangle((x, y + 8, x + marker, y + 8 + marker),
                       fill=color, outline=(85, 85, 85), width=1)
        draw.text((x + marker + marker_gap, y), name, fill=(20, 20, 20), font=item_font)
        x += item_width + item_gap


def compose_candidate(paths):
    width = 2 * MARGIN_X + len(COLUMNS) * CELL_WIDTH + (len(COLUMNS) - 1) * GAP_X
    height = HEADER_HEIGHT + 2 * MARGIN_Y + CELL_HEIGHT + LEGEND_HEIGHT
    image = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(image)
    font = load_font(46, bold=True)
    for column, title in enumerate(COLUMNS):
        x = MARGIN_X + column * (CELL_WIDTH + GAP_X)
        bbox = draw.textbbox((0, 0), title, font=font)
        draw.text((x + (CELL_WIDTH - (bbox[2] - bbox[0])) / 2, 8), title,
                  fill=(20, 20, 20), font=font)
        image.paste(compose_panel(paths[column]), (x, HEADER_HEIGHT + MARGIN_Y))
    draw_legend(draw, width, HEADER_HEIGHT + MARGIN_Y + CELL_HEIGHT + 18)
    return image


def compose_contact_sheet(items):
    thumb_w, thumb_h, columns = 320, 180, 4
    header, label_h, gap, margin = 48, 44, 14, 16
    rows = (len(items) + columns - 1) // columns
    width = 2 * margin + columns * thumb_w + (columns - 1) * gap
    height = header + margin + rows * (thumb_h + label_h) + (rows - 1) * gap + margin
    sheet = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(sheet)
    title_font, label_font = load_font(30, bold=True), load_font(20)
    title = 'ORAD-3D farm qualitative candidates'
    title_box = draw.textbbox((0, 0), title, font=title_font)
    draw.text(((width - title_box[2]) // 2, 6), title, fill=(20, 20, 20), font=title_font)
    for index, (stem, paths) in enumerate(items):
        row, col = divmod(index, columns)
        x = margin + col * (thumb_w + gap)
        y = header + margin + row * (thumb_h + label_h + gap)
        thumb = ImageOps.fit(Image.open(paths[0]).convert('RGB'), (thumb_w, thumb_h),
                             method=Image.Resampling.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.rectangle((x, y, x + thumb_w - 1, y + thumb_h - 1), outline=(190, 190, 190), width=2)
        _, route, frame_id = parse_stem(stem)
        draw.text((x, y + thumb_h + 4), f'{index:02d}: {route}', fill=(20, 20, 20), font=label_font)
        draw.text((x, y + thumb_h + 23), frame_id, fill=(20, 20, 20), font=label_font)
    return sheet


def main():
    args = parse_args()
    stems = sorted(path.stem.removesuffix('_prediction')
                   for path in args.scratch_dir.glob('*_prediction.png'))
    items = []
    for stem in stems:
        paths = sources_for_stem(args, stem)
        if all(path.is_file() for path in paths):
            items.append((stem, paths))
    if not items:
        raise FileNotFoundError('No complete farm qualitative candidates found')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    index = []
    for number, (stem, paths) in enumerate(items):
        output = args.output_dir / f'candidate_{number:02d}_{stem}.png'
        compose_candidate(paths).save(output, optimize=True, dpi=(300, 300))
        index.append({'id': number, 'sample': stem, 'figure': output.name})
    compose_contact_sheet(items).save(args.output_dir / 'contact_sheet_rgb.png',
                                      optimize=True, dpi=(300, 300))
    (args.output_dir / 'index.json').write_text(json.dumps(index, indent=2) + '\n',
                                                 encoding='utf-8')
    print(f'Wrote {len(items)} candidate panels to {args.output_dir}')


if __name__ == '__main__':
    main()
