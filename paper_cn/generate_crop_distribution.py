#!/usr/bin/env python3
"""Generate the AgriOcc-Dataset crop-composition pie chart from split metadata."""

import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
REPORT_PATH = ROOT / 'data' / 'farmsim' / 'splits' / 'split_report.json'
OUTPUT_PATH = Path(__file__).resolve().with_name('crop_distribution.pdf')


def main():
    with REPORT_PATH.open(encoding='utf-8') as file:
        report = json.load(file)
    crops = report['valid']['crop_type']
    names = list(crops)
    counts = [crops[name] for name in names]
    total = sum(counts)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
        'font.size': 9,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    colors = [
        '#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161',
        '#ECE133', '#56B4E9', '#949494', '#D55E00', '#7F7F7F',
    ]
    fig, axis = plt.subplots(figsize=(7.2, 3.8))
    wedges, _, _ = axis.pie(
        counts,
        colors=colors[:len(counts)],
        startangle=90,
        counterclock=False,
        autopct=lambda percentage: f'{percentage:.1f}%',
        pctdistance=0.72,
        wedgeprops={'linewidth': 0.6, 'edgecolor': 'white'},
        textprops={'fontsize': 8},
    )
    legend_labels = [
        f'{name.capitalize()}  {count} ({count / total * 100:.1f}%)'
        for name, count in zip(names, counts)
    ]
    axis.legend(
        wedges, legend_labels,
        loc='center left', bbox_to_anchor=(0.92, 0.5),
        frameon=False, handlelength=1.1, handletextpad=0.5,
        labelspacing=0.55, fontsize=8,
    )
    axis.set_aspect('equal')
    fig.savefig(OUTPUT_PATH, format='pdf', bbox_inches='tight', pad_inches=0.02)
    print(f'Saved {OUTPUT_PATH}')


if __name__ == '__main__':
    main()
