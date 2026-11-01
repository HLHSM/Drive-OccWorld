#!/usr/bin/env python3
"""Build the measured row table; optionally export the archived 2D BEV figure."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results-root', type=Path, required=True)
    parser.add_argument('--paper-dir', type=Path, default=Path('paper_cn_v2'))
    parser.add_argument('--display-threshold', type=float, default=.3,
                        help='Distance tolerance in metres for all plot colours and displayed P/R')
    parser.add_argument('--sample-indices', type=int, nargs=3, default=[60, 262, 1399])
    parser.add_argument('--bev-figure', action='store_true',
                        help='Explicitly regenerate the archived 2D figure; the manuscript uses user-selected 3D views')
    args = parser.parse_args()
    original = json.loads((args.results_root / 'original' / 'metrics.json').read_text())
    summaries = {'original': original}
    threshold = f'{args.display_threshold:.2f}'
    if threshold not in original['results']['overall']['all']['localization']['gt']:
        parser.error('Display threshold must be one of the evaluated thresholds')
    if len(set(args.sample_indices)) != 3:
        parser.error('Three distinct sample indices are required')
    paper = args.paper_dir
    (paper / 'data').mkdir(exist_ok=True)
    (paper / 'figures').mkdir(exist_ok=True)
    # Bundle the new metric and its provenance only. Source artifacts retain
    # the historical metrics; the manuscript focuses on the new task definition.
    bundle = {name: {k: v for k, v in summary.items()
        if k not in ('results', 'retained_full_extent_protocol')} for name, summary in summaries.items()}
    for name, summary in summaries.items():
        bundle[name]['results'] = {group: {value: metrics['localization']
            for value, metrics in members.items()} for group, members in summary['results'].items()}

    paths = sorted((args.results_root / 'original' / 'per_frame').glob('*.json'))
    records = {p.name: json.loads(p.read_text()) for p in paths}
    if len(records) != original['frames']:
        raise ValueError('Per-frame cohort is incomplete')
    for side in ('gt', 'pred'):
        for delta, overall in original['results']['overall']['all']['localization'][side].items():
            total = np.sum([[d['localization'][side][delta][k] for k in ('tp', 'fp', 'fn')]
                            for d in records.values()], axis=0)
            np.testing.assert_array_equal(total, [overall[k] for k in ('tp', 'fp', 'fn')])
    example = next(d for d in records.values() if d['frame']['sample_index'] == 1399)
    sources = [('全量：标签占用拟合', original['results']['overall']['all']['localization']['gt']),
               ('全量：AgriOcc 预测拟合', original['results']['overall']['all']['localization']['pred']),
               ('小麦单帧：AgriOcc 预测拟合', example['localization']['pred'])]
    table = [r'\begin{table}[pos=htbp]', r'\centering',
        r'\caption{共同支撑行定位评价，各项以\%计。前两行为全部 1,540 帧的汇总结果；最后一行仅为小麦示例 '
        r'\#001399 的单帧结果，单独列示。精确率与召回率采用 ' + threshold + r' m 容差。}',
        r'\label{tab:crop_row_localization}',
        r'\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}lrrrrr}', r'\toprule',
        r'统计范围与输入 & $F_{1,\mathrm{loc}}^{0.10}$ & $F_{1,\mathrm{loc}}^{0.20}$ & $F_{1,\mathrm{loc}}^{0.30}$ & $P^{'
        + threshold + r'}$ & $R^{' + threshold + r'}$\\', r'\midrule']
    for index, (label, values) in enumerate(sources):
        if index == 2:
            table.append(r'\midrule')
        numbers = [values[t]['f1'] * 100 for t in ('0.10', '0.20', '0.30')]
        numbers += [values[threshold][k] * 100 for k in ('precision', 'recall')]
        table.append(label + ' & ' + ' & '.join(f'{v:.1f}' for v in numbers) + r' \\')
    table += [r'\bottomrule', r'\end{tabular*}', r'\end{table}']
    (paper / 'figures' / 'crop_row_table.tex').write_text('\n'.join(table) + '\n')

    bundle['example_frame'] = example
    bundle['figure_status'] = 'awaiting_user_selection_of_3D_occupancy_views'
    bundle['generator_sha256'] = sha256(Path(__file__))
    (paper / 'data' / 'crop_row_experiment.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + '\n')
    if not args.bev_figure:
        print(json.dumps(dict(table=str(paper / 'figures' / 'crop_row_table.tex'),
            all_frames_f1_at_030=original['results']['overall']['all']['localization']['pred']['0.30']['f1'],
            wheat_example_f1_at_030=example['localization']['pred']['0.30']['f1']), indent=2))
        return
    # Keep all three examples fixed across the checkpoint/tolerance change.
    names = [next(n for n, d in records.items() if d['frame']['sample_index'] == i)
             for i in args.sample_indices]
    selection = dict(rule='fixed_three_examples_preserved_from_previous_figure',
        sample_indices=args.sample_indices, displayed_threshold_m=args.display_threshold,
        threshold_status='exploratory_display_tolerance_not_validation_selected',
        rows=[], model_checkpoint=original['checkpoint'])
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
        'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7, 'ytick.labelsize': 7,
        'mathtext.fontset': 'stix', 'pdf.fonttype': 42, 'savefig.dpi': 300})
    fig, axes = plt.subplots(3, 3, figsize=(7.3, 8.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.085, right=.99, top=.955, bottom=.15, wspace=.10, hspace=.33)
    eval_dir = Path(original['evaluation_dir'])
    for row_no, name in enumerate(names):
        record = records[name]
        row_path = eval_dir / 'row_prediction' / name
        artifact = json.loads(row_path.read_text())
        npz = eval_dir / 'prediction' / artifact['source']['name']
        stat = npz.stat()
        if stat.st_size != artifact['source']['size_bytes'] or stat.st_mtime_ns != artifact['source']['mtime_ns']:
            raise ValueError(f'Stale row source: {npz}')
        with np.load(npz, allow_pickle=False) as data:
            density = [(data[k] == 1).sum(axis=2).T for k in ('current_gt', 'current_pred')]
            np.testing.assert_allclose(data['point_cloud_range'], artifact['point_cloud_range'])
        pc = artifact['point_cloud_range']
        truth = artifact['current']['geometry']['rows']
        vmax = max(float(np.sqrt(x).max()) for x in density)
        for col, ax in enumerate(axes[row_no]):
            if col:
                raster = np.ma.masked_where(density[col - 1] == 0, np.sqrt(density[col - 1]))
                ax.imshow(raster, origin='lower', extent=[pc[0], pc[3], pc[1], pc[4]],
                    cmap='Greens', alpha=.45, vmin=0, vmax=max(vmax, 1), interpolation='nearest', rasterized=True)
            for row in truth:
                pts = np.asarray(row['points'])
                ax.plot(pts[:, 0], pts[:, 1], color='#0072B2' if col == 0 else '#555555',
                    lw=.95 if col == 0 else .7, ls='-' if col == 0 else '--', zorder=3)
            if col:
                side = 'gt' if col == 1 else 'pred'
                m = record['localization'][side][threshold]
                matched = {x['estimate_index'] for x in m['matches']}
                for index, row in enumerate(artifact['current'][side]['rows']):
                    pts = np.asarray(row['points'])
                    ax.plot(pts[:, 0], pts[:, 1], color='#9B2C9D' if index in matched else '#D55E00',
                        ls='-' if index in matched else '-.', lw=1.05, zorder=4)
                footer = (rf'$F_{{1,\mathrm{{loc}}}}^{{{threshold}}}={100*m["f1"]:.1f}\%$'
                          + f'  ({m["tp"]}/{m["fp"]}/{m["fn"]})')
            else:
                footer = f'{len(truth)} geometric rows'
            ax.text(.5, -.14, footer, transform=ax.transAxes, ha='center', va='top', fontsize=7)
            ax.set(xlim=(pc[0], pc[3]), ylim=(pc[1], pc[4]),
                   xticks=[0, 5, 10, 15], yticks=[-5, 0, 5])
            ax.set_aspect('equal')
            ax.grid(alpha=.12, lw=.4)
            ax.tick_params(length=2, pad=2, labelbottom=True)
            if col == 0:
                ax.set_ylabel('Right y (m)')
        crop = record['frame']['crop_type'].capitalize()
        axes[row_no, 0].text(0, 1.055,
            f'({chr(97 + row_no)}) {crop}, #{record["frame"]["sample_index"]:06d}',
            transform=axes[row_no, 0].transAxes, fontsize=8)
        selection['rows'].append(dict(frame=record['frame'], row_artifact=str(row_path.resolve()),
            row_artifact_sha256=sha256(row_path), occupancy=str(npz.resolve()), occupancy_sha256=sha256(npz),
            localization=record['localization']))
    for col, text in enumerate(['Geometric GT', 'Label-occupancy fit', 'Predicted-occupancy fit']):
        box = axes[0, col].get_position()
        fig.text((box.x0 + box.x1) / 2, .987, text, ha='center', va='top', fontsize=8.5)
    handles = [Line2D([0], [0], color='#0072B2', lw=1, label='Geometric GT'),
        Line2D([0], [0], color='#555555', ls='--', lw=.8, label='GT overlay'),
        Line2D([0], [0], color='#9B2C9D', lw=1.2, label='Matched fit'),
        Line2D([0], [0], color='#D55E00', ls='-.', lw=1.2, label='Unmatched fit')]
    fig.text(.54, .065, 'Forward x (m)', ha='center', fontsize=8)
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.54, .022),
        ncol=4, fontsize=7, frameon=False, handlelength=2., columnspacing=1.)
    for extension in ('pdf', 'png'):
        fig.savefig(paper / f'crop_row_fitting.{extension}', bbox_inches='tight', pad_inches=.06)
    plt.close(fig)
    bundle['figure_selection'] = selection
    bundle['generator_sha256'] = sha256(Path(__file__))
    (paper / 'data' / 'crop_row_experiment.json').write_text(json.dumps(bundle, ensure_ascii=False, indent=2) + '\n')
    snippet = ('\\begin{figure}[pos=htbp]\n\\centering\\includegraphics[width=\\linewidth]{crop_row_fitting.pdf}\n'
        '\\caption{作物行拟合对比。每行一个场景，三列依次为几何真值、语义占用标签拟合和预测占用拟合。'
        '底图为作物占用投影；紫色实线和橙色点划线分别表示在 $' + threshold + '\\,\\mathrm{m}$ 容差下匹配及未匹配的拟合行，'
        '灰色虚线叠加几何真值。面板下方给出共同支撑行定位 F1 和 TP/FP/FN。'
        '三行分别为甘蓝、胡萝卜和小麦的固定示例；定量结果采用全部 1,540 帧。'
        '预测使用与主实验相同的 AgriOcc 检查点，所有拟合线均保留有限长度。}\n'
        '\\label{fig:crop_row_fitting}\n\\end{figure}\n')
    (paper / 'figures' / 'crop_row_figure.tex').write_text(snippet)
    print(json.dumps(dict(figure=str(paper / 'crop_row_fitting.pdf'),
                          samples=[r['frame']['sample_index'] for r in selection['rows']]), indent=2))


if __name__ == '__main__':
    main()
