#!/usr/bin/env python3
"""Render the three user-selected RGB/occupancy/row examples for the paper.

Uses the browser viewer's point extraction, validity mask, palette, and
above-canopy overlays. No fitted coordinates or occupancy arrays are changed.
"""

import argparse
import base64
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import time
from urllib.request import ProxyHandler, build_opener

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from PIL import Image
import plotly.graph_objects as go

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.visualize_farmsim_predictions import DEFAULT_PALETTE, load_sample

DEFAULT_EVAL = ROOT / (
    'work_dirs/front3_s_base_agri_amoe96_cropsample0.1_min500_nohis_b24_ep15_'
    '20261012_214738/croprow_evaluation_20261020')
ROW_COLOR, CENTER_COLOR = '#f0009b', '#007dff'
MARKER_SIZE, MARKER_OPACITY = 4.6, 1.0
CAMERA = dict(eye=dict(x=1.45, y=0, z=2.9), up=dict(x=0, y=0, z=1),
              center=dict(x=0, y=0, z=0))


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_line(fig, points, canopy_top, color, width):
    points = np.asarray(points, dtype=float)
    height = max(float(points[:, 2].max()),
                 canopy_top if canopy_top is not None else -np.inf) + .2
    # This is a display lift, identical to highlightedLine() in the viewer.
    for dz, stroke, size in [(-.005, 'white', width + 3), (0, color, width)]:
        fig.add_trace(go.Scatter3d(
            x=points[:, 0], y=points[:, 1], z=np.full(len(points), height + dz),
            mode='lines', line=dict(color=stroke, width=size),
            showlegend=False, hoverinfo='skip'))


def write_panel(sample, side, panel_dir, basename):
    points = sample['current'][side]
    rows = sample['current']['row_prediction'][side]
    pc = sample['point_cloud_range']
    fig = go.Figure(go.Scatter3d(
        x=points['x'], y=points['y'], z=points['z'], mode='markers',
        marker=dict(size=MARKER_SIZE, color=points['color'],
                    opacity=MARKER_OPACITY, symbol='square'),
        showlegend=False, hoverinfo='skip'))
    for row in rows['rows']:
        add_line(fig, row['points'], points['crop_top_z'], ROW_COLOR, 5)
    centerline = rows.get('geometric_centerline')
    if centerline and centerline.get('points'):
        add_line(fig, centerline['points'], points['crop_top_z'], CENTER_COLOR, 7)
    # Identical scales and camera for both sides, across all three examples.
    def axis(title, limits, ticks):
        return dict(title=dict(text=title, font=dict(size=19)),
                    range=limits, tickvals=ticks, tickfont=dict(size=16),
                    showbackground=False, gridcolor='#dddddd', zeroline=False,
                    showspikes=False, color='#555555')
    fig.update_layout(
        width=800, height=640, paper_bgcolor='white', showlegend=False,
        font=dict(family='Times New Roman', size=18),
        margin=dict(l=6, r=6, t=0, b=0),
        scene=dict(
            xaxis=axis('Forward x (m)', [pc[0], pc[3]], [0, 5, 10, 15]),
            yaxis=axis('Right y (m)', [pc[1], pc[4]], [-5, 0, 5]),
            zaxis=axis('Up z (m)', [pc[2], pc[5]], [-2, 0, 2]),
            aspectmode='data', camera=CAMERA))
    html = panel_dir / f'{basename}_{side}.html'
    fig.write_html(html, include_plotlyjs=True, full_html=True,
                   post_script='window.renderReady = true;',
                   config=dict(displayModeBar=False, staticPlot=True))
    # Eliminate HTML body margins so the screenshot matches the plot viewport.
    content = html.read_text()
    html.write_text(content.replace('<head>', '<head><style>'
        'html,body{margin:0;padding:0;background:white;overflow:hidden}'
        '</style>', 1))
    return html


def screenshot(task):
    chrome, html = task
    png = html.with_suffix('.png')
    command = [chrome, '--headless', '--no-sandbox', '--disable-dev-shm-usage',
               '--hide-scrollbars', '--no-first-run', '--no-default-browser-check',
               '--use-gl=angle', '--use-angle=swiftshader',
               '--enable-unsafe-swiftshader', '--remote-debugging-port=0',
               '--remote-allow-origins=*', 'about:blank']
    # Independent Chrome profiles prevent an existing desktop browser session
    # from swallowing the screenshot invocation.
    import tempfile
    import websocket
    with tempfile.TemporaryDirectory(prefix='croprow-chrome-') as profile:
        command.insert(1, f'--user-data-dir={profile}')
        process = subprocess.Popen(command, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL)
        connection = None
        try:
            port_file = Path(profile) / 'DevToolsActivePort'
            deadline = time.monotonic() + 45
            while not port_file.exists():
                if process.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError('Chrome did not start its debugging endpoint')
                time.sleep(.1)
            port = port_file.read_text().splitlines()[0]
            opener = build_opener(ProxyHandler({}))
            with opener.open(f'http://127.0.0.1:{port}/json/list', timeout=5) as response:
                page = next(x for x in json.load(response) if x['type'] == 'page')
            connection = websocket.create_connection(
                page['webSocketDebuggerUrl'], timeout=60,
                socket=socket.create_connection(('127.0.0.1', int(port)), timeout=10),
                http_no_proxy=['localhost', '127.0.0.1'])
            counter = 0
            def call(method, params=None):
                nonlocal counter
                counter += 1
                connection.send(json.dumps(dict(id=counter, method=method, params=params or {})))
                while True:
                    result = json.loads(connection.recv())
                    if result.get('id') == counter:
                        if 'error' in result:
                            raise RuntimeError(result['error'])
                        return result.get('result', {})
            call('Emulation.setDeviceMetricsOverride',
                 dict(width=800, height=640, deviceScaleFactor=2, mobile=False))
            call('Page.navigate', dict(url=html.as_uri()))
            while True:
                result = call('Runtime.evaluate', dict(expression='window.renderReady === true'))
                if result.get('result', {}).get('value'):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f'Plotly did not finish rendering {html}')
                time.sleep(.2)
            # Plotly exports the WebGL buffer directly, avoiding compositor
            # screenshots that can be blank on headless SwiftShader.
            result = call('Runtime.evaluate', dict(
                expression="Plotly.toImage(document.querySelector('.plotly-graph-div'),"
                           "{format:'png',width:800,height:640,scale:2})",
                awaitPromise=True, returnByValue=True))
            value = result.get('result', {}).get('value', '')
            if not value.startswith('data:image/png;base64,'):
                raise RuntimeError(f'Plotly image export failed: {result}')
            png.write_bytes(base64.b64decode(value.split(',', 1)[1]))
        finally:
            if connection:
                connection.close()
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    with Image.open(png) as image:
        pixels = np.asarray(image.convert('RGB'))
        if np.mean(np.any(pixels < 230, axis=2)) < .05:
            raise RuntimeError(f'Blank WebGL render: {png}')
    html.unlink()
    return png


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--eval-dir', type=Path, default=DEFAULT_EVAL)
    parser.add_argument('--data-root', type=Path, default=Path('/data/HL/SimData-croprow'))
    parser.add_argument('--paper-dir', type=Path, default=ROOT / 'paper_cn_v2')
    parser.add_argument('--sample-indices', type=int, nargs=3, default=[255, 288, 1241])
    parser.add_argument('--reuse-panels', action='store_true')
    args = parser.parse_args()
    chrome = shutil.which('google-chrome') or shutil.which('chromium')
    if not chrome:
        raise RuntimeError('A local Chrome/Chromium executable is required.')
    panel_dir = args.paper_dir / 'figures' / 'crop_row_3d_panels'
    panel_dir.mkdir(parents=True, exist_ok=True)
    examples, tasks = [], []
    for index in args.sample_indices:
        matches = list((args.eval_dir / 'prediction').glob(f'{index:06d}_*.npz'))
        if len(matches) != 1:
            raise ValueError(f'Expected one NPZ for sample {index}: {matches}')
        npz = matches[0]
        row_path = args.eval_dir / 'row_prediction_centerline' / f'{npz.stem}.json'
        artifact = json.loads(row_path.read_text())
        if artifact['frame']['sample_index'] != index:
            raise ValueError(f'Wrong sample: {row_path}')
        sample = load_sample(npz, 30000, row_path.parent)
        if sample['row_overlay_status'] != 'Row result loaded':
            raise ValueError(sample['row_overlay_status'])
        rgb = args.data_root / artifact['frame']['sequence_path'] / 'front_rgb' / (
            artifact['frame']['frame_id'] + '.jpg')
        if not rgb.is_file():
            raise FileNotFoundError(rgb)
        row = dict(sample_index=index, frame=artifact['frame'],
                   occupancy=str(npz.resolve()), occupancy_sha256=sha256(npz),
                   rows=str(row_path.resolve()), rows_sha256=sha256(row_path),
                   rgb=str(rgb.resolve()), rgb_sha256=sha256(rgb), panels={})
        for side in ['gt', 'pred']:
            html = panel_dir / f'{index:06d}_{side}.html'
            if not args.reuse_panels or not html.with_suffix('.png').is_file():
                html = write_panel(sample, side, panel_dir, f'{index:06d}')
                tasks.append((chrome, html))
            row['panels'][side] = str(html.with_suffix('.png'))
        examples.append(row)
    with ThreadPoolExecutor(max_workers=2) as executor:
        for png in executor.map(screenshot, tasks):
            print(f'Rendered {png}', flush=True)

    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
                         'font.size': 10, 'pdf.fonttype': 42})
    # Apply one shared crop to all occupancy panels, removing export margins
    # without changing relative scales or cropping individual scenes differently.
    boxes = []
    for example in examples:
        for path in example['panels'].values():
            with Image.open(path) as source:
                pixels = np.asarray(source.convert('RGB'))
                yy, xx = np.nonzero(np.any(pixels < 235, axis=2))
                boxes.append((xx.min(), yy.min(), xx.max()+1, yy.max()+1))
    crop_box = (max(0, min(b[0] for b in boxes)-18),
                max(0, min(b[1] for b in boxes)-18),
                min(1600, max(b[2] for b in boxes)+18),
                min(1280, max(b[3] for b in boxes)+18))
    fig, axes = plt.subplots(3, 3, figsize=(9, 7.1),
                             gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
    fig.subplots_adjust(left=.01, right=.995, bottom=.075, top=.945,
                        wspace=0, hspace=.03)
    for i, example in enumerate(examples):
        images = [example['rgb'], example['panels']['gt'], example['panels']['pred']]
        for j, path in enumerate(images):
            with Image.open(path) as source:
                image = source.convert('RGB')
                # Preserve the complete RGB field of view; crop only the
                # identical export margins around the occupancy views.
                if j:
                    image = image.crop(crop_box)
                axes[i, j].imshow(image)
            axes[i, j].set_axis_off()
        crop = example['frame']['crop_type'].capitalize()
        axes[i, 0].text(0, 1.025, f'({chr(97+i)}) {crop}',
                        transform=axes[i, 0].transAxes, fontsize=10)
    for j, title in enumerate(['Front RGB', 'Label-occupancy fit', 'Predicted-occupancy fit']):
        box = axes[0, j].get_position()
        fig.text((box.x0+box.x1)/2, .99, title, ha='center', va='top', fontsize=11)
    handles = [Line2D([0], [0], color=ROW_COLOR, lw=2, label='Fitted crop rows'),
               Line2D([0], [0], color=CENTER_COLOR, lw=2.5, label='Navigation centerline')]
    names = ['Crop', 'Soil', 'Drivable', 'Other vegetation', 'Obstacle']
    handles += [Line2D([0], [0], marker='s', linestyle='', markersize=6,
                      color=np.array(DEFAULT_PALETTE[k])/255, label=name)
                for k, name in enumerate(names, 1)]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .005),
               ncol=4, frameon=False, fontsize=9, columnspacing=1.5)
    for ext in ['pdf', 'png']:
        output = args.paper_dir / 'figures' / f'crop_row_3d_rgb.{ext}'
        temporary = output.with_name(f'.{output.stem}.tmp.{ext}')
        fig.savefig(temporary, dpi=300, facecolor='white')
        temporary.replace(output)
        print(f'Saved {output}', flush=True)
    plt.close(fig)
    manifest = dict(
        selection='user_selected_screenshot_samples', samples=examples,
        camera=CAMERA, viewport=[800, 640], device_scale_factor=2,
        shared_occupancy_crop_pixels=list(map(int, crop_box)),
        max_points=30000, valid_mask='current_gt != 255 for both occupancy panels',
        marker_size=MARKER_SIZE, marker_opacity=MARKER_OPACITY, marker_symbol='square',
        row_display='viewer above-canopy lift by 0.2 m; saved XY/Z are unchanged',
        navigation='geometric_centerline; traversability is not checked',
        generator_sha256=sha256(Path(__file__)))
    output = args.paper_dir / 'data' / 'crop_row_3d_rgb.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
