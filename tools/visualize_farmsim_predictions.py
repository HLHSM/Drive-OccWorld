#!/usr/bin/env python3
"""Create an interactive HTML comparison of saved FarmSim occupancy results."""

import argparse
import html
import json
from pathlib import Path

import numpy as np


DEFAULT_PALETTE = [
    (0, 0, 0), (91, 181, 75), (120, 72, 30), (90, 90, 90),
    (55, 150, 80), (160, 80, 190),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Create a Plotly HTML viewer for FarmSim prediction NPZ files.')
    parser.add_argument('prediction_dir', help='directory created by tools/test.py --save-predictions')
    parser.add_argument('--output', help='HTML output path (default: <prediction_dir>/index.html)')
    parser.add_argument('--max-points', type=int, default=30000,
                        help='maximum non-free voxels per label/prediction panel (default: 30000)')
    return parser.parse_args()


def voxel_points(labels, point_cloud_range, max_points):
    mask = (labels != 0) & (labels != 255)
    indices = np.argwhere(mask)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points, dtype=np.int64)]
    if not len(indices):
        return dict(x=[], y=[], z=[], color=[])
    labels = labels[tuple(indices.T)]
    x0, y0, z0, x1, y1, z1 = point_cloud_range
    shape = np.asarray(mask.shape, dtype=np.float32)
    xyz = np.empty_like(indices, dtype=np.float32)
    xyz[:, 0] = x0 + (indices[:, 0] + .5) * (x1 - x0) / shape[0]
    xyz[:, 1] = y0 + (indices[:, 1] + .5) * (y1 - y0) / shape[1]
    xyz[:, 2] = z0 + (indices[:, 2] + .5) * (z1 - z0) / shape[2]
    colors = ['rgb(%d,%d,%d)' % DEFAULT_PALETTE[int(label) % len(DEFAULT_PALETTE)]
              for label in labels]
    return dict(x=xyz[:, 0].round(3).tolist(), y=xyz[:, 1].round(3).tolist(),
                z=xyz[:, 2].round(3).tolist(), color=colors)


def load_sample(path, max_points):
    with np.load(path, allow_pickle=False) as item:
        pc_range = item['point_cloud_range'].astype(float)
        sample = dict(name=path.stem,
                      current=dict(pred=voxel_points(item['current_pred'], pc_range, max_points),
                                   gt=voxel_points(item['current_gt'], pc_range, max_points)),
                      future=[])
        if 'future_pred' in item and item['future_pred'].size:
            for pred, gt in zip(item['future_pred'], item['future_gt']):
                sample['future'].append(dict(
                    pred=voxel_points(pred, pc_range, max_points),
                    gt=voxel_points(gt, pc_range, max_points)))
        if 'trajectory_pred' in item:
            sample['trajectory_pred'] = item['trajectory_pred'].astype(float).tolist()
        if 'trajectory_gt' in item:
            sample['trajectory_gt'] = item['trajectory_gt'].astype(float).tolist()
    return sample


def main():
    args = parse_args()
    if args.max_points < 1:
        raise ValueError('--max-points must be positive')
    prediction_dir = Path(args.prediction_dir)
    paths = sorted(prediction_dir.glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'No .npz files found in {prediction_dir}')
    samples = [load_sample(path, args.max_points) for path in paths]
    output = Path(args.output) if args.output else prediction_dir / 'index.html'
    options = ''.join('<option value="%d">%s</option>' %
                      (index, html.escape(sample['name']))
                      for index, sample in enumerate(samples))
    payload = json.dumps(samples, separators=(',', ':'))
    page = f'''<!doctype html>
<html><head><meta charset="utf-8"><title>FarmSim occupancy comparison</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:sans-serif;margin:16px}} #plot{{width:100%;height:78vh}}
select{{min-width:340px}} .hint{{color:#555}}</style></head>
<body><h2>FarmSim 3D Occupancy: label vs prediction</h2>
<label>Sample <select id="sample">{options}</select></label>
<label> Time <select id="time"></select></label>
<span class="hint">Each panel is capped at {args.max_points:,} non-free voxels.</span>
<div id="plot"></div>
<script>
const samples={payload}; const sampleSelect=document.querySelector('#sample');
const timeSelect=document.querySelector('#time');
function trace(points,name,scene) {{ return {{type:'scatter3d',mode:'markers',name:name,scene:scene,
 x:points.x,y:points.y,z:points.z,marker:{{size:2,color:points.color,opacity:.78}}}}; }}
function trajectory(points,name,scene,color) {{ return {{type:'scatter3d',mode:'lines+markers',name:name,scene:scene,
 x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(_=>0),line:{{color:color,width:6}},marker:{{size:3}}}}; }}
function render() {{ const s=samples[+sampleSelect.value]; const t=+timeSelect.value;
 const frame=t===0?s.current:s.future[t-1]; const label=[trace(frame.gt,'Label','scene')];
 const pred=[trace(frame.pred,'Prediction','scene2')];
 if(s.trajectory_gt) {{ label.push(trajectory(s.trajectory_gt,'GT trajectory','scene','#111')); }}
 if(s.trajectory_pred) {{ pred.push(trajectory(s.trajectory_pred,'Predicted trajectory','scene2','#e60000')); }}
 const scene={{xaxis:{{title:'forward x'}},yaxis:{{title:'right y'}},zaxis:{{title:'up z'}},aspectmode:'data'}};
 Plotly.react('plot',label.concat(pred),{{title:s.name+(t?' — future '+t:' — current'),
 grid:{{rows:1,columns:2,pattern:'independent'}},scene:scene,scene2:scene,
 margin:{{l:0,r:0,t:45,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}}); }}
function setTimes() {{ const s=samples[+sampleSelect.value]; timeSelect.innerHTML='<option value="0">current</option>';
 for(let i=0;i<s.future.length;i++) timeSelect.innerHTML+='<option value="'+(i+1)+'">future '+(i+1)+'</option>';
 render(); }} sampleSelect.onchange=setTimes; timeSelect.onchange=render; setTimes();
</script></body></html>'''
    output.write_text(page, encoding='utf-8')
    print(f'Wrote {output} from {len(samples)} samples.')


if __name__ == '__main__':
    main()
