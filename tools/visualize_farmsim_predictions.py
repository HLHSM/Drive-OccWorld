#!/usr/bin/env python3
"""Serve an interactive browser for saved FarmSim occupancy predictions."""

import argparse
import json
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

import numpy as np


DEFAULT_PALETTE = [
    (0, 0, 0), (154, 205, 50), (120, 72, 30), (135, 206, 235),
    (55, 150, 80), (160, 80, 190),
]
CLASS_NAMES = [
    'free (not rendered)', 'crop', 'soil_ground', 'drivable',
    'other_vegetation', 'other_obstacle',
]


def parse_args():
    parser = argparse.ArgumentParser(
        description='Serve a Plotly viewer for FarmSim prediction NPZ files.')
    parser.add_argument('prediction_dir',
                        help='directory created by tools/test.py --save-predictions')
    parser.add_argument('--host', default='127.0.0.1',
                        help='server bind address (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=8000,
                        help='server port; use 0 to choose a free port (default: 8000)')
    parser.add_argument('--max-points', type=int, default=30000,
                        help='maximum non-free voxels per label/prediction panel (default: 30000)')
    parser.add_argument('--open-browser', action='store_true',
                        help='open the viewer URL in the default browser')
    return parser.parse_args()


def voxel_points(labels, point_cloud_range, max_points, valid_mask=None):
    """Convert visible occupancy voxels to point-cloud payloads.

    ``valid_mask`` is derived from the GT ignore ID, so predictions are shown
    only where the label is evaluable.  This makes the two panels comparable.
    """
    mask = (labels != 0) & (labels != 255)
    if valid_mask is not None:
        if valid_mask.shape != labels.shape:
            raise ValueError('valid_mask shape must match occupancy labels')
        mask &= valid_mask
    indices = np.argwhere(mask)
    total = len(indices)
    if len(indices) > max_points:
        indices = indices[np.linspace(0, len(indices) - 1, max_points,
                                      dtype=np.int64)]
    if not len(indices):
        return dict(x=[], y=[], z=[], color=[], total=0, drawn=0)
    labels = labels[tuple(indices.T)]
    x0, y0, z0, x1, y1, z1 = point_cloud_range
    shape = np.asarray(mask.shape, dtype=np.float32)
    xyz = np.empty_like(indices, dtype=np.float32)
    xyz[:, 0] = x0 + (indices[:, 0] + .5) * (x1 - x0) / shape[0]
    xyz[:, 1] = y0 + (indices[:, 1] + .5) * (y1 - y0) / shape[1]
    xyz[:, 2] = z0 + (indices[:, 2] + .5) * (z1 - z0) / shape[2]
    colors = ['rgb(%d,%d,%d)' % DEFAULT_PALETTE[int(label) % len(DEFAULT_PALETTE)]
              for label in labels]
    return dict(x=xyz[:, 0].round(3).tolist(),
                y=xyz[:, 1].round(3).tolist(),
                z=xyz[:, 2].round(3).tolist(), color=colors,
                total=total, drawn=len(indices))


def load_sample(path, max_points):
    """Read one NPZ only when the browser requests it."""
    with np.load(path, allow_pickle=False) as item:
        pc_range = item['point_cloud_range'].astype(float)
        current_gt = item['current_gt']
        current_valid = current_gt != 255
        sample = dict(name=path.stem,
                      current=dict(pred=voxel_points(item['current_pred'], pc_range, max_points,
                                                     valid_mask=current_valid),
                                   gt=voxel_points(current_gt, pc_range, max_points,
                                                   valid_mask=current_valid)),
                      future=[])
        if 'future_pred' in item and item['future_pred'].size:
            for pred, gt in zip(item['future_pred'], item['future_gt']):
                sample['future'].append(dict(
                    pred=voxel_points(pred, pc_range, max_points,
                                      valid_mask=gt != 255),
                    gt=voxel_points(gt, pc_range, max_points,
                                    valid_mask=gt != 255)))
        if 'trajectory_pred' in item:
            sample['trajectory_pred'] = item['trajectory_pred'].astype(float).tolist()
        if 'trajectory_gt' in item:
            sample['trajectory_gt'] = item['trajectory_gt'].astype(float).tolist()
    return sample


def viewer_page(max_points):
    class_legend = ''.join(
        '<span class="class-item"><i style="background:rgb(%d,%d,%d)"></i>%s</span>' %
        (*color, name) for color, name in zip(DEFAULT_PALETTE, CLASS_NAMES))
    return f'''<!doctype html>
<html><head><meta charset="utf-8"><title>FarmSim occupancy viewer</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:sans-serif;margin:16px}} #plot{{width:100%;height:78vh}}
select{{min-width:340px}} button{{margin-left:6px}} .hint{{color:#555}} #status{{margin-left:8px}}
#class-legend{{display:flex;flex-wrap:wrap;gap:10px 18px;margin:4px 0 0;font-size:14px}}
.class-item{{display:inline-flex;align-items:center;gap:5px}} .class-item i{{width:14px;height:14px;border:1px solid #777;display:inline-block}}</style></head>
<body><h2>FarmSim 3D Occupancy: label vs prediction</h2>
<label>Sample <select id="sample"></select></label><button id="previous">Previous</button><button id="next">Next</button>
<label> Time <select id="time"></select></label>
<span class="hint">Each panel is capped at {max_points:,} valid non-free voxels. All samples are preloaded.</span><span id="status"></span><span id="preload-status" class="hint"></span>
<div id="plot"></div><div id="class-legend"><strong>Semantic colours:</strong>{class_legend}</div><script>
const sampleSelect=document.querySelector('#sample'), timeSelect=document.querySelector('#time'), status=document.querySelector('#status'), preloadStatus=document.querySelector('#preload-status');
let samples=[], current=null; const sampleCache=new Map();
function trace(points,name,scene) {{ return {{type:'scatter3d',mode:'markers',name:name,scene:scene,x:points.x,y:points.y,z:points.z,marker:{{size:2,color:points.color,opacity:.78}}}}; }}
function trajectory(points,name,scene,color) {{ return {{type:'scatter3d',mode:'lines+markers',name:name,scene:scene,x:points.map(p=>p[0]),y:points.map(p=>p[1]),z:points.map(_=>0),line:{{color:color,width:6}},marker:{{size:3}}}}; }}
function render() {{ if(!current) return; const t=+timeSelect.value, frame=t===0?current.current:current.future[t-1]; const label=[trace(frame.gt,'Label','scene')], pred=[trace(frame.pred,'Prediction','scene2')];
 if(current.trajectory_gt) label.push(trajectory(current.trajectory_gt,'GT trajectory','scene','#111'));
 if(current.trajectory_pred) pred.push(trajectory(current.trajectory_pred,'Predicted trajectory','scene2','#e60000'));
 const makeScene=(domain)=>({{domain:{{x:domain,y:[0,1]}},xaxis:{{title:'forward x'}},yaxis:{{title:'right y'}},zaxis:{{title:'up z'}},aspectmode:'data'}});
 status.textContent=` (${{+sampleSelect.value+1}} / ${{samples.length}}; valid non-free: GT ${{frame.gt.total.toLocaleString()}}, Prediction ${{frame.pred.total.toLocaleString()}})`;
 Plotly.react('plot',label.concat(pred),{{title:current.name+(t?' — future '+t:' — current'),scene:makeScene([0,.48]),scene2:makeScene([.52,1]),annotations:[{{text:'<b>Label (GT)</b>',x:.24,y:1.04,xref:'paper',yref:'paper',showarrow:false}},{{text:'<b>Prediction</b>',x:.76,y:1.04,xref:'paper',yref:'paper',showarrow:false}}],margin:{{l:0,r:0,t:80,b:0}},legend:{{orientation:'h'}}}},{{responsive:true}}); }}
 function setTimes() {{ timeSelect.innerHTML='<option value="0">current</option>'; for(let i=0;i<current.future.length;i++) timeSelect.innerHTML+=`<option value="${{i+1}}">future ${{i+1}}</option>`; render(); }}
 async function getSample(index) {{ if(!sampleCache.has(index)) {{ const request=fetch('/api/sample/'+index).then(async response=>{{ if(!response.ok) throw new Error(await response.text()); return response.json(); }}); sampleCache.set(index,request); }} return sampleCache.get(index); }}
 async function loadSample(index) {{ if(index<0 || index>=samples.length) return; sampleSelect.value=index; status.textContent='Loading…'; try {{ current=await getSample(index); setTimes(); }} catch(error) {{ status.textContent='Failed to load sample'; console.error(error); }} }}
 async function preloadSamples() {{ let nextIndex=1, completed=1; const update=()=>preloadStatus.textContent=` Preloading ${{completed}} / ${{samples.length}}…`; update(); async function worker() {{ while(nextIndex<samples.length) {{ const index=nextIndex++; try {{ await getSample(index); }} catch(error) {{ console.error('Failed to preload sample',index,error); }} completed++; update(); }} }} await Promise.all(Array.from({{length:4}},worker)); preloadStatus.textContent=' All samples preloaded.'; }}
 sampleSelect.onchange=()=>loadSample(+sampleSelect.value); timeSelect.onchange=render;
 document.querySelector('#previous').onclick=()=>loadSample(+sampleSelect.value-1);
 document.querySelector('#next').onclick=()=>loadSample(+sampleSelect.value+1);
 (async()=>{{ const response=await fetch('/api/samples'); samples=await response.json(); sampleSelect.innerHTML=samples.map((name,i)=>`<option value="${{i}}">${{name}}</option>`).join(''); await loadSample(0); preloadSamples(); }})();
</script></body></html>'''


def preload_samples(paths, max_points):
    """Materialise all visualisation payloads once before accepting requests."""
    payloads = []
    total = len(paths)
    for index, path in enumerate(paths, 1):
        sample = load_sample(path, max_points)
        payloads.append(json.dumps(sample, separators=(',', ':')).encode('utf-8'))
        if index == total or index % 10 == 0:
            print(f'Preloaded {index}/{total} prediction files.', flush=True)
    return payloads


def make_handler(paths, payloads, max_points):
    class ViewerHandler(BaseHTTPRequestHandler):
        def send_body(self, body, status=HTTPStatus.OK):
            self.send_response(status)
            self.send_header('Content-Type', 'application/json; charset=utf-8')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, payload, status=HTTPStatus.OK):
            self.send_body(json.dumps(payload, separators=(',', ':')).encode('utf-8'), status)

        def do_GET(self):
            route = urlparse(self.path).path
            if route == '/':
                body = viewer_page(max_points).encode('utf-8')
                self.send_response(HTTPStatus.OK)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            if route == '/api/samples':
                self.send_json([path.stem for path in paths])
                return
            if route.startswith('/api/sample/'):
                try:
                    index = int(route.rsplit('/', 1)[1])
                    if index < 0:
                        raise ValueError
                    payload = payloads[index]
                except (ValueError, IndexError):
                    self.send_json({'error': 'unknown sample index'}, HTTPStatus.NOT_FOUND)
                    return
                self.send_body(payload)
                return
            self.send_json({'error': 'not found'}, HTTPStatus.NOT_FOUND)

    return ViewerHandler


def main():
    args = parse_args()
    if args.max_points < 1:
        raise ValueError('--max-points must be positive')
    if not 0 <= args.port <= 65535:
        raise ValueError('--port must be between 0 and 65535')
    prediction_dir = Path(args.prediction_dir)
    paths = sorted(prediction_dir.glob('*.npz'))
    if not paths:
        raise FileNotFoundError(f'No .npz files found in {prediction_dir}')

    print(f'Preloading {len(paths)} prediction files from {prediction_dir.resolve()} ...')
    payloads = preload_samples(paths, args.max_points)
    server = ThreadingHTTPServer((args.host, args.port),
                                 make_handler(paths, payloads, args.max_points))
    address, port = server.server_address[:2]
    host_for_url = 'localhost' if address == '0.0.0.0' else address
    url = f'http://{host_for_url}:{port}/'
    print(f'Serving {len(paths)} preloaded prediction files.')
    print(f'Open {url} in a browser. Press Ctrl-C to stop the server.')
    if args.open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nViewer stopped.')
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
