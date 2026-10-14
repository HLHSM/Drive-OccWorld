"""Export three connected static ONNX graphs, compatible with TRT 8.4."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import onnx
from deploy.common import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_model, samples, write_json
from deploy.model import ImageEncoder, BEVEncoder, OccupancyHead

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',default=DEFAULT_CHECKPOINT)
    parser.add_argument('--output',default='deploy/artifacts/model')
    parser.add_argument('--calibration-json',help='Actual robot lidar2img matrices at image_size=[width,height]; rescaled to model input')
    args=parser.parse_args()
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    cfg,model=load_model(args.config,args.checkpoint)
    image,metadata,target,_=next(samples(args.config,1))
    if args.calibration_json:
        calibration=json.loads(Path(args.calibration_json).read_text())
        projection=np.asarray(calibration['lidar2img'],dtype=np.float64)
        if projection.shape!=(3,4,4): raise ValueError('Expected three 4x4 lidar2img matrices in camera_order')
        native_width,native_height=calibration['image_size']
        projection[:,0]*=image.shape[-1]/native_width
        projection[:,1]*=image.shape[-2]/native_height
        metadata['lidar2img']=projection
    with torch.no_grad():
        image=image.cuda()
        backbone=ImageEncoder(model).eval()
        features=backbone(image)
        bev=BEVEncoder(model,metadata,features).eval()
        head=OccupancyHead(model,for_export=True).eval()
        encoded=bev(*features)
        logits=head(encoded)
        feature_names=[f'feature_{i}' for i in range(len(features))]
        graphs=[('backbone',backbone,(image,),['images'],feature_names),
                ('bev',bev,features,feature_names,['bev']),
                ('head',head,(encoded,),['bev'],['logits'])]
        for name,module,inputs,in_names,out_names in graphs:
            torch.onnx.export(module,inputs,str(root/f'{name}.onnx'),opset_version=13,
                input_names=in_names,output_names=out_names,dynamic_axes=None,
                do_constant_folding=True,dynamo=False,custom_opsets={'agriocc':1})
            onnx.checker.check_model(onnx.load(str(root/f'{name}.onnx')))
            print('Exported',name,flush=True)
        from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import FARMSIM_CLASSES
        specification={'config':str(Path(args.config).resolve()),'checkpoint':str(Path(args.checkpoint).resolve()),
            'opset':13,'batch_size':1,'image_shape':list(image.shape),
            'feature_shapes':[list(f.shape) for f in features], 'bev_shape':list(encoded.shape),
            'logits_shape':list(logits.shape),'occupancy_shape':list(target.shape),
            'classes':list(FARMSIM_CLASSES),'geometry_hash':bev.geometry_hash,
            'lidar2img':np.asarray(metadata['lidar2img']).tolist(),
            'img_shape':metadata['img_shape'],'point_cloud_range':cfg.model.point_cloud_range,
            'preprocessing':{'camera_order':['front','front_left','front_right'],
                             'note':'See actual dataset camera_names; camera order is recorded below.',
                             'color':'BGR','mean':[103.53,116.28,123.675],'std':[1.,1.,1.],
                             'resize':'OpenCV INTER_LINEAR, exact width/height'},
            'target':{'device':'Jetson AGX Orin','jetpack':'5.0.2','tensorrt':'8.4.1','sm':87}}
        specification['calibration_source']=args.calibration_json or 'FarmSim validation first frame'
        from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import FRONT_RGB_CAMERAS
        specification['preprocessing']['camera_order']=list(FRONT_RGB_CAMERAS)
        specification['preprocessing'].pop('note')
        write_json(root/'model.json',specification)
        np.savez(root/'example.npz',images=image.cpu().numpy(),logits=logits.cpu().numpy(),target=target.numpy())

if __name__=='__main__': main()
