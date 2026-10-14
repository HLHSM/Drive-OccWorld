"""Representative training-set INT8 Q/DQ PTQ of the image encoder/FPN."""
import argparse
import json
import numpy as np
from pathlib import Path
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod
from deploy.common import DEFAULT_CONFIG, samples, write_json

class Reader(CalibrationDataReader):
    def __init__(self,config,count):
        self.iterator=iter(samples(config,count,split='train'))
        self.count=0
    def get_next(self):
        try:
            image,_,_,_=next(self.iterator)
        except StopIteration:
            return None
        self.count+=1
        if self.count%8==0: print(f'Calibration frame {self.count}',flush=True)
        return {'images':image.numpy()}

class NpzReader(CalibrationDataReader):
    def __init__(self,directory,count):
        paths=sorted(Path(directory).glob('*.npz'))
        if not paths: raise FileNotFoundError(f'No calibration NPZs: {directory}')
        chosen=np.linspace(0,len(paths)-1,min(count,len(paths)),dtype=np.int64)
        self.paths=iter(paths[index] for index in chosen); self.count=0
    def get_next(self):
        path=next(self.paths,None)
        if path is None: return None
        with np.load(path) as data:
            image=np.ascontiguousarray(data['images'],dtype=np.float32)
        if image.ndim!=4 or image.shape[:2]!=(3,3): raise ValueError(f'Expected [3,3,H,W]: {path}')
        self.count+=1
        return {'images':image}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--model-dir',default='deploy/artifacts/model')
    parser.add_argument('--samples',type=int,default=128)
    parser.add_argument('--calibration-dir',help='Optional NPZ directory with normalized images; no labels required')
    parser.add_argument('--method',choices=['minmax','entropy'],default='minmax',help='MinMax bounds calibration RAM; entropy stores large intermediate activations')
    parser.add_argument('--exclude-convs',nargs='*',default=['/backbone/conv1/Conv'],help='Preserve the image stem (sensitive small-channel convolution) in floating point')
    args=parser.parse_args()
    if args.samples<1: parser.error('--samples must be positive')
    root=Path(args.model_dir)
    reader=NpzReader(args.calibration_dir,args.samples) if args.calibration_dir else Reader(args.config,args.samples)
    quantize_static(str(root/'backbone.onnx'),str(root/'backbone_int8.onnx'),reader,
        quant_format=QuantFormat.QDQ,activation_type=QuantType.QInt8,weight_type=QuantType.QInt8,
        per_channel=True,op_types_to_quantize=['Conv'],reduce_range=False,
        nodes_to_exclude=args.exclude_convs,
        calibrate_method=CalibrationMethod.Entropy if args.method=='entropy' else CalibrationMethod.MinMax,
        extra_options={'ActivationSymmetric':True,'WeightSymmetric':True,'QuantizeBias':False})
    from deploy.onnx_utils import normalize_qdq
    normalize_qdq(root/'backbone_int8.onnx')
    write_json(root/'quantization.json',{'format':'QDQ','precision':'INT8 image backbone/FPN, FP16/FP32 BEV and AMoE',
               'calibration_split':'npz' if args.calibration_dir else 'train',
               'calibration_source':args.calibration_dir or args.config,
               'calibration_samples':reader.count,'seed':0,'method':args.method,
               'excluded_convs':args.exclude_convs,
               'config':str(Path(args.config).resolve())})

if __name__=='__main__': main()
