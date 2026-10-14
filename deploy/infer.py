"""Portable Orin inference/benchmark on a preprocessed NPZ or three BGR images."""
import argparse
import json
from pathlib import Path
import statistics
import os
import shutil
import subprocess
import numpy as np
from deploy.runtime import Pipeline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-dir',required=True)
    parser.add_argument('--plugin',required=True)
    parser.add_argument('--input',help='NPZ containing normalized images; model/example.npz is available')
    parser.add_argument('--images',nargs=3,help='Three raw image paths, in model.json camera_order')
    parser.add_argument('--warmup',type=int,default=20)
    parser.add_argument('--iterations',type=int,default=100)
    parser.add_argument('--output',default='prediction.npz')
    parser.add_argument('--cuda-graph',action='store_true')
    parser.add_argument('--gpu-preprocess',action='store_true',help='CUDA float resize + Caffe BGR normalization of --images')
    args=parser.parse_args()
    if bool(args.input)==bool(args.images): parser.error('Specify --input or --images')
    if args.gpu_preprocess and not args.images: parser.error('--gpu-preprocess requires --images')
    if args.warmup<0 or args.iterations<1: parser.error('Invalid timing counts')
    pipeline=Pipeline(args.engine_dir,args.plugin)
    if args.input:
        images=np.load(args.input)['images']
    else:
        import cv2
        _,_,height,width=pipeline.image_shape
        mean=np.array(pipeline.spec['preprocessing']['mean'],dtype=np.float64).reshape(1,-1)
        inverse_std=1./np.array(pipeline.spec['preprocessing']['std'],dtype=np.float64).reshape(1,-1)
        images=[]
        for path in args.images:
            image=cv2.imread(path)
            if image is None: raise FileNotFoundError(path)
            if args.gpu_preprocess:
                images.append(image)
            else:
                # FarmSim casts BEFORE resize, avoiding uint8 interpolation rounding.
                image=cv2.resize(image.astype(np.float32),(width,height),interpolation=cv2.INTER_LINEAR)
                cv2.subtract(image,mean,image)
                cv2.multiply(image,inverse_std,image)
                images.append(image.transpose(2,0,1))
        images=np.stack(images)
    if args.gpu_preprocess: pipeline.upload_bgr(images)
    else: pipeline.upload(images)
    pipeline.cuda.synchronize()
    if args.cuda_graph: pipeline.capture()
    for _ in range(args.warmup): pipeline.enqueue()
    pipeline.cuda.synchronize()
    after_warmup=pipeline.cuda.free_bytes()[0]
    times=[pipeline.timed() for _ in range(args.iterations)]
    occupancy=pipeline.download_labels()
    Path(args.output).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(args.output,prediction=occupancy,point_cloud_range=pipeline.spec['point_cloud_range'])
    result={'tensorrt':__import__('tensorrt').__version__,'iterations':args.iterations,
            'cuda_graph':args.cuda_graph,
            'mean_ms':statistics.mean(t[0] for t in times),'cuda_mean_ms':statistics.mean(t[1] for t in times),
            'p95_ms':float(np.percentile([t[0] for t in times],95)),
            'timing_scope':'GPU-resident normalized image -> final logits; excludes preprocessing/transfers',
            'engine_context_buffer_delta_mib':(pipeline.memory_before-pipeline.memory_after)/2**20}
    result['after_warmup_device_memory_delta_mib']=(pipeline.memory_before-after_warmup)/2**20
    result['memory_scope']='CUDA free-memory delta excludes CUDA context created before baseline; external process changes can affect it'
    if shutil.which('nvidia-smi'):
        try:
            output_memory=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True)
            result['process_gpu_memory_mib']=sum(float(line.split(',')[1]) for line in output_memory.splitlines() if int(line.split(',')[0])==os.getpid())
        except (ValueError,subprocess.CalledProcessError): pass
    print(json.dumps(result,indent=2))
    Path(args.output).with_suffix('.json').write_text(json.dumps(result,indent=2)+'\n')
    pipeline.close()

if __name__=='__main__': main()
