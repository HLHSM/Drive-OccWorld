"""Compare deployment speed/accuracy using identical FarmSim validation frames."""
import argparse
import json
import statistics
import time
from pathlib import Path
import numpy as np
import torch
from mmcv.runner import wrap_fp16_model
from deploy.common import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_model, samples, labels, metrics, write_json
from deploy.model import ImageEncoder, BEVEncoder, OccupancyHead, geometry_hash

def original(model,image,metadata):
    features=model.extract_feat(image.unsqueeze(0),[metadata])
    bev=model.pts_bbox_head(features,[metadata],None,only_bev=True)
    repeated=bev[None,None].repeat(1,len(model.future_pred_head.bev_pred_head),1,1,1)
    return model.future_pred_head.forward_head(repeated)[0,-1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',default=DEFAULT_CHECKPOINT)
    parser.add_argument('--backend',choices=['original','reference','cuda','trt'],default='cuda')
    parser.add_argument('--precision',choices=['fp32','fp16'],default='fp32',help='Torch static graph precision; original uses checkpoint AMP settings')
    parser.add_argument('--engine-dir',default='deploy/artifacts/fp16')
    parser.add_argument('--plugin',default='deploy/build/plugins/libagriocc_plugins.so')
    parser.add_argument('--samples',type=int,default=100,help='0 evaluates full validation set')
    parser.add_argument('--warmup',type=int,default=10)
    parser.add_argument('--compare-original',action='store_true')
    parser.add_argument('--sampling',choices=['leading','uniform'],default='uniform')
    parser.add_argument('--output',default='deploy/artifacts/evaluation.json')
    args=parser.parse_args()
    if args.samples<0 or args.warmup<0: parser.error('counts must be non-negative')
    from projects.mmdet3d_plugin.datasets.farmsim_world_dataset import FARMSIM_CLASSES
    names=list(FARMSIM_CLASSES); classes=len(names)
    hist=np.zeros((classes,classes),dtype=np.int64)
    oracle_hist=np.zeros_like(hist)
    disagreement=0; valid_total=0; logit_difference=0.; count=0
    cfg,model=load_model(args.config,args.checkpoint) if args.backend!='trt' or args.compare_original else (None,None)
    if model is not None and (args.backend=='original' or args.compare_original): wrap_fp16_model(model)
    pipeline=None; backbone=bev=head=None
    wall_times=[]; gpu_times=[]
    if args.backend=='trt':
        from deploy.runtime import Pipeline
        pipeline=Pipeline(args.engine_dir,args.plugin)
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for image,metadata,target,_ in samples(args.config,args.samples,sampling=args.sampling):
            image=image.cuda()
            height,width,depth=target.shape
            if args.backend in ['cuda','reference'] and backbone is None:
                backbone=ImageEncoder(model)
                features=backbone(image)
                bev=BEVEncoder(model,metadata,features,args.backend)
                head=OccupancyHead(model)
            if bev is not None and geometry_hash(metadata,height,width)!=bev.geometry_hash:
                raise ValueError('Calibration changed; export/select a separate geometry bundle')
            if pipeline is not None and geometry_hash(metadata,height,width)!=pipeline.spec['geometry_hash']:
                raise ValueError('Calibration changed; use a matching geometry bundle')
            def forward():
                if args.backend=='original': return original(model,image,metadata)
                if args.backend=='trt':
                    return pipeline.enqueue(image.data_ptr())
                with torch.autocast('cuda',dtype=torch.float16,enabled=args.precision=='fp16'):
                    return head(bev(*backbone(image)))
            if count==0:
                for _ in range(args.warmup): forward()
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall=time.perf_counter()
            if pipeline is not None:
                wall_ms,gpu_ms=pipeline.timed(image.data_ptr())
                output=torch.from_numpy(pipeline.download())
            else:
                start.record(); output=forward(); end.record(); torch.cuda.synchronize()
                wall_ms=(time.perf_counter()-wall)*1000; gpu_ms=start.elapsed_time(end)
                output=output.cpu()
            wall_times.append(wall_ms); gpu_times.append(gpu_ms)
            if not torch.isfinite(output).all():
                raise RuntimeError('Non-finite deployment logits; check FP32 normalization/gradient-energy precision')
            prediction=labels(output,height,width)[0].numpy()
            target=target.numpy()
            valid=target!=255
            hist+=np.bincount(classes*target[valid]+prediction[valid],minlength=classes**2).reshape(classes,classes)
            if args.compare_original:
                oracle=original(model,image,metadata).cpu()
                oracle_prediction=labels(oracle,height,width)[0].numpy()
                oracle_hist+=np.bincount(classes*target[valid]+oracle_prediction[valid],minlength=classes**2).reshape(classes,classes)
                disagreement+=int((oracle_prediction[valid]!=prediction[valid]).sum())
                valid_total+=int(valid.sum())
                logit_difference=max(logit_difference,float((oracle-output.float()).abs().max()))
            count+=1
            if count%20==0: print(f'Evaluated {count} frames, mean={statistics.mean(wall_times):.2f} ms',flush=True)
    result={'backend':args.backend,'precision':args.precision if args.backend!='trt' else json.loads((Path(args.engine_dir)/'build.json').read_text())['precision'],
            'samples':count,'warmup':args.warmup,'gpu':torch.cuda.get_device_name(),
            'sampling':args.sampling,
            'timing_scope':'GPU-resident normalized image -> final logits; excludes I/O, H2D, D2H, metrics',
            'wall_mean_ms':statistics.mean(wall_times),'wall_median_ms':statistics.median(wall_times),
            'wall_p95_ms':float(np.percentile(wall_times,95)),'cuda_mean_ms':statistics.mean(gpu_times),
            'wall_ms':wall_times,'peak_torch_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'peak_torch_reserved_mib':torch.cuda.max_memory_reserved()/2**20,
            'torch_memory_note':'Includes original comparison model when --compare-original is set; TensorRT allocations are outside PyTorch allocator',
            'metrics':metrics(hist,names),'config':args.config,'checkpoint':args.checkpoint}
    if pipeline is not None:
        result['trt_initial_device_memory_delta_mib']=(pipeline.memory_before-pipeline.memory_after)/2**20
        result['trt_memory_note']='CUDA free-memory difference across engine/context/buffer creation; excludes context loaded before baseline and may include external process variation'
        pipeline.close()
    if args.compare_original:
        result['original_metrics']=metrics(oracle_hist,names)
        result['label_disagreement_ratio']=disagreement/max(valid_total,1)
        result['max_logit_difference']=logit_difference
        result['miou_delta']=result['metrics']['miou']-result['original_metrics']['miou']
    write_json(args.output,result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['wall_ms']},indent=2,ensure_ascii=False))

if __name__=='__main__': main()
