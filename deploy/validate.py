"""Portable validation on exported NPZs; requires only NumPy, CUDA and TensorRT."""
import argparse
import json
from pathlib import Path
import statistics
import numpy as np
from deploy.runtime import Pipeline

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine-dir',required=True)
    parser.add_argument('--plugin',default='deploy/build/plugins/libagriocc_plugins.so')
    parser.add_argument('--data',default='deploy/artifacts/validation')
    parser.add_argument('--output',default='deploy/artifacts/orin_validation.json')
    args=parser.parse_args()
    paths=sorted(Path(args.data).glob('*.npz'))
    if not paths: raise FileNotFoundError(f'No validation NPZs in {args.data}')
    pipeline=Pipeline(args.engine_dir,args.plugin)
    names=pipeline.spec['classes']; classes=len(names)
    hist=np.zeros((classes,classes),dtype=np.int64); oracle_hist=np.zeros_like(hist)
    times=[]; different=0; total=0
    for index,path in enumerate(paths):
        sample=np.load(path)
        if str(sample['geometry_hash'])!=pipeline.spec['geometry_hash']:
            raise ValueError(f'Geometry mismatch: {path}')
        pipeline.upload(sample['images']); pipeline.cuda.synchronize()
        if index==0:
            for _ in range(20): pipeline.enqueue()
        times.append(pipeline.timed()[0]); logits=pipeline.download()
        if not np.isfinite(logits).all(): raise RuntimeError(f'Non-finite logits: {path}')
        height,width,depth=pipeline.spec['occupancy_shape']
        prediction=logits.transpose(0,3,1,2).reshape(1,classes,width,height,depth).transpose(0,1,3,2,4).argmax(1)[0]
        target=sample['target'].astype(np.int64); valid=target!=255
        hist+=np.bincount(classes*target[valid]+prediction[valid],minlength=classes**2).reshape(classes,classes)
        reference=sample['original_labels'].astype(np.int64)
        oracle_hist+=np.bincount(classes*target[valid]+reference[valid],minlength=classes**2).reshape(classes,classes)
        different+=int((prediction[valid]!=reference[valid]).sum()); total+=int(valid.sum())
    def summarize(matrix):
        union=matrix.sum(0)+matrix.sum(1)-np.diag(matrix)
        iou=np.divide(np.diag(matrix),union,out=np.zeros(classes),where=union>0)
        result={'miou':float(iou[union>0].mean()),'class_iou':dict(zip(names,map(float,iou)))}
        crop=names.index('crop')
        result['crop_precision']=float(matrix[crop,crop]/max(matrix[:,crop].sum(),1))
        result['crop_recall']=float(matrix[crop,crop]/max(matrix[crop,:].sum(),1))
        return result
    result={'samples':len(paths),'mean_ms':statistics.mean(times),'p95_ms':float(np.percentile(times,95)),
            'timing_scope':'GPU-resident images -> logits; excludes transfers/metrics',
            'metrics':summarize(hist),'original_metrics':summarize(oracle_hist),
            'label_disagreement_ratio':different/max(total,1),'tensorrt':__import__('tensorrt').__version__}
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2)+'\n'); print(json.dumps(result,indent=2))
    pipeline.close()

if __name__=='__main__': main()
