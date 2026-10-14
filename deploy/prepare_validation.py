"""Export validation frames/GT/original-AMP labels for PyTorch-free Orin checks."""
import argparse
from pathlib import Path
import numpy as np
import torch
from mmcv.runner import wrap_fp16_model
from deploy.common import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_model, samples, labels
from deploy.evaluate import original
from deploy.model import geometry_hash

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',default=DEFAULT_CHECKPOINT)
    parser.add_argument('--samples',type=int,default=20)
    parser.add_argument('--output',default='deploy/artifacts/validation')
    args=parser.parse_args()
    if args.samples<1: parser.error('At least one sample required')
    root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    cfg,model=load_model(args.config,args.checkpoint); wrap_fp16_model(model)
    with torch.inference_mode():
        for index,(image,metadata,target,_) in enumerate(samples(args.config,args.samples,sampling='uniform')):
            height,width,_=target.shape
            reference=labels(original(model,image.cuda(),metadata),height,width)[0].cpu().numpy().astype(np.uint8)
            np.savez_compressed(root/f'{index:06d}.npz',images=image.numpy(),target=target.numpy().astype(np.uint8),
                                original_labels=reference,geometry_hash=geometry_hash(metadata,height,width))
            print(f'Prepared {index+1}',flush=True)

if __name__=='__main__': main()
