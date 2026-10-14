"""Verify CUDA sampler and static model against the original FP32 graph."""
import argparse
import torch
from deploy.common import DEFAULT_CONFIG, DEFAULT_CHECKPOINT, load_model, samples, write_json
from deploy.model import ImageEncoder, BEVEncoder, OccupancyHead
from deploy.cuda_ops import attention

def error(a,b):
    difference=(a.float()-b.float()).abs()
    return {'max_abs':float(difference.max()),'mean_abs':float(difference.mean())}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=DEFAULT_CONFIG)
    parser.add_argument('--checkpoint',default=DEFAULT_CHECKPOINT)
    parser.add_argument('--skip-cuda',action='store_true')
    parser.add_argument('--output',default='deploy/artifacts/verification.json')
    args=parser.parse_args()
    torch.manual_seed(0)
    result={}
    with torch.inference_mode():
        if not args.skip_cuda:
            for dtype in [torch.float32,torch.float16]:
                shapes=torch.tensor([[7,9],[3,4]],device='cuda',dtype=torch.int32)
                starts=torch.tensor([0,63],device='cuda',dtype=torch.int32)
                value=torch.randn(2,75,8,32,device='cuda',dtype=dtype)
                locations=torch.rand(2,23,8,2,4,2,device='cuda',dtype=dtype)*1.4-.2
                weights=torch.rand(2,23,8,2,4,device='cuda',dtype=dtype)
                reference=attention(value.float(),shapes,starts,locations.float(),weights.float(),'reference')
                output=attention(value,shapes,starts,locations,weights)
                stats=error(output,reference)
                torch.testing.assert_close(output.float(),reference,atol=2e-5 if dtype==torch.float32 else .005,
                                           rtol=2e-5 if dtype==torch.float32 else .003)
                result[str(dtype)]=stats
        cfg,model=load_model(args.config,args.checkpoint)
        image,metadata,target,_=next(samples(args.config,1))
        image=image.cuda()
        backbone=ImageEncoder(model)
        features=backbone(image)
        original=model.pts_bbox_head([f.unsqueeze(0) for f in features],[metadata],None,only_bev=True)
        original_logits=model.future_pred_head._agri_amoe_decode(original.unsqueeze(0))[0]
        bev=BEVEncoder(model,metadata,features,'reference')
        head=OccupancyHead(model)
        static=bev(*features)
        static_logits=head(static)
        torch.testing.assert_close(static,original,atol=.001,rtol=.001)
        torch.testing.assert_close(static_logits,original_logits,atol=.003,rtol=.003)
        result['static_bev']=error(static,original)
        result['static_logits']=error(static_logits,original_logits)
        if not args.skip_cuda:
            bev.backend='cuda'
            cuda_bev=bev(*features)
            cuda_logits=head(cuda_bev)
            torch.testing.assert_close(cuda_bev,original,atol=.001,rtol=.001)
            torch.testing.assert_close(cuda_logits,original_logits,atol=.005,rtol=.003)
            result['cuda_bev']=error(cuda_bev,original)
            result['cuda_logits']=error(cuda_logits,original_logits)
        result['geometry_hash']=bev.geometry_hash
        result['class_agreement']=float((static_logits.argmax(-1)==original_logits.argmax(-1)).float().mean())
    write_json(args.output,result)
    print(result)

if __name__=='__main__': main()
