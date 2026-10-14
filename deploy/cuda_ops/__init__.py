import os
from pathlib import Path
import torch

_extension = None

def extension():
    global _extension
    if _extension is None:
        os.environ.setdefault('CUDA_HOME', '/usr/local/cuda')
        os.environ.setdefault('MAX_JOBS', '2')
        from torch.utils.cpp_extension import load
        root = Path(__file__).parent
        build = root.parent / 'build' / 'torch_msda'
        build.mkdir(parents=True, exist_ok=True)
        _extension = load('agriocc_msda', [str(root/'torch_binding.cpp'), str(root/'msda.cu')],
                          build_directory=str(build), extra_cflags=['-O3'],
                          extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr'], verbose=True)
    return _extension

class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, shapes, starts, locations, weights):
        return extension().forward(value.contiguous(), shapes.int().contiguous(),
                                   starts.int().contiguous(), locations.to(value.dtype).contiguous(),
                                   weights.to(value.dtype).contiguous())

    @staticmethod
    def symbolic(g, value, shapes, starts, locations, weights):
        output = g.op('agriocc::AgriOccMSDeformAttn', value, shapes, starts, locations, weights,
                      plugin_version_s='1', plugin_namespace_s='agriocc')
        sizes = value.type().sizes()
        loc_sizes = locations.type().sizes()
        if sizes and loc_sizes:
            output.setType(value.type().with_sizes([sizes[0], loc_sizes[1], sizes[2]*sizes[3]]))
        return output

def attention(value, shapes, starts, locations, weights, backend='cuda'):
    if backend == 'reference':
        from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
        return multi_scale_deformable_attn_pytorch(value, shapes.long(), locations, weights)
    return _Attention.apply(value, shapes, starts, locations, weights)
