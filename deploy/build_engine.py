"""Build engines locally using TRT 8.4 or 10; never ship server engines to Orin."""
import argparse
import shutil
from pathlib import Path
import tensorrt as trt
from deploy.runtime import load_plugins

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',default='deploy/artifacts/model')
    parser.add_argument('--output',default='deploy/artifacts/fp16')
    parser.add_argument('--plugin',default='deploy/build/plugins/libagriocc_plugins.so')
    parser.add_argument('--precision',choices=['fp32','fp16','int8'],default='fp16')
    parser.add_argument('--workspace-mib',type=int,default=1024)
    parser.add_argument('--stage',choices=['all','backbone','bev','head'],default='all')
    args=parser.parse_args()
    plugin=load_plugins(args.plugin)
    root,output=Path(args.model_dir),Path(args.output)
    output.mkdir(parents=True,exist_ok=True)
    for name in ['model.json','example.npz','quantization.json']:
        if (root/name).exists(): shutil.copy2(root/name,output/name)
    for stage in ['backbone','bev','head'] if args.stage=='all' else [args.stage]:
        logger=trt.Logger(trt.Logger.INFO)
        builder=trt.Builder(logger)
        network=builder.create_network(1<<int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser=trt.OnnxParser(network,logger)
        source=root/('backbone_int8.onnx' if stage=='backbone' and args.precision=='int8' else f'{stage}.onnx')
        if not parser.parse(source.read_bytes()):
            raise RuntimeError('\n'.join(str(parser.get_error(i)) for i in range(parser.num_errors)))
        config=builder.create_builder_config()
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE,args.workspace_mib*2**20)
        config.clear_flag(trt.BuilderFlag.TF32)
        config.profiling_verbosity=trt.ProfilingVerbosity.DETAILED
        if args.precision!='fp32': config.set_flag(trt.BuilderFlag.FP16)
        if args.precision=='int8' and stage=='backbone': config.set_flag(trt.BuilderFlag.INT8)
        if args.precision!='fp32':
            config.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for index in range(network.num_layers):
                layer=network.get_layer(index)
                norm_layer=any(tag in layer.name for tag in ['/norm/', '/norms.', '/out_norm/'])
                arithmetic=layer.type in [trt.LayerType.REDUCE,trt.LayerType.ELEMENTWISE,trt.LayerType.UNARY]
                if arithmetic and (norm_layer or layer.type==trt.LayerType.REDUCE):
                    if layer.num_outputs and layer.get_output(0).dtype==trt.float32:
                        layer.precision=trt.float32
                        layer.set_output_type(0,trt.float32)
        # Start with native operator tactics. A target-specific timing cache
        # is deliberately not copied between Ada and Orin.
        plan=builder.build_serialized_network(network,config)
        if plan is None: raise RuntimeError(f'Failed to build {stage} ({trt.__version__})')
        (output/f'{stage}.engine').write_bytes(bytes(plan))
        runtime=trt.Runtime(logger)
        inspection_engine=runtime.deserialize_cuda_engine(bytes(plan))
        inspector=inspection_engine.create_engine_inspector()
        (output/f'{stage}_layers.json').write_text(inspector.get_engine_information(trt.LayerInformationFormat.JSON))
        print('Built',stage,flush=True)
    import json
    (output/'build.json').write_text(json.dumps({'tensorrt':trt.__version__,'precision':args.precision,
        'workspace_mib':args.workspace_mib,'target_build_required':True},indent=2)+'\n')

if __name__=='__main__': main()
