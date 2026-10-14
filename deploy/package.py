"""Package source/ONNX/calibration for target rebuild, excluding engines/binaries."""
import argparse
from pathlib import Path
import tarfile

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir',default='deploy/artifacts/model')
    parser.add_argument('--validation-dir')
    parser.add_argument('--output',default='deploy/artifacts/agriocc_orin_bundle.tar.gz')
    args=parser.parse_args()
    root=Path(__file__).resolve().parent
    output=Path(args.output); output.parent.mkdir(parents=True,exist_ok=True)
    with tarfile.open(output,'w:gz') as archive:
        for name in ['__init__.py','runtime.py','infer.py','validate.py','build_engine.py','check_preprocess.py','README.md','SERVER_RESULTS.md']:
            archive.add(root/name,arcname=f'deploy/{name}')
        for folder in ['trt_plugins','scripts']:
            for path in (root/folder).iterdir():
                if path.is_file(): archive.add(path,arcname=f'deploy/{folder}/{path.name}')
        for name in ['msda.cu','msda.h','preprocess.cu']:
            archive.add(root/'cuda_ops'/name,arcname=f'deploy/cuda_ops/{name}')
        for path in Path(args.model_dir).iterdir():
            if path.suffix in ['.onnx','.json','.npz']:
                archive.add(path,arcname=f'deploy/artifacts/model/{path.name}')
        if args.validation_dir:
            for path in Path(args.validation_dir).glob('*.npz'):
                archive.add(path,arcname=f'deploy/artifacts/validation/{path.name}')
    print(output)

if __name__=='__main__': main()
