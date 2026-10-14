"""Check fused CUDA resize/normalize against FarmSim's OpenCV float pipeline."""
import argparse
import ctypes as C
import cv2
import numpy as np
from deploy.runtime import CUDA, load_plugins

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plugin',default='deploy/build/plugins/libagriocc_plugins.so')
    args=parser.parse_args()
    plugin=load_plugins(args.plugin); cuda=CUDA()
    images=np.random.default_rng(0).integers(0,256,(3,37,61,3),dtype=np.uint8)
    reference=[]
    for image in images:
        image=cv2.resize(image.astype(np.float32),(32,16))
        # mmcv.imnormalize uses OpenCV scalar arithmetic with float64 means.
        cv2.subtract(image,np.array([[103.53,116.28,123.675]],dtype=np.float64),image)
        reference.append(image.transpose(2,0,1))
    expected=np.ascontiguousarray(np.stack(reference))
    source=cuda.malloc(images.nbytes); destination=cuda.malloc(expected.nbytes)
    cuda.copy(source,images.ctypes.data,images.nbytes,1)
    function=plugin.agrioccPreprocess
    function.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p]
    cuda.check(function(source,destination,3,61,37,32,16,cuda.stream))
    output=np.empty_like(expected); cuda.copy(output.ctypes.data,destination,output.nbytes,2); cuda.synchronize()
    difference=float(np.max(abs(output-expected)))
    np.testing.assert_allclose(output,expected,rtol=1e-4,atol=1e-3)
    print({'max_absolute_difference':difference})
    cuda.close()

if __name__=='__main__': main()
