"""TensorRT 8.4/10 runtime with shared CUDA buffers; no PyTorch/MMCV needed."""
import ctypes as C
import ctypes.util
import json
from pathlib import Path
import time
import numpy as np
import tensorrt as trt

class CUDA:
    def __init__(self):
        library=ctypes.util.find_library('cudart')
        if not library:
            for candidate in ['/usr/local/cuda/lib64/libcudart.so','/usr/local/cuda/lib64/libcudart.so.12',
                              '/usr/local/cuda/lib64/libcudart.so.11.0']:
                if Path(candidate).exists(): library=candidate; break
        if not library: raise RuntimeError('CUDA runtime not found; set LD_LIBRARY_PATH to CUDA lib64')
        self.lib=C.CDLL(library)
        self.allocations=[]
        self.stream=C.c_void_p()
        self.check(self.lib.cudaStreamCreate(C.byref(self.stream)))
    @staticmethod
    def check(code):
        if code: raise RuntimeError(f'CUDA error code {code}')
    def malloc(self,size):
        pointer=C.c_void_p(); self.check(self.lib.cudaMalloc(C.byref(pointer),C.c_size_t(size)))
        self.allocations.append(pointer)
        return pointer.value
    def copy(self,destination,source,size,kind):
        self.check(self.lib.cudaMemcpyAsync(C.c_void_p(destination),C.c_void_p(source),C.c_size_t(size),C.c_int(kind),self.stream))
    def synchronize(self): self.check(self.lib.cudaStreamSynchronize(self.stream))
    def event(self):
        event=C.c_void_p(); self.check(self.lib.cudaEventCreate(C.byref(event))); return event
    def record(self,event): self.check(self.lib.cudaEventRecord(event,self.stream))
    def elapsed(self,start,end):
        value=C.c_float(); self.check(self.lib.cudaEventElapsedTime(C.byref(value),start,end)); return value.value
    def free_bytes(self):
        free,total=C.c_size_t(),C.c_size_t()
        self.check(self.lib.cudaMemGetInfo(C.byref(free),C.byref(total)))
        return free.value,total.value
    def close(self):
        self.synchronize()
        for pointer in self.allocations: self.check(self.lib.cudaFree(pointer))
        self.allocations.clear()
        self.check(self.lib.cudaStreamDestroy(self.stream))

def load_plugins(path):
    library=C.CDLL(str(Path(path).resolve()),mode=C.RTLD_GLOBAL)
    initialize=library.initLibAgriOccPlugins
    initialize.restype=C.c_bool
    if not initialize(): raise RuntimeError('Failed to register AgriOcc plugin')
    trt.init_libnvinfer_plugins(trt.Logger(trt.Logger.WARNING),'')
    return library

class Engine:
    def __init__(self,path,cuda):
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(path).read_bytes())
        if self.engine is None: raise RuntimeError(f'Cannot load {path}; rebuild on target device/TRT version')
        self.context=self.engine.create_execution_context()
        if self.context is None: raise RuntimeError(f'Cannot create context for {path}')
        self.cuda=cuda
        self.new_api=int(trt.__version__.split('.')[0])>=10
        if self.new_api:
            self.names=[self.engine.get_tensor_name(i) for i in range(self.engine.num_io_tensors)]
            self.inputs=[n for n in self.names if self.engine.get_tensor_mode(n)==trt.TensorIOMode.INPUT]
            self.shapes={n:tuple(self.engine.get_tensor_shape(n)) for n in self.names}
            self.dtypes={n:np.dtype(trt.nptype(self.engine.get_tensor_dtype(n))) for n in self.names}
        else:
            self.names=[self.engine.get_binding_name(i) for i in range(self.engine.num_bindings)]
            self.inputs=[n for i,n in enumerate(self.names) if self.engine.binding_is_input(i)]
            self.shapes={n:tuple(self.engine.get_binding_shape(i)) for i,n in enumerate(self.names)}
            self.dtypes={n:np.dtype(trt.nptype(self.engine.get_binding_dtype(i))) for i,n in enumerate(self.names)}
        self.outputs=[n for n in self.names if n not in self.inputs]
        self.pointers={}
        for name in self.outputs:
            if min(self.shapes[name])<0: raise ValueError('Only static engines are supported')
            self.pointers[name]=cuda.malloc(int(np.prod(self.shapes[name]))*self.dtypes[name].itemsize)
    def enqueue(self,inputs):
        pointers={**self.pointers,**inputs}
        if set(pointers)!=set(self.names): raise ValueError(f'Inputs mismatch: {set(inputs)} vs {set(self.inputs)}')
        if self.new_api:
            for name,pointer in pointers.items(): self.context.set_tensor_address(name,int(pointer))
            success=self.context.execute_async_v3(self.cuda.stream.value)
        else:
            success=self.context.execute_async_v2([int(pointers[n]) for n in self.names],self.cuda.stream.value)
        if not success: raise RuntimeError('TensorRT execution failed')
        return self.pointers

class Pipeline:
    def __init__(self,directory,plugin):
        directory=Path(directory)
        self.plugin=load_plugins(plugin)
        self.spec=json.loads((directory/'model.json').read_text())
        self.cuda=CUDA()
        self.memory_before=self.cuda.free_bytes()[0]
        self.backbone=Engine(directory/'backbone.engine',self.cuda)
        self.bev=Engine(directory/'bev.engine',self.cuda)
        self.head=Engine(directory/'head.engine',self.cuda)
        self.image_dtype=self.backbone.dtypes['images']
        self.image_shape=self.backbone.shapes['images']
        self.input_pointer=self.cuda.malloc(int(np.prod(self.image_shape))*self.image_dtype.itemsize)
        self.memory_after=self.cuda.free_bytes()[0]
        self.last_host_input=None
        self.raw_pointer=None; self.raw_size=0
        self.label_pointer=None; self.invalid_pointer=None
        self.graph=None; self.graph_exec=None
        self.start,self.end=self.cuda.event(),self.cuda.event()
    def enqueue(self,pointer=None):
        if self.graph_exec is not None:
            if pointer is not None: raise ValueError('CUDA graph uses its fixed internal input buffer')
            self.cuda.check(self.cuda.lib.cudaGraphLaunch(self.graph_exec,self.cuda.stream))
            return self.head.pointers['logits']
        features=self.backbone.enqueue({'images':pointer or self.input_pointer})
        encoded=self.bev.enqueue(features)
        return self.head.enqueue(encoded)['logits']
    def capture(self):
        if self.graph_exec is not None: return
        for _ in range(3): self.enqueue()
        self.cuda.synchronize()
        self.cuda.check(self.cuda.lib.cudaStreamBeginCapture(self.cuda.stream,C.c_int(0)))
        self.enqueue()
        graph=C.c_void_p()
        self.cuda.check(self.cuda.lib.cudaStreamEndCapture(self.cuda.stream,C.byref(graph)))
        executable=C.c_void_p()
        if hasattr(self.cuda.lib,'cudaGraphInstantiateWithFlags'):
            self.cuda.check(self.cuda.lib.cudaGraphInstantiateWithFlags(C.byref(executable),graph,C.c_ulonglong(0)))
        else:
            self.cuda.check(self.cuda.lib.cudaGraphInstantiate(C.byref(executable),graph,None,None,C.c_size_t(0)))
        self.graph,self.graph_exec=graph,executable
    def upload(self,images):
        images=np.ascontiguousarray(images,dtype=self.image_dtype)
        if tuple(images.shape)!=self.image_shape: raise ValueError(f'Expected {self.image_shape}, got {images.shape}')
        self.last_host_input=images  # Keep alive through async H2D.
        self.cuda.copy(self.input_pointer,images.ctypes.data,images.nbytes,1)
    def upload_bgr(self,images):
        images=np.ascontiguousarray(images,dtype=np.uint8)
        if images.ndim!=4 or images.shape[0]!=self.image_shape[0] or images.shape[-1]!=3:
            raise ValueError('Expected equal-size BGR images [camera,height,width,3]')
        if self.image_dtype!=np.float32: raise ValueError('GPU preprocessing requires FP32 image I/O')
        if self.raw_pointer is None:
            self.raw_size=images.nbytes; self.raw_pointer=self.cuda.malloc(self.raw_size)
        if images.nbytes!=self.raw_size: raise ValueError('Raw camera shape changed; recreate pipeline')
        self.last_host_input=images
        self.cuda.copy(self.raw_pointer,images.ctypes.data,images.nbytes,1)
        function=self.plugin.agrioccPreprocess
        function.argtypes=[C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p]
        self.cuda.check(function(self.raw_pointer,self.input_pointer,images.shape[0],images.shape[2],images.shape[1],
                                 self.image_shape[-1],self.image_shape[-2],self.cuda.stream))
    def download(self):
        output=np.empty(self.head.shapes['logits'],dtype=self.head.dtypes['logits'])
        self.cuda.copy(output.ctypes.data,self.head.pointers['logits'],output.nbytes,2)
        self.cuda.synchronize()
        return output
    def download_labels(self):
        height,width,depth=self.spec['occupancy_shape']
        labels=np.empty((height,width,depth),dtype=np.uint8)
        status=np.empty(1,dtype=np.int32)
        if self.label_pointer is None:
            self.label_pointer=self.cuda.malloc(labels.nbytes)
            self.invalid_pointer=self.cuda.malloc(status.nbytes)
        dtype=self.head.dtypes['logits']
        if dtype not in [np.dtype('float32'),np.dtype('float16')]: raise TypeError(dtype)
        function=self.plugin.agrioccArgmax
        function.argtypes=[C.c_void_p,C.c_void_p,C.c_void_p,C.c_int,C.c_int,C.c_int,C.c_int,C.c_int,C.c_void_p]
        self.cuda.check(function(self.head.pointers['logits'],self.label_pointer,self.invalid_pointer,
                                 height,width,depth,len(self.spec['classes']),int(dtype==np.float16),self.cuda.stream))
        self.cuda.copy(labels.ctypes.data,self.label_pointer,labels.nbytes,2)
        self.cuda.copy(status.ctypes.data,self.invalid_pointer,status.nbytes,2)
        self.cuda.synchronize()
        if status[0]: raise RuntimeError('Non-finite logits; rebuild with safe FP32 normalization')
        return labels
    def timed(self,pointer=None):
        self.cuda.synchronize()
        wall=time.perf_counter()
        self.cuda.record(self.start); self.enqueue(pointer); self.cuda.record(self.end)
        self.cuda.synchronize()
        return (time.perf_counter()-wall)*1000,self.cuda.elapsed(self.start,self.end)
    def close(self):
        self.cuda.synchronize()
        if self.graph_exec is not None:
            self.cuda.lib.cudaGraphExecDestroy(self.graph_exec)
            self.cuda.lib.cudaGraphDestroy(self.graph)
        self.cuda.lib.cudaEventDestroy(self.start); self.cuda.lib.cudaEventDestroy(self.end)
        # Contexts own workspace; release before freeing shared buffers.
        for engine in [self.head,self.bev,self.backbone]:
            engine.context=None; engine.engine=None; engine.runtime=None
        self.cuda.close()
