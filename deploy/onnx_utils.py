"""Normalize Q/DQ graphs for TensorRT 8.4/10 parser restrictions."""
import numpy as np
import onnx
from onnx import numpy_helper

def normalize_qdq(path):
    model=onnx.load(str(path))
    initializers={x.name:numpy_helper.to_array(x) for x in model.graph.initializer}
    nodes=[]; folded=[]
    for node in model.graph.node:
        if node.op_type=='DequantizeLinear' and node.input[0] in initializers:
            value=initializers[node.input[0]]
            if value.dtype==np.int32:
                # ORT quantizes Conv biases to int32. TensorRT accepts FP32
                # bias and INT8 weight/activation QDQ, not INT32 DQ nodes.
                scale=initializers[node.input[1]]
                zero=initializers[node.input[2]] if len(node.input)>2 else 0
                folded.append(numpy_helper.from_array(((value.astype(np.float32)-zero)*scale).astype(np.float32),node.output[0]))
                continue
        nodes.append(node)
    del model.graph.node[:]; model.graph.node.extend(nodes)
    model.graph.initializer.extend(folded)
    used={name for node in nodes for name in node.input}
    kept=[]
    for initializer in model.graph.initializer:
        if initializer.name not in used: continue
        array=numpy_helper.to_array(initializer)
        if array.dtype==np.float64:
            initializer=numpy_helper.from_array(array.astype(np.float32),initializer.name)
        kept.append(initializer)
    del model.graph.initializer[:]; model.graph.initializer.extend(kept)
    onnx.checker.check_model(model)
    onnx.save(model,str(path))
    return len(folded)

def restore_convolutions(quantized_path,float_path,names):
    """Restore sensitive Conv nodes to floating point, keeping other QDQ nodes."""
    model,reference=onnx.load(str(quantized_path)),onnx.load(str(float_path))
    source={node.name:node for node in reference.graph.node}
    parameters={value.name:value for value in reference.graph.initializer}
    current={value.name:value for value in model.graph.initializer}
    for node in model.graph.node:
        if node.name in names:
            original=source[node.name]
            del node.input[:]; node.input.extend(original.input)
            for name in original.input:
                if name in parameters: current[name]=parameters[name]
    needed={output.name for output in model.graph.output}
    kept=[]
    for node in reversed(model.graph.node):
        if any(output in needed for output in node.output):
            kept.append(node); needed.update(node.input)
    del model.graph.node[:]; model.graph.node.extend(reversed(kept))
    del model.graph.initializer[:]
    model.graph.initializer.extend(value for name,value in current.items() if name in needed)
    onnx.checker.check_model(model); onnx.save(model,str(quantized_path))
