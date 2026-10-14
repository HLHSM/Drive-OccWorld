#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include "msda.h"

torch::Tensor forward(torch::Tensor value, torch::Tensor shapes,
    torch::Tensor starts, torch::Tensor locations, torch::Tensor weights) {
  TORCH_CHECK(value.is_cuda() && shapes.is_cuda() && starts.is_cuda() && locations.is_cuda() && weights.is_cuda(), "CUDA inputs required");
  TORCH_CHECK(value.is_contiguous() && shapes.is_contiguous() && starts.is_contiguous() && locations.is_contiguous() && weights.is_contiguous(), "contiguous inputs required");
  TORCH_CHECK(value.dim()==4 && locations.dim()==6 && weights.dim()==5, "invalid MSDA ranks");
  TORCH_CHECK(shapes.scalar_type()==torch::kInt32 && starts.scalar_type()==torch::kInt32, "int32 shapes/starts required");
  TORCH_CHECK(value.scalar_type()==torch::kFloat32 || value.scalar_type()==torch::kFloat16, "FP32/FP16 required");
  TORCH_CHECK(locations.scalar_type()==value.scalar_type() && weights.scalar_type()==value.scalar_type(), "matching data types required");
  c10::cuda::CUDAGuard guard(value.device());
  auto output = torch::empty({value.size(0),locations.size(1),value.size(2)*value.size(3)},value.options());
  auto error=agriocc_msda(value.data_ptr(),shapes.data_ptr<int>(),starts.data_ptr<int>(),locations.data_ptr(),weights.data_ptr(),output.data_ptr(),value.size(0),value.size(1),value.size(2),value.size(3),locations.size(1),locations.size(3),locations.size(4),value.scalar_type()==torch::kFloat16,at::cuda::getCurrentCUDAStream());
  TORCH_CHECK(error==cudaSuccess,cudaGetErrorString(error));
  return output;
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) { m.def("forward",&forward); }
