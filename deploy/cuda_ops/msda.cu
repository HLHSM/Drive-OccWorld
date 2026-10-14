#include "msda.h"
#include <cuda_fp16.h>

template <typename T> __device__ inline float read_value(T x) { return float(x); }
template <> __device__ inline float read_value(__half x) { return __half2float(x); }
template <typename T> __device__ inline T write_value(float x) { return T(x); }
template <> __device__ inline __half write_value(float x) { return __float2half(x); }

// grid_sample(mode=bilinear, padding=zeros, align_corners=False), followed by
// weighted reduction over levels/points. FP16 storage uses FP32 accumulation.
template <typename T>
__global__ void sample_kernel(const T* value, const int* shapes, const int* starts,
    const T* locations, const T* weights, T* output, int total, int keys,
    int heads, int channels, int queries, int levels, int points) {
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= total) return;
  int channel = index % channels;
  int head = (index / channels) % heads;
  int query = (index / channels / heads) % queries;
  int batch = index / channels / heads / queries;
  float sum = 0.f;
  for (int level = 0; level < levels; ++level) {
    int height = shapes[2 * level], width = shapes[2 * level + 1];
    for (int point = 0; point < points; ++point) {
      int offset = (((batch * queries + query) * heads + head) * levels + level) * points + point;
      float x = read_value(locations[2 * offset]) * width - .5f;
      float y = read_value(locations[2 * offset + 1]) * height - .5f;
      if (x <= -1.f || x >= width || y <= -1.f || y >= height) continue;
      int x0 = int(floorf(x)), y0 = int(floorf(y));
      float dx = x - x0, dy = y - y0, sampled = 0.f;
      for (int iy = 0; iy < 2; ++iy) {
        int yy = y0 + iy;
        if (yy < 0 || yy >= height) continue;
        for (int ix = 0; ix < 2; ++ix) {
          int xx = x0 + ix;
          if (xx < 0 || xx >= width) continue;
          int address = ((batch * keys + starts[level] + yy * width + xx) * heads + head) * channels + channel;
          sampled += read_value(value[address]) * (ix ? dx : 1.f-dx) * (iy ? dy : 1.f-dy);
        }
      }
      sum += sampled * read_value(weights[offset]);
    }
  }
  output[index] = write_value<T>(sum);
}

cudaError_t agriocc_msda(const void* value, const int* shapes, const int* starts,
    const void* locations, const void* weights, void* output, int batch,
    int keys, int heads, int channels, int queries, int levels, int points,
    bool half, cudaStream_t stream) {
  int total = batch * queries * heads * channels;
  if (!total) return cudaSuccess;
  if (half)
    sample_kernel<<<(total+255)/256,256,0,stream>>>(
      static_cast<const __half*>(value), shapes, starts,
      static_cast<const __half*>(locations), static_cast<const __half*>(weights),
      static_cast<__half*>(output),total,keys,heads,channels,queries,levels,points);
  else
    sample_kernel<<<(total+255)/256,256,0,stream>>>(
      static_cast<const float*>(value),shapes,starts,
      static_cast<const float*>(locations),static_cast<const float*>(weights),
      static_cast<float*>(output),total,keys,heads,channels,queries,levels,points);
  return cudaGetLastError();
}
