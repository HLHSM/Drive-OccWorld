#pragma once
#include <cuda_runtime.h>

cudaError_t agriocc_msda(const void* value, const int* shapes, const int* starts,
    const void* locations, const void* weights, void* output,
    int batch, int keys, int heads, int channels, int queries, int levels,
    int points, bool half, cudaStream_t stream);
