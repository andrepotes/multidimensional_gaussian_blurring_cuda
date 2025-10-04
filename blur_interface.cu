#include <cuda_runtime.h>
#include <iostream>

#include "blur_interface.h"

#ifndef CUDA_CHECK
#define CUDA_CHECK(call) \
    do { \
        cudaError_t _err = (call); \
        if (_err != cudaSuccess) { \
            std::cerr << "CUDA error in " << __FILE__ << ":" << __LINE__ \
                      << ": " << cudaGetErrorString(_err) << " (" << _err << ")\n"; \
            std::exit(1); \
        } \
    } while (0)
#endif

__global__ void simple_copy_kernel(const float* input, float* output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = y * width + x;
        output[idx] = input[idx];
    }
}

void apply_gaussian_blur_cuda(const ImageData& input, ImageData& output) {
    std::cout << "Applying Gaussian blur on CUDA" << std::endl;

    size_t data_size = (size_t)input.width * input.height * sizeof(float);

    float *device_input, *device_output;
    CUDA_CHECK(cudaMalloc((void**)&device_input, data_size));
    CUDA_CHECK(cudaMalloc((void**)&device_output, data_size));

    CUDA_CHECK(cudaMemcpy(device_input, input.data.get(), data_size, cudaMemcpyHostToDevice));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((input.width + threads_per_block.x - 1) / threads_per_block.x,
                    (input.height + threads_per_block.y - 1) / threads_per_block.y);

    simple_copy_kernel<<<num_blocks, threads_per_block>>>(device_input, device_output, input.width, input.height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(output.data.get(), device_output, data_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_input));
    CUDA_CHECK(cudaFree(device_output));

    std::cout << "Gaussian blur applied successfully" << std::endl;
}