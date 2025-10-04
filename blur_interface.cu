#include <cuda_runtime.h>
#include <iostream>

#include "blur_interface.h"

// gaussian blur kernel in GPU constant memory
__constant__ float device_gaussian_blur_kernel[225];

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

__global__ void gaussian_blur_kernel(const float* input, float* output, int width, int height, int kernel_dim) {
    // Get the current thread's position (pixel) in the grid.
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: ensure we don't work on pixels outside the image.
    if (x >= width || y >= height) {
        return;
    }

    float sum = 0.0f;
    int radius = kernel_dim / 2;

    for (int row = 0; row < kernel_dim; row++) {
        for (int col = 0; col < kernel_dim; col++) {
            int neighbor_x = x + col - radius;
            int neighbor_y = y + row - radius;

            if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                int neighbor_idx = neighbor_y * width + neighbor_x;
                int kernel_idx = row * kernel_dim + col;
                sum += input[neighbor_idx] * device_gaussian_blur_kernel[kernel_idx];
            }
        }
    }

    int idx = y * width + x;
    output[idx] = sum;
}

void apply_gaussian_blur_cuda(const ImageData& host_img_input, ImageData& host_img_output, const float* host_gaussian_blur_kernel, int kernel_dim) {
    std::cout << "Applying Gaussian blur on CUDA" << std::endl;

    size_t data_size = (size_t)host_img_input.width * host_img_input.height * sizeof(float);

    float *device_img_input, *device_img_output;
    CUDA_CHECK(cudaMalloc((void**)&device_img_input, data_size));
    CUDA_CHECK(cudaMalloc((void**)&device_img_output, data_size));

    // pass the image data to the device
    CUDA_CHECK(cudaMemcpy(device_img_input, host_img_input.data.get(), data_size, cudaMemcpyHostToDevice));

    // pass the gaussian blur kernel matrix to the device
    CUDA_CHECK(cudaMemcpyToSymbol(device_gaussian_blur_kernel, host_gaussian_blur_kernel, kernel_dim * kernel_dim * sizeof(float)));

    dim3 threads_per_block(16, 16);
    dim3 num_blocks((host_img_input.width + threads_per_block.x - 1) / threads_per_block.x,
                    (host_img_input.height + threads_per_block.y - 1) / threads_per_block.y);

    // execute the CUDA kernel in input host data
    gaussian_blur_kernel<<<num_blocks, threads_per_block>>>(device_img_input, device_img_output, host_img_input.width, host_img_input.height, kernel_dim);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // get the resulting computed output data from the device
    CUDA_CHECK(cudaMemcpy(host_img_output.data.get(), device_img_output, data_size, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(device_img_input));
    CUDA_CHECK(cudaFree(device_img_output));

    std::cout << "Gaussian blur applied successfully" << std::endl;
}