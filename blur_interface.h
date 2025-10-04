#pragma once

#include "image_data.h"

void apply_gaussian_blur_cuda(
    const ImageData& host_img_input, 
    ImageData& host_img_output, 
    const float* host_gaussian_blur_kernel,
    int kernel_dim);