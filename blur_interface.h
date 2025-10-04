#pragma once

#include "image_data.h"

void apply_gaussian_blur_cuda(const ImageData& input, ImageData& output);