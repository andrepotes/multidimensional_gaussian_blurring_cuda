#pragma once

#define IMAGE_FILE_PATH "/home/andrepotes/projects/learning_concurrent_programming/gaussian_blurring_cuda/test_image_1920_1080.jpg"
#define OUTPUT_IMAGE_FILE_PATH "/home/andrepotes/projects/learning_concurrent_programming/gaussian_blurring_cuda/blurred_image.jpg"

#define NUM_OF_CHANNELS 1

#include <iostream>
#include <cstdlib>

#include <opencv2/opencv.hpp>

#include "image_data.h"

void loadImageData(ImageData &image_data);
void storeImageData(ImageData &image_data);