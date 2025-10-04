#pragma once

#define IMAGE_FILE_PATH "/home/andrepotes/projects/learning_concurrent_programming/gaussian_blurring_cuda/test_image_1920_1080.jpg"
#define OUTPUT_IMAGE_FILE_PATH "/home/andrepotes/projects/learning_concurrent_programming/gaussian_blurring_cuda/blurred_image.jpg"

#define NUM_OF_CHANNELS 1

#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

struct ImageData {
    // Unique pointer for the float array, using a custom deleter for arrays
    std::unique_ptr<float[], std::default_delete<float[]>> data;
    int width = 0;
    int height = 0;
};

void loadImageData(ImageData &image_data);

void storeImageData(ImageData &image_data);
