#pragma once

// Paths derived from the user's HOME directory to avoid hardcoded usernames
#include <cstdlib>
#include <string>

#define NUM_OF_CHANNELS 1

#include <iostream>
#include <memory>

#include <opencv2/opencv.hpp>

inline std::string getHomeDirectory() {
    const char *home_env = std::getenv("HOME");
    if (home_env != nullptr && home_env[0] != '\0') {
        return std::string(home_env);
    }
    std::cerr << "Error: HOME environment variable is not set." << std::endl;
    return std::string();
}

inline std::string getImageFilePath() {
    return getHomeDirectory() + "/projects/learning_concurrent_programming/gaussian_blurring_cuda/test_image_1920_1080.jpg";
}

inline std::string getOutputImageFilePath() {
    return getHomeDirectory() + "/projects/learning_concurrent_programming/gaussian_blurring_cuda/blurred_image.jpg";
}

struct ImageData {
    // Unique pointer for the float array, using a custom deleter for arrays
    std::unique_ptr<float[], std::default_delete<float[]>> data;
    int width = 0;
    int height = 0;
};

void loadImageData(ImageData &image_data);

void storeImageData(ImageData &image_data);
