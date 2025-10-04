#pragma once

#include <memory> // For std::unique_ptr

// Definition for the image data structure
struct ImageData {
    std::unique_ptr<float[], std::default_delete<float[]>> data;
    int width = 0;
    int height = 0;
};