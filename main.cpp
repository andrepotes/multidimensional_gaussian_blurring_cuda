#include "gaussian_blurring.h"

#include "blur_interface.h"

void loadImageData(ImageData &image_data) {
    cv::Mat host_image = cv::imread(IMAGE_FILE_PATH, cv::IMREAD_GRAYSCALE);

    if (host_image.empty()) {
        std::cerr << "Error: Could not load image" << std::endl;
        exit(1);
    }

    std::cout << "Image loaded successfully" << std::endl;

    // display image
    cv::imshow("Image", host_image);
    cv::waitKey(0);

    // convert from uchar to float [0.0, 1.0]
    cv::Mat float_host_image;
    host_image.convertTo(float_host_image, CV_32FC(NUM_OF_CHANNELS), 1.0/255.0); // Convert uchar to float [0.0, 1.0]

    // save dimensions
    image_data.width = float_host_image.cols;
    image_data.height = float_host_image.rows;
    size_t total_elements = float_host_image.total();

    // allocate memory for the data and assign to struct with unique pointer
    image_data.data = std::make_unique<float[]>(total_elements);

    // pass data to the struct with memcpy
    memcpy(image_data.data.get(), float_host_image.data, total_elements * sizeof(float));
}

void storeImageData(ImageData &image_data) {
    // get the raw pointer from data
    float *data = image_data.data.get();

    // get mat type 
    int mat_type = CV_32FC(NUM_OF_CHANNELS);

    cv::Mat float_image_view(image_data.height, image_data.width, mat_type, data);

    // Displaying a float Mat usually requires conversion first
    cv::Mat display_image;
    float_image_view.convertTo(display_image, CV_8UC(NUM_OF_CHANNELS), 255.0); 

    cv::imshow("Image", display_image);
    cv::waitKey(0);

    cv::imwrite(OUTPUT_IMAGE_FILE_PATH, display_image);
    std::cout << "Image stored successfully" << std::endl;
}

int main(int argc, char *argv[]) {
    ImageData host_input_img_data;
    ImageData host_output_img_data;

    const std::vector<float> host_gaussian_blur_kernel = {
        1.f/256,  4.f/256,  6.f/256,  4.f/256, 1.f/256,
        4.f/256, 16.f/256, 24.f/256, 16.f/256, 4.f/256,
        6.f/256, 24.f/256, 36.f/256, 24.f/256, 6.f/256,
        4.f/256, 16.f/256, 24.f/256, 16.f/256, 4.f/256,
        1.f/256,  4.f/256,  6.f/256,  4.f/256, 1.f/256
    };

    size_t kernel_size = host_gaussian_blur_kernel.size();
    int kernel_dim = static_cast<int>(std::sqrt(kernel_size));

    // Basic check to make sure it's a valid square kernel
    if (kernel_dim * kernel_dim != kernel_size) {
        std::cerr << "Error: Kernel size is not a perfect square!" << std::endl;
        return 1;
    }

    loadImageData(host_input_img_data);

    // Prepare the output host struct
    host_output_img_data.width = host_input_img_data.width;
    host_output_img_data.height = host_input_img_data.height;
    size_t total_elements = (size_t)host_output_img_data.width * host_output_img_data.height;
    
    host_output_img_data.data = std::make_unique<float[]>(total_elements);

    // Call the wrapper function to run the CUDA part
    apply_gaussian_blur_cuda(
        host_input_img_data, 
        host_output_img_data, 
        host_gaussian_blur_kernel.data(),
        kernel_dim); 

    storeImageData(host_output_img_data);

    return 0;
}