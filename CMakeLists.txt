cmake_minimum_required(VERSION 3.18)

project(gaussian_blurring LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_STANDARD 17)

find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED) # New: Find the CUDA toolkit

add_executable(gaussian_blurring
    main.cpp
    blur_interface.cu
)

target_include_directories(gaussian_blurring PRIVATE ${OpenCV_INCLUDE_DIRS})

target_link_libraries(gaussian_blurring PRIVATE 
    ${OpenCV_LIBS} 
    CUDA::cudart
)

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "CUDA toolkit found: ${CUDAToolkit_FOUND}")
