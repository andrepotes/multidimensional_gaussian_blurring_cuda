cmake_minimum_required(VERSION 3.18)

project(gaussian_blurring LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(gaussian_blurring
    gaussian_blurring.cpp
)

target_include_directories(gaussian_blurring PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(gaussian_blurring PRIVATE ${OpenCV_LIBS})

message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
