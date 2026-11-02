cmake_minimum_required(VERSION 3.20)
project(ncnn_dynamic_mask_runner LANGUAGES CXX)

find_package(ncnn CONFIG REQUIRED)

add_executable(ncnn_dynamic_mask_runner main.cpp)
target_link_libraries(ncnn_dynamic_mask_runner PRIVATE ncnn)
target_compile_features(ncnn_dynamic_mask_runner PRIVATE cxx_std_17)
