cmake_minimum_required(VERSION 3.20)
project(ncnn_dynamic_mask_runner LANGUAGES CXX)

find_package(ncnn CONFIG REQUIRED PATHS "D:/AIProjects/seedvr2-ncnn/build/ncnn-cpu/install/lib/cmake/ncnn" NO_DEFAULT_PATH)

add_executable(ncnn_dynamic_mask_runner main.cpp)
target_link_libraries(ncnn_dynamic_mask_runner PRIVATE ncnn)
target_compile_features(ncnn_dynamic_mask_runner PRIVATE cxx_std_17)
