cmake_minimum_required(VERSION 3.20)
project(seedvr2_vae_ncnn LANGUAGES CXX)

find_package(ncnn CONFIG REQUIRED PATHS "D:/AIProjects/seedvr2-ncnn/build/ncnn-cpu/install/lib/cmake/ncnn" NO_DEFAULT_PATH)

add_executable(vae_stage_runner vae_stage_runner.cpp)
target_link_libraries(vae_stage_runner PRIVATE ncnn)
target_compile_options(vae_stage_runner PRIVATE /EHsc /UNDEBUG)