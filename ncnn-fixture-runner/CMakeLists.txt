cmake_minimum_required(VERSION 3.20)
project(ncnn_fixture_runner LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn CONFIG REQUIRED PATHS "D:/AIProjects/seedvr2-ncnn/build/ncnn-cpu/install/lib/cmake/ncnn" NO_DEFAULT_PATH)

add_executable(ncnn_fixture_runner main.cpp tensor_io.cpp)
target_link_libraries(ncnn_fixture_runner PRIVATE ncnn)
