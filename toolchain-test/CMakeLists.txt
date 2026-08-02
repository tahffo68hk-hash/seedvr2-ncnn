cmake_minimum_required(VERSION 3.20)
project(seedvr2_toolchain_test LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(seedvr2_toolchain_test main.cpp)

if(MSVC)
    target_compile_options(seedvr2_toolchain_test PRIVATE /W4 /permissive-)
endif()
