cmake_minimum_required(VERSION 3.22.1)
project(seededge LANGUAGES C CXX)
if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
  message(FATAL_ERROR "Phase 18A supports arm64-v8a only")
endif()
set(NCNN_BUILD_BENCHMARK OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
set(NCNN_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(NCNN_VULKAN ON CACHE BOOL "" FORCE)
set(NCNN_OPENMP ON CACHE BOOL "" FORCE)
set(NCNN_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_EXCEPTION OFF CACHE BOOL "" FORCE)
set(NCNN_DISABLE_RTTI ON CACHE BOOL "" FORCE)
set(NCNN_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/../.phase18_ncnn_src" CACHE PATH "Pinned ncnn source")
if(NOT EXISTS "${NCNN_SOURCE_DIR}/CMakeLists.txt")
  message(FATAL_ERROR "Pinned ncnn source missing; run scripts/prepare_ncnn.ps1")
endif()
add_subdirectory("${NCNN_SOURCE_DIR}" ncnn-build)
add_library(seededge SHARED native/src/jni_runtime.cpp native/src/runtime_info.cpp native/src/runtime_progress.cpp native/src/synthetic_cpu_test.cpp native/src/vulkan_probe.cpp native/src/model_manifest.cpp native/src/phase18c_contract.cpp native/src/minimal_video_pipeline.cpp
  ../tests/attention_dispatcher/artifact_registry.cpp
  ../tests/attention_dispatcher/model_cache.cpp
  ../tests/seedvr2_dit_block_ncnn/hybrid_dit_block_runtime.cpp)
target_include_directories(seededge PRIVATE native/include ../tests/attention_dispatcher ../tests/seedvr2_dit_block_ncnn)
target_link_libraries(seededge PRIVATE ncnn android log)
target_compile_features(seededge PRIVATE cxx_std_17)
target_compile_options(seededge PRIVATE -fopenmp)
target_link_options(seededge PRIVATE -fopenmp -static-openmp)
# Device validation uses a Debug APK for run-as access, but the real-weight
# manifest hashes more than 14 GB.  Keep assertions/debuggability while
# compiling the native runtime's numerical and SHA-256 hot loops efficiently.
target_compile_options(seededge PRIVATE -O2)
