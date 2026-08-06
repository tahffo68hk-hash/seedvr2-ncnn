cmake_minimum_required(VERSION 3.22)
project(seededge_host_tests LANGUAGES CXX)
add_executable(model_manifest_test model_manifest_test.cpp ../src/model_manifest.cpp)
target_include_directories(model_manifest_test PRIVATE ../include)
target_compile_features(model_manifest_test PRIVATE cxx_std_17)
find_package(ncnn CONFIG REQUIRED PATHS "D:/AIProjects/seedvr2-ncnn/build/ncnn-cpu/install/lib/cmake/ncnn" NO_DEFAULT_PATH)
add_executable(synthetic_cpu_host_test synthetic_cpu_host_test.cpp ../src/synthetic_cpu_test.cpp)
target_include_directories(synthetic_cpu_host_test PRIVATE ../include)
target_link_libraries(synthetic_cpu_host_test PRIVATE ncnn)
target_compile_features(synthetic_cpu_host_test PRIVATE cxx_std_17)
add_executable(phase18c_contract_test phase18c_contract_test.cpp ../src/phase18c_contract.cpp)
target_include_directories(phase18c_contract_test PRIVATE ../include)
target_compile_features(phase18c_contract_test PRIVATE cxx_std_17)
add_executable(phase18d_tile_core_test phase18d_tile_core_test.cpp ../src/phase18d_tile_core.cpp)
target_include_directories(phase18d_tile_core_test PRIVATE ../include)
target_compile_features(phase18d_tile_core_test PRIVATE cxx_std_17)
add_executable(phase18d_single_tile_contract_test
  phase18d_single_tile_contract_test.cpp
  ../src/minimal_video_pipeline.cpp ../src/model_manifest.cpp ../src/phase18c_contract.cpp
  ../src/runtime_progress.cpp ../src/fast_convolution3d.cpp ../src/phase18d_tile_core.cpp
  ../../../tests/attention_dispatcher/artifact_registry.cpp
  ../../../tests/attention_dispatcher/model_cache.cpp
  ../../../tests/seedvr2_dit_block_ncnn/hybrid_dit_block_runtime.cpp)
target_include_directories(phase18d_single_tile_contract_test PRIVATE ../include ../../../tests/attention_dispatcher ../../../tests/seedvr2_dit_block_ncnn)
target_link_libraries(phase18d_single_tile_contract_test PRIVATE ncnn)
target_compile_features(phase18d_single_tile_contract_test PRIVATE cxx_std_17)

add_executable(phase18d_full_photo_contract_test
  phase18d_full_photo_contract_test.cpp
  ../src/minimal_video_pipeline.cpp
  ../src/model_manifest.cpp
  ../src/phase18c_contract.cpp
  ../src/runtime_progress.cpp
  ../src/fast_convolution3d.cpp
  ../src/phase18d_tile_core.cpp
  ../../../tests/attention_dispatcher/artifact_registry.cpp
  ../../../tests/attention_dispatcher/model_cache.cpp
  ../../../tests/seedvr2_dit_block_ncnn/hybrid_dit_block_runtime.cpp)

target_include_directories(
  phase18d_full_photo_contract_test
  PRIVATE
  ../include
  ../../../tests/attention_dispatcher
  ../../../tests/seedvr2_dit_block_ncnn)

target_link_libraries(
  phase18d_full_photo_contract_test
  PRIVATE
  ncnn)

target_compile_features(
  phase18d_full_photo_contract_test
  PRIVATE
  cxx_std_17)
add_executable(phase18d_model_bundle_test phase18d_model_bundle_test.cpp ../src/model_manifest.cpp)
target_include_directories(phase18d_model_bundle_test PRIVATE ../include)
target_compile_features(phase18d_model_bundle_test PRIVATE cxx_std_17)
add_executable(phase18d_single_tile_runner
  phase18d_single_tile_runner.cpp
  ../src/minimal_video_pipeline.cpp ../src/model_manifest.cpp ../src/phase18c_contract.cpp
  ../src/runtime_progress.cpp ../src/fast_convolution3d.cpp ../src/phase18d_tile_core.cpp
  ../../../tests/attention_dispatcher/artifact_registry.cpp
  ../../../tests/attention_dispatcher/model_cache.cpp
  ../../../tests/seedvr2_dit_block_ncnn/hybrid_dit_block_runtime.cpp)
target_include_directories(phase18d_single_tile_runner PRIVATE ../include ../../../tests/attention_dispatcher ../../../tests/seedvr2_dit_block_ncnn)
target_link_libraries(phase18d_single_tile_runner PRIVATE ncnn)
target_compile_features(phase18d_single_tile_runner PRIVATE cxx_std_17)
add_executable(runtime_progress_test runtime_progress_test.cpp ../src/runtime_progress.cpp)
target_include_directories(runtime_progress_test PRIVATE ../include)
target_compile_features(runtime_progress_test PRIVATE cxx_std_17)
