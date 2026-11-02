cmake_minimum_required(VERSION 3.20)
project(attention_dispatcher LANGUAGES CXX)

# ncnn is an external build. Point CMAKE_PREFIX_PATH at its install prefix
# (see README). Host numeric tests can configure with -DSEEDVR2_REQUIRE_NCNN=OFF.
option(SEEDVR2_REQUIRE_NCNN "Configure targets that link ncnn" ON)

if(SEEDVR2_REQUIRE_NCNN)
  find_package(ncnn CONFIG REQUIRED)
endif()

function(seedvr2_cxx_flags target)
  target_compile_features(${target} PUBLIC cxx_std_17)
  if(MSVC)
    target_compile_options(${target} PRIVATE /EHsc)
  endif()
endfunction()

function(seedvr2_keep_asserts target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /UNDEBUG)
  else()
    target_compile_options(${target} PRIVATE -UNDEBUG)
  endif()
endfunction()

add_library(attention_dispatcher_lib attention_dispatcher.cpp)
target_include_directories(attention_dispatcher_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
seedvr2_cxx_flags(attention_dispatcher_lib)

add_executable(dispatcher_tests dispatcher_tests.cpp)
target_link_libraries(dispatcher_tests PRIVATE attention_dispatcher_lib artifact_registry_lib)
seedvr2_keep_asserts(dispatcher_tests)

add_library(artifact_registry_lib artifact_registry.cpp)
target_include_directories(artifact_registry_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
seedvr2_cxx_flags(artifact_registry_lib)

add_executable(artifact_registry_tests artifact_registry_tests.cpp)
target_link_libraries(artifact_registry_tests PRIVATE artifact_registry_lib)
seedvr2_keep_asserts(artifact_registry_tests)

add_library(hybrid_window_adapter_lib hybrid_window_adapter.cpp)
target_include_directories(hybrid_window_adapter_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
seedvr2_cxx_flags(hybrid_window_adapter_lib)

add_library(dynamic_padding_mask_lib dynamic_attention_inputs.cpp dynamic_padding_mask.cpp)
target_include_directories(dynamic_padding_mask_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
seedvr2_cxx_flags(dynamic_padding_mask_lib)

add_library(dynamic_rope3d_lib dynamic_rope3d.cpp)
target_include_directories(dynamic_rope3d_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(dynamic_rope3d_lib PUBLIC dynamic_padding_mask_lib)
seedvr2_cxx_flags(dynamic_rope3d_lib)

add_executable(hybrid_window_adapter_tests hybrid_window_adapter_tests.cpp)
target_link_libraries(hybrid_window_adapter_tests PRIVATE hybrid_window_adapter_lib)
seedvr2_keep_asserts(hybrid_window_adapter_tests)

add_executable(dynamic_padding_mask_tests dynamic_padding_mask_tests.cpp)
target_link_libraries(dynamic_padding_mask_tests PRIVATE dynamic_padding_mask_lib)
seedvr2_keep_asserts(dynamic_padding_mask_tests)

add_executable(dynamic_rope3d_tests dynamic_rope3d_tests.cpp)
target_link_libraries(dynamic_rope3d_tests PRIVATE dynamic_rope3d_lib)
seedvr2_keep_asserts(dynamic_rope3d_tests)

if(SEEDVR2_REQUIRE_NCNN)
  add_executable(attention_dispatcher_runner EXCLUDE_FROM_ALL runner.cpp)
  target_link_libraries(attention_dispatcher_runner PRIVATE attention_dispatcher_lib artifact_registry_lib ncnn)
  seedvr2_cxx_flags(attention_dispatcher_runner)

  add_library(model_cache_lib model_cache.cpp)
  target_include_directories(model_cache_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(model_cache_lib PUBLIC artifact_registry_lib ncnn)
  seedvr2_cxx_flags(model_cache_lib)

  add_library(attention_runtime_lib attention_runtime.cpp)
  target_include_directories(attention_runtime_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(attention_runtime_lib PUBLIC attention_dispatcher_lib model_cache_lib ncnn)
  seedvr2_cxx_flags(attention_runtime_lib)

  add_library(ncnn_mat_contract_lib ncnn_mat_contract.cpp)
  target_include_directories(ncnn_mat_contract_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
  target_link_libraries(ncnn_mat_contract_lib PUBLIC ncnn)
  seedvr2_cxx_flags(ncnn_mat_contract_lib)

  add_library(hybrid_attention_runtime_lib hybrid_model_bundle.cpp hybrid_attention_runtime.cpp ../seedvr2_attention_fidelity/host_head_merge.cpp)
  target_include_directories(hybrid_attention_runtime_lib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/../seedvr2_attention_fidelity")
  target_link_libraries(hybrid_attention_runtime_lib PUBLIC attention_dispatcher_lib model_cache_lib hybrid_window_adapter_lib dynamic_padding_mask_lib dynamic_rope3d_lib ncnn_mat_contract_lib ncnn)
  seedvr2_cxx_flags(hybrid_attention_runtime_lib)

  add_executable(ncnn_mat_contract_tests ncnn_mat_contract_tests.cpp)
  target_link_libraries(ncnn_mat_contract_tests PRIVATE ncnn_mat_contract_lib)
  seedvr2_keep_asserts(ncnn_mat_contract_tests)

  add_executable(model_cache_tests EXCLUDE_FROM_ALL model_cache_tests.cpp)
  target_link_libraries(model_cache_tests PRIVATE model_cache_lib)
  seedvr2_keep_asserts(model_cache_tests)

  add_executable(runtime_validation EXCLUDE_FROM_ALL runtime_validation.cpp)
  target_link_libraries(runtime_validation PRIVATE attention_runtime_lib)
  seedvr2_cxx_flags(runtime_validation)

  add_executable(hybrid_runtime_validation EXCLUDE_FROM_ALL hybrid_runtime_validation.cpp)
  target_link_libraries(hybrid_runtime_validation PRIVATE hybrid_attention_runtime_lib)
  seedvr2_cxx_flags(hybrid_runtime_validation)

  add_executable(dynamic_multi_shape_validation EXCLUDE_FROM_ALL dynamic_multi_shape_validation.cpp)
  target_link_libraries(dynamic_multi_shape_validation PRIVATE hybrid_attention_runtime_lib)
  seedvr2_keep_asserts(dynamic_multi_shape_validation)
endif()
