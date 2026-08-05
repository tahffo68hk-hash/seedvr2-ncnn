#pragma once

#include <string>

std::string run_phase18c_minimal_video_demo(const std::string& model_root,
                                             const std::string& input_root,
                                             const std::string& output_root,
                                             const std::string& config_json);
std::string run_phase18c_decoder_only(const std::string& model_root,
                                       const std::string& checkpoint_root,
                                       const std::string& output_root,
                                       const std::string& config_json);
std::string run_phase18c_decoder_stage_benchmark(const std::string& model_root,
                                                  const std::string& output_root);
