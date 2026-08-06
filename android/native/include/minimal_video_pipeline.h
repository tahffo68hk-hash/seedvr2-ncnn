#pragma once

#include <cstddef>
#include <cstdint>
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

// Starts a new task-local pipeline from normalized RGB8.  This entry never
// accepts nor reads a Phase 18C decoder checkpoint; its only persistent state
// is rooted at <task_root>/tiles/tile_0000.
std::string run_phase18d_single_photo_tile(const std::string& model_root,
                                           const std::string& task_root,
                                           const uint8_t* normalized_rgb,
                                           size_t byte_count, int width, int height,
                                           const std::string& request_json);
