#pragma once

#include "minimal_video_pipeline.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace seededge::phase18d {

// Generates input.png, output.png and side_by_side.png from a completed
// full-photo task. The function validates task status and output_full.rgb
// before creating any final artifact.
std::string write_phase18d_full_photo_artifacts(
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height);

// Runs the testable full-photo orchestration and generates the Host PNG
// artifacts only when the orchestration completed successfully.
std::string run_phase18d_full_photo_artifacts_with_hooks(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json,
    const Phase18dFullPhotoHooks& hooks);

// Production wrapper using the real full-photo hooks.
std::string run_phase18d_full_photo_artifacts(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json);

}  // namespace seededge::phase18d