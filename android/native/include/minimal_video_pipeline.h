#pragma once

#include "phase18d_tile_core.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

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

namespace seededge::phase18d {

// Result of one already-validated tile run, produced by a tile executor hook.
struct Phase18dTileExecutionResult
{
    bool encoder_completed = false;
    int dit_block_completed = 0;
    bool scheduler_completed = false;
    bool decoder_completed = false;
    bool finite = false;
    size_t nan_count = 0;
    size_t inf_count = 0;

    // Contains valid_width * valid_height * 3 RGB8 bytes without padding.
    std::vector<uint8_t> output_valid_rgb;

    // Must be a JSON object; embedded in the tile status.json.
    std::string pipeline_result_json;
};

using Phase18dManifestValidator = bool (*)(
    void* context,
    const std::string& model_root,
    std::string* manifest_sha256,
    std::string* error);

using Phase18dTileExecutor = bool (*)(
    void* context,
    const std::string& model_root,
    const std::string& tile_root,
    const TileSpec& tile,
    const PaddedTile& padded,
    Phase18dTileExecutionResult* result,
    std::string* error);

using Phase18dCancellationCheck = bool (*)(void* context);

struct Phase18dFullPhotoHooks
{
    void* context = nullptr;
    Phase18dManifestValidator validate_manifest = nullptr;
    Phase18dTileExecutor execute_tile = nullptr;
    Phase18dCancellationCheck is_cancelled = nullptr;
};

// Testable full-photo orchestration entry.  The model manifest is validated
// exactly once, then every planned tile runs through the provided executor.
std::string run_phase18d_full_photo_with_hooks(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json,
    const Phase18dFullPhotoHooks& hooks);

// Production full-photo entry: validates the model manifest once, runs every
// planned tile through the real pipeline, fuses the valid regions and writes
// <task_root>/output_full.rgb.
std::string run_phase18d_full_photo(const std::string& model_root,
                                    const std::string& task_root,
                                    const uint8_t* normalized_rgb,
                                    size_t byte_count,
                                    int width,
                                    int height,
                                    const std::string& request_json);

}  // namespace seededge::phase18d
