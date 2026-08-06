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

// Optional cooperative progress callback: stage is a stable ASCII label
// (PREPARING / CHECKING_MODEL / RUNNING_TILE / FUSING / ENCODING_PNG /
// COMPLETED / FAILED / CANCELLED), completed/total describe tile progress.
using Phase18dProgressCallback = void (*)(void* context, const char* stage,
                                          int completed, int total);

struct Phase18dFullPhotoHooks
{
    void* context = nullptr;
    Phase18dManifestValidator validate_manifest = nullptr;
    Phase18dTileExecutor execute_tile = nullptr;
    Phase18dCancellationCheck is_cancelled = nullptr;
    Phase18dProgressCallback on_progress = nullptr;
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

// Context used by the real production hooks.  The manifest validator and tile
// executor read/write manifest; jni_bridge is an opaque caller handle that
// the Android JNI progress/cancellation callbacks use to reach their bridge.
// The context passed to make_real_full_photo_hooks must be this type; passing
// a different object (for example a JNI bridge struct) corrupts the bridge.
struct Phase18dRealContext
{
    std::string manifest;
    void* jni_bridge = nullptr;
};

bool phase18d_real_validate_manifest(
    void* opaque,
    const std::string& model_root,
    std::string* manifest_sha256,
    std::string* error);

bool phase18d_real_execute_tile(
    void* opaque,
    const std::string& model_root,
    const std::string& tile_root,
    const TileSpec& tile,
    const PaddedTile& padded,
    Phase18dTileExecutionResult* result,
    std::string* error);

// Builds the real production hooks around the supplied context and callbacks.
// The manifest validator and tile executor are the real SeedVR2 pipeline; the
// caller supplies the cooperative cancellation check and progress callback.
Phase18dFullPhotoHooks make_real_full_photo_hooks(
    Phase18dRealContext* context,
    Phase18dCancellationCheck is_cancelled,
    Phase18dProgressCallback on_progress);

}  // namespace seededge::phase18d
