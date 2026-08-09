#pragma once

#include "phase18d_stage_major.h"

#include <string>
#include <vector>
#include <cstdint>

namespace seededge::phase18d {

enum class RuntimeMode { TileMajor, StageMajor, Hybrid, Auto };
enum class CheckpointMode { Production, Debug };

struct RuntimeRequest
{
    RuntimeMode mode = RuntimeMode::Auto;
    int cohort_size = 0;
    bool resume_enabled = false;
    CheckpointMode checkpoint_mode = CheckpointMode::Production;
};

struct TileCohort { int begin = 0; int end = 0; };

struct RuntimeCalibration
{
    // Host CPU FP32 calibration. These fields are deliberately data, so Android
    // can replace them after device measurements instead of inheriting guesses.
    int tile_major_max_tiles = 2;
    int default_cohort_size = 4;
    double fixed_runtime_ms = 41799.59302326;
    double per_tile_runtime_ms = 13743.10465116;
    double model_reload_per_cohort_ms = 8413.613;
    uint64_t peak_intermediate_bytes_per_tile = 84918726;
    uint64_t final_bytes_per_tile = 2114073;
    uint64_t safety_margin_bytes = 1073741824;
    double peak_rss_mb = 1480.41;
    std::string source_label = "HOST_MEASURED_256_512_1024_2026-08-09";
};

struct RuntimeSelection
{
    RuntimeMode requested_mode = RuntimeMode::Auto;
    RuntimeMode selected_mode = RuntimeMode::Auto;
    int cohort_size = 0;
    std::string reason;
};

struct RuntimeResourceEstimate
{
    int width = 0;
    int height = 0;
    uint64_t tile_count = 0;
    RuntimeSelection selection;
    uint64_t estimated_peak_task_bytes = 0;
    uint64_t estimated_final_task_bytes = 0;
    uint64_t required_free_bytes = 0;
    double estimated_runtime_ms = 0.0;
    double estimated_peak_rss_mb = 0.0;
    std::string source_label;
};

const char* runtime_mode_name(RuntimeMode mode);
bool runtime_uses_staged_scheduler(RuntimeMode mode);
bool parse_runtime_request(const std::string& json, RuntimeRequest* request,
                           std::string* error);
std::vector<TileCohort> plan_tile_cohorts(int tile_count, int cohort_size,
                                          std::string* error);
bool run_hybrid_schedule(StageMajorCheckpoint* checkpoint, int cohort_size,
                         const StageMajorExecutionHooks& hooks,
                         StageMajorRunStats* stats, std::string* error);
bool estimate_task_resources(int width, int height, const RuntimeRequest& request,
                             uint64_t available_disk_bytes,
                             uint64_t memory_budget_bytes,
                             const RuntimeCalibration& calibration,
                             RuntimeResourceEstimate* estimate,
                             std::string* error);

}  // namespace seededge::phase18d
