#include "phase18d_runtime_strategy.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <regex>

namespace seededge::phase18d {
namespace {
bool fail(std::string* error, const char* code)
{
    if (error) *error = code;
    return false;
}

bool same_stage(const StageMajorWork& a, const StageMajorWork& b)
{
    return a.stage == b.stage && a.stage_index == b.stage_index &&
           a.block_index == b.block_index;
}

bool checked_add(uint64_t a, uint64_t b, uint64_t* output)
{
    if (a > std::numeric_limits<uint64_t>::max() - b) return false;
    *output = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t* output)
{
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *output = a * b;
    return true;
}

uint64_t axis_tiles(int dimension)
{
    if (dimension <= 128) return 1;
    return (static_cast<uint64_t>(dimension - 128) + 95u) / 96u + 1u;
}

bool expected_in_range(const StageMajorCheckpoint& checkpoint, int begin, int end,
                       StageMajorWork* work)
{
    for (int stage = 0; stage < 7; ++stage)
        for (int tile = begin; tile < end; ++tile)
            if (checkpoint.encoder_completed_stage_per_tile[tile] < stage)
            { *work = {"encoder", stage, -1, tile}; return true; }
    for (int tile = begin; tile < end; ++tile)
        if (!checkpoint.dit_input_completed_tiles[tile])
        { *work = {"dit_input", -1, -1, tile}; return true; }
    for (int block = 0; block < 32; ++block)
        for (int tile = begin; tile < end; ++tile)
            if (checkpoint.dit_completed_block_per_tile[tile] < block)
            { *work = {"dit", -1, block, tile}; return true; }
    for (int tile = begin; tile < end; ++tile)
        if (!checkpoint.scheduler_completed_tiles[tile])
        { *work = {"scheduler", -1, -1, tile}; return true; }
    for (int stage = 0; stage < 7; ++stage)
        for (int tile = begin; tile < end; ++tile)
            if (checkpoint.decoder_completed_stage_per_tile[tile] < stage)
            { *work = {"decoder", stage, -1, tile}; return true; }
    return false;
}

bool next_hybrid_work(const StageMajorCheckpoint& checkpoint, int cohort_size,
                      StageMajorWork* work, std::string* error)
{
    if (checkpoint.state == "paused") return fail(error, "TASK_PAUSED");
    if (checkpoint.state == "cancelled") return fail(error, "TASK_CANCELLED");
    if (checkpoint.state == "completed") return false;
    const int tiles = checkpoint.identity.tile_count;
    for (int begin = 0; begin < tiles; begin += cohort_size)
        if (expected_in_range(checkpoint, begin, std::min(begin + cohort_size, tiles), work))
            return true;
    if (!checkpoint.fusion_completed) { *work = {"fusion", -1, -1, -1}; return true; }
    return false;
}

bool mark_hybrid_complete(StageMajorCheckpoint* checkpoint, int cohort_size,
                          const StageMajorWork& work, std::string* error)
{
    StageMajorWork expected;
    if (!next_hybrid_work(*checkpoint, cohort_size, &expected, error) ||
        expected.stage != work.stage || expected.stage_index != work.stage_index ||
        expected.block_index != work.block_index || expected.tile_index != work.tile_index)
        return fail(error, "STALE_WORK_ITEM");
    checkpoint->current_stage = work.stage;
    checkpoint->current_stage_index = work.stage_index;
    checkpoint->current_block = work.block_index;
    if (work.stage == "encoder") {
        checkpoint->encoder_completed_stage_per_tile[work.tile_index] = work.stage_index;
        checkpoint->encoder_completed_tiles[work.tile_index] = work.stage_index == 6;
    } else if (work.stage == "dit_input") checkpoint->dit_input_completed_tiles[work.tile_index] = true;
    else if (work.stage == "dit") checkpoint->dit_completed_block_per_tile[work.tile_index] = work.block_index;
    else if (work.stage == "scheduler") checkpoint->scheduler_completed_tiles[work.tile_index] = true;
    else if (work.stage == "decoder") {
        checkpoint->decoder_completed_stage_per_tile[work.tile_index] = work.stage_index;
        if (work.stage_index == 6) { checkpoint->decoder_completed_tiles[work.tile_index] = true; ++checkpoint->completed_tiles; }
    } else if (work.stage == "fusion") {
        checkpoint->fusion_completed = true; checkpoint->state = "completed";
        checkpoint->current_stage = "completed";
    } else return fail(error, "INVALID_WORK_ITEM");
    return true;
}
}

const char* runtime_mode_name(RuntimeMode mode)
{
    switch (mode) {
    case RuntimeMode::TileMajor: return "tile_major";
    case RuntimeMode::StageMajor: return "stage_major";
    case RuntimeMode::Hybrid: return "hybrid";
    case RuntimeMode::Auto: return "auto";
    }
    return "auto";
}

bool runtime_uses_staged_scheduler(RuntimeMode mode)
{
    return mode == RuntimeMode::StageMajor || mode == RuntimeMode::Hybrid;
}

bool parse_runtime_request(const std::string& json, RuntimeRequest* request,
                           std::string* error)
{
    if (error) error->clear();
    if (!request) return fail(error, "RUNTIME_REQUEST_OUTPUT_NULL");
    RuntimeRequest parsed;
    std::smatch match;
    if (std::regex_search(json, match, std::regex(R"rx("runtime_mode"\s*:\s*"([^"]+)")rx"))) {
        const std::string value = match[1];
        if (value == "tile_major") parsed.mode = RuntimeMode::TileMajor;
        else if (value == "stage_major") parsed.mode = RuntimeMode::StageMajor;
        else if (value == "hybrid") parsed.mode = RuntimeMode::Hybrid;
        else if (value == "auto") parsed.mode = RuntimeMode::Auto;
        else return fail(error, "INVALID_RUNTIME_MODE");
    }
    if (std::regex_search(json, match, std::regex(R"rx("tile_cohort_size"\s*:\s*(-?[0-9]+))rx")))
        parsed.cohort_size = std::stoi(match[1]);
    if (std::regex_search(json, match, std::regex(R"rx("checkpoint_mode"\s*:\s*"([^"]+)")rx"))) {
        const std::string value = match[1];
        if (value == "production") parsed.checkpoint_mode = CheckpointMode::Production;
        else if (value == "debug") parsed.checkpoint_mode = CheckpointMode::Debug;
        else return fail(error, "INVALID_CHECKPOINT_MODE");
    }
    parsed.resume_enabled = std::regex_search(json, std::regex(R"rx("resume"\s*:\s*true)rx"));
    if (parsed.mode == RuntimeMode::Hybrid && parsed.cohort_size <= 0)
        return fail(error, "INVALID_COHORT_SIZE");
    *request = parsed;
    return true;
}

std::vector<TileCohort> plan_tile_cohorts(int tile_count, int cohort_size,
                                          std::string* error)
{
    if (error) error->clear();
    if (tile_count <= 0) { fail(error, "INVALID_TILE_COUNT"); return {}; }
    if (cohort_size <= 0) { fail(error, "INVALID_COHORT_SIZE"); return {}; }
    std::vector<TileCohort> result;
    for (int begin = 0; begin < tile_count; begin += cohort_size)
        result.push_back({begin, std::min(begin + cohort_size, tile_count)});
    return result;
}

bool run_hybrid_schedule(StageMajorCheckpoint* checkpoint, int cohort_size,
                         const StageMajorExecutionHooks& hooks,
                         StageMajorRunStats* stats, std::string* error)
{
    if (error) error->clear();
    if (!checkpoint || !stats || cohort_size <= 0)
        return fail(error, cohort_size <= 0 ? "INVALID_COHORT_SIZE" : "CHECKPOINT_CORRUPT");
    if (!hooks.begin_stage || !hooks.execute_work || !hooks.end_stage || !hooks.persist_checkpoint)
        return fail(error, "HYBRID_HOOKS_REQUIRED");
    *stats = StageMajorRunStats{};
    StageMajorWork work;
    std::string next_error;
    while (next_hybrid_work(*checkpoint, cohort_size, &work, &next_error)) {
        const StageMajorWork stage = work;
        if (!hooks.begin_stage(hooks.context, stage, error)) return false;
        const bool model_stage = stage.stage != "fusion";
        if (model_stage) { ++stats->model_stage_loads; stats->peak_resident_model_stages = 1; }
        bool ok = true;
        while (ok) {
            if (!hooks.execute_work(hooks.context, work, error) ||
                !mark_hybrid_complete(checkpoint, cohort_size, work, error) ||
                !hooks.persist_checkpoint(hooks.context, *checkpoint, error)) { ok = false; break; }
            ++stats->work_items_completed; ++stats->checkpoint_persists;
            if (hooks.checkpoint_committed) hooks.checkpoint_committed(hooks.context, work);
            StageMajorWork next; next_error.clear();
            if (!next_hybrid_work(*checkpoint, cohort_size, &next, &next_error) || !same_stage(next, stage)) break;
            work = next;
        }
        hooks.end_stage(hooks.context, stage);
        if (model_stage) ++stats->model_stage_unloads;
        if (!ok) return false;
    }
    if (!next_error.empty() && checkpoint->state == "running") return fail(error, next_error.c_str());
    return checkpoint->state == "completed";
}

bool estimate_task_resources(int width, int height, const RuntimeRequest& request,
                             uint64_t available_disk_bytes,
                             uint64_t memory_budget_bytes,
                             const RuntimeCalibration& calibration,
                             RuntimeResourceEstimate* estimate,
                             std::string* error)
{
    if (error) error->clear();
    if (!estimate) return fail(error, "RESOURCE_ESTIMATE_OUTPUT_NULL");
    if (width <= 0 || height <= 0)
        return fail(error, "INVALID_RESOURCE_DIMENSIONS");
    if (calibration.tile_major_max_tiles < 1 ||
        calibration.stage_major_max_tiles < calibration.tile_major_max_tiles ||
        calibration.default_cohort_size < 1 ||
        calibration.per_tile_runtime_ms < 0.0 ||
        calibration.model_reload_per_cohort_ms < 0.0)
        return fail(error, "INVALID_RESOURCE_CALIBRATION");

    uint64_t tiles = 0;
    if (!checked_mul(axis_tiles(width), axis_tiles(height), &tiles) || tiles == 0)
        return fail(error, "RESOURCE_ESTIMATE_OVERFLOW");
    RuntimeSelection selection;
    selection.requested_mode = request.mode;
    if (request.mode != RuntimeMode::Auto)
    {
        selection.selected_mode = request.mode;
        selection.cohort_size = request.mode == RuntimeMode::Hybrid ? request.cohort_size : 0;
        if (request.mode == RuntimeMode::Hybrid && request.cohort_size <= 0)
            return fail(error, "INVALID_COHORT_SIZE");
        selection.reason = "EXPLICIT_RUNTIME_OVERRIDE";
    }
    else if (request.resume_enabled)
    {
        selection.selected_mode = RuntimeMode::Hybrid;
        selection.cohort_size = std::min<uint64_t>(tiles, calibration.default_cohort_size);
        selection.reason = "RESUME_REQUIRES_DURABLE_HYBRID";
    }
    else if (tiles <= static_cast<uint64_t>(calibration.tile_major_max_tiles))
    {
        selection.selected_mode = RuntimeMode::TileMajor;
        selection.reason = "MEASURED_SMALL_TASK_TILE_MAJOR";
    }
    else if (tiles <= static_cast<uint64_t>(calibration.stage_major_max_tiles))
    {
        selection.selected_mode = RuntimeMode::StageMajor;
        selection.reason = "MEASURED_MEDIUM_TASK_STAGE_MAJOR";
    }
    else
    {
        selection.selected_mode = RuntimeMode::Hybrid;
        selection.cohort_size = std::min<uint64_t>(tiles, calibration.default_cohort_size);
        selection.reason = "MEASURED_LONG_TASK_BOUNDED_DISK";
    }

    uint64_t live_tiles = selection.selected_mode == RuntimeMode::StageMajor ? tiles :
        (selection.selected_mode == RuntimeMode::Hybrid ?
            static_cast<uint64_t>(selection.cohort_size) : 1u);
    uint64_t intermediate = 0, final_bytes = 0, peak = 0, required = 0;
    if (!checked_mul(live_tiles, calibration.peak_intermediate_bytes_per_tile, &intermediate) ||
        !checked_mul(tiles, calibration.final_bytes_per_tile, &final_bytes) ||
        !checked_add(intermediate, final_bytes, &peak) ||
        !checked_add(peak, calibration.safety_margin_bytes, &required))
        return fail(error, "RESOURCE_ESTIMATE_OVERFLOW");

    // StageMajor is the measured latency choice through 25 tiles, but Auto must
    // not select it when its all-tile intermediate state cannot fit. Fall back
    // to the measured default Hybrid cohort before considering smaller cohorts.
    if (request.mode == RuntimeMode::Auto &&
        selection.selected_mode == RuntimeMode::StageMajor &&
        required > available_disk_bytes)
    {
        selection.selected_mode = RuntimeMode::Hybrid;
        selection.cohort_size = std::min<uint64_t>(tiles, calibration.default_cohort_size);
        selection.reason = "AVAILABLE_DISK_SELECTED_HYBRID";
        live_tiles = static_cast<uint64_t>(selection.cohort_size);
        if (!checked_mul(live_tiles, calibration.peak_intermediate_bytes_per_tile, &intermediate) ||
            !checked_add(intermediate, final_bytes, &peak) ||
            !checked_add(peak, calibration.safety_margin_bytes, &required))
            return fail(error, "RESOURCE_ESTIMATE_OVERFLOW");
    }

    // Auto may reduce K only from measured/default K downwards to fit the disk;
    // it never invents a larger cohort or hides an insufficient final-state budget.
    if (request.mode == RuntimeMode::Auto && selection.selected_mode == RuntimeMode::Hybrid &&
        required > available_disk_bytes && selection.cohort_size > 1)
    {
        while (selection.cohort_size > 1 && required > available_disk_bytes)
        {
            --selection.cohort_size;
            if (!checked_mul(static_cast<uint64_t>(selection.cohort_size),
                             calibration.peak_intermediate_bytes_per_tile, &intermediate) ||
                !checked_add(intermediate, final_bytes, &peak) ||
                !checked_add(peak, calibration.safety_margin_bytes, &required))
                return fail(error, "RESOURCE_ESTIMATE_OVERFLOW");
        }
        if (required <= available_disk_bytes)
            selection.reason = "AVAILABLE_DISK_REDUCED_HYBRID_COHORT";
    }
    if (required > available_disk_bytes)
        return fail(error, "INSUFFICIENT_ESTIMATED_DISK");
    const uint64_t rss_bytes = static_cast<uint64_t>(
        std::ceil(calibration.peak_rss_mb * 1024.0 * 1024.0));
    if (memory_budget_bytes && rss_bytes > memory_budget_bytes)
        return fail(error, "INSUFFICIENT_MEMORY_BUDGET");

    double runtime_ms = 0.0;
    if (selection.selected_mode == RuntimeMode::TileMajor)
        runtime_ms = 15190.4 * static_cast<double>(tiles); // measured 2-tile median / 2
    else
    {
        const uint64_t cohorts = selection.selected_mode == RuntimeMode::Hybrid ?
            (tiles + static_cast<uint64_t>(selection.cohort_size) - 1u) /
                static_cast<uint64_t>(selection.cohort_size) : 1u;
        runtime_ms = calibration.fixed_runtime_ms +
            calibration.per_tile_runtime_ms * static_cast<double>(tiles) +
            calibration.model_reload_per_cohort_ms * static_cast<double>(cohorts - 1u);
    }
    *estimate = RuntimeResourceEstimate{};
    estimate->width = width; estimate->height = height; estimate->tile_count = tiles;
    estimate->selection = selection;
    estimate->estimated_peak_task_bytes = peak;
    estimate->estimated_final_task_bytes = final_bytes;
    estimate->required_free_bytes = required;
    estimate->estimated_runtime_ms = runtime_ms;
    estimate->estimated_peak_rss_mb = calibration.peak_rss_mb;
    estimate->source_label = calibration.source_label;
    return true;
}

}  // namespace seededge::phase18d
