#include "phase18d_runtime_strategy.h"

#include <iostream>
#include <string>
#include <vector>
#include <limits>

namespace p18d = seededge::phase18d;

namespace {
int failed = 0;
void expect(bool value, const char* message)
{
    if (!value) { ++failed; std::cerr << "FAIL " << message << '\n'; }
}

p18d::StageMajorIdentity identity(int tiles)
{
    return {"hybrid-task", "model-v1", std::string(64, 'a'),
            std::string(64, 'b'), tiles, std::string(64, 'c')};
}

struct Trace { std::vector<p18d::StageMajorWork> work; };
bool begin(void*, const p18d::StageMajorWork&, std::string*) { return true; }
bool execute(void* opaque, const p18d::StageMajorWork& work, std::string*)
{
    static_cast<Trace*>(opaque)->work.push_back(work);
    return true;
}
void end(void*, const p18d::StageMajorWork&) {}
bool persist(void*, const p18d::StageMajorCheckpoint&, std::string*) { return true; }
}

int main()
{
    p18d::RuntimeRequest request;
    std::string error;
    expect(p18d::parse_runtime_request(
        R"({"runtime_mode":"hybrid","tile_cohort_size":4,"resume":true})",
        &request, &error), "hybrid request parses");
    expect(request.mode == p18d::RuntimeMode::Hybrid && request.cohort_size == 4 &&
           request.resume_enabled && request.checkpoint_mode == p18d::CheckpointMode::Production,
           "parsed hybrid fields and default production mode are exact");
    expect(p18d::parse_runtime_request(
        R"({"runtime_mode":"stage_major","checkpoint_mode":"debug"})",
        &request, &error) && request.checkpoint_mode == p18d::CheckpointMode::Debug,
        "debug checkpoint retention requires an explicit request");
    expect(!p18d::parse_runtime_request(
        R"({"runtime_mode":"stage_major","checkpoint_mode":"lossy"})",
        &request, &error) && error == "INVALID_CHECKPOINT_MODE",
        "unknown or lossy checkpoint mode is rejected");
    expect(!p18d::parse_runtime_request(
        R"({"runtime_mode":"hybrid","tile_cohort_size":0})", &request, &error) &&
           error == "INVALID_COHORT_SIZE", "zero cohort is rejected stably");
    expect(!p18d::parse_runtime_request(
        R"({"runtime_mode":"mystery"})", &request, &error) &&
           error == "INVALID_RUNTIME_MODE", "unknown mode is rejected stably");
    expect(p18d::runtime_uses_staged_scheduler(p18d::RuntimeMode::StageMajor) &&
           p18d::runtime_uses_staged_scheduler(p18d::RuntimeMode::Hybrid) &&
           !p18d::runtime_uses_staged_scheduler(p18d::RuntimeMode::TileMajor),
           "only stage-major and hybrid use staged real hooks");

    const std::vector<p18d::TileCohort> cohorts = p18d::plan_tile_cohorts(9, 4, &error);
    expect(cohorts.size() == 3 && cohorts[0].begin == 0 && cohorts[0].end == 4 &&
           cohorts[1].begin == 4 && cohorts[1].end == 8 &&
           cohorts[2].begin == 8 && cohorts[2].end == 9,
           "nine tiles split into deterministic consecutive cohorts");
    expect(p18d::plan_tile_cohorts(0, 4, &error).empty() &&
           error == "INVALID_TILE_COUNT", "zero tiles are rejected stably");

    p18d::RuntimeCalibration calibration;
    p18d::RuntimeResourceEstimate estimate;
    p18d::RuntimeRequest automatic;
    expect(p18d::estimate_task_resources(160, 96, automatic,
        std::numeric_limits<uint64_t>::max(), 2ull * 1024 * 1024 * 1024,
        calibration, &estimate, &error) && estimate.tile_count == 2 &&
        estimate.selection.selected_mode == p18d::RuntimeMode::TileMajor &&
        estimate.selection.reason == "MEASURED_SMALL_TASK_TILE_MAJOR",
        "Auto uses measured two-tile crossover");
    expect(p18d::estimate_task_resources(640, 427, automatic,
        std::numeric_limits<uint64_t>::max(), 2ull * 1024 * 1024 * 1024,
        calibration, &estimate, &error) && estimate.tile_count == 35 &&
        estimate.selection.selected_mode == p18d::RuntimeMode::Hybrid &&
        estimate.selection.cohort_size == 4 &&
        estimate.required_free_bytes >= estimate.estimated_peak_task_bytes,
        "Auto uses bounded-disk Hybrid for a long task");
    p18d::RuntimeRequest explicit_stage;
    explicit_stage.mode = p18d::RuntimeMode::StageMajor;
    expect(p18d::estimate_task_resources(4032, 3024, explicit_stage,
        std::numeric_limits<uint64_t>::max(), 2ull * 1024 * 1024 * 1024,
        calibration, &estimate, &error) && estimate.tile_count == 1344 &&
        estimate.selection.selected_mode == p18d::RuntimeMode::StageMajor &&
        estimate.selection.reason == "EXPLICIT_RUNTIME_OVERRIDE",
        "explicit StageMajor overrides Auto at maximum supported photo size");
    const int dimensions[][2] = {{1,1},{127,127},{128,128},{129,129},
        {255,257},{511,513},{640,427},{1920,1080},{1976,1267},
        {3200,2136},{4032,3024}};
    for (const auto& dimension : dimensions)
        expect(p18d::estimate_task_resources(dimension[0], dimension[1], automatic,
            std::numeric_limits<uint64_t>::max(), 2ull * 1024 * 1024 * 1024,
            calibration, &estimate, &error) && estimate.tile_count > 0,
            "all required planner dimensions are overflow-safe");
    expect(!p18d::estimate_task_resources(0, 1, automatic,
        std::numeric_limits<uint64_t>::max(), 0, calibration, &estimate, &error) &&
        error == "INVALID_RESOURCE_DIMENSIONS", "invalid dimensions fail stably");
    expect(!p18d::estimate_task_resources(640, 427, automatic, 1,
        2ull * 1024 * 1024 * 1024, calibration, &estimate, &error) &&
        error == "INSUFFICIENT_ESTIMATED_DISK", "disk budget fails before execution");
    expect(!p18d::estimate_task_resources(640, 427, automatic,
        std::numeric_limits<uint64_t>::max(), 512ull * 1024 * 1024,
        calibration, &estimate, &error) && error == "INSUFFICIENT_MEMORY_BUDGET",
        "memory budget fails before execution");

    p18d::StageMajorCheckpoint checkpoint = p18d::make_stage_major_checkpoint(identity(5));
    Trace trace;
    p18d::StageMajorExecutionHooks hooks;
    hooks.context = &trace; hooks.begin_stage = &begin; hooks.execute_work = &execute;
    hooks.end_stage = &end; hooks.persist_checkpoint = &persist;
    p18d::StageMajorRunStats stats;
    expect(p18d::run_hybrid_schedule(&checkpoint, 2, hooks, &stats, &error),
           "hybrid schedule completes");
    expect(checkpoint.state == "completed" && checkpoint.completed_tiles == 5,
           "hybrid completes every tile exactly once");
    expect(stats.model_stage_loads == 144 && stats.model_stage_unloads == 144,
           "three cohorts each load exactly 48 weighted stages");
    expect(stats.work_items_completed == 241,
           "five tiles execute 48 work items each plus one fusion");
    expect(stats.peak_resident_model_stages == 1,
           "hybrid keeps at most one weighted stage resident");
    expect(trace.work.size() == 241 && trace.work.front().stage == "encoder" &&
           trace.work.front().tile_index == 0 && trace.work[1].tile_index == 1 &&
           trace.work[2].stage == "encoder" && trace.work[2].stage_index == 1,
           "cohort zero advances stage before cohort one begins");

    std::cout << "phase18d_runtime_strategy_test failed=" << failed << '\n';
    return failed ? 1 : 0;
}
