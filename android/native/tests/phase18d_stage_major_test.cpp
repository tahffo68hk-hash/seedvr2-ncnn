#include "phase18d_stage_major.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
namespace p18d = seededge::phase18d;

namespace {
int failed = 0;

void expect(bool value, const char* message)
{
    if (!value)
    {
        std::cerr << "FAIL " << message << '\n';
        ++failed;
    }
}

p18d::StageMajorIdentity identity(int tiles = 2)
{
    p18d::StageMajorIdentity value;
    value.task_id = "task-stage-major";
    value.model_version = "phase18d-model-v1";
    value.source_sha256 = std::string(64, 'a');
    value.normalized_image_sha256 = std::string(64, 'b');
    value.tile_count = tiles;
    value.tile_plan_sha256 = std::string(64, 'c');
    return value;
}

void complete_until(p18d::StageMajorCheckpoint* checkpoint,
                    const std::string& stage, int stage_index, int block_index)
{
    std::string error;
    for (int guard = 0; guard < 10000; ++guard)
    {
        p18d::StageMajorWork work;
        if (!p18d::next_stage_major_work(*checkpoint, &work, &error)) return;
        if (work.stage == stage && work.stage_index == stage_index &&
            work.block_index == block_index)
            return;
        expect(p18d::mark_stage_major_work_complete(checkpoint, work, &error),
               "work item can be completed");
    }
    expect(false, "state machine terminates");
}

void test_stage_order_and_resume()
{
    p18d::StageMajorCheckpoint checkpoint =
        p18d::make_stage_major_checkpoint(identity());
    std::string error;
    p18d::StageMajorWork work;
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "encoder" && work.stage_index == 0 &&
               work.tile_index == 0,
           "new task starts at encoder stage 0 tile 0");
    expect(p18d::mark_stage_major_work_complete(&checkpoint, work, &error),
           "first encoder work completes");
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "encoder" && work.stage_index == 0 &&
               work.tile_index == 1,
           "encoder is stage-major across tiles");

    complete_until(&checkpoint, "dit", -1, 5);
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "dit" && work.block_index == 5 && work.tile_index == 0,
           "interruption before DiT block 5 resumes at block 5 tile 0");
    expect(p18d::mark_stage_major_work_complete(&checkpoint, work, &error),
           "DiT block 5 tile 0 completes");
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.block_index == 5 && work.tile_index == 1,
           "DiT block stays loaded for the second tile");

    complete_until(&checkpoint, "scheduler", -1, -1);
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "scheduler" && work.tile_index == 0,
           "DiT block 31 advances to scheduler");
    complete_until(&checkpoint, "decoder", 6, -1);
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "decoder" && work.stage_index == 6,
           "decoder interruption resumes at final decoder stage");
    complete_until(&checkpoint, "fusion", -1, -1);
    expect(p18d::next_stage_major_work(checkpoint, &work, &error) &&
               work.stage == "fusion" && work.tile_index == -1,
           "pre-fusion interruption resumes at fusion only");
    expect(p18d::mark_stage_major_work_complete(&checkpoint, work, &error),
           "fusion completes");
    expect(!p18d::next_stage_major_work(checkpoint, &work, &error) &&
               checkpoint.state == "completed" && checkpoint.fusion_completed,
           "completed task has no more work");
}

void test_atomic_persistence_and_identity()
{
    const fs::path root = fs::temp_directory_path() / "phase18d_stage_major_state_test";
    std::error_code ec;
    fs::remove_all(root, ec);
    fs::create_directories(root, ec);
    p18d::StageMajorCheckpoint checkpoint =
        p18d::make_stage_major_checkpoint(identity(4));
    std::string error;
    expect(p18d::persist_stage_major_checkpoint(root.string(), checkpoint, &error),
           "checkpoint persists atomically");
    expect(fs::is_regular_file(root / "stage_major_checkpoint.json") &&
               !fs::exists(root / "stage_major_checkpoint.json.tmp"),
           "atomic persistence leaves only final checkpoint");
    p18d::StageMajorCheckpoint loaded;
    expect(p18d::load_stage_major_checkpoint(root.string(), &loaded, &error),
           "checkpoint reload succeeds");
    expect(p18d::validate_stage_major_checkpoint(identity(4), loaded, &error),
           "matching task identity is accepted");
    p18d::StageMajorIdentity wrong = identity(4);
    wrong.task_id = "wrong-task";
    expect(!p18d::validate_stage_major_checkpoint(wrong, loaded, &error) &&
               error == "CHECKPOINT_TASK_MISMATCH",
           "wrong task is rejected");
    wrong = identity(4);
    wrong.model_version = "wrong-model";
    expect(!p18d::validate_stage_major_checkpoint(wrong, loaded, &error) &&
               error == "CHECKPOINT_MODEL_MISMATCH",
           "wrong model is rejected");
    wrong = identity(4);
    wrong.normalized_image_sha256 = std::string(64, 'd');
    expect(!p18d::validate_stage_major_checkpoint(wrong, loaded, &error) &&
               error == "CHECKPOINT_INPUT_MISMATCH",
           "wrong normalized image is rejected");
    wrong = identity(4);
    wrong.tile_plan_sha256 = std::string(64, 'e');
    expect(!p18d::validate_stage_major_checkpoint(wrong, loaded, &error) &&
               error == "CHECKPOINT_PLAN_MISMATCH",
           "wrong tile plan is rejected");

    std::ofstream(root / "stage_major_checkpoint.json", std::ios::trunc)
        << "{corrupt";
    expect(!p18d::load_stage_major_checkpoint(root.string(), &loaded, &error) &&
               error == "CHECKPOINT_CORRUPT",
           "corrupted checkpoint is rejected");
    fs::remove_all(root, ec);
}

void test_pause_cancel_and_duplicate_work()
{
    p18d::StageMajorCheckpoint checkpoint =
        p18d::make_stage_major_checkpoint(identity());
    std::string error;
    p18d::StageMajorWork work;
    expect(p18d::set_stage_major_control(&checkpoint, "paused", &error),
           "running task pauses at a boundary");
    expect(!p18d::next_stage_major_work(checkpoint, &work, &error) &&
               error == "TASK_PAUSED",
           "paused task dispatches no work");
    expect(p18d::set_stage_major_control(&checkpoint, "running", &error),
           "paused task resumes");
    expect(p18d::next_stage_major_work(checkpoint, &work, &error),
           "resumed task dispatches work");
    const p18d::StageMajorWork duplicate = work;
    expect(p18d::mark_stage_major_work_complete(&checkpoint, work, &error),
           "dispatched work completes once");
    expect(!p18d::mark_stage_major_work_complete(&checkpoint, duplicate, &error) &&
               error == "STALE_WORK_ITEM",
           "duplicate completion is rejected");
    expect(p18d::set_stage_major_control(&checkpoint, "cancelled", &error),
           "running task cancels at a boundary");
    expect(!p18d::next_stage_major_work(checkpoint, &work, &error) &&
               error == "TASK_CANCELLED",
           "cancelled task dispatches no work");
    expect(!p18d::set_stage_major_control(&checkpoint, "running", &error),
           "cancelled task cannot resume implicitly");
}

struct HookContext
{
    int model_loads = 0;
    int model_unloads = 0;
    int work_items = 0;
    int persists = 0;
    int committed = 0;
    std::string active_key;
};

std::string work_key(const p18d::StageMajorWork& work)
{
    return work.stage + ":" + std::to_string(work.stage_index) + ":" +
           std::to_string(work.block_index);
}

bool hook_begin(void* opaque, const p18d::StageMajorWork& work, std::string*)
{
    auto* context = static_cast<HookContext*>(opaque);
    context->active_key = work_key(work);
    if (work.stage != "fusion") ++context->model_loads;
    return true;
}

bool hook_execute(void* opaque, const p18d::StageMajorWork& work, std::string*)
{
    auto* context = static_cast<HookContext*>(opaque);
    if (context->active_key != work_key(work)) return false;
    ++context->work_items;
    return true;
}

void hook_end(void* opaque, const p18d::StageMajorWork& work)
{
    auto* context = static_cast<HookContext*>(opaque);
    if (work.stage != "fusion") ++context->model_unloads;
    context->active_key.clear();
}

bool hook_persist(void* opaque, const p18d::StageMajorCheckpoint&, std::string*)
{
    ++static_cast<HookContext*>(opaque)->persists;
    return true;
}

void hook_committed(void* opaque, const p18d::StageMajorWork&)
{
    ++static_cast<HookContext*>(opaque)->committed;
}

void test_grouped_stage_scheduler()
{
    for (int tiles : {2, 4})
    {
        p18d::StageMajorCheckpoint checkpoint =
            p18d::make_stage_major_checkpoint(identity(tiles));
        HookContext context;
        p18d::StageMajorExecutionHooks hooks;
        hooks.context = &context;
        hooks.begin_stage = &hook_begin;
        hooks.execute_work = &hook_execute;
        hooks.end_stage = &hook_end;
        hooks.persist_checkpoint = &hook_persist;
        hooks.checkpoint_committed = &hook_committed;
        p18d::StageMajorRunStats stats;
        std::string error;
        expect(p18d::run_stage_major_schedule(&checkpoint, hooks, &stats, &error),
               "grouped stage-major schedule completes");
        const int expected_work = tiles * (7 + 1 + 32 + 1 + 7) + 1;
        expect(context.model_loads == 48 && context.model_unloads == 48,
               "model stages load and unload once independent of tile count");
        expect(context.work_items == expected_work && context.persists == expected_work &&
                   context.committed == expected_work,
               "every tile boundary persists exactly once");
        expect(stats.model_stage_loads == 48 && stats.work_items_completed == expected_work &&
                   stats.peak_resident_model_stages == 1,
               "scheduler reports bounded residency and exact counts");
    }
}
}  // namespace

int main()
{
    test_stage_order_and_resume();
    test_atomic_persistence_and_identity();
    test_pause_cancel_and_duplicate_work();
    test_grouped_stage_scheduler();
    std::cout << "phase18d_stage_major_test failed=" << failed << '\n';
    return failed == 0 ? 0 : 1;
}
