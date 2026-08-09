#include "phase18d_checkpoint_store.h"

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
    if (!value) { ++failed; std::cerr << "FAIL " << message << '\n'; }
}
}

int main()
{
    const fs::path root = fs::temp_directory_path() / "phase18d_checkpoint_store_test";
    fs::remove_all(root);
    const std::string identity(64, 'a');
    std::string error;
    uint64_t generation = 0;
    expect(p18d::publish_checkpoint_generation(root.string(), R"({"state":"one"})",
        identity, p18d::CheckpointCrashPoint::None, &generation, &error) && generation == 1,
        "first generation publishes");

    p18d::LoadedCheckpoint loaded;
    expect(p18d::load_newest_valid_generation(root.string(), identity, &loaded, &error) &&
           loaded.generation == 1 && loaded.payload == R"({"state":"one"})",
           "first generation loads exactly");
    p18d::LoadedCheckpoint newest_any;
    expect(p18d::load_newest_valid_generation_any(root.string(), &newest_any, &error) &&
           newest_any.generation == 1 && newest_any.identity_sha256 == identity,
           "startup can scan newest valid generation without trusting stale control");

    for (p18d::CheckpointCrashPoint point : {
            p18d::CheckpointCrashPoint::BeforeWrite,
            p18d::CheckpointCrashPoint::HalfWrite,
            p18d::CheckpointCrashPoint::BeforeRename}) {
        expect(!p18d::publish_checkpoint_generation(root.string(), R"({"state":"two"})",
            identity, point, &generation, &error), "pre-publish crash is reported");
        expect(p18d::load_newest_valid_generation(root.string(), identity, &loaded, &error) &&
               loaded.generation == 1, "pre-publish crash preserves generation one");
    }

    expect(!p18d::publish_checkpoint_generation(root.string(), R"({"state":"two"})",
        identity, p18d::CheckpointCrashPoint::BeforeControlUpdate, &generation, &error) &&
        generation == 2, "control-update crash occurs after generation two publish");
    expect(p18d::load_newest_valid_generation(root.string(), identity, &loaded, &error) &&
           loaded.generation == 2 && loaded.payload == R"({"state":"two"})",
           "startup scan recovers published generation despite stale control");

    expect(p18d::publish_checkpoint_generation(root.string(), R"({"state":"three"})",
        identity, p18d::CheckpointCrashPoint::None, &generation, &error) && generation == 3,
        "generation three publishes and GC runs");
    p18d::CheckpointInventory inventory;
    expect(p18d::inspect_checkpoint_generations(root.string(), &inventory, &error) &&
           inventory.valid_generations == 2 && inventory.temporary_files == 0 &&
           inventory.newest_generation == 3,
           "GC retains only newest and one rollback generation and removes tmp files");

    std::ofstream(root / "checkpoints" / "generation_0000000003.json", std::ios::trunc)
        << "corrupt";
    expect(p18d::load_newest_valid_generation(root.string(), identity, &loaded, &error) &&
           loaded.generation == 2, "corrupt newest generation falls back to prior valid generation");
    expect(!p18d::load_newest_valid_generation(root.string(), std::string(64, 'b'),
           &loaded, &error) && error == "CHECKPOINT_IDENTITY_MISMATCH",
           "wrong task identity cannot resume");

    const fs::path before_gc = fs::temp_directory_path() / "phase18d_checkpoint_before_gc";
    fs::remove_all(before_gc);
    expect(p18d::publish_checkpoint_generation(before_gc.string(), R"({"n":1})", identity,
           p18d::CheckpointCrashPoint::None, &generation, &error), "before-GC fixture publishes one");
    expect(!p18d::publish_checkpoint_generation(before_gc.string(), R"({"n":2})", identity,
           p18d::CheckpointCrashPoint::BeforeGc, &generation, &error) && generation == 2,
           "GC-before crash is injected after control publication");
    expect(p18d::load_newest_valid_generation(before_gc.string(), identity, &loaded, &error) &&
           loaded.generation == 2, "GC-before crash resumes newest valid generation");

    const fs::path after_gc = fs::temp_directory_path() / "phase18d_checkpoint_after_gc";
    fs::remove_all(after_gc);
    expect(p18d::publish_checkpoint_generation(after_gc.string(), R"({"n":1})", identity,
           p18d::CheckpointCrashPoint::None, &generation, &error) &&
           p18d::publish_checkpoint_generation(after_gc.string(), R"({"n":2})", identity,
           p18d::CheckpointCrashPoint::None, &generation, &error), "after-GC fixture publishes two");
    expect(!p18d::publish_checkpoint_generation(after_gc.string(), R"({"n":3})", identity,
           p18d::CheckpointCrashPoint::AfterGc, &generation, &error) && generation == 3,
           "GC-after crash is injected only after obsolete generation removal");
    expect(p18d::load_newest_valid_generation(after_gc.string(), identity, &loaded, &error) &&
           loaded.generation == 3, "GC-after crash resumes newest valid generation");

    fs::remove_all(root);
    fs::remove_all(before_gc);
    fs::remove_all(after_gc);
    std::cout << "phase18d_checkpoint_store_test failed=" << failed << '\n';
    return failed ? 1 : 0;
}
