#pragma once

#include <cstdint>
#include <string>

namespace seededge::phase18d {

enum class CheckpointCrashPoint
{
    None,
    BeforeWrite,
    HalfWrite,
    BeforeRename,
    BeforeControlUpdate,
    BeforeGc,
    AfterGc
};

struct LoadedCheckpoint
{
    uint64_t generation = 0;
    std::string identity_sha256;
    std::string payload_sha256;
    std::string payload;
};

struct CheckpointInventory
{
    int valid_generations = 0;
    int temporary_files = 0;
    uint64_t newest_generation = 0;
    uint64_t total_bytes = 0;
};

bool publish_checkpoint_generation(const std::string& task_root,
                                   const std::string& payload,
                                   const std::string& identity_sha256,
                                   CheckpointCrashPoint crash_point,
                                   uint64_t* published_generation,
                                   std::string* error);
bool load_newest_valid_generation(const std::string& task_root,
                                  const std::string& identity_sha256,
                                  LoadedCheckpoint* loaded,
                                  std::string* error);
bool load_newest_valid_generation_any(const std::string& task_root,
                                      LoadedCheckpoint* loaded,
                                      std::string* error);
bool inspect_checkpoint_generations(const std::string& task_root,
                                    CheckpointInventory* inventory,
                                    std::string* error);

}  // namespace seededge::phase18d
