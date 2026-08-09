#pragma once

#include <string>
#include <vector>

namespace seededge::phase18d {

struct StageMajorIdentity
{
    std::string task_id;
    std::string model_version;
    std::string source_sha256;
    std::string normalized_image_sha256;
    int tile_count = 0;
    std::string tile_plan_sha256;
};

struct StageMajorCheckpoint
{
    int schema_version = 1;
    StageMajorIdentity identity;
    std::string state = "running";
    std::string current_stage = "encoder";
    int current_stage_index = 0;
    int current_block = -1;
    int completed_tiles = 0;
    std::vector<int> encoder_completed_stage_per_tile;
    std::vector<bool> encoder_completed_tiles;
    std::vector<int> dit_completed_block_per_tile;
    std::vector<bool> scheduler_completed_tiles;
    std::vector<int> decoder_completed_stage_per_tile;
    std::vector<bool> decoder_completed_tiles;
    bool fusion_completed = false;
};

struct StageMajorWork
{
    std::string stage;
    int stage_index = -1;
    int block_index = -1;
    int tile_index = -1;
};

StageMajorCheckpoint make_stage_major_checkpoint(const StageMajorIdentity& identity);

bool validate_stage_major_checkpoint(const StageMajorIdentity& expected,
                                     const StageMajorCheckpoint& actual,
                                     std::string* error);

bool persist_stage_major_checkpoint(const std::string& task_root,
                                    const StageMajorCheckpoint& checkpoint,
                                    std::string* error);

bool load_stage_major_checkpoint(const std::string& task_root,
                                 StageMajorCheckpoint* checkpoint,
                                 std::string* error);

bool next_stage_major_work(const StageMajorCheckpoint& checkpoint,
                           StageMajorWork* work,
                           std::string* error);

bool mark_stage_major_work_complete(StageMajorCheckpoint* checkpoint,
                                    const StageMajorWork& work,
                                    std::string* error);

bool set_stage_major_control(StageMajorCheckpoint* checkpoint,
                             const std::string& state,
                             std::string* error);

}  // namespace seededge::phase18d
