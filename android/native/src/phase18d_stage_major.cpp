#include "phase18d_stage_major.h"
#include "phase18d_checkpoint_store.h"
#include "phase18d_tile_core.h"

#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace seededge::phase18d {
namespace {

bool fail(std::string* error, const std::string& message)
{
    if (error) *error = message;
    return false;
}

bool valid_hash(const std::string& value)
{
    return value.size() == 64 &&
           value.find_first_not_of("0123456789abcdef") == std::string::npos;
}

bool valid_structure(const StageMajorCheckpoint& checkpoint)
{
    const size_t count = checkpoint.identity.tile_count > 0
        ? static_cast<size_t>(checkpoint.identity.tile_count) : 0u;
    return checkpoint.schema_version == 1 && count > 0 &&
           checkpoint.encoder_completed_stage_per_tile.size() == count &&
           checkpoint.encoder_completed_tiles.size() == count &&
           checkpoint.dit_input_completed_tiles.size() == count &&
           checkpoint.dit_completed_block_per_tile.size() == count &&
           checkpoint.scheduler_completed_tiles.size() == count &&
           checkpoint.decoder_completed_stage_per_tile.size() == count &&
           checkpoint.decoder_completed_tiles.size() == count &&
           valid_hash(checkpoint.identity.source_sha256) &&
           valid_hash(checkpoint.identity.normalized_image_sha256) &&
           valid_hash(checkpoint.identity.tile_plan_sha256) &&
           !checkpoint.identity.task_id.empty() &&
           !checkpoint.identity.model_version.empty();
}

std::string escape_json(const std::string& value)
{
    std::string escaped;
    for (char c : value)
    {
        if (c == '\\' || c == '"') escaped += '\\';
        if (static_cast<unsigned char>(c) >= 0x20u) escaped += c;
    }
    return escaped;
}

template <typename T>
void append_array(std::ostringstream& output, const std::vector<T>& values)
{
    output << '[';
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void append_bool_array(std::ostringstream& output, const std::vector<bool>& values)
{
    output << '[';
    for (size_t index = 0; index < values.size(); ++index)
    {
        if (index) output << ',';
        output << (values[index] ? "true" : "false");
    }
    output << ']';
}

std::string serialize(const StageMajorCheckpoint& checkpoint)
{
    const StageMajorIdentity& id = checkpoint.identity;
    std::ostringstream output;
    output << "{\"schema_version\":" << checkpoint.schema_version
           << ",\"task_id\":\"" << escape_json(id.task_id)
           << "\",\"model_version\":\"" << escape_json(id.model_version)
           << "\",\"source_sha256\":\"" << id.source_sha256
           << "\",\"normalized_image_sha256\":\"" << id.normalized_image_sha256
           << "\",\"tile_count\":" << id.tile_count
           << ",\"tile_plan_sha256\":\"" << id.tile_plan_sha256
           << "\",\"state\":\"" << checkpoint.state
           << "\",\"current_stage\":\"" << checkpoint.current_stage
           << "\",\"current_stage_index\":" << checkpoint.current_stage_index
           << ",\"current_block\":" << checkpoint.current_block
           << ",\"completed_tiles\":" << checkpoint.completed_tiles
           << ",\"encoder_completed_stage_per_tile\":";
    append_array(output, checkpoint.encoder_completed_stage_per_tile);
    output << ",\"encoder_completed_tiles\":";
    append_bool_array(output, checkpoint.encoder_completed_tiles);
    output << ",\"dit_input_completed_tiles\":";
    append_bool_array(output, checkpoint.dit_input_completed_tiles);
    output << ",\"dit_completed_block_per_tile\":";
    append_array(output, checkpoint.dit_completed_block_per_tile);
    output << ",\"scheduler_completed_tiles\":";
    append_bool_array(output, checkpoint.scheduler_completed_tiles);
    output << ",\"decoder_completed_stage_per_tile\":";
    append_array(output, checkpoint.decoder_completed_stage_per_tile);
    output << ",\"decoder_completed_tiles\":";
    append_bool_array(output, checkpoint.decoder_completed_tiles);
    output << ",\"fusion_completed\":"
           << (checkpoint.fusion_completed ? "true" : "false") << '}';
    return output.str();
}

bool string_field(const std::string& json, const char* key, std::string* value)
{
    std::smatch match;
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*\\\"([^\\\"]*)\\\"");
    if (!std::regex_search(json, match, pattern)) return false;
    *value = match[1].str();
    return true;
}

bool int_field(const std::string& json, const char* key, int* value)
{
    std::smatch match;
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*(-?[0-9]+)");
    if (!std::regex_search(json, match, pattern)) return false;
    try { *value = std::stoi(match[1].str()); return true; }
    catch (...) { return false; }
}

bool bool_field(const std::string& json, const char* key, bool* value)
{
    std::smatch match;
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*(true|false)");
    if (!std::regex_search(json, match, pattern)) return false;
    *value = match[1].str() == "true";
    return true;
}

bool int_array(const std::string& json, const char* key, std::vector<int>* values)
{
    std::smatch match;
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (!std::regex_search(json, match, pattern)) return false;
    values->clear();
    std::stringstream input(match[1].str());
    std::string token;
    while (std::getline(input, token, ','))
    {
        if (token.empty()) continue;
        try { values->push_back(std::stoi(token)); }
        catch (...) { return false; }
    }
    return true;
}

bool bool_array(const std::string& json, const char* key, std::vector<bool>* values)
{
    std::smatch match;
    const std::regex pattern(std::string("\\\"") + key + "\\\"\\s*:\\s*\\[([^\\]]*)\\]");
    if (!std::regex_search(json, match, pattern)) return false;
    values->clear();
    std::stringstream input(match[1].str());
    std::string token;
    while (std::getline(input, token, ','))
    {
        token = std::regex_replace(token, std::regex("\\s"), "");
        if (token == "true") values->push_back(true);
        else if (token == "false") values->push_back(false);
        else if (!token.empty()) return false;
    }
    return true;
}

bool replace_file(const fs::path& temporary, const fs::path& target)
{
#ifdef _WIN32
    return MoveFileExW(temporary.wstring().c_str(), target.wstring().c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    std::error_code error;
    fs::rename(temporary, target, error);
    return !error;
#endif
}

bool expected_work(const StageMajorCheckpoint& checkpoint, StageMajorWork* work)
{
    const int tiles = checkpoint.identity.tile_count;
    for (int stage = 0; stage < 7; ++stage)
        for (int tile = 0; tile < tiles; ++tile)
            if (checkpoint.encoder_completed_stage_per_tile[tile] < stage)
            {
                *work = {"encoder", stage, -1, tile};
                return true;
            }
    for (int tile = 0; tile < tiles; ++tile)
        if (!checkpoint.dit_input_completed_tiles[tile])
        {
            *work = {"dit_input", -1, -1, tile};
            return true;
        }
    for (int block = 0; block < 32; ++block)
        for (int tile = 0; tile < tiles; ++tile)
            if (checkpoint.dit_completed_block_per_tile[tile] < block)
            {
                *work = {"dit", -1, block, tile};
                return true;
            }
    for (int tile = 0; tile < tiles; ++tile)
        if (!checkpoint.scheduler_completed_tiles[tile])
        {
            *work = {"scheduler", -1, -1, tile};
            return true;
        }
    for (int stage = 0; stage < 7; ++stage)
        for (int tile = 0; tile < tiles; ++tile)
            if (checkpoint.decoder_completed_stage_per_tile[tile] < stage)
            {
                *work = {"decoder", stage, -1, tile};
                return true;
            }
    if (!checkpoint.fusion_completed)
    {
        *work = {"fusion", -1, -1, -1};
        return true;
    }
    return false;
}

}  // namespace

StageMajorCheckpoint make_stage_major_checkpoint(const StageMajorIdentity& identity)
{
    StageMajorCheckpoint checkpoint;
    checkpoint.identity = identity;
    const size_t count = identity.tile_count > 0
        ? static_cast<size_t>(identity.tile_count) : 0u;
    checkpoint.encoder_completed_stage_per_tile.assign(count, -1);
    checkpoint.encoder_completed_tiles.assign(count, false);
    checkpoint.dit_input_completed_tiles.assign(count, false);
    checkpoint.dit_completed_block_per_tile.assign(count, -1);
    checkpoint.scheduler_completed_tiles.assign(count, false);
    checkpoint.decoder_completed_stage_per_tile.assign(count, -1);
    checkpoint.decoder_completed_tiles.assign(count, false);
    return checkpoint;
}

bool validate_stage_major_checkpoint(const StageMajorIdentity& expected,
                                     const StageMajorCheckpoint& actual,
                                     std::string* error)
{
    if (error) error->clear();
    if (!valid_structure(actual)) return fail(error, "CHECKPOINT_CORRUPT");
    if (expected.task_id != actual.identity.task_id)
        return fail(error, "CHECKPOINT_TASK_MISMATCH");
    if (expected.model_version != actual.identity.model_version)
        return fail(error, "CHECKPOINT_MODEL_MISMATCH");
    if (expected.source_sha256 != actual.identity.source_sha256 ||
        expected.normalized_image_sha256 != actual.identity.normalized_image_sha256)
        return fail(error, "CHECKPOINT_INPUT_MISMATCH");
    if (expected.tile_count != actual.identity.tile_count ||
        expected.tile_plan_sha256 != actual.identity.tile_plan_sha256)
        return fail(error, "CHECKPOINT_PLAN_MISMATCH");
    return true;
}

bool persist_stage_major_checkpoint(const std::string& task_root,
                                    const StageMajorCheckpoint& checkpoint,
                                    std::string* error)
{
    if (error) error->clear();
    if (!valid_structure(checkpoint)) return fail(error, "CHECKPOINT_CORRUPT");
    const fs::path root(task_root);
    std::error_code directory_error;
    fs::create_directories(root, directory_error);
    if (directory_error) return fail(error, "CHECKPOINT_DIRECTORY_FAILED");
    const fs::path target = root / "stage_major_checkpoint.json";
    const fs::path temporary = root / "stage_major_checkpoint.json.tmp";
    const std::string json = serialize(checkpoint);
    const std::string identity_text = checkpoint.identity.task_id + "\n" +
        checkpoint.identity.model_version + "\n" + checkpoint.identity.source_sha256 + "\n" +
        checkpoint.identity.normalized_image_sha256 + "\n" +
        std::to_string(checkpoint.identity.tile_count) + "\n" + checkpoint.identity.tile_plan_sha256;
    const std::string identity_sha = sha256_hex(
        reinterpret_cast<const uint8_t*>(identity_text.data()), identity_text.size());
    uint64_t generation = 0;
    if (!publish_checkpoint_generation(task_root, json, identity_sha,
            CheckpointCrashPoint::None, &generation, error))
        return false;
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        output.write(json.data(), static_cast<std::streamsize>(json.size()));
        output.flush();
        if (!output) return fail(error, "CHECKPOINT_WRITE_FAILED");
    }
    if (!replace_file(temporary, target))
    {
        std::error_code ignored;
        fs::remove(temporary, ignored);
        return fail(error, "CHECKPOINT_RENAME_FAILED");
    }
    return true;
}

bool load_stage_major_checkpoint(const std::string& task_root,
                                 StageMajorCheckpoint* checkpoint,
                                 std::string* error)
{
    if (error) error->clear();
    if (!checkpoint) return fail(error, "CHECKPOINT_OUTPUT_NULL");
    LoadedCheckpoint generation;
    std::string generation_error;
    std::string json;
    if (load_newest_valid_generation_any(task_root, &generation, &generation_error))
        json = generation.payload;
    else {
        std::ifstream input(fs::path(task_root) / "stage_major_checkpoint.json", std::ios::binary);
        std::ostringstream buffer; buffer << input.rdbuf(); json = buffer.str();
        if (!input && json.empty()) return fail(error, "CHECKPOINT_CORRUPT");
    }
    StageMajorCheckpoint loaded;
    if (json.empty() || json.front() != '{' || json.back() != '}' ||
        !int_field(json, "schema_version", &loaded.schema_version) ||
        !string_field(json, "task_id", &loaded.identity.task_id) ||
        !string_field(json, "model_version", &loaded.identity.model_version) ||
        !string_field(json, "source_sha256", &loaded.identity.source_sha256) ||
        !string_field(json, "normalized_image_sha256", &loaded.identity.normalized_image_sha256) ||
        !int_field(json, "tile_count", &loaded.identity.tile_count) ||
        !string_field(json, "tile_plan_sha256", &loaded.identity.tile_plan_sha256) ||
        !string_field(json, "state", &loaded.state) ||
        !string_field(json, "current_stage", &loaded.current_stage) ||
        !int_field(json, "current_stage_index", &loaded.current_stage_index) ||
        !int_field(json, "current_block", &loaded.current_block) ||
        !int_field(json, "completed_tiles", &loaded.completed_tiles) ||
        !int_array(json, "encoder_completed_stage_per_tile", &loaded.encoder_completed_stage_per_tile) ||
        !bool_array(json, "encoder_completed_tiles", &loaded.encoder_completed_tiles) ||
        !bool_array(json, "dit_input_completed_tiles", &loaded.dit_input_completed_tiles) ||
        !int_array(json, "dit_completed_block_per_tile", &loaded.dit_completed_block_per_tile) ||
        !bool_array(json, "scheduler_completed_tiles", &loaded.scheduler_completed_tiles) ||
        !int_array(json, "decoder_completed_stage_per_tile", &loaded.decoder_completed_stage_per_tile) ||
        !bool_array(json, "decoder_completed_tiles", &loaded.decoder_completed_tiles) ||
        !bool_field(json, "fusion_completed", &loaded.fusion_completed) ||
        !valid_structure(loaded))
        return fail(error, "CHECKPOINT_CORRUPT");
    *checkpoint = std::move(loaded);
    return true;
}

bool next_stage_major_work(const StageMajorCheckpoint& checkpoint,
                           StageMajorWork* work,
                           std::string* error)
{
    if (error) error->clear();
    if (!work || !valid_structure(checkpoint)) return fail(error, "CHECKPOINT_CORRUPT");
    if (checkpoint.state == "paused") return fail(error, "TASK_PAUSED");
    if (checkpoint.state == "cancelled") return fail(error, "TASK_CANCELLED");
    if (checkpoint.state == "completed") return false;
    if (checkpoint.state != "running") return fail(error, "INVALID_TASK_STATE");
    return expected_work(checkpoint, work);
}

bool mark_stage_major_work_complete(StageMajorCheckpoint* checkpoint,
                                    const StageMajorWork& work,
                                    std::string* error)
{
    if (error) error->clear();
    if (!checkpoint || !valid_structure(*checkpoint) || checkpoint->state != "running")
        return fail(error, "INVALID_TASK_STATE");
    StageMajorWork expected;
    if (!expected_work(*checkpoint, &expected) || expected.stage != work.stage ||
        expected.stage_index != work.stage_index || expected.block_index != work.block_index ||
        expected.tile_index != work.tile_index)
        return fail(error, "STALE_WORK_ITEM");
    checkpoint->current_stage = work.stage;
    checkpoint->current_stage_index = work.stage_index;
    checkpoint->current_block = work.block_index;
    if (work.stage == "encoder")
    {
        checkpoint->encoder_completed_stage_per_tile[work.tile_index] = work.stage_index;
        checkpoint->encoder_completed_tiles[work.tile_index] = work.stage_index == 6;
    }
    else if (work.stage == "dit_input")
        checkpoint->dit_input_completed_tiles[work.tile_index] = true;
    else if (work.stage == "dit")
        checkpoint->dit_completed_block_per_tile[work.tile_index] = work.block_index;
    else if (work.stage == "scheduler")
        checkpoint->scheduler_completed_tiles[work.tile_index] = true;
    else if (work.stage == "decoder")
    {
        checkpoint->decoder_completed_stage_per_tile[work.tile_index] = work.stage_index;
        if (work.stage_index == 6)
        {
            checkpoint->decoder_completed_tiles[work.tile_index] = true;
            ++checkpoint->completed_tiles;
        }
    }
    else if (work.stage == "fusion")
    {
        checkpoint->fusion_completed = true;
        checkpoint->state = "completed";
        checkpoint->current_stage = "completed";
    }
    else return fail(error, "INVALID_WORK_ITEM");
    return true;
}

bool set_stage_major_control(StageMajorCheckpoint* checkpoint,
                             const std::string& state,
                             std::string* error)
{
    if (error) error->clear();
    if (!checkpoint || !valid_structure(*checkpoint)) return fail(error, "CHECKPOINT_CORRUPT");
    if (checkpoint->state == "completed" || checkpoint->state == "cancelled")
        return fail(error, "TERMINAL_TASK_STATE");
    const bool allowed = (checkpoint->state == "running" && (state == "paused" || state == "cancelled")) ||
                         (checkpoint->state == "paused" && (state == "running" || state == "cancelled"));
    if (!allowed) return fail(error, "INVALID_CONTROL_TRANSITION");
    checkpoint->state = state;
    return true;
}

bool run_stage_major_schedule(StageMajorCheckpoint* checkpoint,
                              const StageMajorExecutionHooks& hooks,
                              StageMajorRunStats* stats,
                              std::string* error)
{
    if (error) error->clear();
    if (!checkpoint || !stats || !valid_structure(*checkpoint))
        return fail(error, "CHECKPOINT_CORRUPT");
    if (!hooks.begin_stage || !hooks.execute_work || !hooks.end_stage ||
        !hooks.persist_checkpoint)
        return fail(error, "STAGE_MAJOR_HOOKS_REQUIRED");
    *stats = StageMajorRunStats{};
    StageMajorWork work;
    std::string next_error;
    while (next_stage_major_work(*checkpoint, &work, &next_error))
    {
        const StageMajorWork stage_key = work;
        if (!hooks.begin_stage(hooks.context, stage_key, error)) return false;
        const bool model_stage = stage_key.stage != "fusion";
        if (model_stage)
        {
            ++stats->model_stage_loads;
            stats->peak_resident_model_stages = 1;
        }
        bool stage_ok = true;
        while (stage_ok)
        {
            if (!hooks.execute_work(hooks.context, work, error) ||
                !mark_stage_major_work_complete(checkpoint, work, error) ||
                !hooks.persist_checkpoint(hooks.context, *checkpoint, error))
            {
                stage_ok = false;
                break;
            }
            ++stats->work_items_completed;
            ++stats->checkpoint_persists;
            if (hooks.checkpoint_committed)
                hooks.checkpoint_committed(hooks.context, work);
            StageMajorWork next;
            next_error.clear();
            if (!next_stage_major_work(*checkpoint, &next, &next_error)) break;
            if (next.stage != stage_key.stage ||
                next.stage_index != stage_key.stage_index ||
                next.block_index != stage_key.block_index)
                break;
            work = next;
        }
        hooks.end_stage(hooks.context, stage_key);
        if (model_stage) ++stats->model_stage_unloads;
        if (!stage_ok) return false;
        if (!next_error.empty() && checkpoint->state != "completed")
            return fail(error, next_error);
    }
    if (!next_error.empty() && checkpoint->state != "completed")
        return fail(error, next_error);
    return checkpoint->state == "completed";
}

}  // namespace seededge::phase18d
