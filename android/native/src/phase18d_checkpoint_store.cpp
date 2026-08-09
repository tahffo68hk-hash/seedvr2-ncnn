#include "phase18d_checkpoint_store.h"

#include "phase18d_tile_core.h"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

namespace fs = std::filesystem;
namespace seededge::phase18d {
namespace {
constexpr const char* kHeader = "seededge_checkpoint_generation_v1";

bool fail(std::string* error, const char* code)
{
    if (error) *error = code;
    return false;
}

bool valid_sha(const std::string& value)
{
    return value.size() == 64 && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

std::string generation_name(uint64_t generation)
{
    std::ostringstream out;
    out << "generation_" << std::setw(10) << std::setfill('0') << generation << ".json";
    return out.str();
}

bool parse_generation_name(const fs::path& path, uint64_t* generation)
{
    const std::string name = path.filename().string();
    if (name.size() != 26 || name.rfind("generation_", 0) != 0 ||
        name.substr(21) != ".json") return false;
    try { *generation = std::stoull(name.substr(11, 10)); return true; }
    catch (...) { return false; }
}

std::string envelope(uint64_t generation, const std::string& identity,
                     const std::string& payload)
{
    const std::string digest = sha256_hex(
        reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
    std::ostringstream out;
    out << kHeader << '\n' << generation << '\n' << identity << '\n' << digest << '\n'
        << payload.size() << '\n' << payload;
    return out.str();
}

bool durable_write(const fs::path& path, const std::string& bytes, size_t limit)
{
    FILE* file = nullptr;
#ifdef _WIN32
    if (fopen_s(&file, path.string().c_str(), "wb") != 0) return false;
#else
    file = std::fopen(path.string().c_str(), "wb");
    if (!file) return false;
#endif
    const size_t count = std::min(limit, bytes.size());
    const bool wrote = std::fwrite(bytes.data(), 1, count, file) == count &&
                       std::fflush(file) == 0;
#ifdef _WIN32
    const bool synced = wrote && _commit(_fileno(file)) == 0;
#else
    const bool synced = wrote && fsync(fileno(file)) == 0;
#endif
    const bool closed = std::fclose(file) == 0;
    return synced && closed;
}

bool parse_file(const fs::path& path, LoadedCheckpoint* loaded)
{
    std::ifstream input(path, std::ios::binary);
    std::string header, generation, identity, digest, size;
    if (!std::getline(input, header) || !std::getline(input, generation) ||
        !std::getline(input, identity) || !std::getline(input, digest) ||
        !std::getline(input, size) || header != kHeader || !valid_sha(identity) ||
        !valid_sha(digest)) return false;
    std::ostringstream payload_stream; payload_stream << input.rdbuf();
    const std::string payload = payload_stream.str();
    uint64_t parsed_generation = 0, declared_size = 0;
    try { parsed_generation = std::stoull(generation); declared_size = std::stoull(size); }
    catch (...) { return false; }
    const std::string actual = sha256_hex(
        reinterpret_cast<const uint8_t*>(payload.data()), payload.size());
    if (declared_size != payload.size() || actual != digest || payload.empty() ||
        payload.front() != '{' || payload.back() != '}') return false;
    *loaded = {parsed_generation, identity, digest, payload};
    return true;
}

std::vector<LoadedCheckpoint> scan_valid(const fs::path& directory)
{
    std::vector<LoadedCheckpoint> result;
    std::error_code error;
    if (!fs::is_directory(directory, error)) return result;
    for (const auto& entry : fs::directory_iterator(directory, error)) {
        uint64_t named = 0; LoadedCheckpoint loaded;
        if (entry.is_regular_file() && parse_generation_name(entry.path(), &named) &&
            parse_file(entry.path(), &loaded) && named == loaded.generation)
            result.push_back(std::move(loaded));
    }
    std::sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
        return a.generation < b.generation;
    });
    return result;
}

bool replace_file(const fs::path& source, const fs::path& target)
{
    std::error_code error;
    fs::rename(source, target, error);
    if (!error) return true;
    fs::remove(target, error); error.clear(); fs::rename(source, target, error);
    return !error;
}
}

bool publish_checkpoint_generation(const std::string& task_root,
                                   const std::string& payload,
                                   const std::string& identity_sha256,
                                   CheckpointCrashPoint crash_point,
                                   uint64_t* published_generation,
                                   std::string* error)
{
    if (error) error->clear();
    if (published_generation) *published_generation = 0;
    if (!valid_sha(identity_sha256) || payload.empty() || payload.front() != '{' ||
        payload.back() != '}') return fail(error, "CHECKPOINT_GENERATION_INPUT_INVALID");
    const fs::path directory = fs::path(task_root) / "checkpoints";
    std::error_code io_error; fs::create_directories(directory, io_error);
    if (io_error) return fail(error, "CHECKPOINT_DIRECTORY_FAILED");
    const auto existing = scan_valid(directory);
    const uint64_t generation = existing.empty() ? 1 : existing.back().generation + 1;
    if (crash_point == CheckpointCrashPoint::BeforeWrite)
        return fail(error, "INJECTED_BEFORE_WRITE");
    const fs::path temporary = directory / (generation_name(generation) + ".tmp");
    const fs::path target = directory / generation_name(generation);
    const std::string bytes = envelope(generation, identity_sha256, payload);
    const size_t write_count = crash_point == CheckpointCrashPoint::HalfWrite
        ? bytes.size() / 2 : bytes.size();
    if (!durable_write(temporary, bytes, write_count)) return fail(error, "CHECKPOINT_WRITE_FAILED");
    if (crash_point == CheckpointCrashPoint::HalfWrite)
        return fail(error, "INJECTED_HALF_WRITE");
    LoadedCheckpoint verified;
    if (!parse_file(temporary, &verified)) return fail(error, "CHECKPOINT_VALIDATION_FAILED");
    if (crash_point == CheckpointCrashPoint::BeforeRename)
        return fail(error, "INJECTED_BEFORE_RENAME");
    if (!replace_file(temporary, target)) return fail(error, "CHECKPOINT_RENAME_FAILED");
    if (published_generation) *published_generation = generation;
    if (crash_point == CheckpointCrashPoint::BeforeControlUpdate)
        return fail(error, "INJECTED_BEFORE_CONTROL_UPDATE");
    const fs::path control_tmp = directory / "current.tmp";
    const fs::path control = directory / "current";
    const std::string control_text = std::to_string(generation) + "\n";
    if (!durable_write(control_tmp, control_text, control_text.size()) ||
        !replace_file(control_tmp, control)) return fail(error, "CHECKPOINT_CONTROL_WRITE_FAILED");
    if (crash_point == CheckpointCrashPoint::BeforeGc)
        return fail(error, "INJECTED_BEFORE_GC");
    const auto valid = scan_valid(directory);
    for (size_t i = 0; i + 2 < valid.size(); ++i)
        fs::remove(directory / generation_name(valid[i].generation), io_error);
    for (const auto& entry : fs::directory_iterator(directory, io_error))
        if (entry.path().extension() == ".tmp") fs::remove(entry.path(), io_error);
    if (crash_point == CheckpointCrashPoint::AfterGc)
        return fail(error, "INJECTED_AFTER_GC");
    return true;
}

bool load_newest_valid_generation(const std::string& task_root,
                                  const std::string& identity_sha256,
                                  LoadedCheckpoint* loaded,
                                  std::string* error)
{
    if (error) error->clear();
    if (!loaded || !valid_sha(identity_sha256)) return fail(error, "CHECKPOINT_LOAD_INPUT_INVALID");
    const auto valid = scan_valid(fs::path(task_root) / "checkpoints");
    bool other_identity = false;
    for (auto it = valid.rbegin(); it != valid.rend(); ++it) {
        if (it->identity_sha256 == identity_sha256) { *loaded = *it; return true; }
        other_identity = true;
    }
    return fail(error, other_identity ? "CHECKPOINT_IDENTITY_MISMATCH" : "CHECKPOINT_GENERATION_NOT_FOUND");
}

bool load_newest_valid_generation_any(const std::string& task_root,
                                      LoadedCheckpoint* loaded,
                                      std::string* error)
{
    if (error) error->clear();
    if (!loaded) return fail(error, "CHECKPOINT_LOAD_OUTPUT_NULL");
    const auto valid = scan_valid(fs::path(task_root) / "checkpoints");
    if (valid.empty()) return fail(error, "CHECKPOINT_GENERATION_NOT_FOUND");
    *loaded = valid.back();
    return true;
}

bool inspect_checkpoint_generations(const std::string& task_root,
                                    CheckpointInventory* inventory,
                                    std::string* error)
{
    if (error) error->clear();
    if (!inventory) return fail(error, "CHECKPOINT_INVENTORY_OUTPUT_NULL");
    *inventory = CheckpointInventory{};
    const fs::path directory = fs::path(task_root) / "checkpoints";
    const auto valid = scan_valid(directory);
    inventory->valid_generations = static_cast<int>(valid.size());
    if (!valid.empty()) inventory->newest_generation = valid.back().generation;
    std::error_code io_error;
    if (fs::is_directory(directory, io_error))
        for (const auto& entry : fs::directory_iterator(directory, io_error))
            if (entry.is_regular_file()) {
                inventory->total_bytes += entry.file_size(io_error);
                if (entry.path().extension() == ".tmp") ++inventory->temporary_files;
            }
    return true;
}

}  // namespace seededge::phase18d
