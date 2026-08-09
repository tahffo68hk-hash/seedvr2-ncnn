#include "model_manifest.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace
{
constexpr const char* kCacheName = ".seededge_manifest_verified_v1";
constexpr const char* kCacheHeader = "seededge_verified_manifest_v1";

std::string fail(const char* error)
{
    return std::string("{\"valid\":false,\"error\":\"") + error +
           "\",\"silent_fallback\":false}";
}

uint32_t rotate_right(uint32_t value, unsigned bits)
{
    return (value >> bits) | (value << (32 - bits));
}

class Sha256
{
public:
    void update(const uint8_t* data, size_t size)
    {
        total_bytes_ += size;
        while (size > 0)
        {
            const size_t copied = (std::min)(size, block_.size() - block_size_);
            std::memcpy(block_.data() + block_size_, data, copied);
            block_size_ += copied;
            data += copied;
            size -= copied;
            if (block_size_ == block_.size())
            {
                compress(block_.data());
                block_size_ = 0;
            }
        }
    }

    std::string finish()
    {
        const uint64_t bit_count = total_bytes_ * 8u;
        block_[block_size_++] = 0x80;
        if (block_size_ > 56)
        {
            std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_), block_.end(), 0);
            compress(block_.data());
            block_size_ = 0;
        }
        std::fill(block_.begin() + static_cast<std::ptrdiff_t>(block_size_),
                  block_.begin() + 56, 0);
        for (int index = 0; index < 8; ++index)
            block_[56 + index] = static_cast<uint8_t>(bit_count >> ((7 - index) * 8));
        compress(block_.data());

        std::ostringstream output;
        output << std::hex << std::setfill('0');
        for (uint32_t value : state_) output << std::setw(8) << value;
        return output.str();
    }

private:
    void compress(const uint8_t* bytes)
    {
        static const uint32_t constants[64] = {
            0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
            0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
            0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
            0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
            0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
            0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
            0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
            0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
        uint32_t words[64] = {};
        for (int index = 0; index < 16; ++index)
            words[index] = (uint32_t(bytes[index * 4]) << 24) |
                           (uint32_t(bytes[index * 4 + 1]) << 16) |
                           (uint32_t(bytes[index * 4 + 2]) << 8) |
                           uint32_t(bytes[index * 4 + 3]);
        for (int index = 16; index < 64; ++index)
        {
            const uint32_t s0 = rotate_right(words[index - 15], 7) ^
                                rotate_right(words[index - 15], 18) ^
                                (words[index - 15] >> 3);
            const uint32_t s1 = rotate_right(words[index - 2], 17) ^
                                rotate_right(words[index - 2], 19) ^
                                (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        uint32_t a=state_[0],b=state_[1],c=state_[2],d=state_[3];
        uint32_t e=state_[4],f=state_[5],g=state_[6],h=state_[7];
        for (int index = 0; index < 64; ++index)
        {
            const uint32_t s1 = rotate_right(e,6) ^ rotate_right(e,11) ^ rotate_right(e,25);
            const uint32_t choose = (e & f) ^ ((~e) & g);
            const uint32_t first = h + s1 + choose + constants[index] + words[index];
            const uint32_t s0 = rotate_right(a,2) ^ rotate_right(a,13) ^ rotate_right(a,22);
            const uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
            const uint32_t second = s0 + majority;
            h=g; g=f; f=e; e=d+first; d=c; c=b; b=a; a=first+second;
        }
        state_[0]+=a; state_[1]+=b; state_[2]+=c; state_[3]+=d;
        state_[4]+=e; state_[5]+=f; state_[6]+=g; state_[7]+=h;
    }

    std::array<uint32_t, 8> state_ = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    std::array<uint8_t, 64> block_{};
    size_t block_size_ = 0;
    uint64_t total_bytes_ = 0;
};

std::string sha256_bytes(const void* data, size_t size)
{
    Sha256 sha;
    sha.update(static_cast<const uint8_t*>(data), size);
    return sha.finish();
}

std::string sha256_file(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    // Keep the streaming buffer comfortably below the Android and Windows
    // thread-stack limits; unlike the previous implementation this never
    // allocates the complete (potentially multi-GB) model file.
    std::array<uint8_t, 64 * 1024> buffer{};
    Sha256 sha;
    while (input)
    {
        input.read(reinterpret_cast<char*>(buffer.data()),
                   static_cast<std::streamsize>(buffer.size()));
        const std::streamsize count = input.gcount();
        if (count > 0) sha.update(buffer.data(), static_cast<size_t>(count));
    }
    if (!input.eof()) return {};
    return sha.finish();
}

bool is_safe_relative_path(const std::string& relative)
{
    if (relative.empty() || relative.find(':') != std::string::npos) return false;
    const fs::path path(relative);
    if (path.is_absolute() || path.has_root_name() || path.has_root_directory()) return false;
    for (const auto& component : path)
        if (component == "..") return false;
    return true;
}

long long modified_ticks(const fs::path& path, std::error_code& error)
{
    const auto value = fs::last_write_time(path, error);
    return error ? 0LL : static_cast<long long>(value.time_since_epoch().count());
}

struct ManifestEntry
{
    std::string relative;
    fs::path target;
    uintmax_t size = 0;
    std::string expected_hash;
    long long mtime_ticks = 0;
};

std::string normalized_root_identity(const fs::path& base)
{
    std::error_code error;
    fs::path canonical = fs::weakly_canonical(base, error);
    if (error) canonical = fs::absolute(base, error).lexically_normal();
    const std::string normalized = canonical.generic_string();
    return sha256_bytes(normalized.data(), normalized.size());
}

bool cache_matches(const fs::path& cache_path, const std::string& root_identity,
                   const std::string& manifest_hash,
                   const std::vector<ManifestEntry>& entries)
{
    std::ifstream input(cache_path, std::ios::binary);
    if (!input) return false;
    std::string line;
    if (!std::getline(input, line) || line != kCacheHeader) return false;
    if (!std::getline(input, line) || line != "root_sha256=" + root_identity) return false;
    if (!std::getline(input, line) || line != "manifest_sha256=" + manifest_hash) return false;
    if (!std::getline(input, line) || line != "file_count=" + std::to_string(entries.size())) return false;
    for (const ManifestEntry& entry : entries)
    {
        if (!std::getline(input, line)) return false;
        const std::string expected = sha256_bytes(entry.relative.data(), entry.relative.size()) + "\t" +
                                     std::to_string(entry.size) + "\t" +
                                     std::to_string(entry.mtime_ticks);
        if (line != expected) return false;
    }
    return !std::getline(input, line);
}

bool replace_cache_file(const fs::path& temporary, const fs::path& target)
{
#ifdef _WIN32
    return MoveFileExW(temporary.c_str(), target.c_str(),
                       MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
    return std::rename(temporary.c_str(), target.c_str()) == 0;
#endif
}

bool persist_cache(const fs::path& cache_path, const std::string& root_identity,
                   const std::string& manifest_hash,
                   const std::vector<ManifestEntry>& entries)
{
    const fs::path temporary = cache_path.string() + ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) return false;
        output << kCacheHeader << '\n'
               << "root_sha256=" << root_identity << '\n'
               << "manifest_sha256=" << manifest_hash << '\n'
               << "file_count=" << entries.size() << '\n';
        for (const ManifestEntry& entry : entries)
            output << sha256_bytes(entry.relative.data(), entry.relative.size()) << '\t'
                   << entry.size << '\t' << entry.mtime_ticks << '\n';
        output.flush();
        if (!output) return false;
    }
    if (replace_cache_file(temporary, cache_path)) return true;
    std::error_code ignored;
    fs::remove(temporary, ignored);
    return false;
}
}

std::string runtime_sha256_file(const std::string& path)
{
    return sha256_file(fs::path(path));
}

std::string validate_model_manifest(const std::string& root)
{
    const auto started = std::chrono::steady_clock::now();
    const fs::path base(root);
    const fs::path manifest = base / "manifest.json";
    if (!fs::exists(manifest)) return fail("manifest missing");
    std::ifstream file(manifest, std::ios::binary);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string json = buffer.str();
    if (json.empty() || json.front() != '{' || json.back() != '}') return fail("invalid JSON");
    if (!std::regex_search(json, std::regex(R"("schema_version"\s*:\s*1(?:\s*[,}]))")))
        return fail("schema_version must be 1");
    if (!std::regex_search(json, std::regex(R"("model_set"\s*:\s*"phase18c_minimal_video_demo")")))
        return fail("model_set mismatch");

    const std::regex pattern(R"xx("relative_path"\s*:\s*"([^"]+)"\s*,\s*"size_bytes"\s*:\s*([0-9]+)\s*,\s*"sha256"\s*:\s*"([^"]+)")xx");
    const std::sregex_iterator begin(json.begin(), json.end(), pattern), end;
    if (begin == end) return fail("files must not be empty");

    std::vector<ManifestEntry> entries;
    uintmax_t total_bytes = 0;
    for (auto iterator = begin; iterator != end; ++iterator)
    {
        ManifestEntry entry;
        entry.relative = (*iterator)[1];
        entry.size = std::stoull((*iterator)[2]);
        entry.expected_hash = (*iterator)[3];
        if (!is_safe_relative_path(entry.relative))
            return fail(entry.relative.find("..") != std::string::npos ? "path traversal" : "absolute path");
        if (!std::regex_match(entry.expected_hash, std::regex("[0-9a-f]{64}")))
            return fail("sha256 must be 64 lowercase hex");
        entry.target = base / fs::path(entry.relative);
        if (!fs::exists(entry.target)) return fail("model file missing");
        for (fs::path current = entry.target; current != base && !current.empty(); current = current.parent_path())
            if (fs::is_symlink(fs::symlink_status(current))) return fail("symbolic link rejected");
        if (!fs::is_regular_file(entry.target)) return fail("model path is not a regular file");
        if (fs::file_size(entry.target) != entry.size) return fail("size mismatch");
        std::error_code time_error;
        entry.mtime_ticks = modified_ticks(entry.target, time_error);
        if (time_error) return fail("model mtime unavailable");
        total_bytes += entry.size;
        entries.push_back(std::move(entry));
    }

    const std::string manifest_hash = sha256_bytes(json.data(), json.size());
    const std::string root_identity = normalized_root_identity(base);
    const fs::path cache_path = base / kCacheName;
    const bool cache_hit = cache_matches(cache_path, root_identity, manifest_hash, entries);
    if (!cache_hit)
    {
        for (const ManifestEntry& entry : entries)
            if (sha256_file(entry.target) != entry.expected_hash) return fail("sha256 mismatch");
        persist_cache(cache_path, root_identity, manifest_hash, entries);
    }

    const double validation_ms = std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
    std::ostringstream result;
    result << "{\"valid\":true,\"schema_version\":1,"
           << "\"model_set\":\"phase18c_minimal_video_demo\","
           << "\"files_verified\":" << entries.size() << ','
           << "\"model_bytes_declared\":" << total_bytes << ','
           << "\"manifest_sha256\":\"" << manifest_hash << "\","
           << "\"validation_mode\":\"" << (cache_hit ? "identity_cache" : "full_sha256") << "\","
           << "\"cache_hit\":" << (cache_hit ? "true" : "false") << ','
           << "\"full_sha256_files\":" << (cache_hit ? 0 : entries.size()) << ','
           << "\"full_sha256_bytes\":" << (cache_hit ? 0 : total_bytes) << ','
           << "\"validation_ms\":" << std::fixed << std::setprecision(3) << validation_ms << ','
           << "\"silent_fallback\":false}";
    return result.str();
}
