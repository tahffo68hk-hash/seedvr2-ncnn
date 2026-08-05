#include "model_manifest.h"

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{
std::string fail(const char* error)
{
    return std::string("{\"valid\":false,\"error\":\"") + error +
           "\",\"silent_fallback\":false}";
}

uint32_t rotate_right(uint32_t value, unsigned bits)
{
    return (value >> bits) | (value << (32 - bits));
}

std::string sha256_file(const fs::path& path)
{
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
    std::ifstream input(path, std::ios::binary);
    if (!input) return {};
    input.seekg(0, std::ios::end);
    const std::streamoff file_size = input.tellg();
    if (file_size < 0) return {};
    input.seekg(0, std::ios::beg);
    std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
    if (file_size > 0 &&
        !input.read(reinterpret_cast<char*>(bytes.data()), file_size))
        return {};
    const uint64_t bit_count = static_cast<uint64_t>(bytes.size()) * 8;
    bytes.push_back(0x80);
    while ((bytes.size() % 64) != 56) bytes.push_back(0);
    for (int i = 7; i >= 0; --i) bytes.push_back(static_cast<uint8_t>(bit_count >> (i * 8)));
    std::array<uint32_t, 8> h = {0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
                                  0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    for (size_t offset = 0; offset < bytes.size(); offset += 64)
    {
        uint32_t w[64] = {};
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(bytes[offset+i*4]) << 24) | (uint32_t(bytes[offset+i*4+1]) << 16) |
                   (uint32_t(bytes[offset+i*4+2]) << 8) | uint32_t(bytes[offset+i*4+3]);
        for (int i = 16; i < 64; ++i)
        {
            const uint32_t s0 = rotate_right(w[i-15],7) ^ rotate_right(w[i-15],18) ^ (w[i-15] >> 3);
            const uint32_t s1 = rotate_right(w[i-2],17) ^ rotate_right(w[i-2],19) ^ (w[i-2] >> 10);
            w[i] = w[i-16] + s0 + w[i-7] + s1;
        }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
        for (int i = 0; i < 64; ++i)
        {
            const uint32_t s1=rotate_right(e,6)^rotate_right(e,11)^rotate_right(e,25);
            const uint32_t ch=(e&f)^((~e)&g);
            const uint32_t t1=hh+s1+ch+k[i]+w[i];
            const uint32_t s0=rotate_right(a,2)^rotate_right(a,13)^rotate_right(a,22);
            const uint32_t maj=(a&b)^(a&c)^(b&c);
            const uint32_t t2=s0+maj;
            hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
        }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
    }
    std::ostringstream output;
    output << std::hex << std::setfill('0');
    for (uint32_t value : h) output << std::setw(8) << value;
    return output.str();
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
}

std::string validate_model_manifest(const std::string& root)
{
    const fs::path base(root);
    const fs::path manifest = base / "manifest.json";
    if (!fs::exists(manifest)) return fail("manifest missing");
    std::ifstream file(manifest);
    std::stringstream buffer;
    buffer << file.rdbuf();
    const std::string json = buffer.str();
    if (json.empty() || json.front() != '{' || json.back() != '}') return fail("invalid JSON");
    if (!std::regex_search(json, std::regex(R"("schema_version"\s*:\s*1(?:\s*[,}]))")))
        return fail("schema_version must be 1");
    if (!std::regex_search(json, std::regex(R"("model_set"\s*:\s*"phase18c_minimal_video_demo")")))
        return fail("model_set mismatch");

    const std::regex entry(R"xx("relative_path"\s*:\s*"([^"]+)"\s*,\s*"size_bytes"\s*:\s*([0-9]+)\s*,\s*"sha256"\s*:\s*"([^"]+)")xx");
    const std::sregex_iterator begin(json.begin(), json.end(), entry), end;
    if (begin == end) return fail("files must not be empty");
    size_t verified = 0;
    for (auto iterator = begin; iterator != end; ++iterator)
    {
        const std::string relative = (*iterator)[1];
        const uintmax_t expected_size = std::stoull((*iterator)[2]);
        const std::string expected_hash = (*iterator)[3];
        if (!is_safe_relative_path(relative))
            return fail(relative.find("..") != std::string::npos ? "path traversal" : "absolute path");
        if (!std::regex_match(expected_hash, std::regex("[0-9a-f]{64}")))
            return fail("sha256 must be 64 lowercase hex");
        const fs::path target = base / fs::path(relative);
        if (!fs::exists(target)) return fail("model file missing");
        for (fs::path current = target; current != base && !current.empty(); current = current.parent_path())
            if (fs::is_symlink(fs::symlink_status(current))) return fail("symbolic link rejected");
        if (!fs::is_regular_file(target)) return fail("model path is not a regular file");
        if (fs::file_size(target) != expected_size) return fail("size mismatch");
        if (sha256_file(target) != expected_hash) return fail("sha256 mismatch");
        ++verified;
    }
    return std::string("{\"valid\":true,\"schema_version\":1,\"model_set\":\"phase18c_minimal_video_demo\",\"files_verified\":") +
           std::to_string(verified) + ",\"silent_fallback\":false}";
}
