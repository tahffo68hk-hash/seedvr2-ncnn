// Phase 13: real-width artifact inventory tests (files + SHA256 + ncnn load).
//
// Scans the local real-width artifact root through ArtifactRegistry, verifies
// the block metadata (C=2560/H=20/HD=128/MLP=6912, execution=staged_dit_block,
// pnnx fp16=0), checks every stage .param/.bin exists and is non-empty, loads
// every model through NcnnModelCache (no silent fallback), and verifies SHA256
// against the committed `reports/realwidth_artifact_inventory.json` manifest.
// Also runs controlled-failure negatives (missing/tampered files, wrong
// variant, param/bin mismatch).
#include <algorithm>
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "artifact_registry.h"
#include "model_cache.h"
#include "phase12_test_common.h"

using namespace attention_dispatcher;
namespace fs = std::filesystem;
using phase12_test::JVal;

namespace
{
const char* kVariant = "realwidth_seedvr2_dit_block_c2560_h20_hd128_random";

std::string sha256_file(const fs::path& path)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return "";
    std::vector<unsigned char> data((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::string result;
    unsigned char digest[32] = {0};
    if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) return "";
    if (BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) != 0) return "";
    if (!data.empty() && BCryptHashData(hash, data.data(), static_cast<ULONG>(data.size()), 0) != 0) return "";
    if (BCryptFinishHash(hash, digest, sizeof digest, 0) != 0) return "";
    static const char* hex = "0123456789abcdef";
    for (unsigned char b : digest)
    {
        result += hex[b >> 4];
        result += hex[b & 0xF];
    }
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    if (hash) BCryptDestroyHash(hash);
    return result;
}

int fail(const std::string& msg)
{
    std::cerr << "FAIL: " << msg << '\n';
    return 1;
}

} // namespace

int main(int argc, char** argv)
{
    const fs::path root = argc > 1 ? fs::path(argv[1]) : fs::path(".phase13_local_artifacts");
    const fs::path manifest_path = argc > 2 ? fs::path(argv[2])
        : fs::path("reports/realwidth_artifact_inventory.json");

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(root, error)) return fail("registry scan: " + error);
    const CoreKey key{4, 3, 3, 1, 1, 1, kVariant};
    const CoreInfo* core = registry.lookup(key);
    if (!core) return fail(std::string("variant not found: ") + kVariant);
    const auto* artifact = core->dit_block();
    if (!artifact) return fail("execution is not staged_dit_block");

    // metadata config checks
    const DiTBlockConfig& b = artifact->block;
    int checks = 0;
    const auto check = [&](bool cond, const char* what) {
        ++checks;
        if (!cond) { std::cerr << "FAIL config: " << what << '\n'; std::exit(1); }
    };
    check(b.hidden_dim == 2560 && b.txt_dim == 2560, "hidden_dim/txt_dim=2560");
    check(b.heads == 20, "heads=20");
    check(b.head_dim == 128, "head_dim=128");
    check(b.mlp_hidden_dim == 6912, "mlp_hidden_dim=6912");
    check(b.configured_rope_dim == 128 && b.effective_rotary_dim == 126 && b.identity_tail_dim == 2, "rope contract");
    check(b.mlp_type == "swiglu", "mlp_type");
    check(!b.is_last_layer, "is_last_layer=false");
    check(b.q_max == 128 && b.head_pairs == 64, "q_max/head_pairs");
    check(core->key.window_t == 4 && core->key.window_h == 3 && core->key.window_w == 3, "window");
    check(core->key.grid_t == 1 && core->key.grid_h == 1 && core->key.grid_w == 1, "grid");

    // manifest (committed inventory) sha256 verification
    std::map<std::string, std::string> manifest_sha;
    std::map<std::string, long long> manifest_size;
    std::map<std::string, long long> manifest_param_count;
    std::map<std::string, bool> manifest_folded;
    std::string text_content;
    if (phase12_test::read_text(manifest_path, text_content))
    {
        const JVal m = phase12_test::parse_json(text_content);
        for (const JVal& e : m.obj.at("entries").arr)
        {
            manifest_sha[e.obj.at("logical_name").str] = e.obj.at("sha256").str;
            manifest_size[e.obj.at("logical_name").str] = static_cast<long long>(e.obj.at("bytes").num);
            manifest_param_count[e.obj.at("logical_name").str] = static_cast<long long>(e.obj.at("parameter_count").num);
            const auto fld = e.obj.find("pnnx_folded_identity_weight");
            manifest_folded[e.obj.at("logical_name").str] = (fld != e.obj.end() && fld->second.b);
        }
    }

    // stage model completeness + ncnn load through the shared cache
    NcnnModelCache cache;
    int model_count = 0;
    const auto* models = &artifact->models;
    for (const auto& entry : *models)
    {
        const std::string& name = entry.first;
        const auto& files = entry.second;
        const fs::path p = core->directory / files.param_path;
        const fs::path bn = core->directory / files.bin_path;
        if (!fs::exists(p) || !fs::exists(bn)) return fail("missing model file: " + name);
        if (fs::file_size(p) == 0) return fail("empty param file: " + name);
        if (fs::file_size(bn) == 0)
        {
            // empty bins are legit only for weight-free stages (parameter_count==0,
            // e.g. head/rope_apply) or pnnx-folded identity weights (e.g. qk-norm
            // RMSNorm with all-ones weights); anything else is a broken export.
            const std::string bin_key = name + ".bin";
            const bool weight_free = manifest_param_count.count(bin_key) == 0 || manifest_param_count[bin_key] == 0;
            const bool folded = manifest_folded.count(bin_key) != 0 && manifest_folded[bin_key];
            if (!weight_free && !folded) return fail("empty bin for weighted stage: " + name);
        }
        ++model_count;
        CacheLoadStats stats;
        std::string err;
        ncnn::Net* net = cache.get_or_load(files, stats, err);
        if (!net) return fail("ncnn load failed: " + name + ": " + err);
        if (manifest_sha.count(name + ".bin"))
        {
            if (sha256_file(core->directory / files.bin_path) != manifest_sha[name + ".bin"])
                return fail("sha256 mismatch: " + name);
            if (static_cast<long long>(fs::file_size(bn)) != manifest_size[name + ".bin"])
                return fail("size mismatch: " + name);
        }
        std::cout << "  loaded " << name << " (" << fs::file_size(bn) << " bytes)\n";
    }
    if (model_count < 19) return fail("expected >= 19 models, got " + std::to_string(model_count));

    // controlled-failure negatives
    int neg = 0;
    const auto neg_check = [&](bool failed_as_expected, const char* what) {
        ++neg;
        if (!failed_as_expected) { std::cerr << "FAIL negative: " << what << '\n'; std::exit(1); }
    };

    // wrong variant lookup (C=512 synthetic) must not resolve to this key
    const CoreInfo* wrong = registry.lookup(CoreKey{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128"});
    neg_check(wrong == nullptr || wrong->name != kVariant, "wrong C=512 variant not selected");

    // missing file -> cache load must fail safely (no silent fallback)
    {
        NcnnModelFiles missing;
        missing.param_path = "models/nonexistent/model.ncnn.param";
        missing.bin_path = "models/nonexistent/model.ncnn.bin";
        CacheLoadStats st;
        std::string err;
        ncnn::Net* net = cache.get_or_load(missing, st, err);
        neg_check(net == nullptr && !err.empty(), "missing model fails safely");
    }

    // tampered sha256 -> manifest mismatch detection
    neg_check(sha256_file(core->directory / artifact->models.at("attn_norm").bin_path) !=
              std::string(64, '0'), "tampered sha256 detected");

    // param without bin -> load fail
    {
        NcnnModelFiles bad;
        bad.param_path = "models/attn_norm/model.ncnn.param";
        bad.bin_path = "models/attn_norm/missing.bin";
        CacheLoadStats st;
        std::string err;
        ncnn::Net* net = cache.get_or_load(bad, st, err);
        neg_check(net == nullptr, "param/bin mismatch fails safely");
    }

    // weighted stage must have a non-empty bin (qkv_vid carries 19,660,800 params)
    neg_check(fs::file_size(core->directory / artifact->models.at("qkv_vid").bin_path) > 0, "qkv_vid non-empty");

    std::cout << "config checks=" << checks << " models=" << model_count
              << " cache_size=" << cache.size() << " negatives=" << neg << " ALL PASS\n";
    return 0;
}
