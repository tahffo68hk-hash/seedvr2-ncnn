// Phase 12: ArtifactRegistry contract tests for the synthetic SeedVR2 DiT block
// variant, plus registry negative tests (wrong metadata configs, missing
// artifacts) and SHA256 manifest verification.
#include <cstdio>
#include <functional>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <bcrypt.h>

#include "artifact_registry.h"
#include "phase12_test_common.h"

using namespace attention_dispatcher;
namespace fs = std::filesystem;
using phase12_test::JVal;

namespace
{
int g_failures = 0;
int g_checks = 0;

void check(bool ok, const std::string& what)
{
    ++g_checks;
    std::cout << (ok ? "  PASS " : "  FAIL ") << what << '\n';
    if (!ok) ++g_failures;
}

bool read_all(const fs::path& path, std::vector<unsigned char>& out)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto bytes = file.tellg();
    if (bytes < 0) return false;
    out.resize(static_cast<std::size_t>(bytes));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(out.data()), bytes);
    return bool(file);
}

std::string sha256_file(const fs::path& path)
{
    std::vector<unsigned char> data;
    if (!read_all(path, data)) return "";
    BCRYPT_ALG_HANDLE alg = nullptr;
    BCRYPT_HASH_HANDLE hash = nullptr;
    std::string result;
    do
    {
        if (BCryptOpenAlgorithmProvider(&alg, BCRYPT_SHA256_ALGORITHM, nullptr, 0) != 0) break;
        if (BCryptCreateHash(alg, &hash, nullptr, 0, nullptr, 0, 0) != 0) break;
        if (!data.empty() && BCryptHashData(hash, data.data(), static_cast<ULONG>(data.size()), 0) != 0) break;
        BYTE digest[32] = {};
        if (BCryptFinishHash(hash, digest, sizeof digest, 0) != 0) break;
        static const char* hex = "0123456789abcdef";
        for (BYTE b : digest)
        {
            result += hex[b >> 4];
            result += hex[b & 0xF];
        }
    } while (false);
    if (hash) BCryptDestroyHash(hash);
    if (alg) BCryptCloseAlgorithmProvider(alg, 0);
    return result;
}

// Builds a temp artifact dir with metadata.json derived from the base metadata
// and an (optionally empty) params dir.  Returns false on setup failure.
bool make_temp_artifact(const fs::path& root, const fs::path& src_metadata,
                        const std::function<void(std::map<std::string, JVal>&)>& mutate,
                        const std::map<std::string, std::string>& extra_files,
                        fs::path& out_dir)
{
    const fs::path dir = root / "bad";
    std::error_code ec;
    fs::remove_all(dir, ec);
    fs::create_directories(dir / "params", ec);
    std::string text_content;
    if (!phase12_test::read_text(src_metadata, text_content)) return false;
    JVal meta = phase12_test::parse_json(text_content);
    if (mutate) mutate(meta.obj);
    std::ofstream file(dir / "metadata.json");
    file << phase12_test::to_json(meta);
    for (const auto& kv : extra_files)
    {
        const fs::path target = dir / kv.first;
        fs::create_directories(target.parent_path(), ec);
        std::ofstream f(target, std::ios::binary);
        f.write(kv.second.data(), static_cast<std::streamsize>(kv.second.size()));
    }
    out_dir = dir;
    return true;
}

JVal& block_of(std::map<std::string, JVal>& meta)
{
    return meta.at("block");
}
} // namespace

int main(int argc, char** argv)
{
    std::cout << std::unitbuf;
    const fs::path artifacts_root = argc > 1 ? fs::path(argv[1]) : fs::path("tests/seedvr2_dit_block_ncnn/artifacts");
    const fs::path base_dir = artifacts_root / "synthetic_seedvr2_dit_block_c512_h4_hd128";
    const fs::path ll_dir = artifacts_root / "synthetic_seedvr2_dit_block_c512_h4_hd128_last_layer";
    const fs::path base_metadata = base_dir / "metadata.json";

    std::cout << "== registry scan ==\n";
    ArtifactRegistry registry;
    std::string error;
    check(registry.scan(artifacts_root, error), "scan base+last-layer (" + error + ")");
    check(registry.size() == 2, "registry size == 2");

    const CoreKey base_key{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128"};
    const CoreInfo* base = registry.lookup(base_key);
    check(base != nullptr, "base core lookup");
    if (base)
    {
        const DiTBlockArtifact* art = base->dit_block();
        check(art != nullptr, "base execution is StagedDiTBlock");
        check(base->execution_type == ExecutionType::StagedDiTBlock, "base execution_type");
        check(art && art->block.hidden_dim == 512 && art->block.txt_dim == 512, "base hidden/txt dim");
        check(art && art->block.heads == 4 && art->block.head_dim == 128, "base heads/head_dim");
        check(art && art->block.configured_rope_dim == 128 && art->block.effective_rotary_dim == 126, "base rope dims");
        check(art && art->block.identity_tail_dim == 2, "base identity tail");
        check(art && art->block.mlp_hidden_dim == 1536, "base mlp hidden");
        check(art && art->block.q_max == 128 && art->block.head_pairs == 64, "base q_max/head_pairs");
        check(art && !art->block.is_last_layer, "base not last layer");
        check(art && art->model("head3") != nullptr && art->model("head0") != nullptr, "base head models");
        check(art && art->model("swiglu_gate_txt") != nullptr && art->model("swiglu_up_txt") != nullptr && art->model("swiglu_out_txt") != nullptr,
              "base txt swiglu models present");
        check(art && art->model("swiglu_gate_vid") != nullptr, "base vid swiglu model present");
        check(art && fs::exists(art->params_dir / "ada_txt_mlp_shift.bin"), "base txt ada params present");
        check(art && fs::exists(art->params_dir / "ada_vid_attn_gate.bin"), "base vid ada params present");
    }

    const CoreKey ll_key{4, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128_last_layer"};
    const CoreInfo* ll = registry.lookup(ll_key);
    check(ll != nullptr, "last-layer core lookup");
    if (ll)
    {
        const DiTBlockArtifact* art = ll->dit_block();
        check(art && art->block.is_last_layer, "last-layer is_last_layer");
        check(art && art->model("swiglu_gate_txt") == nullptr && art->model("swiglu_out_txt") == nullptr, "last-layer lacks txt swiglu");
        check(art && art->model("swiglu_gate_vid") != nullptr, "last-layer vid swiglu present");
        check(art && art->model("attn_norm") != nullptr, "last-layer shared attn_norm via .. path");
        check(!fs::exists(art->params_dir / "ada_txt_mlp_shift.bin"), "last-layer lacks txt ada params");
        check(fs::exists(art->params_dir / "ada_vid_attn_shift.bin"), "last-layer vid ada params present");
    }

    check(registry.lookup(CoreKey{4, 3, 3, 1, 1, 1, "does_not_exist"}) == nullptr, "wrong variant -> nullptr");
    check(registry.lookup(CoreKey{8, 3, 3, 1, 1, 1, "synthetic_seedvr2_dit_block_c512_h4_hd128"}) == nullptr, "wrong window -> nullptr");

    std::cout << "== sha256 manifest verification ==\n";
    {
        std::string manifest_text;
        const bool read_base_manifest = phase12_test::read_text(base_dir / "sha256_manifest.json", manifest_text);
        check(read_base_manifest, "read base sha256 manifest");
        if (read_base_manifest)
        {
            const JVal manifest = phase12_test::parse_json(manifest_text);
            const JVal& files = manifest.obj.at("files");
            bool all_ok = true;
            std::size_t count = 0;
            for (const auto& kv : files.obj)
            {
                const std::string got = sha256_file(base_dir / kv.first);
                all_ok = all_ok && got == kv.second.str;
                ++count;
            }
            check(all_ok, "base sha256 all match (" + std::to_string(count) + " files)");
        }
        manifest_text.clear();
        const bool read_ll_manifest = phase12_test::read_text(ll_dir / "sha256_manifest.json", manifest_text);
        check(read_ll_manifest, "read ll sha256 manifest");
        if (read_ll_manifest)
        {
            const JVal manifest = phase12_test::parse_json(manifest_text);
            const JVal& files = manifest.obj.at("files");
            bool all_ok = true;
            std::size_t count = 0;
            for (const auto& kv : files.obj)
            {
                const std::string got = sha256_file(ll_dir / kv.first);
                all_ok = all_ok && got == kv.second.str;
                ++count;
            }
            check(all_ok, "last-layer sha256 all match (" + std::to_string(count) + " files)");
        }
    }

    std::cout << "== registry negative tests ==\n";
    const fs::path tmp = fs::temp_directory_path() / "phase12_registry_neg";
    std::error_code ec;
    fs::remove_all(tmp, ec);
    fs::create_directories(tmp, ec);
    const auto scan_expect_fail = [&](const char* what, const std::function<void(std::map<std::string, JVal>&)>& mutate) {
        fs::path dir;
        if (!make_temp_artifact(tmp, base_metadata, mutate, {}, dir)) { check(false, std::string(what) + " setup"); return; }
        ArtifactRegistry r2;
        std::string e2;
        const bool ok = !r2.scan(dir.parent_path(), e2);
        check(ok, std::string(what) + " -> rejected (" + e2 + ")");
    };
    scan_expect_fail("heads*head_dim != hidden", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["heads"] = JVal{JVal::Number, false, 5, {}, {}, {}}; });
    scan_expect_fail("head_dim*heads != hidden", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["head_dim"] = JVal{JVal::Number, false, 64, {}, {}, {}}; });
    scan_expect_fail("configured_rope_dim > head_dim", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["configured_rope_dim"] = JVal{JVal::Number, false, 256, {}, {}, {}}; });
    scan_expect_fail("effective+tail != configured", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["effective_rotary_dim"] = JVal{JVal::Number, false, 124, {}, {}, {}}; });
    scan_expect_fail("identity_tail wrong", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["identity_tail_dim"] = JVal{JVal::Number, false, 4, {}, {}, {}}; });
    scan_expect_fail("effective not divisible by 3", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["effective_rotary_dim"] = JVal{JVal::Number, false, 125, {}, {}, {}}; });
    scan_expect_fail("mlp_type unsupported", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["mlp_type"] = JVal{JVal::String, false, 0, "gelu", {}, {}}; });
    scan_expect_fail("q_max < head_dim", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["q_max"] = JVal{JVal::Number, false, 64, {}, {}, {}}; });
    scan_expect_fail("head_pairs mismatch", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["head_pairs"] = JVal{JVal::Number, false, 32, {}, {}, {}}; });
    scan_expect_fail("norm_eps not positive", [](std::map<std::string, JVal>& meta) { block_of(meta).obj["norm_eps"] = JVal{JVal::Number, false, 0.0, {}, {}, {}}; });

    // missing model file: valid metadata, empty params, models reference nonexistent files
    {
        fs::path dir;
        if (make_temp_artifact(tmp, base_metadata, nullptr, {}, dir))
        {
            ArtifactRegistry r2;
            std::string e2;
            const bool ok = !r2.scan(dir.parent_path(), e2);
            check(ok, "missing model file -> rejected (" + e2 + ")");
        }
        else check(false, "missing model file setup");
    }
    // missing params dir
    {
        const fs::path dir = tmp / "bad2";
        fs::remove_all(dir, ec);
        fs::create_directories(dir, ec);
        std::string text_content;
        phase12_test::read_text(base_metadata, text_content);
        std::ofstream file(dir / "metadata.json");
        file << text_content;
        ArtifactRegistry r2;
        std::string e2;
        const bool ok = !r2.scan(dir.parent_path(), e2);
        check(ok, "missing params dir -> rejected (" + e2 + ")");
    }

    // sha256 tamper detection
    {
        const fs::path tampered = tmp / "tampered.bin";
        std::vector<unsigned char> data;
        const bool read_ok = read_all(base_dir / "models" / "qkv_vid" / "model.ncnn.bin", data);
        check(read_ok && data.size() > 4, "tamper test source bin readable");
        if (read_ok && data.size() > 4)
        {
            std::ofstream f(tampered, std::ios::binary);
            f.write(reinterpret_cast<char*>(data.data()), static_cast<std::streamsize>(data.size()));
            f.close();
            const std::string before = sha256_file(tampered);
            {
                std::fstream f2(tampered, std::ios::binary | std::ios::in | std::ios::out);
                f2.seekp(static_cast<std::streamoff>(data.size() / 2));
                const unsigned char flipped = static_cast<unsigned char>(data[data.size() / 2] ^ 0xFF);
                f2.write(reinterpret_cast<const char*>(&flipped), 1);
            }
            const std::string after = sha256_file(tampered);
            check(!before.empty() && before != after, "sha256 detects tampered file");
        }
    }

    fs::remove_all(tmp, ec);

    std::cout << "\n=== registry tests: " << (g_failures == 0 ? "PASS" : "FAIL")
              << " (" << g_checks << " checks, " << g_failures << " failures) ===\n";
    return g_failures == 0 ? 0 : 1;
}
