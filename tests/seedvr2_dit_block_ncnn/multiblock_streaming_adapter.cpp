// Phase 15: multi-block streaming hybrid adapter executable.
//
// Runs a contiguous chain of real-weight SeedVR2 blocks (1/2/4) through the
// HybridDiTBlockRuntime with two load modes:
//   streaming:  LOAD block i -> EXECUTE block i -> UNLOAD block i (destroy
//               runtime, clear the ncnn model cache, verify cache==0, release
//               file handles) before LOAD block i+1.  max weighted resident=1.
//   resident:   LOAD all blocks first, EXECUTE in chain order, UNLOAD all at
//               the end (diagnostic mode only, never the acceptance mode).
// The adapter is the ncnn path: there is no PyTorch fallback, so
// original_block_call_count is always 0 and every chain emits `HYBRID BLOCK
// EXECUTED` per block with content hashes (no silent fallback).
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>
#pragma comment(lib, "psapi.lib")
#endif

#include "artifact_registry.h"
#include "hybrid_dit_block_runtime.h"
#include "phase12_test_common.h"

using namespace attention_dispatcher;
namespace fs = std::filesystem;
using seedvr2_dit_block::DiTBlockInput;
using seedvr2_dit_block::DiTBlockOutput;
using seedvr2_dit_block::DiTBlockRunStats;
using seedvr2_dit_block::DiTBlockTrace;
using seedvr2_dit_block::HybridDiTBlockRuntime;
using seedvr2_dit_block::VideoShape;
using phase12_test::JVal;
using phase12_test::parse_json;

namespace
{
JVal number(double v) { JVal j; j.kind = JVal::Number; j.num = v; return j; }
JVal boolean(bool v) { JVal j; j.kind = JVal::Bool; j.b = v; return j; }
JVal text(const std::string& v) { JVal j; j.kind = JVal::String; j.str = v; return j; }

uint64_t fnv1a64(const void* data, std::size_t n)
{
    uint64_t h = 1469598103934665603ull;
    const unsigned char* p = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < n; ++i) { h ^= p[i]; h *= 1099511628211ull; }
    return h;
}

std::string hash_hex(const std::vector<float>& v)
{
    char buf[17];
    std::snprintf(buf, sizeof(buf), "%016llx", static_cast<unsigned long long>(fnv1a64(v.data(), v.size() * sizeof(float))));
    return buf;
}

bool read_json_ints(const fs::path& p, std::vector<int>& out)
{
    std::string s;
    if (!phase12_test::read_text(p, s)) return false;
    out = phase12_test::int_array(parse_json(s));
    return true;
}

bool write_bin(const fs::path& p, const std::vector<float>& v)
{
    std::ofstream f(p, std::ios::binary);
    if (!f) return false;
    f.write(reinterpret_cast<const char*>(v.data()), static_cast<std::streamsize>(v.size() * sizeof(float)));
    return bool(f);
}

struct MemSnap
{
    double private_bytes_mb = 0.0, working_set_mb = 0.0, peak_working_set_mb = 0.0;
};
MemSnap mem_snap()
{
    MemSnap m;
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS_EX pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&pmc), sizeof(pmc)))
    {
        m.private_bytes_mb = static_cast<double>(pmc.PrivateUsage) / (1024.0 * 1024.0);
        m.working_set_mb = static_cast<double>(pmc.WorkingSetSize) / (1024.0 * 1024.0);
        m.peak_working_set_mb = static_cast<double>(pmc.PeakWorkingSetSize) / (1024.0 * 1024.0);
    }
#else
    (void)0;
#endif
    return m;
}

int metadata_block_index(const fs::path& root, const std::string& variant, std::string& error)
{
    // Cross-check the CLI-declared block index against the artifact metadata:
    // the variant encodes the real checkpoint block, so a mismatched binding
    // (e.g. `99:real_seedvr2_block16_...`) must be rejected, not silently run
    // the wrong weights under a fabricated index.
    std::string text;
    if (!phase12_test::read_text(root / variant / "metadata.json", text))
    {
        error = "metadata.json unreadable for " + variant;
        return -1;
    }
    const JVal meta = phase12_test::parse_json(text);
    if (meta.kind != JVal::Object) { error = "metadata.json not an object for " + variant; return -1; }
    const auto it_block = meta.obj.find("block");
    if (it_block == meta.obj.end() || it_block->second.kind != JVal::Object)
    {
        error = "metadata.json missing block object for " + variant;
        return -1;
    }
    const auto it_idx = it_block->second.obj.find("block_index");
    if (it_idx == it_block->second.obj.end() || it_idx->second.kind != JVal::Number)
    {
        error = "metadata.json missing block.block_index for " + variant;
        return -1;
    }
    return static_cast<int>(it_idx->second.num);
}

int usage()
{
    std::cerr << "usage: multiblock_streaming_adapter --root <artifacts_root>"
                 " --blocks <idx:variant[,idx:variant...]> --input-dir <dir> --output-dir <dir>"
                 " [--load-mode streaming|resident] [--report <json>] [--trace-dir <dir>]"
                 " [--memory-profile] [--no-file-probe]\n";
    return 2;
}

struct BlockSpec { int index; std::string variant; };

bool file_handle_probe(const std::vector<fs::path>& files, std::string& error)
{
    // After unload the artifact files must be openable and renameable:
    // a leftover OS handle makes the rename fail on Windows (sharing violation).
    for (const auto& p : files)
    {
        {
            std::ifstream f(p, std::ios::binary);
            if (!f) { error = "file not openable after unload: " + p.string(); return false; }
        }
        const fs::path tmp = fs::path(p.string() + ".probe_tmp");
        std::error_code ec;
        fs::rename(p, tmp, ec);
        if (ec) { error = "rename probe failed (file handle still held?): " + p.string() + " : " + ec.message(); return false; }
        fs::rename(tmp, p, ec);
        if (ec) { error = "rename-back failed: " + p.string() + " : " + ec.message(); return false; }
    }
    return true;
}

int run_chain(const std::vector<BlockSpec>& blocks, const fs::path& root,
              const fs::path& input_dir, const fs::path& output_dir,
              const fs::path& report_path, const fs::path& trace_dir,
              bool resident, bool memory_profile, bool probe_files)
{
    ArtifactRegistry registry;
    std::string error;
    // Range/binding contract mirrored from the Python orchestrator: block
    // indices must be 0..30 (last layer 31 excluded) and must match the
    // artifact metadata (no fabricated index / wrong-variant binding).
    for (const auto& b : blocks)
    {
        if (b.index < 0 || b.index > 30)
        {
            std::cerr << "block index out of range: " << b.index << "\n";
            return 1;
        }
        const int meta_index = metadata_block_index(root, b.variant, error);
        if (meta_index < 0)
        {
            std::cerr << "metadata block_index unavailable: " << error << "\n";
            return 1;
        }
        if (meta_index != b.index)
        {
            std::cerr << "block index mismatch: declared " << b.index
                      << " but artifact " << b.variant << " is block " << meta_index << "\n";
            return 1;
        }
    }
    if (!registry.scan(root, error)) { std::cerr << "registry scan failed: " << error << "\n"; return 1; }


    std::vector<const CoreInfo*> cores;
    for (const auto& b : blocks)
    {
        const CoreKey key{4, 3, 3, 1, 1, 1, b.variant};
        const CoreInfo* core = registry.lookup(key);
        if (!core) { std::cerr << "variant not found: " << b.variant << '\n'; return 1; }
        cores.push_back(core);
    }

    DiTBlockInput in;
    std::vector<float> vid, txt, emb;
    if (!phase12_test::read_bin(input_dir / "input_vid.bin", vid) ||
        !phase12_test::read_bin(input_dir / "input_txt.bin", txt) ||
        !phase12_test::read_bin(input_dir / "emb.bin", emb))
    {
        std::cerr << "input bins missing in " << input_dir << '\n';
        return 1;
    }
    std::vector<int> shapes_flat;
    if (!read_json_ints(input_dir / "vid_shapes.json", shapes_flat) || shapes_flat.empty() || shapes_flat.size() % 3 != 0)
    {
        std::cerr << "vid_shapes.json invalid\n";
        return 1;
    }
    std::vector<int> txt_lengths;
    if (!read_json_ints(input_dir / "txt_lengths.json", txt_lengths) || txt_lengths.empty())
    {
        std::cerr << "txt_lengths.json invalid\n";
        return 1;
    }
    const int dim = cores.front()->dit_block()->block.hidden_dim;
    for (std::size_t i = 0; i < shapes_flat.size(); i += 3)
        in.vid_shapes.push_back({shapes_flat[i], shapes_flat[i + 1], shapes_flat[i + 2]});
    in.txt_lengths = txt_lengths;
    in.vid = ncnn::Mat(dim, static_cast<int>(vid.size() / dim));
    std::memcpy(in.vid.data, vid.data(), vid.size() * sizeof(float));
    in.txt = ncnn::Mat(dim, static_cast<int>(txt.size() / dim));
    std::memcpy(in.txt.data, txt.data(), txt.size() * sizeof(float));
    in.emb = ncnn::Mat(6 * dim, static_cast<int>(emb.size() / (6 * dim)));
    std::memcpy(in.emb.data, emb.data(), emb.size() * sizeof(float));

    NcnnModelCache cache;
    const auto t_start = std::chrono::steady_clock::now();
    const auto now_ms = [&]() {
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_start).count();
    };

    JVal report;
    report.kind = JVal::Object;
    report.obj["mode"] = text(resident ? "resident" : "streaming");
    JVal bidx; bidx.kind = JVal::Array; for (const auto& b : blocks) { JVal n; n.kind = JVal::Number; n.num = static_cast<double>(b.index); bidx.arr.push_back(n); } report.obj["block_indices"] = bidx;
    report.obj["events"] = phase12_test::JVal{}; report.obj["events"].kind = JVal::Array;
    report.obj["per_block"] = phase12_test::JVal{}; report.obj["per_block"].kind = JVal::Array;

    std::vector<JVal> events;
    std::vector<JVal> per_block;
    std::size_t total_load_ms = 0, total_execute_ms = 0, total_unload_ms = 0;
    bool order_ok = true;

    // Resident mode: initialize every runtime first (diagnostic).
    std::vector<HybridDiTBlockRuntime*> resident_runtimes;
    std::vector<NcnnModelCache*> resident_caches;
    if (resident)
    {
        for (std::size_t i = 0; i < blocks.size(); ++i)
        {
            auto* cache_i = new NcnnModelCache();
            auto* rt = new HybridDiTBlockRuntime(*cache_i);
            const auto t0 = std::chrono::steady_clock::now();
            if (!rt->initialize(*cores[i], error))
            {
                std::cerr << "resident init failed block " << blocks[i].index << ": " << error << '\n';
                delete rt; delete cache_i;
                for (std::size_t j = 0; j < resident_runtimes.size(); ++j) { delete resident_runtimes[j]; delete resident_caches[j]; }
                return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            total_load_ms += static_cast<std::size_t>(ms);
            JVal ev; ev.kind = JVal::Object;
            ev.obj["stage"] = text("LOAD");
            ev.obj["block_index"] = number(blocks[i].index);
            ev.obj["variant"] = text(blocks[i].variant);
            ev.obj["t_ms"] = number(now_ms());
            ev.obj["duration_ms"] = number(ms);
            ev.obj["cache_size_before"] = number(0);
            ev.obj["cache_size_after"] = number(static_cast<double>(cache_i->size()));
            ev.obj["resident_bytes_after"] = number(static_cast<double>(cache_i->resident_bytes_estimate()));
            events.push_back(ev);
            resident_runtimes.push_back(rt);
            resident_caches.push_back(cache_i);
        }
    }

    // Chain execution (streaming: one runtime alive at a time).
    DiTBlockOutput chain_out;
    bool first_block = true;
    std::size_t adapter_call_count = 0;
    for (std::size_t i = 0; i < blocks.size(); ++i)
    {
        const int bi = blocks[i].index;
        const std::string& variant = blocks[i].variant;
        JVal pbin; pbin.kind = JVal::Object;
        pbin.obj["block_index"] = number(bi);
        pbin.obj["variant"] = text(variant);

        HybridDiTBlockRuntime* rt = nullptr;
        NcnnModelCache* cache_i = &cache;
        if (resident) { rt = resident_runtimes[i]; cache_i = resident_caches[i]; }
        else { rt = new HybridDiTBlockRuntime(cache); }

        // LOAD (streaming) / already loaded (resident).
        if (!resident)
        {
            const auto t0 = std::chrono::steady_clock::now();
            const std::size_t cache_before = cache.size();
            if (!rt->initialize(*cores[i], error))
            {
                std::cerr << "runtime init failed block " << bi << ": " << error << '\n';
                delete rt; cache.clear();
                return 1;
            }
            const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            total_load_ms += static_cast<std::size_t>(ms);
            JVal ev; ev.kind = JVal::Object;
            ev.obj["stage"] = text("LOAD");
            ev.obj["block_index"] = number(bi);
            ev.obj["variant"] = text(variant);
            ev.obj["t_ms"] = number(now_ms());
            ev.obj["duration_ms"] = number(ms);
            ev.obj["cache_size_before"] = number(static_cast<double>(cache_before));
            ev.obj["cache_size_after"] = number(static_cast<double>(cache.size()));
            ev.obj["resident_bytes_after"] = number(static_cast<double>(cache.resident_bytes_estimate()));
            if (memory_profile) { const auto m = mem_snap(); ev.obj["mem_private_mb"] = number(m.private_bytes_mb); ev.obj["mem_working_set_mb"] = number(m.working_set_mb); ev.obj["mem_peak_ws_mb"] = number(m.peak_working_set_mb); }
            events.push_back(ev);
            std::cout << "LOAD block " << bi << " variant=" << variant << " cache=" << cache.size() << "\n";
        }

        // EXECUTE.
        DiTBlockInput block_in;
        if (first_block) { block_in = in; }
        else { block_in = in; block_in.vid = chain_out.vid; block_in.txt = chain_out.txt; }
        DiTBlockTrace trace;
        DiTBlockOutput out;
        DiTBlockRunStats stats;
        const auto t0 = std::chrono::steady_clock::now();
        if (!rt->run(block_in, out, &trace, stats))
        {
            std::cerr << "runtime error block " << bi << ": " << stats.error_message << '\n';
            if (!resident) { delete rt; cache.clear(); }
            return 1;
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        total_execute_ms += static_cast<std::size_t>(ms);
        ++adapter_call_count;
        chain_out = out;

        const std::vector<float> out_vid(static_cast<const float*>(out.vid.data),
                                         static_cast<const float*>(out.vid.data) + out.vid.total());
        const std::vector<float> out_txt(static_cast<const float*>(out.txt.data),
                                         static_cast<const float*>(out.txt.data) + out.txt.total());
        const fs::path bdir = output_dir / ("block_" + std::to_string(bi));
        fs::create_directories(bdir);
        if (!write_bin(bdir / "output_vid.bin", out_vid) || !write_bin(bdir / "output_txt.bin", out_txt))
        {
            std::cerr << "cannot write block outputs\n";
            if (!resident) { delete rt; cache.clear(); }
            return 1;
        }
        if (first_block)
        {
            write_bin(bdir / "input_vid.bin", vid);
            write_bin(bdir / "input_txt.bin", txt);
            write_bin(bdir / "emb.bin", emb);
            first_block = false;
        }
        else
        {
            const std::vector<float> prev_vid(static_cast<const float*>(chain_out.vid.data),
                                              static_cast<const float*>(chain_out.vid.data) + chain_out.vid.total());
            const std::vector<float> prev_txt(static_cast<const float*>(chain_out.txt.data),
                                              static_cast<const float*>(chain_out.txt.data) + chain_out.txt.total());
            // block i input == block (i-1) output (written under block i-1 dir).
            write_bin(output_dir / ("block_" + std::to_string(blocks[i - 1].index)) / "output_next_input_vid.bin", prev_vid);
            write_bin(output_dir / ("block_" + std::to_string(blocks[i - 1].index)) / "output_next_input_txt.bin", prev_txt);
        }

        JVal ev; ev.kind = JVal::Object;
        ev.obj["stage"] = text("EXECUTE");
        ev.obj["block_index"] = number(bi);
        ev.obj["variant"] = text(variant);
        ev.obj["t_ms"] = number(now_ms());
        ev.obj["duration_ms"] = number(ms);
        ev.obj["cache_size_before"] = number(static_cast<double>(cache_i->size()));
        ev.obj["cache_size_after"] = number(static_cast<double>(cache_i->size()));
        ev.obj["vid_hash"] = text(hash_hex(out_vid));
        ev.obj["txt_hash"] = text(hash_hex(out_txt));
        ev.obj["total_packed"] = number(static_cast<double>(stats.total_packed));
        events.push_back(ev);

        pbin.obj["load_ms"] = number(0);
        pbin.obj["execute_ms"] = number(ms);
        pbin.obj["unload_ms"] = number(0);
        pbin.obj["vid_hash"] = text(hash_hex(out_vid));
        pbin.obj["txt_hash"] = text(hash_hex(out_txt));
        per_block.push_back(pbin);

        if (!trace_dir.empty())
        {
            const fs::path tdir = trace_dir / ("block_" + std::to_string(bi));
            fs::create_directories(tdir);
            std::vector<std::pair<const char*, const std::vector<float>*>> stages = {
                {"attn_norm_vid", &trace.attn_norm_vid}, {"attn_norm_txt", &trace.attn_norm_txt},
                {"qkv_vid", &trace.qkv_vid}, {"qkv_txt", &trace.qkv_txt},
                {"q_vid", &trace.q_vid}, {"k_vid", &trace.k_vid}, {"v_vid", &trace.v_vid},
                {"q_txt", &trace.q_txt}, {"k_txt", &trace.k_txt}, {"v_txt", &trace.v_txt},
                {"q_norm_vid", &trace.q_norm_vid}, {"k_norm_vid", &trace.k_norm_vid},
                {"q_norm_txt", &trace.q_norm_txt}, {"k_norm_txt", &trace.k_norm_txt},
                {"rope_after_q_vid", &trace.rope_after_q_vid}, {"rope_after_k_vid", &trace.rope_after_k_vid},
                {"rope_after_q_txt", &trace.rope_after_q_txt}, {"rope_after_k_txt", &trace.rope_after_k_txt},
                {"attention_packed_output", &trace.attention_packed_output},
                {"attention_vid_windows", &trace.attention_vid_windows},
                {"attention_txt_windows", &trace.attention_txt_windows},
                {"attention_vid_merged", &trace.attention_vid_merged},
                {"attention_txt_mean_merged", &trace.attention_txt_mean_merged},
                {"attention_proj_vid", &trace.attention_proj_vid}, {"attention_proj_txt", &trace.attention_proj_txt},
                {"attention_residual_vid", &trace.attention_residual_vid}, {"attention_residual_txt", &trace.attention_residual_txt},
                {"mlp_norm_vid", &trace.mlp_norm_vid}, {"mlp_norm_txt", &trace.mlp_norm_txt},
                {"swiglu_gate_vid", &trace.swiglu_gate_vid}, {"swiglu_up_vid", &trace.swiglu_up_vid},
                {"swiglu_product_vid", &trace.swiglu_product_vid}, {"swiglu_output_vid", &trace.swiglu_output_vid},
                {"swiglu_gate_txt", &trace.swiglu_gate_txt}, {"swiglu_up_txt", &trace.swiglu_up_txt},
                {"swiglu_product_txt", &trace.swiglu_product_txt}, {"swiglu_output_txt", &trace.swiglu_output_txt},
                {"final_vid", &trace.final_vid}, {"final_txt", &trace.final_txt},
            };
            for (const auto& s : stages)
                write_bin(tdir / (std::string(s.first) + ".bin"), *s.second);
        }

        // UNLOAD (streaming: destroy runtime, clear cache, verify, file probe).
        if (!resident)
        {
            const auto u0 = std::chrono::steady_clock::now();
            delete rt;                     // destroy runtime (its net pointers die with scope)
            const std::size_t erased = cache.size();
            cache.clear();
            const double ums = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();
            total_unload_ms += static_cast<std::size_t>(ums);
            const std::size_t after = cache.size();
            const std::size_t bytes_after = cache.resident_bytes_estimate();
            if (after != 0 || bytes_after != 0)
            {
                std::cerr << "cache not cleared after unload block " << bi << " size=" << after << " bytes=" << bytes_after << '\n';
                return 1;
            }
            JVal evu; evu.kind = JVal::Object;
            evu.obj["stage"] = text("UNLOAD");
            evu.obj["block_index"] = number(bi);
            evu.obj["variant"] = text(variant);
            evu.obj["t_ms"] = number(now_ms());
            evu.obj["duration_ms"] = number(ums);
            evu.obj["cache_size_before"] = number(static_cast<double>(cache.size() + erased));
            evu.obj["cache_size_after"] = number(static_cast<double>(after));
            evu.obj["resident_bytes_after"] = number(static_cast<double>(bytes_after));
            evu.obj["cache_cleared"] = boolean(after == 0 && bytes_after == 0);
            if (memory_profile) { const auto m = mem_snap(); evu.obj["mem_private_mb"] = number(m.private_bytes_mb); evu.obj["mem_working_set_mb"] = number(m.working_set_mb); }
            events.push_back(evu);
            std::cout << "UNLOAD block " << bi << " cache_after=" << after << " resident_bytes_after=" << bytes_after << "\n";

            // File-handle release probe (unless disabled).
            if (probe_files)
            {
                std::vector<fs::path> files;
                // probe every model file of this block's artifact
                const fs::path vdir = root / variant;
                for (const auto& entry : fs::recursive_directory_iterator(vdir))
                {
                    if (entry.is_regular_file())
                    {
                        const std::string ext = entry.path().extension().string();
                        if (ext == ".bin" || ext == ".param") files.push_back(entry.path());
                    }
                }
                std::string perr;
                if (!file_handle_probe(files, perr))
                {
                    std::cerr << "file handle probe failed block " << bi << ": " << perr << '\n';
                    return 1;
                }
                JVal evp; evp.kind = JVal::Object;
                evp.obj["stage"] = text("FILE_PROBE");
                evp.obj["block_index"] = number(bi);
                evp.obj["t_ms"] = number(now_ms());
                evp.obj["files_probed"] = number(static_cast<double>(files.size()));
                evp.obj["handles_released"] = boolean(true);
                events.push_back(evp);
                std::cout << "FILE_PROBE block " << bi << " files=" << files.size() << " handles_released=true\n";
            }
        }
        std::cout << "HYBRID BLOCK EXECUTED block_index=" << bi << " runtime=" << variant
                  << " vid_hash=" << hash_hex(out_vid) << " txt_hash=" << hash_hex(out_txt)
                  << " ncnn_adapter_call_count=" << adapter_call_count << " original_block_call_count=0"
                  << " silent_fallback=false execute_ms=" << ms << '\n';
    }

    // Final outputs.
    fs::create_directories(output_dir);
    const std::vector<float> fvid(static_cast<const float*>(chain_out.vid.data),
                                  static_cast<const float*>(chain_out.vid.data) + chain_out.vid.total());
    const std::vector<float> ftxt(static_cast<const float*>(chain_out.txt.data),
                                  static_cast<const float*>(chain_out.txt.data) + chain_out.txt.total());
    write_bin(output_dir / "final_vid.bin", fvid);
    write_bin(output_dir / "final_txt.bin", ftxt);

    // Resident UNLOAD all at the end.
    if (resident)
    {
        for (std::size_t i = 0; i < resident_runtimes.size(); ++i)
        {
            const auto u0 = std::chrono::steady_clock::now();
            delete resident_runtimes[i];
            const std::size_t erased = resident_caches[i]->size();
            resident_caches[i]->clear();
            const double ums = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - u0).count();
            total_unload_ms += static_cast<std::size_t>(ums);
            JVal evu; evu.kind = JVal::Object;
            evu.obj["stage"] = text("UNLOAD");
            evu.obj["block_index"] = number(blocks[i].index);
            evu.obj["t_ms"] = number(now_ms());
            evu.obj["duration_ms"] = number(ums);
            evu.obj["cache_size_before"] = number(static_cast<double>(erased));
            evu.obj["cache_size_after"] = number(0);
            events.push_back(evu);
            delete resident_caches[i];
        }
    }

    // Order validation: streaming must show strict LOAD,EXECUTE,UNLOAD per block
    // with no LOAD B before UNLOAD A.  Resident mode loads all blocks up front
    // (diagnostic), so the strict per-block interleave applies only to streaming.
    if (!resident)
    {
        bool seen_unload = true;
        int expected_block = -1;
        for (const auto& ev : events)
        {
            const std::string stage = ev.obj.at("stage").str;
            const int bi = static_cast<int>(ev.obj.at("block_index").num);
            if (stage == "LOAD")
            {
                if (!seen_unload) { order_ok = false; break; }
                seen_unload = false;
                expected_block = bi;
            }
            else if (stage == "EXECUTE")
            {
                if (expected_block != bi) { order_ok = false; break; }
            }
            else if (stage == "UNLOAD")
            {
                if (expected_block != bi || seen_unload) { order_ok = false; break; }
                seen_unload = true;
            }
        }
        if (!seen_unload) order_ok = false;
    }
    JVal events_json; events_json.kind = JVal::Array; events_json.arr = events;
    JVal per_block_json; per_block_json.kind = JVal::Array; per_block_json.arr = per_block;
    report.obj["events"] = events_json;
    report.obj["per_block"] = per_block_json;
    report.obj["order_ok"] = boolean(order_ok);
    report.obj["max_resident_weighted_blocks"] = number(resident ? static_cast<double>(blocks.size()) : 1.0);
    report.obj["total_load_ms"] = number(static_cast<double>(total_load_ms));
    report.obj["total_execute_ms"] = number(static_cast<double>(total_execute_ms));
    report.obj["total_unload_ms"] = number(static_cast<double>(total_unload_ms));
    report.obj["total_wall_ms"] = number(now_ms());
    report.obj["ncnn_adapter_call_count"] = number(static_cast<double>(adapter_call_count));
    report.obj["original_block_call_count"] = number(0);
    report.obj["silent_fallback"] = boolean(false);
    report.obj["streaming_mode_validated"] = boolean(!resident && order_ok);
    report.obj["full_32_block_ncnn"] = boolean(false);
    report.obj["full_pipeline_pytorch"] = boolean(true);
    report.obj["android_vulkan"] = boolean(false);

    if (!report_path.empty())
    {
        std::ofstream f(report_path);
        if (f) f << phase12_test::to_json(report);
    }
    return order_ok ? 0 : 3;
}

} // namespace

int main(int argc, char** argv)
{
    fs::path root, input_dir, output_dir, report_path, trace_dir;
    std::string blocks_arg;
    bool resident = false, memory_profile = false, probe_files = true;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        const auto next = [&](const char* n) -> std::string {
            if (i + 1 >= argc) { std::cerr << "missing value for " << n << '\n'; std::exit(2); }
            return argv[++i];
        };
        if (a == "--root") root = next("--root");
        else if (a == "--blocks") blocks_arg = next("--blocks");
        else if (a == "--input-dir") input_dir = next("--input-dir");
        else if (a == "--output-dir") output_dir = next("--output-dir");
        else if (a == "--load-mode") { const std::string m = next("--load-mode"); if (m != "streaming" && m != "resident") { std::cerr << "invalid load-mode " << m << '\n'; return usage(); } resident = (m == "resident"); }
        else if (a == "--report") report_path = next("--report");
        else if (a == "--trace-dir") trace_dir = next("--trace-dir");
        else if (a == "--memory-profile") memory_profile = true;
        else if (a == "--no-file-probe") probe_files = false;
        else { std::cerr << "unknown arg " << a << '\n'; return usage(); }
    }
    if (root.empty() || blocks_arg.empty() || input_dir.empty() || output_dir.empty()) return usage();

    std::vector<BlockSpec> blocks;
    std::stringstream ss(blocks_arg);
    std::string item;
    while (std::getline(ss, item, ','))
    {
        const auto colon = item.find(':');
        if (colon == std::string::npos) { std::cerr << "bad block spec: " << item << '\n'; return usage(); }
        blocks.push_back({std::atoi(item.substr(0, colon).c_str()), item.substr(colon + 1)});
    }
    if (blocks.empty()) { std::cerr << "no blocks\n"; return usage(); }

    return run_chain(blocks, root, input_dir, output_dir, report_path, trace_dir,
                     resident, memory_profile, probe_files);
}
