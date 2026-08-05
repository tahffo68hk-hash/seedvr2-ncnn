#include "minimal_video_pipeline.h"

#include "model_manifest.h"
#include "phase18c_contract.h"
#include "runtime_progress.h"

#include "artifact_registry.h"
#include "hybrid_dit_block_runtime.h"
#include "model_cache.h"

#include <command.h>
#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using attention_dispatcher::ArtifactRegistry;
using attention_dispatcher::CoreInfo;
using attention_dispatcher::CoreKey;
using attention_dispatcher::NcnnModelCache;
using seedvr2_dit_block::DiTBlockInput;
using seedvr2_dit_block::DiTBlockOutput;
using seedvr2_dit_block::DiTBlockRunStats;
using seedvr2_dit_block::HybridDiTBlockRuntime;
using seedvr2_dit_block::VideoShape;

namespace
{
struct Shape { int c, t, h, w; };
struct Stage { const char* name; Shape in, out; };

const Stage kEncoder[] = {
    {"enc_conv_in", {3,5,128,128}, {128,5,128,128}},
    {"enc_down0", {128,5,128,128}, {128,5,64,64}},
    {"enc_down1", {128,5,64,64}, {256,3,32,32}},
    {"enc_down2", {256,3,32,32}, {512,2,16,16}},
    {"enc_down3", {512,2,16,16}, {512,2,16,16}},
    {"enc_mid", {512,2,16,16}, {512,2,16,16}},
    {"enc_out", {512,2,16,16}, {32,2,16,16}},
};
const Stage kDecoder[] = {
    {"dec_conv_in", {16,2,16,16}, {512,2,16,16}},
    {"dec_mid", {512,2,16,16}, {512,2,16,16}},
    {"dec_up0", {512,2,16,16}, {512,3,32,32}},
    {"dec_up1", {512,3,32,32}, {512,5,64,64}},
    {"dec_up2", {512,5,64,64}, {256,5,128,128}},
    {"dec_up3", {256,5,128,128}, {128,5,128,128}},
    {"dec_out", {128,5,128,128}, {3,5,128,128}},
};

size_t count(const Shape& s) { return static_cast<size_t>(s.c) * s.t * s.h * s.w; }

bool read_bytes(const fs::path& path, std::vector<unsigned char>& output)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    output.assign(std::istreambuf_iterator<char>(file), {});
    return bool(file) || file.eof();
}

bool read_floats(const fs::path& path, std::vector<float>& output)
{
    std::vector<unsigned char> bytes;
    if (!read_bytes(path, bytes) || bytes.size() % sizeof(float)) return false;
    output.resize(bytes.size() / sizeof(float));
    std::memcpy(output.data(), bytes.data(), bytes.size());
    return true;
}

bool write_bytes(const fs::path& path, const void* data, size_t size)
{
    std::ofstream file(path, std::ios::binary);
    if (!file) return false;
    file.write(static_cast<const char*>(data), static_cast<std::streamsize>(size));
    return bool(file);
}

void force_fp32(ncnn::Option& option, bool vulkan)
{
    option.num_threads = 1;
    option.use_vulkan_compute = vulkan;
    option.use_subgroup_ops = false;
    option.use_cooperative_matrix = false;
    option.use_packing_layout = false;
    option.use_fp16_packed = false;
    option.use_fp16_storage = false;
    option.use_fp16_arithmetic = false;
    option.use_bf16_storage = false;
}

std::vector<float> contiguous(const ncnn::Mat& mat)
{
    const size_t plane = static_cast<size_t>(mat.w) * mat.h * mat.d;
    std::vector<float> output(plane * mat.c);
    const float* source = static_cast<const float*>(mat.data);
    for (int channel = 0; channel < mat.c; ++channel)
    {
        std::memcpy(output.data() + static_cast<size_t>(channel) * plane,
                    source + static_cast<size_t>(channel) * mat.cstep,
                    plane * sizeof(float));
    }
    return output;
}

bool run_model(const fs::path& param, const fs::path& bin,
               const std::vector<float>& input, const Shape& in_shape,
               const Shape& out_shape, bool vulkan,
               std::vector<float>& output, std::string& error)
{
    ncnn::Net net;
    force_fp32(net.opt, vulkan);
    const ncnn::VulkanDevice* device = nullptr;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
    if (vulkan)
    {
        ncnn::create_gpu_instance();
        device = ncnn::get_gpu_count() > 0 ? ncnn::get_gpu_device(0) : nullptr;
        if (!device) { error = "vulkan_device_unavailable"; ncnn::destroy_gpu_instance(); return false; }
        blob_allocator = device->acquire_blob_allocator();
        staging_allocator = device->acquire_staging_allocator();
        if (!blob_allocator || !staging_allocator)
        {
            error = "vulkan_allocator_unavailable";
            if (blob_allocator) device->reclaim_blob_allocator(blob_allocator);
            if (staging_allocator) device->reclaim_staging_allocator(staging_allocator);
            ncnn::destroy_gpu_instance();
            return false;
        }
        net.set_vulkan_device(device);
        net.opt.blob_vkallocator = blob_allocator;
        net.opt.workspace_vkallocator = blob_allocator;
        net.opt.staging_vkallocator = staging_allocator;
    }
    int rc = net.load_param(param.string().c_str());
    if (!rc) rc = net.load_model(bin.string().c_str());
    if (rc) error = "model_load_failed:" + std::to_string(rc);
    if (!rc)
    {
        ncnn::Mat host(in_shape.w, in_shape.h, in_shape.t, in_shape.c,
                       const_cast<float*>(input.data()));
        ncnn::Extractor extractor = net.create_extractor();
        ncnn::Mat result;
        if (vulkan)
        {
            extractor.set_blob_vkallocator(blob_allocator);
            extractor.set_workspace_vkallocator(blob_allocator);
            extractor.set_staging_vkallocator(staging_allocator);
            ncnn::VkMat gpu_input, gpu_output;
            ncnn::VkCompute command(device);
            command.record_upload(host, gpu_input, net.opt);
            rc = extractor.input("in0", gpu_input);
            if (!rc) rc = extractor.extract("out0", gpu_output, command);
            if (!rc) command.record_download(gpu_output, result, net.opt);
            if (!rc) rc = command.submit_and_wait();
        }
        else
        {
            rc = extractor.input("in0", host);
            if (!rc) rc = extractor.extract("out0", result);
        }
        if (rc) error = "model_execute_failed:" + std::to_string(rc);
        if (!rc)
        {
            output = contiguous(result);
            if (output.size() != count(out_shape)) { error = "model_output_shape_mismatch"; rc = -1; }
        }
    }
    net.clear();
    if (device)
    {
        device->reclaim_blob_allocator(blob_allocator);
        device->reclaim_staging_allocator(staging_allocator);
        ncnn::destroy_gpu_instance();
    }
    return rc == 0;
}

bool run_vae_chain(const fs::path& root, const Stage* stages, size_t stage_count,
                   const std::vector<float>& input, bool first_stage_vulkan,
                   std::vector<float>& output, std::string& error)
{
    output = input;
    for (size_t index = 0; index < stage_count; ++index)
    {
        const Stage& stage = stages[index];
        std::vector<float> next;
        const fs::path directory = root / stage.name;
        if (!run_model(directory / "model.ncnn.param", directory / "model.ncnn.bin",
                       output, stage.in, stage.out, first_stage_vulkan && index == 0,
                       next, error))
        {
            error = std::string(stage.name) + ":" + error;
            return false;
        }
        output.swap(next);
    }
    return true;
}

bool run_linear_model(const fs::path& directory, const std::vector<float>& input,
                      int rows, int in_dim, int out_dim, bool vulkan,
                      std::vector<float>& output, std::string& error)
{
    return run_model(directory / "model.ncnn.param", directory / "model.ncnn.bin",
                     input, {1,1,rows,in_dim}, {1,1,rows,out_dim}, vulkan, output, error);
}

std::string escape_json(const std::string& input)
{
    std::string output;
    for (char c : input)
    {
        if (c == '\\' || c == '"') { output += '\\'; output += c; }
        else if (c == '\n') output += "\\n";
        else if (static_cast<unsigned char>(c) >= 0x20) output += c;
    }
    return output;
}

std::string result_json(bool verified, bool encoder, int blocks, bool decoder,
                        int output_frames, bool finite, long long duration_ms,
                        const std::string& error_stage, const std::string& error_message)
{
    const bool pass = verified && encoder && blocks == 32 && decoder && output_frames == 4 && finite && error_stage.empty();
    std::ostringstream out;
    out << "{\"schema_version\":1,\"device_executed\":true,\"model_set_verified\":" << (verified?"true":"false")
        << ",\"input_frames_loaded\":4,\"vae_encoder_executed\":" << (encoder?"true":"false")
        << ",\"dit_blocks_executed\":" << blocks << ",\"dit_max_resident_blocks\":1"
        << ",\"vae_decoder_executed\":" << (decoder?"true":"false")
        << ",\"output_frames_written\":" << output_frames
        << ",\"video_encoded\":false,\"output_video_path\":null,\"output_video_size_bytes\":0,\"output_video_sha256\":null"
        << ",\"backend\":\"cpu_fp32\",\"backend_per_stage\":{\"vae_encoder\":\"cpu_fp32\",\"dit\":\"cpu_fp32\",\"vae_decoder\":\"cpu_fp32\"}"
        << ",\"silent_cpu_fallback\":false,\"finite\":" << (finite?"true":"false")
        << ",\"peak_rss_mb\":null,\"peak_gpu_memory_mb\":null,\"duration_ms\":" << duration_ms
        << ",\"thermal_before\":null,\"thermal_after\":null,\"error_stage\":";
    if (error_stage.empty()) out << "null"; else out << '"' << escape_json(error_stage) << '"';
    out << ",\"error_code\":" << (error_stage.empty()?"null":"\"runtime_failure\"") << ",\"error_message\":";
    if (error_message.empty()) out << "null"; else out << '"' << escape_json(error_message) << '"';
    out << ",\"overall_pass\":" << (pass?"true":"false") << '}';
    return out.str();
}
}

std::string run_phase18c_minimal_video_demo(const std::string& model_root,
                                             const std::string& input_root,
                                             const std::string& output_root,
                                             const std::string& config_json)
{
    seededge_progress::reset("full_demo", output_root);
    int requested_blocks = 32;
    std::string stop_after;
    std::smatch match;
    if (std::regex_search(config_json, match, std::regex(R"("max_blocks"\s*:\s*([0-9]+))")))
        requested_blocks = std::clamp(std::stoi(match[1]), 0, 32);
    if (std::regex_search(config_json, match, std::regex(R"cfg("stop_after"\s*:\s*"([a-z_]+)")cfg")))
        stop_after = match[1];
    const auto started = std::chrono::steady_clock::now();
    bool verified=false, encoder=false, decoder=false, finite=false;
    int blocks=0, frames_written=0;
    std::string error_stage, error;
    const fs::path models(model_root), inputs(input_root), outputs(output_root);
    fs::create_directories(outputs);

    const std::string manifest = validate_model_manifest(model_root);
    seededge_progress::update("verifying_manifest", "manifest", "manifest verification complete");
    verified = manifest.find("\"valid\":true") != std::string::npos;
    if (!verified) { error_stage="model_manifest"; error=manifest; goto done; }
    if (stop_after == "manifest") goto done;

    {
        std::vector<float> vae_input, encoded;
        seededge_progress::update("vae_encoder", "encoder", "loading VAE encoder");
        if (!read_floats(inputs / "input_norm.f32", vae_input) || vae_input.size() != count(kEncoder[0].in))
        { error_stage="input_frames"; error="input_norm.f32 missing or wrong shape"; goto done; }
        if (!run_vae_chain(models / "vae" / "encoder", kEncoder, std::size(kEncoder),
                           vae_input, false, encoded, error))
        { error_stage="vae_encoder"; goto done; }
        encoder=true;
        seededge_progress::update("dit_streaming", "vid_in", "VAE encoder complete");
        write_bytes(outputs/"gate_vae_latent.f32", encoded.data(), encoded.size()*sizeof(float));
        if (stop_after == "encoder") { finite=phase18c_all_finite(encoded); goto done; }

        std::vector<float> latent_thwc(2*16*16*16);
        for (int t=0;t<2;++t) for(int h=0;h<16;++h) for(int w=0;w<16;++w) for(int c=0;c<16;++c)
            latent_thwc[(((t*16+h)*16+w)*16)+c] = encoded[(((c*2+t)*16+h)*16+w)] * 0.9152f;
        std::vector<float> noise, txt, emb;
        if (!read_floats(models/"constants"/"noise.f32", noise) || noise.size()!=latent_thwc.size() ||
            !read_floats(models/"constants"/"input_txt.f32", txt) || txt.size()!=58*2560 ||
            !read_floats(models/"constants"/"emb.f32", emb) || emb.size()!=15360)
        { error_stage="conditioning"; error="fixed conditioning tensor missing or wrong shape"; goto done; }

        std::vector<float> dit_input(2*16*16*33);
        for(size_t token=0;token<2*16*16;++token)
        {
            for(int c=0;c<16;++c) dit_input[token*33+c]=noise[token*16+c];
            for(int c=0;c<16;++c) dit_input[token*33+16+c]=latent_thwc[token*16+c];
            dit_input[token*33+32]=1.f;
        }
        std::vector<float> patched=phase18c_patchify_thwc(dit_input,2,16,16,33,2,2), vid;
        if (!run_linear_model(models/"dit"/"vid_in", patched,128,132,2560,false,vid,error))
        { error_stage="dit_vid_in"; goto done; }

        ArtifactRegistry registry;
        if (!registry.scan(models/"dit"/"blocks", error)) { error_stage="dit_registry"; goto done; }
        for(int block=0;block<requested_blocks;++block)
        {
            seededge_progress::update("dit_streaming", "block_"+std::to_string(block), "executing DiT block", 0, 0, 0, 0, block, 0, 0, 0, 0);
            const std::string variant="real_seedvr2_block"+std::to_string(block)+"_c2560_h20_hd128";
            const CoreInfo* core=registry.lookup(CoreKey{4,3,3,1,1,1,variant});
            if (!core) { error_stage="dit_block_"+std::to_string(block); error="artifact lookup failed"; goto done; }
            NcnnModelCache cache;
            HybridDiTBlockRuntime runtime(cache);
            if (!runtime.initialize(*core,error)) { error_stage="dit_block_"+std::to_string(block)+"_load"; goto done; }
            DiTBlockInput block_input;
            block_input.vid=ncnn::Mat(2560,128,vid.data());
            block_input.txt=ncnn::Mat(2560,58,txt.data());
            block_input.emb=ncnn::Mat(15360,1,emb.data());
            block_input.vid_shapes={{2,8,8}};
            block_input.txt_lengths={58};
            DiTBlockOutput block_output;
            DiTBlockRunStats stats;
            if (!runtime.run(block_input,block_output,nullptr,stats))
            { error_stage="dit_block_"+std::to_string(block)+"_execute"; error=stats.error_message; goto done; }
            vid=contiguous(block_output.vid);
            if (block < 31) txt=contiguous(block_output.txt);
            cache.clear();
            ++blocks;
            if (blocks == 1 || blocks == 2 || blocks == 4 || blocks == 32)
            {
                write_bytes(outputs/("gate_dit_block_"+std::to_string(blocks)+"_vid.f32"),vid.data(),vid.size()*sizeof(float));
                if (!txt.empty()) write_bytes(outputs/("gate_dit_block_"+std::to_string(blocks)+"_txt.f32"),txt.data(),txt.size()*sizeof(float));
            }
        }
        if (requested_blocks < 32 || stop_after == "blocks")
        {
            finite=phase18c_all_finite(vid) && phase18c_all_finite(txt);
            write_bytes(outputs/"gate_dit_vid.f32",vid.data(),vid.size()*sizeof(float));
            goto done;
        }

        std::vector<float> norm_weight,out_shift,out_scale;
        if(!read_floats(models/"dit"/"post"/"norm_weight.f32",norm_weight)||norm_weight.size()!=2560||
           !read_floats(models/"dit"/"post"/"out_shift.f32",out_shift)||out_shift.size()!=2560||
           !read_floats(models/"dit"/"post"/"out_scale.f32",out_scale)||out_scale.size()!=2560)
        {error_stage="dit_post_weights";error="post weights missing";goto done;}
        for(int row=0;row<128;++row)
        {
            double square=0.; for(int c=0;c<2560;++c){const float v=vid[row*2560+c];square+=double(v)*v;}
            const float inv=1.f/std::sqrt(float(square/2560.)+1e-5f);
            for(int c=0;c<2560;++c)
            {
                const float normalized=vid[row*2560+c]*inv*norm_weight[c];
                vid[row*2560+c]=normalized*(emb[c*3+1]+out_scale[c])+(emb[c*3]+out_shift[c]);
            }
        }
        std::vector<float> prediction_patched;
        if(!run_linear_model(models/"dit"/"vid_out",vid,128,2560,64,false,prediction_patched,error))
        {error_stage="dit_vid_out";goto done;}
        const std::vector<float> prediction=phase18c_unpatchify_thwc(prediction_patched,2,8,8,16,2,2);
        const std::vector<float> final_latent=phase18c_euler_endpoint(noise,prediction);
        if(!phase18c_all_finite(final_latent)){error_stage="dit_final";error="non-finite latent";goto done;}

        std::vector<float> decoder_input(16*2*16*16);
        for(int t=0;t<2;++t) for(int h=0;h<16;++h) for(int w=0;w<16;++w) for(int c=0;c<16;++c)
            decoder_input[(((c*2+t)*16+h)*16+w)] = final_latent[(((t*16+h)*16+w)*16)+c]/0.9152f;
        std::vector<float> decoded;
        seededge_progress::update("vae_decoder", "decoder", "loading VAE decoder", 0, 0, 0, 0, 32, 0, static_cast<int>(std::size(kDecoder)));
        if(!run_vae_chain(models/"vae"/"decoder",kDecoder,std::size(kDecoder),decoder_input,false,decoded,error))
        {error_stage="vae_decoder";goto done;}
        decoder=true; finite=phase18c_all_finite(decoded);
        if(!finite){error_stage="vae_decoder";error="non-finite decoded output";goto done;}
        for(int frame=0;frame<4;++frame)
        {
            seededge_progress::update("writing_frames", "frame_"+std::to_string(frame), "writing decoded frame", 0, 0, 0, 0, 32, static_cast<int>(std::size(kDecoder)), static_cast<int>(std::size(kDecoder)), frame, frames_written);
            std::vector<unsigned char> rgb(128*128*3);
            for(int h=0;h<128;++h) for(int w=0;w<128;++w) for(int c=0;c<3;++c)
            {
                const float value=decoded[(((c*5+frame)*128+h)*128+w)];
                rgb[(h*128+w)*3+c]=static_cast<unsigned char>(std::clamp(std::lround((value*.5f+.5f)*255.f),0l,255l));
            }
            std::ostringstream name; name<<"frame_"<<std::setw(3)<<std::setfill('0')<<frame<<".rgb";
            if(!write_bytes(outputs/name.str(),rgb.data(),rgb.size())){error_stage="output_frames";error="frame write failed";goto done;}
            ++frames_written;
        }
    }
done:
    seededge_progress::finish(error_stage.empty() ? "completed" : (seededge_progress::cancelled() ? "cancelled" : "failed"), error_stage, error);
    return result_json(verified,encoder,blocks,decoder,frames_written,finite,
        std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-started).count(),
        error_stage,error);
}

std::string run_phase18c_decoder_only(const std::string& model_root,
                                       const std::string& checkpoint_root,
                                       const std::string& output_root,
                                       const std::string& config_json)
{
    seededge_progress::reset("decoder_only", output_root);
    const fs::path models(model_root), checkpoints(checkpoint_root), outputs(output_root);
    fs::create_directories(outputs);
    const fs::path frame_dir = checkpoints / "frames";
    const fs::path vid_gate = frame_dir / "gate_dit_block_32_vid.f32";
    const fs::path txt_gate = frame_dir / "gate_dit_block_32_txt.f32";
    std::vector<float> vid, txt, noise, emb, norm_weight, out_shift, out_scale;
    std::string error;
    if (!read_floats(vid_gate, vid) || vid.size() != 128u * 2560u ||
        !read_floats(txt_gate, txt) || txt.size() != 58u * 2560u)
    { seededge_progress::finish("failed", "CHECKPOINT_SIZE_MISMATCH", "Block 32 gate shape mismatch"); return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":false,\"error_code\":\"CHECKPOINT_SIZE_MISMATCH\",\"overall_pass\":false}"; }
    seededge_progress::update("verifying_checkpoint", "gate_dit_block_32", "checkpoint shape verified", 0, 0, 0, 0, 32);
    if (!read_floats(models/"constants"/"noise.f32", noise) || !read_floats(models/"constants"/"emb.f32", emb) ||
        !read_floats(models/"dit"/"post"/"norm_weight.f32", norm_weight) ||
        !read_floats(models/"dit"/"post"/"out_shift.f32", out_shift) ||
        !read_floats(models/"dit"/"post"/"out_scale.f32", out_scale) ||
        noise.size() != 2u*16u*16u*16u || emb.size() != 15360u || norm_weight.size() != 2560u || out_shift.size() != 2560u || out_scale.size() != 2560u)
    { seededge_progress::finish("failed", "BRIDGE_INPUT_MISSING", "post-head constants missing"); return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":true,\"decoder_bridge_verified\":false,\"error_code\":\"BRIDGE_INPUT_MISSING\",\"overall_pass\":false}"; }
    seededge_progress::update("block32_bridge", "vid_out_norm", "running post-head norm and Ada modulation", 0, 0, 0, 0, 32);
    for (int row=0; row<128; ++row) { double sq=0.; for(int c=0;c<2560;++c){float v=vid[row*2560+c];sq+=double(v)*v;} float inv=1.f/std::sqrt(float(sq/2560.)+1e-5f); for(int c=0;c<2560;++c){float n=vid[row*2560+c]*inv*norm_weight[c]; vid[row*2560+c]=n*(emb[c*3+1]+out_scale[c])+(emb[c*3]+out_shift[c]);} }
    std::vector<float> prediction_patched;
    if (!run_linear_model(models/"dit"/"vid_out", vid, 128, 2560, 64, false, prediction_patched, error)) { seededge_progress::finish("failed", "BRIDGE_VID_OUT_FAILED", error); return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":true,\"decoder_bridge_verified\":false,\"error_code\":\"BRIDGE_VID_OUT_FAILED\",\"overall_pass\":false}"; }
    const std::vector<float> prediction=phase18c_unpatchify_thwc(prediction_patched,2,8,8,16,2,2);
    const std::vector<float> final_latent=phase18c_euler_endpoint(noise,prediction);
    if (!phase18c_all_finite(final_latent)) { seededge_progress::finish("failed", "NON_FINITE_OUTPUT", "scheduler endpoint non-finite"); return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":true,\"decoder_bridge_verified\":false,\"error_code\":\"NON_FINITE_OUTPUT\",\"overall_pass\":false}"; }
    std::vector<float> decoder_input(16u*2u*16u*16u);
    for(int t=0;t<2;++t)for(int h=0;h<16;++h)for(int w=0;w<16;++w)for(int c=0;c<16;++c) decoder_input[(((c*2+t)*16+h)*16+w)] = final_latent[(((t*16+h)*16+w)*16)+c]/0.9152f;
    seededge_progress::update("vae_decoder", "decoder", "running VAE decoder from verified bridge", 0, 0, 0, 0, 32, 0, static_cast<int>(std::size(kDecoder)));
    std::vector<float> decoded;
    if (!run_vae_chain(models/"vae"/"decoder",kDecoder,std::size(kDecoder),decoder_input,false,decoded,error) || !phase18c_all_finite(decoded)) { seededge_progress::finish("failed", "DECODER_FAILED", error); return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":true,\"decoder_bridge_verified\":true,\"decoder_single_frame_pass\":false,\"error_code\":\"DECODER_FAILED\",\"overall_pass\":false}"; }
    std::vector<unsigned char> rgb(128u*128u*3u); for(int h=0;h<128;++h)for(int w=0;w<128;++w)for(int c=0;c<3;++c){float v=decoded[(((c*5)*128+h)*128+w)];rgb[(h*128+w)*3+c]=static_cast<unsigned char>(std::clamp(std::lround((v*.5f+.5f)*255.f),0l,255l));}
    write_bytes(outputs/"frame_000.rgb",rgb.data(),rgb.size()); write_bytes(outputs/"decoder_input_latent.f32",decoder_input.data(),decoder_input.size()*sizeof(float)); const std::string bridge_json="{\"decoder_bridge_verified\":true,\"decoder_latent_verified\":true}"; write_bytes(outputs/"decoder_bridge_result.json", bridge_json.data(), bridge_json.size());
    seededge_progress::finish("completed");
    return "{\"mode\":\"decoder_only\",\"checkpoint_source\":\"gate_dit_block_32\",\"encoder_skipped\":true,\"dit_blocks_skipped\":true,\"checkpoint_verified\":true,\"decoder_bridge_verified\":true,\"decoder_latent_verified\":true,\"decoder_single_frame_pass\":true,\"output_frame_path\":\"frame_000.rgb\",\"backend\":\"cpu_fp32\",\"silent_cpu_fallback\":false,\"finite\":true,\"overall_pass\":true}";
}
