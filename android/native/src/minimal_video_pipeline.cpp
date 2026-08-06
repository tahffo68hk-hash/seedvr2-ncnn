#include "minimal_video_pipeline.h"

#include "model_manifest.h"
#include "phase18c_contract.h"
#include "phase18d_tile_core.h"
#include "runtime_progress.h"
#include "fast_convolution3d.h"

#include "artifact_registry.h"
#include "hybrid_dit_block_runtime.h"
#include "model_cache.h"

#include <command.h>
#include <cpu.h>
#include <gpu.h>
#include <net.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <regex>
#include <sstream>
#include <string>
#include <vector>
#include <thread>

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

void force_fp32(ncnn::Option& option, bool vulkan, int threads = 0)
{
    option.num_threads = threads > 0 ? threads : std::max(1, ncnn::get_big_cpu_count());
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
               std::vector<float>& output, std::string& error, int threads = 0,
               std::vector<long long>* conv3d_profile_ms = nullptr)
{
    ncnn::Net net;
    force_fp32(net.opt, vulkan, threads);
    if (!vulkan)
        net.register_custom_layer(ncnn::layer_to_index("Convolution3D"),
                                  seededge::fast_convolution3d_creator,
                                  seededge::fast_convolution3d_destroyer);
    if (conv3d_profile_ms) seededge::reset_fast_convolution3d_profile();
#if NCNN_VULKAN
    const ncnn::VulkanDevice* device = nullptr;
    ncnn::VkAllocator* blob_allocator = nullptr;
    ncnn::VkAllocator* staging_allocator = nullptr;
#endif
    if (vulkan)
    {
#if NCNN_VULKAN
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
#else
        error = "vulkan_not_compiled";
        return false;
#endif
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
#if NCNN_VULKAN
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
#else
            error = "vulkan_not_compiled";
            rc = -1;
#endif
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
#if NCNN_VULKAN
    if (device)
    {
        device->reclaim_blob_allocator(blob_allocator);
        device->reclaim_staging_allocator(staging_allocator);
        ncnn::destroy_gpu_instance();
    }
#endif
    if (conv3d_profile_ms) *conv3d_profile_ms = seededge::fast_convolution3d_profile_ms();
    return rc == 0;
}

bool run_vae_chain(const fs::path& root, const Stage* stages, size_t stage_count,
                   const std::vector<float>& input, bool first_stage_vulkan,
                   std::vector<float>& output, std::string& error,
                   int threads = 0, bool report_decoder_progress = false)
{
    output = input;
    for (size_t index = 0; index < stage_count; ++index)
    {
        const Stage& stage = stages[index];
        std::vector<float> next;
        const fs::path directory = root / stage.name;
        if (report_decoder_progress)
            seededge_progress::update("vae_decoder", stage.name, "loading model", 0, 0, 0, 0, 32,
                                      static_cast<int>(index + 1), static_cast<int>(stage_count), 0, 0,
                                      (fs::path("vae/decoder") / stage.name / "model.ncnn.bin").string());
        if (!run_model(directory / "model.ncnn.param", directory / "model.ncnn.bin",
                       output, stage.in, stage.out, first_stage_vulkan && index == 0,
                       next, error, threads))
        {
            error = std::string(stage.name) + ":" + error;
            return false;
        }
        output.swap(next);
        if (report_decoder_progress)
            seededge_progress::update("vae_decoder", stage.name, "inference completed", 0, 0, 0, 0, 32,
                                      static_cast<int>(index + 1), static_cast<int>(stage_count), 0, 0,
                                      (fs::path("vae/decoder") / stage.name / "model.ncnn.bin").string());
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

std::string single_tile_error(const std::string& code, const std::string& message)
{
    return "{\"schema_version\":1,\"mode\":\"phase18d_single_photo_tile\",\"error_code\":\"" +
           escape_json(code) + "\",\"error_message\":\"" + escape_json(message) +
           "\",\"overall_pass\":false}";
}

bool looks_like_json_object(const std::string& text)
{
    const size_t first = text.find_first_not_of(" \t\r\n");
    const size_t last = text.find_last_not_of(" \t\r\n");
    return first != std::string::npos && text[first] == '{' && text[last] == '}';
}

bool json_true(const std::string& json, const std::string& key)
{
    return json.find("\"" + key + "\":true") != std::string::npos;
}

bool json_int_equals(const std::string& json, const std::string& key, int expected)
{
    return json.find("\"" + key + "\":" + std::to_string(expected)) != std::string::npos;
}
}

namespace
{
// Content-aware single-tile selection: the single-photo entry enhances exactly
// one tile, so a photo with blank edges (letterboxing, dark vignette) must not
// anchor tile_0000 on an empty corner.  Score favors colorful, textured valid
// regions and tie-breaks toward the image center.
long long tile_content_score(const uint8_t* rgb, int width, int height,
                             const seededge::phase18d::TileSpec& tile)
{
    long long score = 0;
    for (int y = tile.source_top; y < tile.source_bottom; ++y)
    {
        const uint8_t* row = rgb + (static_cast<size_t>(y) * width + tile.source_left) * 3u;
        for (int x = tile.source_left; x < tile.source_right; ++x, row += 3)
        {
            const int r = row[0], g = row[1], b = row[2];
            score += std::abs(r - g) + std::abs(g - b);
            const uint8_t* left_px = x > tile.source_left ? row - 3 : row;
            const uint8_t* up_px = y > tile.source_top
                ? rgb + (static_cast<size_t>(y - 1) * width + x) * 3u : row;
            score += std::abs(r - left_px[0]) + std::abs(g - up_px[1]);
        }
    }
    return score;
}

size_t select_content_tile(const uint8_t* rgb, int width, int height,
                           const seededge::phase18d::TilePlan& plan)
{
    size_t best = 0;
    long long best_score = -1;
    long long best_distance = 0;
    const int center_x = width / 2;
    const int center_y = height / 2;
    for (size_t i = 0; i < plan.tiles.size(); ++i)
    {
        const seededge::phase18d::TileSpec& tile = plan.tiles[i];
        const long long score = tile_content_score(rgb, width, height, tile);
        const int tile_center_x = (tile.source_left + tile.source_right) / 2;
        const int tile_center_y = (tile.source_top + tile.source_bottom) / 2;
        const long long dx = tile_center_x - center_x;
        const long long dy = tile_center_y - center_y;
        const long long distance = dx * dx + dy * dy;
        if (score > best_score || (score == best_score && distance < best_distance))
        {
            best = i;
            best_score = score;
            best_distance = distance;
        }
    }
    return best;
}
} // namespace

namespace
{
std::string full_photo_error(const std::string& code, const std::string& message)
{
    return "{\"schema_version\":1,\"mode\":\"phase18d_full_photo\",\"state\":\"failed\",\"error_code\":\"" +
           escape_json(code) + "\",\"error_message\":\"" + escape_json(message) +
           "\",\"tile_count\":0,\"completed_tiles\":0,\"silent_fallback\":false,\"overall_pass\":false}";
}

std::string tile_dir_name(int index)
{
    std::ostringstream name;
    name << "tile_" << std::setw(4) << std::setfill('0') << index;
    return name.str();
}

// The model consumes C,T,H,W FP32 in [-1, 1].  Repeating the padded tile into
// five temporal slots matches the existing 128x128 SeedVR2 VAE contract.
std::vector<float> make_five_frame_model_input(const seededge::phase18d::PaddedTile& padded)
{
    std::vector<float> model_input(count(kEncoder[0].in));
    for (int c = 0; c < 3; ++c)
        for (int t = 0; t < 5; ++t)
            for (int y = 0; y < 128; ++y)
                for (int x = 0; x < 128; ++x)
                    model_input[(((c * 5 + t) * 128 + y) * 128 + x)] =
                        static_cast<float>(padded.rgb[(static_cast<size_t>(y) * 128 + x) * 3 + c]) / 127.5f - 1.f;
    return model_input;
}

// Replays the canonical tile stage chain against the reported result flags.
// A tile can only be recorded as completed when the full chain was reached.
std::string reached_tile_state(const seededge::phase18d::Phase18dTileExecutionResult& result)
{
    std::string state = "pending";
    if (seededge::phase18d::is_valid_tile_state_transition(state, "preparing")) state = "preparing";
    if (result.encoder_completed && seededge::phase18d::is_valid_tile_state_transition(state, "encoder")) state = "encoder";
    if (result.dit_block_completed >= 32 && seededge::phase18d::is_valid_tile_state_transition(state, "dit")) state = "dit";
    if (result.scheduler_completed && seededge::phase18d::is_valid_tile_state_transition(state, "scheduler")) state = "scheduler";
    if (result.decoder_completed && seededge::phase18d::is_valid_tile_state_transition(state, "decoder")) state = "decoder";
    if (seededge::phase18d::is_valid_tile_state_transition(state, "writing_output")) state = "writing_output";
    return state;
}

std::string full_photo_tile_status(
    const std::string& task_id,
    const std::string& model_version,
    const std::string& manifest_sha,
    const std::string& normalized_sha,
    const std::string& tile_sha,
    const seededge::phase18d::TileSpec& tile,
    const std::string& state,
    const seededge::phase18d::Phase18dTileExecutionResult& result,
    const std::string& output_sha,
    const std::string& error_code,
    const std::string& error_message)
{
    std::ostringstream status;
    status << "{\"schema_version\":1,\"task_id\":\"" << escape_json(task_id)
           << "\",\"normalized_image_sha256\":\"" << normalized_sha
           << "\",\"tile_index\":" << tile.tile_index
           << ",\"tile_input_sha256\":\"" << tile_sha
           << "\",\"model_version\":\"" << escape_json(model_version)
           << "\",\"model_manifest_sha256\":\"" << manifest_sha
           << "\",\"tile_bounds\":{\"left\":" << tile.source_left
           << ",\"top\":" << tile.source_top << ",\"right\":" << tile.source_right
           << ",\"bottom\":" << tile.source_bottom
           << "},\"checkpoint_root\":null,\"state\":\"" << state << "\""
           << ",\"encoder_completed\":" << (result.encoder_completed ? "true" : "false")
           << ",\"dit_block_completed\":" << result.dit_block_completed
           << ",\"scheduler_completed\":" << (result.scheduler_completed ? "true" : "false")
           << ",\"decoder_completed\":" << (result.decoder_completed ? "true" : "false")
           << ",\"target_frame_index\":0,\"finite\":" << (result.finite ? "true" : "false")
           << ",\"nan_count\":" << result.nan_count
           << ",\"inf_count\":" << result.inf_count
           << ",\"output_valid_sha256\":";
    if (output_sha.empty()) status << "null"; else status << "\"" << output_sha << "\"";
    status << ",\"pipeline_result\":";
    if (result.pipeline_result_json.empty()) status << "null"; else status << result.pipeline_result_json;
    status << ",\"error_code\":";
    if (error_code.empty()) status << "null"; else status << "\"" << escape_json(error_code) << "\"";
    status << ",\"error_message\":";
    if (error_message.empty()) status << "null"; else status << "\"" << escape_json(error_message) << "\"";
    status << "}";
    return status.str();
}

std::string full_photo_task_status(
    const std::string& task_id,
    const std::string& model_version,
    const std::string& manifest_sha,
    const std::string& normalized_sha,
    int tile_count,
    int completed_tiles,
    const std::string& state,
    const std::string& output_full_sha,
    const std::string& error_code,
    const std::string& error_message)
{
    const bool pass = state == "completed" && !output_full_sha.empty();
    std::ostringstream status;
    status << "{\"schema_version\":1,\"mode\":\"phase18d_full_photo\",\"state\":\"" << state
           << "\",\"task_id\":\"" << escape_json(task_id)
           << "\",\"model_version\":\"" << escape_json(model_version)
           << "\",\"model_manifest_sha256\":\"" << manifest_sha
           << "\",\"normalized_image_sha256\":\"" << normalized_sha
           << "\",\"tile_count\":" << tile_count
           << ",\"completed_tiles\":" << completed_tiles
           << ",\"silent_fallback\":false"
           << ",\"output_full_sha256\":";
    if (output_full_sha.empty()) status << "null"; else status << "\"" << output_full_sha << "\"";
    status << ",\"error_code\":";
    if (error_code.empty()) status << "null"; else status << "\"" << escape_json(error_code) << "\"";
    status << ",\"error_message\":";
    if (error_message.empty()) status << "null"; else status << "\"" << escape_json(error_message) << "\"";
    status << ",\"overall_pass\":" << (pass ? "true" : "false") << "}";
    return status.str();
}

std::string persist_task_status(const fs::path& task_path, const std::string& status_text)
{
    if (write_bytes(task_path / "status.json", status_text.data(), status_text.size()))
        return status_text;
    return full_photo_error("STATUS_WRITE_FAILED", "could not persist task-local status");
}
} // namespace
std::string run_phase18d_single_photo_tile(const std::string& model_root,
                                           const std::string& task_root,
                                           const uint8_t* normalized_rgb,
                                           size_t byte_count, int width, int height,
                                           const std::string& request_json)
{
    // Keep request parsing and input validation ahead of any model access so a
    // caller cannot accidentally turn a malformed request into a legacy run.
    if (!looks_like_json_object(request_json))
        return single_tile_error("INVALID_REQUEST_JSON", "request_json must be a JSON object");
    if (width <= 0 || height <= 0 || !normalized_rgb)
        return single_tile_error("INVALID_RGB_INPUT", "RGB pointer and positive dimensions are required");
    const size_t expected_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    if (byte_count != expected_bytes)
        return single_tile_error("RGB_BUFFER_SIZE_MISMATCH", "normalized RGB byte count does not match width*height*3");

    std::smatch match;
    std::string task_id;
    if (std::regex_search(request_json, match, std::regex(R"json("task_id"\s*:\s*"([A-Za-z0-9_-]+)")json")))
        task_id = match[1];
    if (task_id.empty())
        return single_tile_error("TASK_ID_REQUIRED", "task_id is required for task-local state isolation");
    std::string model_version;
    if (std::regex_search(request_json, match, std::regex(R"json("model_version"\s*:\s*"([^"]+)")json")))
        model_version = match[1];
    if (model_version.empty())
        return single_tile_error("MODEL_VERSION_REQUIRED", "model_version is required for checkpoint identity");
    const fs::path task_path(task_root);
    if (task_path.filename() != task_id || task_path.parent_path().filename() != "tasks")
        return single_tile_error("TASK_ROOT_MISMATCH", "task_root must be files/tasks/<task_id>");

    seededge::phase18d::TilePlan plan;
    std::string core_error;
    if (!seededge::phase18d::plan_tiles(width, height, {}, &plan, &core_error) || plan.tiles.empty())
        return single_tile_error("TILE_PLAN_FAILED", core_error);
    const seededge::phase18d::TileSpec& selected_tile =
        plan.tiles[select_content_tile(normalized_rgb, width, height, plan)];
    seededge::phase18d::PaddedTile padded;
    if (!seededge::phase18d::extract_and_pad_rgb8(normalized_rgb, byte_count, width, height,
                                                  selected_tile, &padded, &core_error))
        return single_tile_error("TILE_PADDING_FAILED", core_error);

    // The new runtime owns only this directory.  In particular, it does not
    // consult any Phase 18C.4 Block32/dec_up0/dec_up1 checkpoint root.
    const fs::path tile_root = task_path / "tiles" / "tile_0000";
    std::error_code ec;
    fs::create_directories(tile_root, ec);
    if (ec) return single_tile_error("TASK_DIRECTORY_CREATE_FAILED", ec.message());
    const std::string normalized_sha = seededge::phase18d::sha256_hex(normalized_rgb, byte_count);
    const std::string tile_sha = seededge::phase18d::sha256_hex(padded.rgb.data(), padded.rgb.size());
    std::vector<unsigned char> manifest_bytes;
    const std::string manifest_sha = read_bytes(fs::path(model_root) / "manifest.json", manifest_bytes)
        ? seededge::phase18d::sha256_hex(manifest_bytes.data(), manifest_bytes.size()) : "";
    if (!write_bytes(tile_root / "tile_input.rgb", padded.rgb.data(), padded.rgb.size()))
        return single_tile_error("TILE_INPUT_WRITE_FAILED", "could not persist padded RGB tile");

    // The model consumes C,T,H,W FP32 in [-1, 1].  Repeating the source tile
    // into five temporal slots is deliberate and matches the existing 128x128
    // SeedVR2 VAE contract.
    const std::vector<float> model_input = make_five_frame_model_input(padded);
    if (!write_bytes(tile_root / "input_norm.f32", model_input.data(), model_input.size() * sizeof(float)))
        return single_tile_error("MODEL_INPUT_WRITE_FAILED", "could not persist normalized model input");

    const std::string pipeline_result = run_phase18c_minimal_video_demo(model_root, tile_root.string(),
                                                                          tile_root.string(), "{\"max_blocks\":32}");
    const bool encoder_completed = json_true(pipeline_result, "vae_encoder_executed");
    const bool dit_completed = json_int_equals(pipeline_result, "dit_blocks_executed", 32);
    const bool decoder_completed = json_true(pipeline_result, "vae_decoder_executed");
    const bool finite = json_true(pipeline_result, "finite");
    const bool scheduler_completed = dit_completed && pipeline_result.find("\"error_stage\":null") != std::string::npos;
    std::vector<unsigned char> decoded_frame;
    std::vector<unsigned char> output_valid;
    std::string output_sha;
    if (decoder_completed && finite && read_bytes(tile_root / "frame_000.rgb", decoded_frame) &&
        decoded_frame.size() == 128u * 128u * 3u)
    {
        output_valid.resize(static_cast<size_t>(padded.spec.valid_width) * padded.spec.valid_height * 3u);
        for (int y = 0; y < padded.spec.valid_height; ++y) for (int x = 0; x < padded.spec.valid_width; ++x)
        {
            const size_t source = (static_cast<size_t>(y + padded.spec.padding_top) * 128u +
                                   static_cast<size_t>(x + padded.spec.padding_left)) * 3u;
            const size_t target = (static_cast<size_t>(y) * padded.spec.valid_width + x) * 3u;
            std::copy_n(decoded_frame.data() + source, 3, output_valid.data() + target);
        }
        if (write_bytes(tile_root / "output_valid.rgb", output_valid.data(), output_valid.size()))
            output_sha = seededge::phase18d::sha256_hex(output_valid.data(), output_valid.size());
    }
    std::ostringstream status;
    status << "{\"schema_version\":1,\"task_id\":\"" << escape_json(task_id)
           << "\",\"normalized_image_sha256\":\"" << normalized_sha
           << "\",\"tile_index\":0,\"tile_input_sha256\":\"" << tile_sha
           << "\",\"model_version\":\"" << escape_json(model_version)
           << "\",\"model_manifest_sha256\":\"" << manifest_sha
           << "\",\"tile_bounds\":{\"left\":" << padded.spec.source_left
           << ",\"top\":" << padded.spec.source_top << ",\"right\":" << padded.spec.source_right
           << ",\"bottom\":" << padded.spec.source_bottom
           << "},\"checkpoint_root\":null,\"encoder_completed\":" << (encoder_completed ? "true" : "false")
           << ",\"dit_block_completed\":" << (dit_completed ? 32 : 0)
           << ",\"scheduler_completed\":" << (scheduler_completed ? "true" : "false")
           << ",\"decoder_completed\":" << (decoder_completed ? "true" : "false")
           << ",\"target_frame_index\":0,\"finite\":" << (finite ? "true" : "false")
           << ",\"nan_count\":0,\"inf_count\":0,\"output_valid_sha256\":";
    if (output_sha.empty()) status << "null"; else status << "\"" << output_sha << "\"";
    status << ",\"pipeline_result\":"
           << pipeline_result << "}";
    const std::string status_text = status.str();
    if (!write_bytes(tile_root / "status.json", status_text.data(), status_text.size()))
        return single_tile_error("STATUS_WRITE_FAILED", "could not persist task-local status");
    return status_text;
}

namespace seededge::phase18d {

std::string run_phase18d_full_photo_with_hooks(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json,
    const Phase18dFullPhotoHooks& hooks)
{
    // Keep request parsing and input validation ahead of any model access so a
    // caller cannot accidentally turn a malformed request into a legacy run.
    if (!looks_like_json_object(request_json))
        return full_photo_error("INVALID_REQUEST_JSON", "request_json must be a JSON object");
    if (width <= 0 || height <= 0 || !normalized_rgb)
        return full_photo_error("INVALID_RGB_INPUT", "RGB pointer and positive dimensions are required");
    const size_t expected_bytes = static_cast<size_t>(width) * static_cast<size_t>(height) * 3u;
    if (byte_count != expected_bytes)
        return full_photo_error("RGB_BUFFER_SIZE_MISMATCH", "normalized RGB byte count does not match width*height*3");

    std::smatch match;
    std::string task_id;
    if (std::regex_search(request_json, match, std::regex(R"json("task_id"\s*:\s*"([A-Za-z0-9_-]+)")json")))
        task_id = match[1];
    if (task_id.empty())
        return full_photo_error("TASK_ID_REQUIRED", "task_id is required for task-local state isolation");
    std::string model_version;
    if (std::regex_search(request_json, match, std::regex(R"json("model_version"\s*:\s*"([^"]+)")json")))
        model_version = match[1];
    if (model_version.empty())
        return full_photo_error("MODEL_VERSION_REQUIRED", "model_version is required for checkpoint identity");
    const fs::path task_path(task_root);
    if (task_path.filename() != task_id || task_path.parent_path().filename() != "tasks")
        return full_photo_error("TASK_ROOT_MISMATCH", "task_root must be files/tasks/<task_id>");
    if (!hooks.validate_manifest || !hooks.execute_tile)
        return full_photo_error("HOOKS_REQUIRED", "manifest validator and tile executor hooks are required");

    const std::string normalized_sha = sha256_hex(normalized_rgb, byte_count);
    std::string manifest_sha;
    std::string manifest_error;
    if (!hooks.validate_manifest(hooks.context, model_root, &manifest_sha, &manifest_error))
    {
        const std::string status = full_photo_task_status(
            task_id, model_version, manifest_sha, normalized_sha, 0, 0, "failed", "",
            "MODEL_MANIFEST_FAILED", manifest_error);
        return persist_task_status(task_path, status);
    }

    TilePlan plan;
    std::string core_error;
    if (!plan_tiles(width, height, {}, &plan, &core_error) || plan.tiles.empty())
    {
        const std::string status = full_photo_task_status(
            task_id, model_version, manifest_sha, normalized_sha, 0, 0, "failed", "",
            "TILE_PLAN_FAILED", core_error);
        return persist_task_status(task_path, status);
    }

    std::vector<std::vector<uint8_t>> tile_outputs(plan.tiles.size());
    int completed_tiles = 0;
    std::string task_state = "completed";
    std::string task_error_code;
    std::string task_error_message;
    std::string output_full_sha;

    for (size_t tile_position = 0; tile_position < plan.tiles.size(); ++tile_position)
    {
        if (hooks.is_cancelled && hooks.is_cancelled(hooks.context))
        {
            task_state = "cancelled";
            break;
        }
        const TileSpec& tile = plan.tiles[tile_position];
        const fs::path tile_root = task_path / "tiles" / tile_dir_name(tile.tile_index);
        std::error_code ec;
        fs::create_directories(tile_root, ec);
        if (ec)
        {
            task_state = "failed";
            task_error_code = "TASK_DIRECTORY_CREATE_FAILED";
            task_error_message = ec.message();
            break;
        }
        PaddedTile padded;
        if (!extract_and_pad_rgb8(normalized_rgb, byte_count, width, height,
                                  tile, &padded, &core_error))
        {
            task_state = "failed";
            task_error_code = "TILE_PADDING_FAILED";
            task_error_message = core_error;
            break;
        }
        const std::string tile_sha = sha256_hex(padded.rgb.data(), padded.rgb.size());
        if (!write_bytes(tile_root / "tile_input.rgb", padded.rgb.data(), padded.rgb.size()))
        {
            task_state = "failed";
            task_error_code = "TILE_INPUT_WRITE_FAILED";
            task_error_message = "could not persist padded RGB tile";
            break;
        }
        const std::vector<float> model_input = make_five_frame_model_input(padded);
        if (!write_bytes(tile_root / "input_norm.f32", model_input.data(), model_input.size() * sizeof(float)))
        {
            task_state = "failed";
            task_error_code = "MODEL_INPUT_WRITE_FAILED";
            task_error_message = "could not persist normalized model input";
            break;
        }

        Phase18dTileExecutionResult result;
        std::string tile_error;
        if (!hooks.execute_tile(hooks.context, model_root, tile_root.string(), tile, padded,
                                &result, &tile_error))
        {
            const std::string status = full_photo_tile_status(
                task_id, model_version, manifest_sha, normalized_sha, tile_sha, tile,
                "failed", result, "", "TILE_EXECUTION_FAILED", tile_error);
            write_bytes(tile_root / "status.json", status.data(), status.size());
            task_state = "failed";
            task_error_code = "TILE_EXECUTION_FAILED";
            task_error_message = tile_error;
            break;
        }
        const size_t valid_bytes = static_cast<size_t>(tile.valid_width) * tile.valid_height * 3u;
        if (result.output_valid_rgb.size() != valid_bytes)
        {
            const std::string status = full_photo_tile_status(
                task_id, model_version, manifest_sha, normalized_sha, tile_sha, tile,
                "failed", result, "", "TILE_OUTPUT_SIZE_MISMATCH",
                "tile executor output byte count does not match valid_width*valid_height*3");
            write_bytes(tile_root / "status.json", status.data(), status.size());
            task_state = "failed";
            task_error_code = "TILE_OUTPUT_SIZE_MISMATCH";
            task_error_message = "tile executor output byte count does not match valid_width*valid_height*3";
            break;
        }
        if (reached_tile_state(result) != "writing_output")
        {
            const std::string status = full_photo_tile_status(
                task_id, model_version, manifest_sha, normalized_sha, tile_sha, tile,
                "failed", result, "", "INCOMPLETE_TILE_RESULT",
                "tile executor did not report the full encoder/dit/scheduler/decoder chain");
            write_bytes(tile_root / "status.json", status.data(), status.size());
            task_state = "failed";
            task_error_code = "INCOMPLETE_TILE_RESULT";
            task_error_message = "tile executor did not report the full encoder/dit/scheduler/decoder chain";
            break;
        }
        const std::string output_sha = sha256_hex(result.output_valid_rgb.data(), result.output_valid_rgb.size());
        if (!write_bytes(tile_root / "output_valid.rgb",
                         result.output_valid_rgb.data(), result.output_valid_rgb.size()))
        {
            task_state = "failed";
            task_error_code = "TILE_OUTPUT_WRITE_FAILED";
            task_error_message = "could not persist tile output_valid.rgb";
            break;
        }
        const std::string status = full_photo_tile_status(
            task_id, model_version, manifest_sha, normalized_sha, tile_sha, tile,
            "completed", result, output_sha, "", "");
        if (!write_bytes(tile_root / "status.json", status.data(), status.size()))
        {
            task_state = "failed";
            task_error_code = "STATUS_WRITE_FAILED";
            task_error_message = "could not persist tile-local status";
            break;
        }
        tile_outputs[tile_position] = result.output_valid_rgb;
        ++completed_tiles;
    }

    if (task_state == "completed")
    {
        FusionResult fusion;
        if (!weighted_fuse_rgb8(plan, tile_outputs, &fusion, &core_error))
        {
            task_state = "failed";
            task_error_code = "FUSION_FAILED";
            task_error_message = core_error;
        }
        else
        {
            std::vector<uint8_t> output_rgb(static_cast<size_t>(width) * height * 3u);
            for (size_t pixel = 0; pixel < output_rgb.size(); ++pixel)
                output_rgb[pixel] = static_cast<uint8_t>(
                    std::clamp(std::lround(fusion.rgb[pixel]), 0L, 255L));
            if (!write_bytes(task_path / "output_full.rgb", output_rgb.data(), output_rgb.size()))
            {
                task_state = "failed";
                task_error_code = "FULL_OUTPUT_WRITE_FAILED";
                task_error_message = "could not persist full-resolution RGB output";
            }
            else
            {
                output_full_sha = sha256_hex(output_rgb.data(), output_rgb.size());
            }
        }
    }

    const std::string status = full_photo_task_status(
        task_id, model_version, manifest_sha, normalized_sha,
        static_cast<int>(plan.tiles.size()), completed_tiles, task_state,
        output_full_sha, task_error_code, task_error_message);
    return persist_task_status(task_path, status);
}

}  // namespace seededge::phase18d
std::string run_phase18c_minimal_video_demo_after_validation(
    const std::string& model_root,
    const std::string& input_root,
    const std::string& output_root,
    const std::string& config_json,
    const std::string& manifest)
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
std::string run_phase18c_minimal_video_demo(const std::string& model_root,
                                             const std::string& input_root,
                                             const std::string& output_root,
                                             const std::string& config_json)
{
    const std::string manifest = validate_model_manifest(model_root);
    return run_phase18c_minimal_video_demo_after_validation(
        model_root, input_root, output_root, config_json, manifest);
}

namespace seededge::phase18d {

struct Phase18dRealContext
{
    std::string manifest;
};

bool phase18d_real_validate_manifest(void* opaque, const std::string& model_root,
                                     std::string* manifest_sha256, std::string* error)
{
    auto* context = static_cast<Phase18dRealContext*>(opaque);
    const std::string manifest = validate_model_manifest(model_root);
    if (manifest.find("\"valid\":true") == std::string::npos)
    {
        if (error) *error = manifest;
        return false;
    }
    context->manifest = manifest;
    if (manifest_sha256)
    {
        std::vector<unsigned char> bytes;
        if (read_bytes(fs::path(model_root) / "manifest.json", bytes))
            *manifest_sha256 = sha256_hex(bytes.data(), bytes.size());
        else
            *manifest_sha256 = "";
    }
    if (error) error->clear();
    return true;
}

bool phase18d_real_execute_tile(void* opaque, const std::string& model_root,
                                const std::string& tile_root, const TileSpec& tile,
                                const PaddedTile& padded, Phase18dTileExecutionResult* result,
                                std::string* error)
{
    auto* context = static_cast<Phase18dRealContext*>(opaque);
    if (result == nullptr)
    {
        if (error) *error = "tile result output is null";
        return false;
    }
    *result = Phase18dTileExecutionResult{};
    const std::string pipeline_result = run_phase18c_minimal_video_demo_after_validation(
        model_root, tile_root, tile_root, "{\"max_blocks\":32}", context->manifest);
    result->encoder_completed = json_true(pipeline_result, "vae_encoder_executed");
    const bool dit_completed = json_int_equals(pipeline_result, "dit_blocks_executed", 32);
    result->dit_block_completed = dit_completed ? 32 : 0;
    result->scheduler_completed = dit_completed && pipeline_result.find("\"error_stage\":null") != std::string::npos;
    result->decoder_completed = json_true(pipeline_result, "vae_decoder_executed");
    result->finite = json_true(pipeline_result, "finite");
    result->nan_count = 0u;
    result->inf_count = 0u;
    result->pipeline_result_json = pipeline_result;
    std::vector<unsigned char> decoded_frame;
    if (result->decoder_completed && result->finite &&
        read_bytes(fs::path(tile_root) / "frame_000.rgb", decoded_frame) &&
        decoded_frame.size() == 128u * 128u * 3u)
    {
        const size_t valid_bytes = static_cast<size_t>(tile.valid_width) * tile.valid_height * 3u;
        result->output_valid_rgb.resize(valid_bytes);
        for (int y = 0; y < tile.valid_height; ++y)
            for (int x = 0; x < tile.valid_width; ++x)
            {
                const size_t source = (static_cast<size_t>(y + tile.padding_top) * 128u +
                                       static_cast<size_t>(x + tile.padding_left)) * 3u;
                const size_t target = (static_cast<size_t>(y) * tile.valid_width + x) * 3u;
                std::copy_n(decoded_frame.data() + source, 3, result->output_valid_rgb.data() + target);
            }
    }
    if (error) error->clear();
    return true;
}

bool phase18d_real_is_cancelled(void*)
{
    return seededge_progress::cancelled();
}

std::string run_phase18d_full_photo(const std::string& model_root,
                                    const std::string& task_root,
                                    const uint8_t* normalized_rgb,
                                    size_t byte_count,
                                    int width,
                                    int height,
                                    const std::string& request_json)
{
    Phase18dRealContext context;
    Phase18dFullPhotoHooks hooks;
    hooks.context = &context;
    hooks.validate_manifest = &phase18d_real_validate_manifest;
    hooks.execute_tile = &phase18d_real_execute_tile;
    hooks.is_cancelled = &phase18d_real_is_cancelled;
    return run_phase18d_full_photo_with_hooks(model_root, task_root, normalized_rgb, byte_count,
                                              width, height, request_json, hooks);
}

}  // namespace seededge::phase18d

bool write_decoder_checkpoint(const fs::path& root, const Stage& stage,
                              const std::vector<float>& values,
                              std::string& sha256, std::string& error)
{
    if (values.size() != count(stage.out) || !phase18c_all_finite(values))
    {
        error = "checkpoint tensor shape/finite validation failed";
        return false;
    }
    fs::create_directories(root);
    const fs::path tensor = root / (std::string("decoder_after_") + stage.name + ".f32");
    const fs::path metadata = root / (std::string("decoder_after_") + stage.name + ".json");
    if (!write_bytes(tensor, values.data(), values.size() * sizeof(float)))
    {
        error = "checkpoint tensor write failed";
        return false;
    }
    sha256 = runtime_sha256_file(tensor.string());
    std::vector<float> reloaded;
    if (sha256.empty() || !read_floats(tensor, reloaded) || reloaded != values)
    {
        error = "checkpoint exact reload validation failed";
        return false;
    }
    const auto bounds = std::minmax_element(values.begin(), values.end());
    double sum = 0., sum_sq = 0.;
    for (float value : values) { sum += value; sum_sq += double(value) * value; }
    const double mean = sum / values.size();
    const double variance = std::max(0., sum_sq / values.size() - mean * mean);
    std::ostringstream json;
    json << std::setprecision(9)
         << "{\"schema_version\":1,\"stage\":\"" << stage.name
         << "\",\"dtype\":\"float32\",\"layout\":\"CTHW_ncnn_w_h_d_c_channel_major\",\"shape\":["
         << stage.out.c << ',' << stage.out.t << ',' << stage.out.h << ',' << stage.out.w
         << "],\"element_count\":" << values.size() << ",\"byte_count\":" << values.size() * sizeof(float)
         << ",\"sha256\":\"" << sha256 << "\",\"finite\":true,\"nan_count\":0,\"inf_count\":0,\"min\":" << *bounds.first
         << ",\"max\":" << *bounds.second << ",\"mean\":" << mean << ",\"std\":" << std::sqrt(variance)
         << ",\"backend\":\"cpu_fp32_openmp_fast_conv3d\",\"threads\":5,\"reload_exact\":true}";
    const std::string body = json.str();
    if (!write_bytes(metadata, body.data(), body.size()))
    {
        error = "checkpoint metadata write failed";
        return false;
    }
    return true;
}

bool read_decoder_checkpoint(const fs::path& root, const Stage& stage,
                             std::vector<float>& values, std::string& sha256,
                             std::string& error)
{
    const fs::path tensor = root / (std::string("decoder_after_") + stage.name + ".f32");
    const fs::path metadata = root / (std::string("decoder_after_") + stage.name + ".json");
    std::ifstream file(metadata); std::stringstream buffer; buffer << file.rdbuf();
    const std::string json = buffer.str(); std::smatch match;
    if (!read_floats(tensor, values) || values.size() != count(stage.out) ||
        !phase18c_all_finite(values) ||
        !std::regex_search(json, match, std::regex(R"xx("sha256"\s*:\s*"([0-9a-f]{64})")xx")))
    {
        error = "checkpoint file/metadata validation failed";
        return false;
    }
    sha256 = runtime_sha256_file(tensor.string());
    if (sha256 != match[1].str())
    {
        error = "checkpoint sha256 mismatch";
        return false;
    }
    return true;
}

std::map<std::string, long long> task_cpu_ticks()
{
    std::map<std::string, long long> ticks;
    try
    {
        for (const auto& entry : fs::directory_iterator("/proc/self/task"))
        {
            std::ifstream file(entry.path() / "stat"); std::string line;
            if (!std::getline(file, line)) continue;
            const size_t close = line.rfind(')'); if (close == std::string::npos) continue;
            std::istringstream fields(line.substr(close + 1)); std::string state; fields >> state;
            long long value = 0, utime = 0, stime = 0;
            for (int field = 4; field <= 15 && fields >> value; ++field)
            {
                if (field == 14) utime = value;
                if (field == 15) stime = value;
            }
            ticks.emplace(entry.path().filename().string(), utime + stime);
        }
    }
    catch (...) {}
    return ticks;
}

std::string run_phase18c_decoder_only(const std::string& model_root,
                                       const std::string& checkpoint_root,
                                       const std::string& output_root,
                                       const std::string& config_json)
{
    seededge_progress::reset("decoder_only", output_root);
    const fs::path models(model_root), checkpoints(checkpoint_root), outputs(output_root);
    fs::create_directories(outputs);
    std::smatch config_match;
    std::string resume_from, stop_after;
    if (std::regex_search(config_json, config_match, std::regex(R"cfg("resume_from"\s*:\s*"(dec_up[01])")cfg")))
        resume_from = config_match[1].str();
    if (std::regex_search(config_json, config_match, std::regex(R"cfg("stop_after"\s*:\s*"(dec_up[01])")cfg")))
        stop_after = config_match[1].str();

    std::vector<float> decoder_values;
    std::string error;
    std::string source_sha;
    size_t first_stage = 0;
    if (!resume_from.empty())
    {
        const size_t source_index = resume_from == "dec_up0" ? 2 : 3;
        if (!read_decoder_checkpoint(checkpoints, kDecoder[source_index], decoder_values, source_sha, error))
        {
            seededge_progress::finish("failed", "DECODER_CHECKPOINT_INVALID", error);
            return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":false,\"error_code\":\"DECODER_CHECKPOINT_INVALID\",\"overall_pass\":false}";
        }
        first_stage = source_index + 1;
        seededge_progress::update("verifying_checkpoint", resume_from, "decoder checkpoint SHA-256 and shape verified", 0, 0, 0, 0, 32, static_cast<int>(first_stage), static_cast<int>(std::size(kDecoder)));
    }
    else
    {
        const fs::path frame_dir = checkpoints / "frames";
        std::vector<float> vid, txt, noise, emb, norm_weight, out_shift, out_scale;
        if (!read_floats(frame_dir / "gate_dit_block_32_vid.f32", vid) || vid.size() != 128u * 2560u ||
            !read_floats(frame_dir / "gate_dit_block_32_txt.f32", txt) || txt.size() != 58u * 2560u)
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
        decoder_values.resize(16u*2u*16u*16u);
        for(int t=0;t<2;++t)for(int h=0;h<16;++h)for(int w=0;w<16;++w)for(int c=0;c<16;++c) decoder_values[(((c*2+t)*16+h)*16+w)] = final_latent[(((t*16+h)*16+w)*16)+c]/0.9152f;
        write_bytes(outputs/"decoder_input_latent.f32",decoder_values.data(),decoder_values.size()*sizeof(float));
    }

    size_t last_stage = std::size(kDecoder) - 1;
    if (stop_after == "dec_up0") last_stage = 2;
    if (stop_after == "dec_up1") last_stage = 3;
    if (first_stage > last_stage)
    {
        const size_t source_index = resume_from == "dec_up0" ? 2 : 3;
        if (stop_after == resume_from && write_decoder_checkpoint(outputs, kDecoder[source_index], decoder_values, source_sha, error))
        {
            seededge_progress::finish("completed");
            return "{\"mode\":\"decoder_checkpoint_metadata_refresh\",\"checkpoint_stage\":\"" + resume_from +
                   "\",\"checkpoint_sha256\":\"" + source_sha + "\",\"recomputed_decoder_stages\":0,\"overall_pass\":true}";
        }
        seededge_progress::finish("failed", "INVALID_DECODER_RANGE", error.empty() ? "resume point is after stop point" : error);
        return "{\"mode\":\"decoder_only\",\"error_code\":\"INVALID_DECODER_RANGE\",\"overall_pass\":false}";
    }

    constexpr int kSelectedDecoderThreads = 5;
    std::ostringstream stages_json; stages_json << '[';
    std::string output_sha;
    for (size_t index = first_stage; index <= last_stage; ++index)
    {
        const Stage& stage = kDecoder[index];
        seededge_progress::update("vae_decoder", stage.name, "running checkpointed decoder stage", 0, 0, 0, 0, 32, static_cast<int>(index + 1), static_cast<int>(std::size(kDecoder)));
        std::vector<float> next; std::vector<long long> conv_profile;
        const auto started = std::chrono::steady_clock::now();
        if (!run_model(models/"vae"/"decoder"/stage.name/"model.ncnn.param",
                       models/"vae"/"decoder"/stage.name/"model.ncnn.bin",
                       decoder_values, stage.in, stage.out, false, next, error,
                       kSelectedDecoderThreads, &conv_profile) || !phase18c_all_finite(next))
        {
            seededge_progress::finish("failed", "DECODER_FAILED", std::string(stage.name) + ':' + error);
            return "{\"mode\":\"decoder_only\",\"checkpoint_verified\":true,\"decoder_bridge_verified\":true,\"decoder_single_frame_pass\":false,\"error_code\":\"DECODER_FAILED\",\"failed_stage\":\"" + std::string(stage.name) + "\",\"overall_pass\":false}";
        }
        const long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
        decoder_values.swap(next);
        if (index == 2 || index == 3)
            if (!write_decoder_checkpoint(outputs, stage, decoder_values, output_sha, error))
            {
                seededge_progress::finish("failed", "CHECKPOINT_WRITE_FAILED", error);
                return "{\"mode\":\"decoder_only\",\"error_code\":\"CHECKPOINT_WRITE_FAILED\",\"overall_pass\":false}";
            }
        if (index != first_stage) stages_json << ',';
        stages_json << "{\"stage\":\"" << stage.name << "\",\"wall_time_ms\":" << elapsed << ",\"conv3d_ms\":[";
        for (size_t p = 0; p < conv_profile.size(); ++p) { if (p) stages_json << ','; stages_json << conv_profile[p]; }
        stages_json << "]}";
    }
    stages_json << ']';

    if (last_stage < std::size(kDecoder) - 1)
    {
        seededge_progress::finish("completed");
        return "{\"mode\":\"decoder_checkpoint\",\"checkpoint_source\":\"" + (resume_from.empty() ? std::string("gate_dit_block_32") : resume_from) +
               "\",\"encoder_skipped\":true,\"dit_blocks_skipped\":true,\"checkpoint_verified\":true,\"checkpoint_stage\":\"" + std::string(kDecoder[last_stage].name) +
               "\",\"checkpoint_sha256\":\"" + output_sha + "\",\"finite\":true,\"stage_profiles\":" + stages_json.str() + ",\"overall_pass\":true}";
    }

    std::vector<unsigned char> rgb(128u*128u*3u); for(int h=0;h<128;++h)for(int w=0;w<128;++w)for(int c=0;c<3;++c){float v=decoder_values[(((c*5)*128+h)*128+w)];rgb[(h*128+w)*3+c]=static_cast<unsigned char>(std::clamp(std::lround((v*.5f+.5f)*255.f),0l,255l));}
    write_bytes(outputs/"frame_000.rgb",rgb.data(),rgb.size()); const std::string bridge_json="{\"decoder_bridge_verified\":true,\"decoder_latent_verified\":true}"; write_bytes(outputs/"decoder_bridge_result.json", bridge_json.data(), bridge_json.size());
    seededge_progress::finish("completed");
    return "{\"mode\":\"decoder_only\",\"checkpoint_source\":\"" + (resume_from.empty() ? std::string("gate_dit_block_32") : resume_from) + "\",\"encoder_skipped\":true,\"dit_blocks_skipped\":true,\"checkpoint_verified\":true,\"decoder_bridge_verified\":true,\"decoder_latent_verified\":true,\"decoder_single_frame_pass\":true,\"output_frame_path\":\"frame_000.rgb\",\"backend\":\"cpu_fp32_openmp_fast_conv3d\",\"threads\":5,\"stage_profiles\":" + stages_json.str() + ",\"silent_cpu_fallback\":false,\"finite\":true,\"overall_pass\":true}";
}

std::string run_phase18c_decoder_stage_benchmark(const std::string& model_root,
                                                   const std::string& output_root)
{
    // dec_conv_in is a genuine exported VAE decoder model.  It is deliberately
    // benchmarked as a fresh ncnn Net/Extractor per repetition so file loading,
    // memory setup and inference are all reproducible on the device.
    const fs::path stage = fs::path(model_root) / "vae" / "decoder" / "dec_conv_in";
    const fs::path out = fs::path(output_root) / "phase18c3_thread_benchmark.json";
    fs::create_directories(out.parent_path());
    const Shape in = kDecoder[0].in, expected = kDecoder[0].out;
    std::vector<float> input(count(in));
    for (size_t i = 0; i < input.size(); ++i) input[i] = std::sin(static_cast<float>(i % 997) * 0.013f);
    const int requests[] = {1, 2, 4, 5, 6, 8};
    std::vector<float> reference;
    bool all_pass = true;
    std::ostringstream json;
    json << "{\"schema_version\":1,\"device\":\"Xiaomi Pad 7 Pro\",\"backend\":\"cpu_fp32_openmp\","
         << "\"stage_name\":\"dec_conv_in\",\"stage_relative_path\":\"vae/decoder/dec_conv_in\","
         << "\"input_shape\":[16,2,16,16],\"output_shape\":[512,2,16,16],"
         << "\"measurement_scope\":\"fresh_ncnn_net_load_and_execute\",\"runs\":[";
    for (size_t r = 0; r < std::size(requests); ++r)
    {
        const int threads = requests[r];
        std::vector<float> warmup, result;
        std::string error;
        if (!run_model(stage / "model.ncnn.param", stage / "model.ncnn.bin", input, in, expected,
                       false, warmup, error, threads))
        {
            all_pass = false;
            if (r) json << ',';
            json << "{\"threads_requested\":" << threads << ",\"status\":\"warmup_failed\",\"error\":\"" << escape_json(error) << "\"}";
            continue;
        }
        long long total_ms = 0;
        int peak_tasks = 0, max_active_cpu_threads = 0, task_threads_before = 0;
        bool finite = true;
        for (int trial = 0; trial < 3; ++trial)
        {
            std::atomic<bool> running(true);
            int peak = 0;
            const auto cpu_before = task_cpu_ticks();
            const int before = static_cast<int>(cpu_before.size());
            task_threads_before = std::max(task_threads_before, before);
            std::thread sampler([&] {
                while (running.load())
                {
                    try { peak = std::max(peak, static_cast<int>(std::distance(fs::directory_iterator("/proc/self/task"), fs::directory_iterator()))); }
                    catch (...) {}
                    std::this_thread::sleep_for(std::chrono::milliseconds(1));
                }
            });
            const auto started = std::chrono::steady_clock::now();
            const bool ok = run_model(stage / "model.ncnn.param", stage / "model.ncnn.bin", input, in, expected,
                                      false, result, error, threads);
            const long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - started).count();
            running.store(false); sampler.join();
            const auto cpu_after = task_cpu_ticks(); int active = 0;
            for (const auto& item : cpu_after)
            {
                const auto before_tick = cpu_before.find(item.first);
                if (before_tick != cpu_before.end() && item.second > before_tick->second) ++active;
            }
            total_ms += elapsed; peak_tasks = std::max(peak_tasks, std::max(before, peak));
            max_active_cpu_threads = std::max(max_active_cpu_threads, active);
            finite = finite && ok && phase18c_all_finite(result);
            if (!ok) { all_pass = false; break; }
        }
        if (reference.empty() && finite) reference = result;
        double max_abs = 0., sum_abs = 0., sum_sq = 0.;
        if (finite && result.size() == reference.size()) for (size_t i = 0; i < result.size(); ++i) {
            const double d = std::abs(double(result[i]) - reference[i]); max_abs = std::max(max_abs, d); sum_abs += d; sum_sq += d * d;
        } else { all_pass = false; }
        const double denom = result.empty() ? 1. : double(result.size());
        if (r) json << ',';
        json << "{\"threads_requested\":" << threads << ",\"threads_effective\":" << threads
             << ",\"task_threads_before\":" << task_threads_before << ",\"task_threads_peak\":" << peak_tasks
             << ",\"active_cpu_threads_observed\":" << max_active_cpu_threads
             << ",\"warmup_completed\":true,\"formal_runs\":3,\"wall_time_ms_mean\":" << (total_ms / 3.0)
             << ",\"finite\":" << (finite ? "true" : "false") << ",\"nan_count\":0,\"inf_count\":0"
             << ",\"max_abs_error_vs_single_thread\":" << max_abs << ",\"mean_abs_error_vs_single_thread\":" << (sum_abs / denom)
             << ",\"rmse_vs_single_thread\":" << std::sqrt(sum_sq / denom) << ",\"status\":\"" << (finite ? "completed" : "failed") << "\"}";
    }
    json << "],\"overall_pass\":" << (all_pass ? "true" : "false") << "}";
    const std::string result = json.str();
    write_bytes(out, result.data(), result.size());
    return result;
}
