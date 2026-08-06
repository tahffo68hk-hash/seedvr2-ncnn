#include "phase18d_tile_core.h"

#include <algorithm>
#include <array>
#include <cmath>

#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>

#include <string>
#include <vector>

namespace seededge::phase18d {
namespace {

bool fixed_config(const TileConfig& config, std::string* error)
{
    if (config.tile_size == 128 && config.overlap == 32 && config.stride == 96)
        return true;
    if (error)
        *error = "INVALID_TILE_CONFIG";
    return false;
}

std::vector<int> axis_starts(int length, int tile_size, int stride)
{
    const int anchored_last = std::max(0, length - tile_size);
    std::vector<int> starts;
    for (int start = 0; start < anchored_last; start += stride)
        starts.push_back(start);
    starts.push_back(anchored_last);
    std::sort(starts.begin(), starts.end());
    starts.erase(std::unique(starts.begin(), starts.end()), starts.end());
    return starts;
}

int reflected_index(int coordinate, int length)
{
    const int period = 2 * (length - 1);
    int reduced = coordinate % period;
    if (reduced < 0)
        reduced += period;
    return reduced < length ? reduced : period - reduced;
}

int source_index(int coordinate, int valid_length, const std::string& mode)
{
    if (mode == "reflect")
        return reflected_index(coordinate, valid_length);
    return std::max(0, std::min(valid_length - 1, coordinate));
}

uint32_t rotr(uint32_t value, int count) { return (value >> count) | (value << (32 - count)); }

std::array<uint8_t, 32> sha256_bytes(const uint8_t* data, size_t size)
{
    static const uint32_t k[64] = {
        0x428a2f98u,0x71374491u,0xb5c0fbcfu,0xe9b5dba5u,0x3956c25bu,0x59f111f1u,0x923f82a4u,0xab1c5ed5u,
        0xd807aa98u,0x12835b01u,0x243185beu,0x550c7dc3u,0x72be5d74u,0x80deb1feu,0x9bdc06a7u,0xc19bf174u,
        0xe49b69c1u,0xefbe4786u,0x0fc19dc6u,0x240ca1ccu,0x2de92c6fu,0x4a7484aau,0x5cb0a9dcu,0x76f988dau,
        0x983e5152u,0xa831c66du,0xb00327c8u,0xbf597fc7u,0xc6e00bf3u,0xd5a79147u,0x06ca6351u,0x14292967u,
        0x27b70a85u,0x2e1b2138u,0x4d2c6dfcu,0x53380d13u,0x650a7354u,0x766a0abbu,0x81c2c92eu,0x92722c85u,
        0xa2bfe8a1u,0xa81a664bu,0xc24b8b70u,0xc76c51a3u,0xd192e819u,0xd6990624u,0xf40e3585u,0x106aa070u,
        0x19a4c116u,0x1e376c08u,0x2748774cu,0x34b0bcb5u,0x391c0cb3u,0x4ed8aa4au,0x5b9cca4fu,0x682e6ff3u,
        0x748f82eeu,0x78a5636fu,0x84c87814u,0x8cc70208u,0x90befffau,0xa4506cebu,0xbef9a3f7u,0xc67178f2u};
    uint32_t h[8] = {0x6a09e667u,0xbb67ae85u,0x3c6ef372u,0xa54ff53au,0x510e527fu,0x9b05688cu,0x1f83d9abu,0x5be0cd19u};
    std::vector<uint8_t> bytes(data, data + size);
    bytes.push_back(0x80);
    while ((bytes.size() % 64u) != 56u) bytes.push_back(0);
    const uint64_t bits = static_cast<uint64_t>(size) * 8u;
    for (int shift = 56; shift >= 0; shift -= 8) bytes.push_back(static_cast<uint8_t>(bits >> shift));
    for (size_t offset = 0; offset < bytes.size(); offset += 64)
    {
        uint32_t w[64]{};
        for (int i = 0; i < 16; ++i)
            w[i] = (uint32_t(bytes[offset+i*4])<<24) | (uint32_t(bytes[offset+i*4+1])<<16) |
                   (uint32_t(bytes[offset+i*4+2])<<8) | uint32_t(bytes[offset+i*4+3]);
        for (int i = 16; i < 64; ++i) { const uint32_t a=w[i-15], b=w[i-2]; w[i]=(rotr(a,7)^rotr(a,18)^(a>>3))+w[i-16]+(rotr(b,17)^rotr(b,19)^(b>>10))+w[i-7]; }
        uint32_t a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],q=h[7];
        for (int i = 0; i < 64; ++i) { const uint32_t s1=rotr(e,6)^rotr(e,11)^rotr(e,25), ch=(e&f)^((~e)&g), t1=q+s1+ch+k[i]+w[i], s0=rotr(a,2)^rotr(a,13)^rotr(a,22), maj=(a&b)^(a&c)^(b&c), t2=s0+maj; q=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2; }
        h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=q;
    }
    std::array<uint8_t,32> output{};
    for (int i=0;i<8;++i) for(int j=0;j<4;++j) output[i*4+j]=static_cast<uint8_t>(h[i] >> (24-j*8));
    return output;
}

float axis_weight(const TilePlan& plan, const TileSpec& tile, int global, bool horizontal)
{
    const int current_start = horizontal ? tile.source_left : tile.source_top;
    const int current_end = horizontal ? tile.source_right : tile.source_bottom;
    const int current_grid = horizontal ? tile.grid_x : tile.grid_y;
    const int other_grid = horizontal ? tile.grid_y : tile.grid_x;
    const TileSpec* previous = nullptr;
    const TileSpec* next = nullptr;
    for (const TileSpec& candidate : plan.tiles)
    {
        const int candidate_grid = horizontal ? candidate.grid_x : candidate.grid_y;
        const int candidate_other = horizontal ? candidate.grid_y : candidate.grid_x;
        if (candidate_other != other_grid) continue;
        if (candidate_grid == current_grid - 1) previous = &candidate;
        if (candidate_grid == current_grid + 1) next = &candidate;
    }
    constexpr float pi = 3.14159265358979323846f;
    if (previous)
    {
        const int previous_end = horizontal ? previous->source_right : previous->source_bottom;
        const int overlap = previous_end - current_start;
        if (overlap > 0 && global < previous_end)
        {
            const float theta = (pi * 0.5f) * (float(global - current_start) + 0.5f) / float(overlap);
            return std::sin(theta) * std::sin(theta);
        }
    }
    if (next)
    {
        const int next_start = horizontal ? next->source_left : next->source_top;
        const int overlap = current_end - next_start;
        if (overlap > 0 && global >= next_start)
        {
            const float theta = (pi * 0.5f) * (float(global - next_start) + 0.5f) / float(overlap);
            return std::cos(theta) * std::cos(theta);
        }
    }
    return 1.f;
}

bool phase18d_fusion_fail(
    std::string* error,
    const std::string& message) {

    if (error != nullptr) {
        *error = message;
    }

    return false;
}

bool phase18d_checked_multiply(
    size_t left,
    size_t right,
    size_t* result) {

    if (result == nullptr) {
        return false;
    }

    if (left != 0u &&
        right > std::numeric_limits<size_t>::max() / left) {
        return false;
    }

    *result = left * right;
    return true;
}

const TileSpec* phase18d_find_tile_by_grid(
    const TilePlan& plan,
    int grid_x,
    int grid_y) {

    for (const TileSpec& tile : plan.tiles) {
        if (tile.grid_x == grid_x &&
            tile.grid_y == grid_y) {
            return &tile;
        }
    }

    return nullptr;
}

bool phase18d_validate_overlap(
    int overlap,
    int current_extent,
    int neighbor_extent,
    const char* label,
    int* validated_overlap,
    std::string* error) {

    if (validated_overlap == nullptr) {
        return phase18d_fusion_fail(
            error,
            "validated overlap output is null");
    }

    if (overlap <= 0) {
        return phase18d_fusion_fail(
            error,
            std::string(label) +
                " neighbor does not overlap the current tile");
    }

    if (overlap > current_extent ||
        overlap > neighbor_extent) {
        return phase18d_fusion_fail(
            error,
            std::string(label) +
                " overlap exceeds a tile valid extent");
    }

    *validated_overlap = overlap;
    return true;
}

}  // namespace

bool plan_tiles(int image_width, int image_height, const TileConfig& config,
                TilePlan* plan, std::string* error)
{
    if (!plan || image_width <= 0 || image_height <= 0)
    {
        if (error)
            *error = "INVALID_IMAGE_DIMENSIONS";
        return false;
    }
    if (!fixed_config(config, error))
        return false;

    const std::vector<int> xs = axis_starts(image_width, config.tile_size, config.stride);
    const std::vector<int> ys = axis_starts(image_height, config.tile_size, config.stride);
    plan->image_width = image_width;
    plan->image_height = image_height;
    plan->config = config;
    plan->tiles.clear();
    int index = 0;
    for (size_t y = 0; y < ys.size(); ++y)
    {
        for (size_t x = 0; x < xs.size(); ++x)
        {
            TileSpec tile;
            tile.tile_index = index++;
            tile.grid_x = static_cast<int>(x);
            tile.grid_y = static_cast<int>(y);
            tile.source_left = xs[x];
            tile.source_top = ys[y];
            tile.source_right = std::min(image_width, tile.source_left + config.tile_size);
            tile.source_bottom = std::min(image_height, tile.source_top + config.tile_size);
            tile.valid_width = tile.source_right - tile.source_left;
            tile.valid_height = tile.source_bottom - tile.source_top;
            tile.padding_left = 0;
            tile.padding_top = 0;
            tile.padding_right = config.tile_size - tile.valid_width;
            tile.padding_bottom = config.tile_size - tile.valid_height;
            plan->tiles.push_back(tile);
        }
    }
    return true;
}

bool extract_and_pad_rgb8(const uint8_t* image_rgb, size_t byte_count,
                           int image_width, int image_height, const TileSpec& tile,
                           PaddedTile* padded, std::string* error)
{
    if (!image_rgb || !padded || image_width <= 0 || image_height <= 0 ||
        byte_count != static_cast<size_t>(image_width) * image_height * 3u ||
        tile.valid_width <= 0 || tile.valid_height <= 0 ||
        tile.valid_width + tile.padding_left + tile.padding_right != 128 ||
        tile.valid_height + tile.padding_top + tile.padding_bottom != 128 ||
        tile.source_left < 0 || tile.source_top < 0 ||
        tile.source_right > image_width || tile.source_bottom > image_height ||
        tile.source_right - tile.source_left != tile.valid_width ||
        tile.source_bottom - tile.source_top != tile.valid_height)
    {
        if (error)
            *error = "INVALID_TILE_BOUNDS";
        return false;
    }

    const bool reflect_x = tile.valid_width > 1 && tile.padding_left < tile.valid_width &&
                           tile.padding_right < tile.valid_width;
    const bool reflect_y = tile.valid_height > 1 && tile.padding_top < tile.valid_height &&
                           tile.padding_bottom < tile.valid_height;
    padded->spec = tile;
    padded->padding_mode_x = reflect_x ? "reflect" : "replicate";
    padded->padding_mode_y = reflect_y ? "reflect" : "replicate";
    padded->rgb.assign(128u * 128u * 3u, 0);

    for (int y = 0; y < 128; ++y)
    {
        const int local_y = source_index(y - tile.padding_top, tile.valid_height,
                                         padded->padding_mode_y);
        for (int x = 0; x < 128; ++x)
        {
            const int local_x = source_index(x - tile.padding_left, tile.valid_width,
                                             padded->padding_mode_x);
            const size_t source = (static_cast<size_t>(tile.source_top + local_y) * image_width +
                                   static_cast<size_t>(tile.source_left + local_x)) * 3u;
            const size_t destination = (static_cast<size_t>(y) * 128u + x) * 3u;
            padded->rgb[destination] = image_rgb[source];
            padded->rgb[destination + 1] = image_rgb[source + 1];
            padded->rgb[destination + 2] = image_rgb[source + 2];
        }
    }
    return true;
}

std::string sha256_hex(const uint8_t* data, size_t byte_count)
{
    if (!data && byte_count != 0) return std::string();
    const auto digest = sha256_bytes(data ? data : reinterpret_cast<const uint8_t*>(""), byte_count);
    std::ostringstream out;
    for (const uint8_t value : digest) out << std::hex << std::setw(2) << std::setfill('0') << int(value);
    return out.str();
}

bool identity_fuse_rgb8(const TilePlan& plan, const uint8_t* image_rgb,
                        size_t byte_count, FusionResult* result, std::string* error)
{
    if (!result || !image_rgb || plan.image_width <= 0 || plan.image_height <= 0 ||
        byte_count != static_cast<size_t>(plan.image_width) * plan.image_height * 3u)
    {
        if (error) *error = "INVALID_RGB_BUFFER";
        return false;
    }
    const size_t pixels = static_cast<size_t>(plan.image_width) * plan.image_height;
    std::vector<double> weighted_sum(pixels * 3u, 0.0);
    std::vector<double> weight_sum(pixels, 0.0);
    result->rgb.assign(pixels * 3u, 0.f);
    result->weight_sum.assign(pixels, 0.f);
    result->uncovered_pixel_count = 0;
    result->zero_weight_pixel_count = 0;
    result->max_abs_error = 0.f;
    for (const TileSpec& tile : plan.tiles)
        for (int y=tile.source_top; y<tile.source_bottom; ++y)
            for (int x=tile.source_left; x<tile.source_right; ++x)
            {
                const float weight = axis_weight(plan,tile,x,true) * axis_weight(plan,tile,y,false);
                const size_t pixel = static_cast<size_t>(y) * plan.image_width + x;
                weight_sum[pixel] += weight;
                for (int c=0;c<3;++c)
                    weighted_sum[pixel*3u+c] += double(image_rgb[pixel*3u+c]) * weight;
            }
    for (size_t pixel=0; pixel<pixels; ++pixel)
    {
        if (weight_sum[pixel] <= 0.0) { ++result->zero_weight_pixel_count; ++result->uncovered_pixel_count; continue; }
        result->weight_sum[pixel] = static_cast<float>(weight_sum[pixel]);
        for (int c=0;c<3;++c)
        {
            result->rgb[pixel*3u+c] = static_cast<float>(weighted_sum[pixel*3u+c] / weight_sum[pixel]);
            result->max_abs_error = std::max(result->max_abs_error,
                std::fabs(result->rgb[pixel*3u+c] - float(image_rgb[pixel*3u+c])));
        }
    }
    return true;
}

bool validate_checkpoint_identity(const CheckpointIdentity& expected,
                                  const CheckpointIdentity& actual,
                                  std::string* error_code, std::string* error)
{
    const auto fail = [&](const char* code, const char* message) {
        if (error_code) *error_code = code;
        if (error) *error = message;
        return false;
    };
    if (expected.task_id != actual.task_id) return fail("CHECKPOINT_TASK_MISMATCH", "task_id differs");
    if (expected.normalized_image_sha256 != actual.normalized_image_sha256 ||
        expected.tile_input_sha256 != actual.tile_input_sha256)
        return fail("CHECKPOINT_INPUT_MISMATCH", "input digest differs");
    const TileSpec& a=expected.tile; const TileSpec& b=actual.tile;
    if (a.tile_index != b.tile_index || a.source_left != b.source_left || a.source_top != b.source_top ||
        a.source_right != b.source_right || a.source_bottom != b.source_bottom)
        return fail("CHECKPOINT_TILE_MISMATCH", "tile bounds differ");
    if (expected.model_manifest_sha256 != actual.model_manifest_sha256)
        return fail("CHECKPOINT_MODEL_MISMATCH", "model manifest differs");
    if (expected.shape != actual.shape || expected.dtype != actual.dtype)
        return fail("CHECKPOINT_SHAPE_MISMATCH", "shape or dtype differs");
    if (expected.byte_count != actual.byte_count)
        return fail("CHECKPOINT_SIZE_MISMATCH", "byte count differs");
    if (expected.checkpoint_sha256 != actual.checkpoint_sha256)
        return fail("CHECKPOINT_SHA_MISMATCH", "checkpoint digest differs");
    if (!actual.finite)
        return fail("CHECKPOINT_NON_FINITE", "checkpoint is non-finite");
    if (error_code) error_code->clear();
    if (error) error->clear();
    return true;
}

bool is_valid_tile_state_transition(const std::string& from, const std::string& to)
{
    if (from == "pending") return to == "preparing" || to == "cancelled" || to == "failed";
    if (from == "preparing") return to == "encoder" || to == "cancelled" || to == "failed";
    if (from == "encoder") return to == "dit" || to == "cancelled" || to == "failed";
    if (from == "dit") return to == "scheduler" || to == "cancelled" || to == "failed";
    if (from == "scheduler") return to == "decoder" || to == "cancelled" || to == "failed";
    if (from == "decoder") return to == "writing_output" || to == "cancelled" || to == "failed";
    if (from == "writing_output") return to == "completed" || to == "failed";
    return false;
}

bool make_axis_fusion_weights(
    int extent,
    int leading_overlap,
    int trailing_overlap,
    std::vector<float>* weights,
    std::string* error) {

    if (error != nullptr) {
        error->clear();
    }

    if (weights == nullptr) {
        return phase18d_fusion_fail(
            error,
            "fusion weights output is null");
    }

    weights->clear();

    if (extent <= 0) {
        return phase18d_fusion_fail(
            error,
            "fusion weight extent must be positive");
    }

    if (leading_overlap < 0 ||
        trailing_overlap < 0) {
        return phase18d_fusion_fail(
            error,
            "fusion overlap must not be negative");
    }

    if (leading_overlap > extent ||
        trailing_overlap > extent) {
        return phase18d_fusion_fail(
            error,
            "fusion overlap exceeds valid extent");
    }

    constexpr float kPi =
        3.14159265358979323846f;

    std::vector<float> generated(
        static_cast<size_t>(extent),
        1.0f);

    // Left/top overlap: sin^2 ramp from near zero toward one.
    for (int index = 0;
         index < leading_overlap;
         ++index) {

        const float t =
            (static_cast<float>(index) + 0.5f) /
            static_cast<float>(leading_overlap);

        const float angle =
            0.5f * kPi * t;

        const float sine = std::sin(angle);
        const float ramp = sine * sine;

        if (!std::isfinite(ramp) || ramp <= 0.0f) {
            return phase18d_fusion_fail(
                error,
                "leading fusion ramp is not finite and positive");
        }

        generated[static_cast<size_t>(index)] *= ramp;
    }

    // Right/bottom overlap: cos^2 ramp from near one toward zero.
    for (int overlap_index = 0;
         overlap_index < trailing_overlap;
         ++overlap_index) {

        const float t =
            (static_cast<float>(overlap_index) + 0.5f) /
            static_cast<float>(trailing_overlap);

        const float angle =
            0.5f * kPi * t;

        const float cosine = std::cos(angle);
        const float ramp = cosine * cosine;

        if (!std::isfinite(ramp) || ramp <= 0.0f) {
            return phase18d_fusion_fail(
                error,
                "trailing fusion ramp is not finite and positive");
        }

        const int index =
            extent - trailing_overlap + overlap_index;

        generated[static_cast<size_t>(index)] *= ramp;
    }

    for (float weight : generated) {
        if (!std::isfinite(weight) || weight <= 0.0f) {
            return phase18d_fusion_fail(
                error,
                "combined fusion weight is not finite and positive");
        }
    }

    *weights = std::move(generated);
    return true;
}

bool weighted_fuse_rgb8(
    const TilePlan& plan,
    const std::vector<std::vector<uint8_t>>& tile_outputs,
    FusionResult* result,
    std::string* error) {

    if (error != nullptr) {
        error->clear();
    }

    if (result == nullptr) {
        return phase18d_fusion_fail(
            error,
            "fusion result output is null");
    }

    *result = FusionResult{};

    if (plan.image_width <= 0 ||
        plan.image_height <= 0) {
        return phase18d_fusion_fail(
            error,
            "fusion image dimensions must be positive");
    }

    if (plan.tiles.empty()) {
        return phase18d_fusion_fail(
            error,
            "fusion tile plan is empty");
    }

    if (tile_outputs.size() != plan.tiles.size()) {
        return phase18d_fusion_fail(
            error,
            "tile output count does not match tile plan");
    }

    size_t pixel_count = 0u;
    size_t rgb_count = 0u;

    if (!phase18d_checked_multiply(
            static_cast<size_t>(plan.image_width),
            static_cast<size_t>(plan.image_height),
            &pixel_count) ||
        !phase18d_checked_multiply(
            pixel_count,
            3u,
            &rgb_count)) {
        return phase18d_fusion_fail(
            error,
            "fusion output size overflows size_t");
    }

    // Reject ambiguous plans. Neighbor lookup is based on grid coordinates.
    for (size_t left = 0u;
         left < plan.tiles.size();
         ++left) {

        for (size_t right = left + 1u;
             right < plan.tiles.size();
             ++right) {

            if (plan.tiles[left].grid_x ==
                    plan.tiles[right].grid_x &&
                plan.tiles[left].grid_y ==
                    plan.tiles[right].grid_y) {
                return phase18d_fusion_fail(
                    error,
                    "tile plan contains duplicate grid coordinates");
            }
        }
    }

    // Double accumulation reduces order-dependent rounding. Public output
    // remains float to preserve the existing FusionResult ABI.
    std::vector<double> rgb_sum(rgb_count, 0.0);
    std::vector<double> weight_sum(pixel_count, 0.0);

    for (size_t tile_position = 0u;
         tile_position < plan.tiles.size();
         ++tile_position) {

        const TileSpec& tile =
            plan.tiles[tile_position];

        if (tile.grid_x < 0 || tile.grid_y < 0) {
            return phase18d_fusion_fail(
                error,
                "tile grid coordinate must not be negative");
        }

        if (tile.source_left < 0 ||
            tile.source_top < 0 ||
            tile.source_right > plan.image_width ||
            tile.source_bottom > plan.image_height ||
            tile.source_right <= tile.source_left ||
            tile.source_bottom <= tile.source_top) {
            return phase18d_fusion_fail(
                error,
                "tile source bounds are invalid");
        }

        const int bounds_width =
            tile.source_right - tile.source_left;

        const int bounds_height =
            tile.source_bottom - tile.source_top;

        if (tile.valid_width != bounds_width ||
            tile.valid_height != bounds_height ||
            tile.valid_width <= 0 ||
            tile.valid_height <= 0) {
            return phase18d_fusion_fail(
                error,
                "tile valid dimensions do not match source bounds");
        }

        size_t tile_pixel_count = 0u;
        size_t expected_tile_bytes = 0u;

        if (!phase18d_checked_multiply(
                static_cast<size_t>(tile.valid_width),
                static_cast<size_t>(tile.valid_height),
                &tile_pixel_count) ||
            !phase18d_checked_multiply(
                tile_pixel_count,
                3u,
                &expected_tile_bytes)) {
            return phase18d_fusion_fail(
                error,
                "tile output size overflows size_t");
        }

        const std::vector<uint8_t>& tile_rgb =
            tile_outputs[tile_position];

        if (tile_rgb.size() != expected_tile_bytes) {
            return phase18d_fusion_fail(
                error,
                "tile output byte count does not match valid dimensions");
        }

        const TileSpec* left_neighbor =
            phase18d_find_tile_by_grid(
                plan,
                tile.grid_x - 1,
                tile.grid_y);

        const TileSpec* right_neighbor =
            phase18d_find_tile_by_grid(
                plan,
                tile.grid_x + 1,
                tile.grid_y);

        const TileSpec* top_neighbor =
            phase18d_find_tile_by_grid(
                plan,
                tile.grid_x,
                tile.grid_y - 1);

        const TileSpec* bottom_neighbor =
            phase18d_find_tile_by_grid(
                plan,
                tile.grid_x,
                tile.grid_y + 1);

        int left_overlap = 0;
        int right_overlap = 0;
        int top_overlap = 0;
        int bottom_overlap = 0;

        if (left_neighbor != nullptr) {
            const int overlap =
                left_neighbor->source_right -
                tile.source_left;

            if (!phase18d_validate_overlap(
                    overlap,
                    tile.valid_width,
                    left_neighbor->valid_width,
                    "left",
                    &left_overlap,
                    error)) {
                return false;
            }
        }

        if (right_neighbor != nullptr) {
            const int overlap =
                tile.source_right -
                right_neighbor->source_left;

            if (!phase18d_validate_overlap(
                    overlap,
                    tile.valid_width,
                    right_neighbor->valid_width,
                    "right",
                    &right_overlap,
                    error)) {
                return false;
            }
        }

        if (top_neighbor != nullptr) {
            const int overlap =
                top_neighbor->source_bottom -
                tile.source_top;

            if (!phase18d_validate_overlap(
                    overlap,
                    tile.valid_height,
                    top_neighbor->valid_height,
                    "top",
                    &top_overlap,
                    error)) {
                return false;
            }
        }

        if (bottom_neighbor != nullptr) {
            const int overlap =
                tile.source_bottom -
                bottom_neighbor->source_top;

            if (!phase18d_validate_overlap(
                    overlap,
                    tile.valid_height,
                    bottom_neighbor->valid_height,
                    "bottom",
                    &bottom_overlap,
                    error)) {
                return false;
            }
        }

        std::vector<float> x_weights;
        std::vector<float> y_weights;
        std::string weight_error;

        if (!make_axis_fusion_weights(
                tile.valid_width,
                left_overlap,
                right_overlap,
                &x_weights,
                &weight_error)) {
            return phase18d_fusion_fail(
                error,
                "failed to build x fusion weights: " +
                    weight_error);
        }

        if (!make_axis_fusion_weights(
                tile.valid_height,
                top_overlap,
                bottom_overlap,
                &y_weights,
                &weight_error)) {
            return phase18d_fusion_fail(
                error,
                "failed to build y fusion weights: " +
                    weight_error);
        }

        for (int local_y = 0;
             local_y < tile.valid_height;
             ++local_y) {

            const int global_y =
                tile.source_top + local_y;

            for (int local_x = 0;
                 local_x < tile.valid_width;
                 ++local_x) {

                const int global_x =
                    tile.source_left + local_x;

                const double weight =
                    static_cast<double>(
                        x_weights[
                            static_cast<size_t>(local_x)]) *
                    static_cast<double>(
                        y_weights[
                            static_cast<size_t>(local_y)]);

                if (!std::isfinite(weight) ||
                    weight <= 0.0) {
                    return phase18d_fusion_fail(
                        error,
                        "tile fusion weight is not finite and positive");
                }

                const size_t global_pixel =
                    static_cast<size_t>(global_y) *
                        static_cast<size_t>(plan.image_width) +
                    static_cast<size_t>(global_x);

                const size_t local_pixel =
                    static_cast<size_t>(local_y) *
                        static_cast<size_t>(tile.valid_width) +
                    static_cast<size_t>(local_x);

                weight_sum[global_pixel] += weight;

                for (size_t channel = 0u;
                     channel < 3u;
                     ++channel) {

                    const size_t global_rgb =
                        global_pixel * 3u + channel;

                    const size_t local_rgb =
                        local_pixel * 3u + channel;

                    rgb_sum[global_rgb] +=
                        static_cast<double>(
                            tile_rgb[local_rgb]) *
                        weight;
                }
            }
        }
    }

    FusionResult fused;
    fused.weight_sum.resize(pixel_count, 0.0f);

    for (size_t pixel = 0u;
         pixel < pixel_count;
         ++pixel) {

        const double weight = weight_sum[pixel];

        if (!std::isfinite(weight) ||
            weight <= 0.0) {

            ++fused.uncovered_pixel_count;

            if (weight == 0.0) {
                ++fused.zero_weight_pixel_count;
            }

            continue;
        }

        fused.weight_sum[pixel] =
            static_cast<float>(weight);
    }

    if (fused.uncovered_pixel_count != 0u ||
        fused.zero_weight_pixel_count != 0u) {

        *result = std::move(fused);

        return phase18d_fusion_fail(
            error,
            "fusion produced uncovered or zero-weight pixels");
    }

    fused.rgb.resize(rgb_count, 0.0f);

    for (size_t pixel = 0u;
         pixel < pixel_count;
         ++pixel) {

        const double weight = weight_sum[pixel];

        for (size_t channel = 0u;
             channel < 3u;
             ++channel) {

            const size_t rgb_index =
                pixel * 3u + channel;

            double value =
                rgb_sum[rgb_index] / weight;

            if (!std::isfinite(value)) {
                return phase18d_fusion_fail(
                    error,
                    "normalized fusion value is not finite");
            }

            value = std::clamp(
                value,
                0.0,
                255.0);

            fused.rgb[rgb_index] =
                static_cast<float>(value);
        }
    }

    // max_abs_error belongs to identity/reference comparisons. Weighted
    // fusion has no independent reference image at this layer.
    fused.max_abs_error = 0.0f;

    *result = std::move(fused);
    return true;
}

}  // namespace seededge::phase18d
