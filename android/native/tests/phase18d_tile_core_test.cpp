#include "phase18d_tile_core.h"

#include <iostream>
#include <cmath>
#include <set>
#include <utility>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>

namespace p18d = seededge::phase18d;

static int failed = 0;

static void expect(bool condition, const char* name)
{
    if (!condition)
    {
        ++failed;
        std::cerr << "FAIL " << name << "\n";
    }
}

static bool has_exact_coverage(const p18d::TilePlan& plan, int width, int height)
{
    std::vector<int> coverage(static_cast<size_t>(width) * height, 0);
    std::set<std::pair<int, int>> origins;
    for (const p18d::TileSpec& tile : plan.tiles)
    {
        if (tile.source_left < 0 || tile.source_top < 0 ||
            tile.source_right > width || tile.source_bottom > height ||
            tile.source_left >= tile.source_right || tile.source_top >= tile.source_bottom ||
            !origins.insert({tile.source_left, tile.source_top}).second)
            return false;
        for (int y = tile.source_top; y < tile.source_bottom; ++y)
            for (int x = tile.source_left; x < tile.source_right; ++x)
                ++coverage[static_cast<size_t>(y) * width + x];
    }
    for (const int value : coverage)
        if (value == 0)
            return false;
    return true;
}

using seededge::phase18d::FusionResult;
using seededge::phase18d::TileConfig;
using seededge::phase18d::TilePlan;
using seededge::phase18d::TileSpec;
using seededge::phase18d::make_axis_fusion_weights;
using seededge::phase18d::plan_tiles;
using seededge::phase18d::weighted_fuse_rgb8;

static TileConfig fusion_test_config() {
    TileConfig config;
    config.tile_size = 128;
    config.overlap = 32;
    config.stride = 96;
    return config;
}

static std::vector<uint8_t> make_solid_valid_tile(
    const TileSpec& tile,
    uint8_t value) {

    const size_t pixel_count =
        static_cast<size_t>(tile.valid_width) *
        static_cast<size_t>(tile.valid_height);

    return std::vector<uint8_t>(pixel_count * 3u, value);
}

static std::vector<std::vector<uint8_t>> make_solid_tile_outputs(
    const TilePlan& plan,
    uint8_t value) {

    std::vector<std::vector<uint8_t>> outputs;
    outputs.reserve(plan.tiles.size());

    for (const TileSpec& tile : plan.tiles) {
        outputs.push_back(make_solid_valid_tile(tile, value));
    }

    return outputs;
}

static bool all_rgb_near(
    const FusionResult& result,
    float expected,
    float tolerance) {

    for (float value : result.rgb) {
        if (!std::isfinite(value) ||
            std::fabs(value - expected) > tolerance) {
            return false;
        }
    }

    return true;
}

static bool all_weights_positive_and_finite(
    const FusionResult& result,
    size_t expected_pixel_count) {

    if (result.weight_sum.size() != expected_pixel_count) {
        return false;
    }

    for (float weight : result.weight_sum) {
        if (!std::isfinite(weight) || weight <= 0.0f) {
            return false;
        }
    }

    return true;
}

static void test_weighted_fuse_single_tile_identity() {
    TilePlan plan;
    std::string error;

    const bool planned = plan_tiles(
        64,
        73,
        fusion_test_config(),
        &plan,
        &error);

    expect(planned, "weighted fusion single tile plan succeeds");
    if (!planned) {
        return;
    }

    expect(
        plan.tiles.size() == 1u,
        "64x73 produces exactly one tile");

    const auto outputs = make_solid_tile_outputs(plan, 73);

    FusionResult result;
    error.clear();

    const bool fused = weighted_fuse_rgb8(
        plan,
        outputs,
        &result,
        &error);

    expect(fused, "weighted fusion single tile succeeds");
    if (!fused) {
        return;
    }

    const size_t expected_pixels = 64u * 73u;

    expect(
        result.rgb.size() == expected_pixels * 3u,
        "single tile fusion restores original image dimensions");

    expect(
        all_rgb_near(result, 73.0f, 1.0e-5f),
        "single tile constant RGB remains unchanged");

    expect(
        all_weights_positive_and_finite(result, expected_pixels),
        "single tile has finite positive weight at every pixel");

    expect(
        result.uncovered_pixel_count == 0u,
        "single tile fusion has no uncovered pixels");

    expect(
        result.zero_weight_pixel_count == 0u,
        "single tile fusion has no zero-weight pixels");
}

static void test_weighted_fuse_two_tiles_constant() {
    TilePlan plan;
    std::string error;

    const bool planned = plan_tiles(
        160,
        96,
        fusion_test_config(),
        &plan,
        &error);

    expect(planned, "weighted fusion two-tile plan succeeds");
    if (!planned) {
        return;
    }

    expect(
        plan.tiles.size() == 2u,
        "160x96 produces two anchored horizontal tiles");

    const auto outputs = make_solid_tile_outputs(plan, 91);

    FusionResult result;
    error.clear();

    const bool fused = weighted_fuse_rgb8(
        plan,
        outputs,
        &result,
        &error);

    expect(fused, "weighted fusion two-tile constant succeeds");
    if (!fused) {
        return;
    }

    expect(
        result.rgb.size() == 160u * 96u * 3u,
        "two-tile fusion output has full image size");

    expect(
        all_rgb_near(result, 91.0f, 1.0e-4f),
        "constant tiles remain constant through overlap");

    expect(
        all_weights_positive_and_finite(
            result,
            160u * 96u),
        "two-tile fusion covers every pixel");

    expect(
        result.uncovered_pixel_count == 0u,
        "two-tile fusion reports no uncovered pixels");

    expect(
        result.zero_weight_pixel_count == 0u,
        "two-tile fusion reports no zero weights");
}

static bool verify_sine_cosine_complement(int overlap) {
    constexpr int kExtent = 128;
    constexpr float kPi = 3.14159265358979323846f;
    constexpr float kTolerance = 1.0e-6f;

    std::vector<float> left_weights;
    std::vector<float> right_weights;
    std::string error;

    if (!make_axis_fusion_weights(
            kExtent,
            0,
            overlap,
            &left_weights,
            &error)) {
        return false;
    }

    error.clear();

    if (!make_axis_fusion_weights(
            kExtent,
            overlap,
            0,
            &right_weights,
            &error)) {
        return false;
    }

    if (left_weights.size() !=
            static_cast<size_t>(kExtent) ||
        right_weights.size() !=
            static_cast<size_t>(kExtent)) {
        return false;
    }

    for (int k = 0; k < overlap; ++k) {
        const float t =
            (static_cast<float>(k) + 0.5f) /
            static_cast<float>(overlap);

        const float angle = 0.5f * kPi * t;

        const float expected_left =
            std::cos(angle) * std::cos(angle);

        const float expected_right =
            std::sin(angle) * std::sin(angle);

        const float actual_left =
            left_weights[
                static_cast<size_t>(
                    kExtent - overlap + k)];

        const float actual_right =
            right_weights[static_cast<size_t>(k)];

        if (!std::isfinite(actual_left) ||
            !std::isfinite(actual_right)) {
            return false;
        }

        if (std::fabs(
                actual_left - expected_left) >
            kTolerance) {
            return false;
        }

        if (std::fabs(
                actual_right - expected_right) >
            kTolerance) {
            return false;
        }

        if (std::fabs(
                actual_left +
                actual_right -
                1.0f) >
            kTolerance) {
            return false;
        }
    }

    return true;
}

static void test_squared_sine_cosine_complement() {
    expect(
        verify_sine_cosine_complement(1),
        "sin2/cos2 weights complement for overlap 1");

    expect(
        verify_sine_cosine_complement(32),
        "sin2/cos2 weights complement for overlap 32");

    // 160 瀹姐€?28 tile銆佹湯鍧楅敋瀹氬埌 x=32锛?
    // 瀹為檯 overlap 涓?96锛屼笉鑳藉亣瀹氭案杩滅瓑浜庨厤缃腑鐨?32銆?
    expect(
        verify_sine_cosine_complement(96),
        "sin2/cos2 weights complement for anchored overlap 96");
}

static void test_weighted_fuse_full_coverage_matrix() {
    struct ImageSize {
        int width;
        int height;
    };

    const ImageSize sizes[] = {
        {1, 1},
        {127, 127},
        {128, 128},
        {129, 128},
        {160, 96},
        {225, 129},
        {320, 240},
    };

    bool all_plans_succeeded = true;
    bool all_fusions_succeeded = true;
    bool all_sizes_correct = true;
    bool all_pixels_covered = true;
    bool all_values_constant = true;

    for (const ImageSize size : sizes) {
        TilePlan plan;
        std::string error;

        const bool planned = plan_tiles(
            size.width,
            size.height,
            fusion_test_config(),
            &plan,
            &error);

        if (!planned) {
            all_plans_succeeded = false;
            continue;
        }

        const auto outputs =
            make_solid_tile_outputs(plan, 37);

        FusionResult result;
        error.clear();

        const bool fused = weighted_fuse_rgb8(
            plan,
            outputs,
            &result,
            &error);

        if (!fused) {
            all_fusions_succeeded = false;
            continue;
        }

        const size_t pixel_count =
            static_cast<size_t>(size.width) *
            static_cast<size_t>(size.height);

        if (result.rgb.size() != pixel_count * 3u ||
            result.weight_sum.size() != pixel_count) {
            all_sizes_correct = false;
        }

        if (!all_weights_positive_and_finite(
                result,
                pixel_count) ||
            result.uncovered_pixel_count != 0u ||
            result.zero_weight_pixel_count != 0u) {
            all_pixels_covered = false;
        }

        if (!all_rgb_near(
                result,
                37.0f,
                1.0e-4f)) {
            all_values_constant = false;
        }
    }

    expect(
        all_plans_succeeded,
        "coverage matrix tile plans all succeed");

    expect(
        all_fusions_succeeded,
        "coverage matrix weighted fusions all succeed");

    expect(
        all_sizes_correct,
        "coverage matrix outputs preserve every image size");

    expect(
        all_pixels_covered,
        "coverage matrix has no uncovered or zero-weight pixels");

    expect(
        all_values_constant,
        "coverage matrix preserves constant RGB");
}

static void test_weighted_fuse_order_independent() {
    TilePlan forward_plan;
    std::string error;

    const bool planned = plan_tiles(
        224,
        192,
        fusion_test_config(),
        &forward_plan,
        &error);

    expect(planned, "order-independence plan succeeds");
    if (!planned) {
        return;
    }

    std::vector<std::vector<uint8_t>> forward_outputs;
    forward_outputs.reserve(forward_plan.tiles.size());

    for (size_t i = 0; i < forward_plan.tiles.size(); ++i) {
        const uint8_t value =
            static_cast<uint8_t>(20u + i * 41u);

        forward_outputs.push_back(
            make_solid_valid_tile(
                forward_plan.tiles[i],
                value));
    }

    FusionResult forward_result;
    error.clear();

    const bool forward_fused = weighted_fuse_rgb8(
        forward_plan,
        forward_outputs,
        &forward_result,
        &error);

    expect(
        forward_fused,
        "forward-order weighted fusion succeeds");

    if (!forward_fused) {
        return;
    }

    TilePlan reverse_plan = forward_plan;
    auto reverse_outputs = forward_outputs;

    std::reverse(
        reverse_plan.tiles.begin(),
        reverse_plan.tiles.end());

    std::reverse(
        reverse_outputs.begin(),
        reverse_outputs.end());

    FusionResult reverse_result;
    error.clear();

    const bool reverse_fused = weighted_fuse_rgb8(
        reverse_plan,
        reverse_outputs,
        &reverse_result,
        &error);

    expect(
        reverse_fused,
        "reverse-order weighted fusion succeeds");

    if (!reverse_fused) {
        return;
    }

    bool same_rgb =
        forward_result.rgb.size() ==
        reverse_result.rgb.size();

    if (same_rgb) {
        for (size_t i = 0;
             i < forward_result.rgb.size();
             ++i) {

            if (std::fabs(
                    forward_result.rgb[i] -
                    reverse_result.rgb[i]) >
                1.0e-4f) {
                same_rgb = false;
                break;
            }
        }
    }

    bool same_weights =
        forward_result.weight_sum.size() ==
        reverse_result.weight_sum.size();

    if (same_weights) {
        for (size_t i = 0;
             i < forward_result.weight_sum.size();
             ++i) {

            if (std::fabs(
                    forward_result.weight_sum[i] -
                    reverse_result.weight_sum[i]) >
                1.0e-5f) {
                same_weights = false;
                break;
            }
        }
    }

    expect(
        same_rgb,
        "weighted fusion RGB is independent of tile iteration order");

    expect(
        same_weights,
        "weighted fusion weight sum is independent of tile order");
}

static int find_tile_position(
    const TilePlan& plan,
    int grid_x,
    int grid_y) {

    for (size_t i = 0; i < plan.tiles.size(); ++i) {
        const TileSpec& tile = plan.tiles[i];

        if (tile.grid_x == grid_x &&
            tile.grid_y == grid_y) {
            return static_cast<int>(i);
        }
    }

    return -1;
}

static bool build_tile_axis_weights(
    const TileSpec& tile,
    int left_overlap,
    int right_overlap,
    int top_overlap,
    int bottom_overlap,
    std::vector<float>* x_weights,
    std::vector<float>* y_weights) {

    std::string error;

    if (!make_axis_fusion_weights(
            tile.valid_width,
            left_overlap,
            right_overlap,
            x_weights,
            &error)) {
        return false;
    }

    error.clear();

    return make_axis_fusion_weights(
        tile.valid_height,
        top_overlap,
        bottom_overlap,
        y_weights,
        &error);
}

static void test_weighted_fuse_two_dimensional_corner() {
    TilePlan plan;
    std::string error;

    const bool planned = plan_tiles(
        224,
        192,
        fusion_test_config(),
        &plan,
        &error);

    expect(planned, "2D corner plan succeeds");
    if (!planned) {
        return;
    }

    const int top_left =
        find_tile_position(plan, 0, 0);

    const int top_right =
        find_tile_position(plan, 1, 0);

    const int bottom_left =
        find_tile_position(plan, 0, 1);

    const int bottom_right =
        find_tile_position(plan, 1, 1);

    const bool found_four_tiles =
        top_left >= 0 &&
        top_right >= 0 &&
        bottom_left >= 0 &&
        bottom_right >= 0;

    expect(
        found_four_tiles,
        "2D corner test finds a 2x2 tile layout");

    if (!found_four_tiles) {
        return;
    }

    std::vector<std::vector<uint8_t>> outputs;
    outputs.reserve(plan.tiles.size());

    for (const TileSpec& tile : plan.tiles) {
        uint8_t value = 0;

        if (tile.grid_x == 0 && tile.grid_y == 0) {
            value = 10;
        } else if (
            tile.grid_x == 1 &&
            tile.grid_y == 0) {
            value = 50;
        } else if (
            tile.grid_x == 0 &&
            tile.grid_y == 1) {
            value = 100;
        } else if (
            tile.grid_x == 1 &&
            tile.grid_y == 1) {
            value = 200;
        }

        outputs.push_back(
            make_solid_valid_tile(tile, value));
    }

    FusionResult result;
    error.clear();

    const bool fused = weighted_fuse_rgb8(
        plan,
        outputs,
        &result,
        &error);

    expect(
        fused,
        "2D corner weighted fusion succeeds");

    if (!fused) {
        return;
    }

    const TileSpec& tl =
        plan.tiles[static_cast<size_t>(top_left)];

    const TileSpec& tr =
        plan.tiles[static_cast<size_t>(top_right)];

    const TileSpec& bl =
        plan.tiles[static_cast<size_t>(bottom_left)];

    const TileSpec& br =
        plan.tiles[static_cast<size_t>(bottom_right)];

    const int overlap_x =
        tl.source_right - tr.source_left;

    const int overlap_y =
        tl.source_bottom - bl.source_top;

    expect(
        overlap_x == 32,
        "2D corner has expected horizontal overlap");

    expect(
        overlap_y == 64,
        "2D corner exercises anchored vertical overlap");

    const int global_x = 100;
    const int global_y = 80;

    std::vector<float> tl_x;
    std::vector<float> tl_y;
    std::vector<float> tr_x;
    std::vector<float> tr_y;
    std::vector<float> bl_x;
    std::vector<float> bl_y;
    std::vector<float> br_x;
    std::vector<float> br_y;

    const bool axes_ok =
        build_tile_axis_weights(
            tl,
            0,
            overlap_x,
            0,
            overlap_y,
            &tl_x,
            &tl_y) &&
        build_tile_axis_weights(
            tr,
            overlap_x,
            0,
            0,
            overlap_y,
            &tr_x,
            &tr_y) &&
        build_tile_axis_weights(
            bl,
            0,
            overlap_x,
            overlap_y,
            0,
            &bl_x,
            &bl_y) &&
        build_tile_axis_weights(
            br,
            overlap_x,
            0,
            overlap_y,
            0,
            &br_x,
            &br_y);

    expect(
        axes_ok,
        "2D corner expected axis weights build successfully");

    if (!axes_ok) {
        return;
    }

    const auto weight_at =
        [global_x, global_y](
            const TileSpec& tile,
            const std::vector<float>& x_weights,
            const std::vector<float>& y_weights) {

            const int local_x =
                global_x - tile.source_left;

            const int local_y =
                global_y - tile.source_top;

            return
                x_weights[static_cast<size_t>(local_x)] *
                y_weights[static_cast<size_t>(local_y)];
        };

    const float weight_tl =
        weight_at(tl, tl_x, tl_y);

    const float weight_tr =
        weight_at(tr, tr_x, tr_y);

    const float weight_bl =
        weight_at(bl, bl_x, bl_y);

    const float weight_br =
        weight_at(br, br_x, br_y);

    const float expected_weight_sum =
        weight_tl +
        weight_tr +
        weight_bl +
        weight_br;

    const float expected_value =
        (
            10.0f * weight_tl +
            50.0f * weight_tr +
            100.0f * weight_bl +
            200.0f * weight_br
        ) / expected_weight_sum;

    const size_t pixel_index =
        static_cast<size_t>(global_y) * 224u +
        static_cast<size_t>(global_x);

    const bool finite_positive_sum =
        std::isfinite(expected_weight_sum) &&
        expected_weight_sum > 0.0f;

    expect(
        finite_positive_sum,
        "2D corner expected weight sum is finite and positive");

    expect(
        std::fabs(expected_weight_sum - 1.0f) <=
            1.0e-5f,
        "2D separable corner weights sum to one");

    bool pixel_matches = true;

    for (int channel = 0; channel < 3; ++channel) {
        const size_t rgb_index =
            pixel_index * 3u +
            static_cast<size_t>(channel);

        if (rgb_index >= result.rgb.size() ||
            !std::isfinite(result.rgb[rgb_index]) ||
            std::fabs(
                result.rgb[rgb_index] -
                expected_value) >
                1.0e-4f) {
            pixel_matches = false;
        }
    }

    expect(
        pixel_matches,
        "2D corner uses separable x*y fusion weights");
}

int main()
{
    expect(p18d::sha256_hex(nullptr, 0) ==
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "sha256 empty vector");
    const std::vector<unsigned char> abc{'a', 'b', 'c'};
    expect(p18d::sha256_hex(abc.data(), abc.size()) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "sha256 abc vector");
    std::string error;
    const std::vector<std::pair<int, int>> sizes{
        {1, 1}, {127, 127}, {128, 128}, {129, 129}, {255, 257},
        {256, 256}, {511, 513}, {1920, 1080}, {1976, 1267},
        {3200, 2136}, {4032, 3024}};
    for (const auto& size : sizes)
    {
        p18d::TilePlan plan;
        std::string error;
        expect(p18d::plan_tiles(size.first, size.second, p18d::TileConfig{}, &plan, &error),
               "planner accepts fixed profile");
        expect(has_exact_coverage(plan, size.first, size.second), "planner covers every pixel");
        p18d::PhotoRuntimeLimits limits;
        expect(p18d::validate_photo_limits(size.first, size.second, plan, limits, &error),
               "runtime limits accept supported size matrix");
        p18d::PhotoStorageEstimate estimate;
        expect(p18d::estimate_photo_storage(size.first, size.second, plan.tiles.size(),
                                             &estimate, &error),
               "storage estimate accepts supported size matrix");
        expect(estimate.required_bytes >= estimate.input_rgb_bytes + estimate.output_rgb_bytes &&
                   estimate.checkpoint_bytes > 0 && estimate.fusion_bytes > 0,
               "storage estimate includes checkpoints and fusion buffers");
    }

    p18d::TilePlan identity_plan;
    expect(p18d::plan_tiles(256, 256, p18d::TileConfig{}, &identity_plan, &error),
           "identity plan exists");
    const std::string plan_sha = p18d::tile_plan_sha256(identity_plan);
    expect(plan_sha.size() == 64 && plan_sha == p18d::tile_plan_sha256(identity_plan),
           "tile plan SHA is deterministic");
    identity_plan.tiles[0].source_left += 1;
    expect(plan_sha != p18d::tile_plan_sha256(identity_plan),
           "tile plan SHA changes with bounds");
    identity_plan.tiles[0].source_left -= 1;
    const std::string plan_json = p18d::tile_plan_json(identity_plan);
    expect(plan_json.find("\"tile_plan_sha256\":\"" + plan_sha + "\"") != std::string::npos &&
               plan_json.find("\"tile_count\":9") != std::string::npos &&
               plan_json.find("\"tile_index\":8") != std::string::npos &&
               plan_json.find("\"source_right\":256") != std::string::npos,
           "tile plan JSON records deterministic identity and every tile");

    p18d::PhotoRuntimeLimits tiny_limits;
    tiny_limits.max_pixels = 100;
    tiny_limits.max_tiles = 1;
    p18d::TilePlan over_limit_plan;
    expect(p18d::plan_tiles(128, 128, p18d::TileConfig{}, &over_limit_plan, &error),
           "over-limit fixture plan exists");
    expect(!p18d::validate_photo_limits(128, 128, over_limit_plan, tiny_limits, &error) &&
               error == "MAX_PIXELS_EXCEEDED",
           "max_pixels is enforced");
    tiny_limits.max_pixels = 128u * 128u;
    expect(!p18d::validate_photo_limits(129, 128, over_limit_plan, tiny_limits, &error) &&
               error == "PLAN_DIMENSION_MISMATCH",
           "plan identity dimensions are enforced");
    p18d::TilePlan two_tile_plan;
    expect(p18d::plan_tiles(160, 96, p18d::TileConfig{}, &two_tile_plan, &error),
           "two-tile limit fixture plan exists");
    expect(!p18d::validate_photo_limits(160, 96, two_tile_plan, tiny_limits, &error) &&
               error == "MAX_TILES_EXCEEDED",
           "max_tiles is enforced");
    expect(!p18d::validate_photo_limits(0, 1, p18d::TilePlan{}, p18d::PhotoRuntimeLimits{}, &error) &&
               error == "INVALID_IMAGE_DIMENSIONS",
           "zero dimensions are rejected by runtime guard");
    expect(!p18d::estimate_photo_storage(-1, 1, 1, nullptr, &error),
           "invalid storage estimate request is rejected");
    expect(p18d::has_required_free_space(1024, 1024, &error),
           "exact required free space is accepted");
    expect(!p18d::has_required_free_space(1025, 1024, &error) &&
               error == "INSUFFICIENT_FREE_SPACE",
           "insufficient free space is rejected");

    p18d::TilePlan small_plan;
    expect(p18d::plan_tiles(1, 1, p18d::TileConfig{}, &small_plan, &error), "one pixel plan");
    const std::vector<unsigned char> one_pixel{7, 11, 19};
    p18d::PaddedTile padded;
    expect(p18d::extract_and_pad_rgb8(one_pixel.data(), one_pixel.size(), 1, 1,
                                      small_plan.tiles.at(0), &padded, &error),
           "one pixel padding");
    expect(padded.rgb.size() == 128u * 128u * 3u, "padded tile size");
    expect(padded.padding_mode_x == "replicate" && padded.padding_mode_y == "replicate",
           "one pixel replicate fallback");
    expect(padded.rgb.front() == 7 && padded.rgb[1] == 11 && padded.rgb[2] == 19,
           "one pixel preserved");

    const int fusion_width = 129;
    const int fusion_height = 128;
    std::vector<unsigned char> fusion_input(static_cast<size_t>(fusion_width) * fusion_height * 3u);
    for (size_t i = 0; i < fusion_input.size(); ++i)
        fusion_input[i] = static_cast<unsigned char>((i * 31u + 17u) % 251u);
    p18d::TilePlan fusion_plan;
    expect(p18d::plan_tiles(fusion_width, fusion_height, p18d::TileConfig{}, &fusion_plan, &error),
           "fusion plan");
    p18d::FusionResult fusion;
    expect(p18d::identity_fuse_rgb8(fusion_plan, fusion_input.data(), fusion_input.size(), &fusion,
                                    &error),
           "identity fusion succeeds");
    expect(fusion.uncovered_pixel_count == 0, "identity fusion no uncovered pixels");
    expect(fusion.zero_weight_pixel_count == 0, "identity fusion no zero weight pixels");
    if (fusion.max_abs_error > 1e-5f)
        std::cerr << "identity max_abs_error=" << fusion.max_abs_error
                  << " tiles=" << fusion_plan.tiles.size() << "\n";
    expect(fusion.max_abs_error <= 1e-5f, "identity fusion reconstruction error");

    p18d::CheckpointIdentity expected;
    expected.task_id = "task-a";
    expected.normalized_image_sha256 = std::string(64, 'a');
    expected.tile_input_sha256 = std::string(64, 'b');
    expected.model_manifest_sha256 = std::string(64, 'c');
    expected.tile = fusion_plan.tiles.at(0);
    expected.shape = {16, 2, 16, 16};
    expected.dtype = "fp32";
    expected.byte_count = 32768;
    expected.checkpoint_sha256 = std::string(64, 'd');
    expected.finite = true;
    p18d::CheckpointIdentity actual = expected;
    std::string error_code;
    expect(p18d::validate_checkpoint_identity(expected, actual, &error_code, &error),
           "matching checkpoint identity accepted");
    actual.tile_input_sha256 = std::string(64, 'e');
    expect(!p18d::validate_checkpoint_identity(expected, actual, &error_code, &error) &&
               error_code == "CHECKPOINT_INPUT_MISMATCH",
           "different tile input rejected");
    expect(p18d::is_valid_tile_state_transition("pending", "preparing") &&
               !p18d::is_valid_tile_state_transition("pending", "decoder") &&
               !p18d::is_valid_tile_state_transition("failed", "completed"),
           "tile state transitions enforced");

test_weighted_fuse_single_tile_identity();
test_weighted_fuse_two_tiles_constant();
test_squared_sine_cosine_complement();
test_weighted_fuse_full_coverage_matrix();
test_weighted_fuse_order_independent();
test_weighted_fuse_two_dimensional_corner();

    std::cout << "phase18d_tile_core_test failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
