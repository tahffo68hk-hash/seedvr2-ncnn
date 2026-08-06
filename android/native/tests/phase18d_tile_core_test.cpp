#include "phase18d_tile_core.h"

#include <iostream>
#include <cmath>
#include <set>
#include <utility>
#include <vector>

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

int main()
{
    expect(p18d::sha256_hex(nullptr, 0) ==
               "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
           "sha256 empty vector");
    const std::vector<unsigned char> abc{'a', 'b', 'c'};
    expect(p18d::sha256_hex(abc.data(), abc.size()) ==
               "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
           "sha256 abc vector");
    const std::vector<std::pair<int, int>> sizes{
        {1, 1}, {64, 64}, {128, 128}, {129, 128}, {256, 256},
        {320, 240}, {512, 512}, {1024, 768}, {3200, 2136}};
    for (const auto& size : sizes)
    {
        p18d::TilePlan plan;
        std::string error;
        expect(p18d::plan_tiles(size.first, size.second, p18d::TileConfig{}, &plan, &error),
               "planner accepts fixed profile");
        expect(has_exact_coverage(plan, size.first, size.second), "planner covers every pixel");
    }

    p18d::TilePlan small_plan;
    std::string error;
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

    std::cout << "phase18d_tile_core_test failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
