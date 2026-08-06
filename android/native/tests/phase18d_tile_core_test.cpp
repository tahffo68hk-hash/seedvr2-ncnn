#include "phase18d_tile_core.h"

#include <iostream>
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

    std::cout << "phase18d_tile_core_test failed=" << failed << "\n";
    return failed == 0 ? 0 : 1;
}
