#include "phase18d_tile_core.h"

#include <algorithm>
#include <limits>

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

}  // namespace seededge::phase18d
