#pragma once

#include "phase18d_tile_core.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace seededge::phase18d {

struct SeamMetrics
{
    bool bounds_valid = false;
    int duplicate_bounds = 0;
    size_t vertical_samples = 0;
    size_t horizontal_samples = 0;
    double vertical_mean_gradient = 0.;
    double vertical_p95_gradient = 0.;
    double vertical_max_gradient = 0.;
    double vertical_reference_mean_gradient = 0.;
    double horizontal_mean_gradient = 0.;
    double horizontal_p95_gradient = 0.;
    double horizontal_max_gradient = 0.;
    double horizontal_reference_mean_gradient = 0.;
};

bool analyze_seams_rgb8(const uint8_t* rgb, size_t byte_count, int width, int height,
                        const TilePlan& plan, SeamMetrics* metrics, std::string* error);

}  // namespace seededge::phase18d
