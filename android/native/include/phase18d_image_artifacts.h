#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace seededge::phase18d {

// Encodes RGB8 pixels into a PNG file.  byte_count must equal
// width * height * 3; any mismatch fails without reading past the buffer.
// The file is written atomically: <output_path>.tmp is encoded first and then
// renamed, so a failed call never leaves a partial target or a .tmp file.
bool write_rgb8_png(
    const std::string& output_path,
    const uint8_t* rgb,
    size_t byte_count,
    int width,
    int height,
    std::string* error);

// Encodes left and right RGB8 images of the same size side by side into one
// PNG of width*2 x height.  Both buffers must be exactly width*height*3 bytes;
// any mismatch fails.  Uses the same atomic write contract as write_rgb8_png.
bool write_side_by_side_png(
    const std::string& output_path,
    const uint8_t* left_rgb,
    size_t left_byte_count,
    const uint8_t* right_rgb,
    size_t right_byte_count,
    int width,
    int height,
    std::string* error);

}  // namespace seededge::phase18d