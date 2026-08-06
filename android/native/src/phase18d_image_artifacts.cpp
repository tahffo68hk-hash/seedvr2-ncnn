#include "phase18d_image_artifacts.h"

#include "stb_image_write.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <new>
#include <string>
#include <system_error>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#endif

namespace fs = std::filesystem;

namespace seededge::phase18d {
namespace {

bool artifact_fail(
    std::string* error,
    const std::string& message)
{
    if (error != nullptr)
        *error = message;

    return false;
}

bool checked_multiply(
    size_t left,
    size_t right,
    size_t* result)
{
    if (result == nullptr)
        return false;

    if (left != 0u &&
        right >
            std::numeric_limits<size_t>::max() /
                left)
    {
        return false;
    }

    *result = left * right;
    return true;
}

bool expected_rgb_byte_count(
    int width,
    int height,
    size_t* expected,
    std::string* error)
{
    if (width <= 0 || height <= 0)
    {
        return artifact_fail(
            error,
            "image dimensions must be positive");
    }

    if (width >
        std::numeric_limits<int>::max() / 3)
    {
        return artifact_fail(
            error,
            "image row stride exceeds int range");
    }

    size_t pixel_count = 0u;

    if (!checked_multiply(
            static_cast<size_t>(width),
            static_cast<size_t>(height),
            &pixel_count) ||
        !checked_multiply(
            pixel_count,
            3u,
            expected))
    {
        return artifact_fail(
            error,
            "RGB image byte count overflows size_t");
    }

    return true;
}

struct PngBuffer {
    std::vector<uint8_t> bytes;
    bool failed = false;
};

void append_png_bytes(
    void* context_pointer,
    void* data_pointer,
    int byte_count)
{
    auto* context =
        static_cast<PngBuffer*>(context_pointer);

    if (context == nullptr ||
        context->failed)
    {
        return;
    }

    if (data_pointer == nullptr ||
        byte_count <= 0)
    {
        context->failed = true;
        return;
    }

    const size_t append_count =
        static_cast<size_t>(byte_count);

    if (append_count >
        std::numeric_limits<size_t>::max() -
            context->bytes.size())
    {
        context->failed = true;
        return;
    }

    const auto* source =
        static_cast<const uint8_t*>(data_pointer);

    try
    {
        context->bytes.insert(
            context->bytes.end(),
            source,
            source + append_count);
    }
    catch (...)
    {
        // No exception may escape through the C callback boundary.
        context->failed = true;
    }
}

bool encode_rgb8_png(
    const uint8_t* rgb,
    int width,
    int height,
    std::vector<uint8_t>* encoded,
    std::string* error)
{
    if (encoded == nullptr)
    {
        return artifact_fail(
            error,
            "encoded PNG output is null");
    }

    encoded->clear();

    if (rgb == nullptr)
    {
        return artifact_fail(
            error,
            "RGB input pointer is null");
    }

    if (width <= 0 || height <= 0)
    {
        return artifact_fail(
            error,
            "image dimensions must be positive");
    }

    if (width >
        std::numeric_limits<int>::max() / 3)
    {
        return artifact_fail(
            error,
            "PNG row stride exceeds int range");
    }

    const int stride_bytes = width * 3;

    PngBuffer buffer;

    const int encoded_ok =
        stbi_write_png_to_func(
            &append_png_bytes,
            &buffer,
            width,
            height,
            3,
            rgb,
            stride_bytes);

    if (encoded_ok == 0 ||
        buffer.failed ||
        buffer.bytes.empty())
    {
        return artifact_fail(
            error,
            "stb failed to encode RGB8 PNG");
    }

    *encoded = std::move(buffer.bytes);
    return true;
}

class TemporaryFileGuard {
public:
    explicit TemporaryFileGuard(fs::path path)
        : path_(std::move(path))
    {
    }

    TemporaryFileGuard(
        const TemporaryFileGuard&) = delete;

    TemporaryFileGuard& operator=(
        const TemporaryFileGuard&) = delete;

    ~TemporaryFileGuard()
    {
        if (committed_)
            return;

        std::error_code ignored;
        fs::remove(path_, ignored);
    }

    void commit()
    {
        committed_ = true;
    }

private:
    fs::path path_;
    bool committed_ = false;
};

bool replace_file_atomically(
    const fs::path& temporary_path,
    const fs::path& output_path,
    std::string* error)
{
#ifdef _WIN32
    const BOOL replaced =
        MoveFileExW(
            temporary_path.c_str(),
            output_path.c_str(),
            MOVEFILE_REPLACE_EXISTING |
                MOVEFILE_WRITE_THROUGH);

    if (replaced == FALSE)
    {
        const std::error_code move_error(
            static_cast<int>(GetLastError()),
            std::system_category());

        return artifact_fail(
            error,
            "could not atomically replace PNG target: " +
                move_error.message());
    }
#else
    std::error_code rename_error;

    // temporary_path is placed beside output_path, so this is a same-
    // filesystem rename. POSIX rename replaces an existing regular file.
    fs::rename(
        temporary_path,
        output_path,
        rename_error);

    if (rename_error)
    {
        return artifact_fail(
            error,
            "could not atomically replace PNG target: " +
                rename_error.message());
    }
#endif

    return true;
}

bool write_encoded_png_atomically(
    const fs::path& output_path,
    const std::vector<uint8_t>& encoded,
    std::string* error)
{
    if (output_path.empty())
    {
        return artifact_fail(
            error,
            "PNG output path is empty");
    }

    if (encoded.empty())
    {
        return artifact_fail(
            error,
            "encoded PNG data is empty");
    }

    if (encoded.size() >
        static_cast<size_t>(
            std::numeric_limits<std::streamsize>::max()))
    {
        return artifact_fail(
            error,
            "encoded PNG is too large for stream write");
    }

    fs::path temporary_path = output_path;
    temporary_path += ".tmp";

    // A stale temporary artifact must never be reused or appended.
    std::error_code remove_error;
    fs::remove(temporary_path, remove_error);

    if (remove_error)
    {
        return artifact_fail(
            error,
            "could not remove stale PNG temporary file: " +
                remove_error.message());
    }

    TemporaryFileGuard temporary_guard(
        temporary_path);

    {
        std::ofstream output(
            temporary_path,
            std::ios::binary |
                std::ios::trunc);

        if (!output)
        {
            return artifact_fail(
                error,
                "could not open PNG temporary file");
        }

        output.write(
            reinterpret_cast<const char*>(
                encoded.data()),
            static_cast<std::streamsize>(
                encoded.size()));

        if (!output)
        {
            return artifact_fail(
                error,
                "could not write complete PNG temporary file");
        }

        output.flush();

        if (!output)
        {
            return artifact_fail(
                error,
                "could not flush PNG temporary file");
        }

        output.close();

        if (!output)
        {
            return artifact_fail(
                error,
                "could not close PNG temporary file");
        }
    }

    if (!replace_file_atomically(
            temporary_path,
            output_path,
            error))
    {
        return false;
    }

    temporary_guard.commit();
    return true;
}

}  // namespace

bool write_rgb8_png(
    const std::string& output_path,
    const uint8_t* rgb,
    size_t byte_count,
    int width,
    int height,
    std::string* error)
{
    if (error != nullptr)
        error->clear();

    if (rgb == nullptr)
    {
        return artifact_fail(
            error,
            "RGB input pointer is null");
    }

    size_t expected_bytes = 0u;

    if (!expected_rgb_byte_count(
            width,
            height,
            &expected_bytes,
            error))
    {
        return false;
    }

    if (byte_count != expected_bytes)
    {
        return artifact_fail(
            error,
            "RGB byte count does not match width*height*3");
    }

    std::vector<uint8_t> encoded;

    if (!encode_rgb8_png(
            rgb,
            width,
            height,
            &encoded,
            error))
    {
        return false;
    }

    return write_encoded_png_atomically(
        fs::path(output_path),
        encoded,
        error);
}

bool write_side_by_side_png(
    const std::string& output_path,
    const uint8_t* left_rgb,
    size_t left_byte_count,
    const uint8_t* right_rgb,
    size_t right_byte_count,
    int width,
    int height,
    std::string* error)
{
    if (error != nullptr)
        error->clear();

    if (left_rgb == nullptr ||
        right_rgb == nullptr)
    {
        return artifact_fail(
            error,
            "side-by-side RGB input pointer is null");
    }

    size_t expected_side_bytes = 0u;

    if (!expected_rgb_byte_count(
            width,
            height,
            &expected_side_bytes,
            error))
    {
        return false;
    }

    if (left_byte_count != expected_side_bytes)
    {
        return artifact_fail(
            error,
            "left RGB byte count does not match width*height*3");
    }

    if (right_byte_count != expected_side_bytes)
    {
        return artifact_fail(
            error,
            "right RGB byte count does not match width*height*3");
    }

    if (width >
        std::numeric_limits<int>::max() / 2)
    {
        return artifact_fail(
            error,
            "side-by-side width exceeds int range");
    }

    const int combined_width = width * 2;

    size_t combined_bytes = 0u;

    if (!expected_rgb_byte_count(
            combined_width,
            height,
            &combined_bytes,
            error))
    {
        return false;
    }

    std::vector<uint8_t> combined;

    try
    {
        combined.resize(combined_bytes);
    }
    catch (const std::bad_alloc&)
    {
        return artifact_fail(
            error,
            "could not allocate side-by-side RGB buffer");
    }
    catch (...)
    {
        return artifact_fail(
            error,
            "unexpected side-by-side RGB allocation failure");
    }

    const size_t side_row_bytes =
        static_cast<size_t>(width) * 3u;

    const size_t combined_row_bytes =
        static_cast<size_t>(combined_width) * 3u;

    for (int y = 0; y < height; ++y)
    {
        const size_t side_offset =
            static_cast<size_t>(y) *
            side_row_bytes;

        const size_t combined_offset =
            static_cast<size_t>(y) *
            combined_row_bytes;

        std::copy_n(
            left_rgb + side_offset,
            side_row_bytes,
            combined.data() + combined_offset);

        std::copy_n(
            right_rgb + side_offset,
            side_row_bytes,
            combined.data() +
                combined_offset +
                side_row_bytes);
    }

    return write_rgb8_png(
        output_path,
        combined.data(),
        combined.size(),
        combined_width,
        height,
        error);
}

}  // namespace seededge::phase18d
