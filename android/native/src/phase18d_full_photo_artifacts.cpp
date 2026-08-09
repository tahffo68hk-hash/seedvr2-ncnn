#include "phase18d_full_photo_artifacts.h"

#include "phase18d_image_artifacts.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <regex>
#include <sstream>
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

struct ArtifactPaths {
    fs::path final_path;
    fs::path pending_path;
    fs::path temporary_path;
    fs::path backup_path;
};

using ArtifactPathArray = std::array<ArtifactPaths, 4>;

bool fail(
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

bool expected_rgb_bytes(
    int width,
    int height,
    size_t* expected,
    std::string* error)
{
    if (width <= 0 || height <= 0)
    {
        return fail(
            error,
            "image dimensions must be positive");
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
        return fail(
            error,
            "RGB image byte count overflows size_t");
    }

    return true;
}

std::string escape_json(
    const std::string& value)
{
    std::ostringstream escaped;

    for (const unsigned char character : value)
    {
        switch (character)
        {
        case '"':
            escaped << "\\\"";
            break;

        case '\\':
            escaped << "\\\\";
            break;

        case '\b':
            escaped << "\\b";
            break;

        case '\f':
            escaped << "\\f";
            break;

        case '\n':
            escaped << "\\n";
            break;

        case '\r':
            escaped << "\\r";
            break;

        case '\t':
            escaped << "\\t";
            break;

        default:
            if (character < 0x20u)
            {
                static constexpr char kHex[] =
                    "0123456789abcdef";

                escaped << "\\u00"
                        << kHex[
                               (character >> 4u) &
                               0x0fu]
                        << kHex[
                               character &
                               0x0fu];
            }
            else
            {
                escaped <<
                    static_cast<char>(character);
            }

            break;
        }
    }

    return escaped.str();
}

std::string error_result(
    const std::string& error_code,
    const std::string& error)
{
    std::ostringstream result;

    result
        << "{"
        << "\"state\":\"failed\","
        << "\"artifacts_written\":false,"
        << "\"silent_fallback\":false,"
        << "\"error_code\":\""
        << escape_json(error_code)
        << "\","
        << "\"error\":\""
        << escape_json(error)
        << "\""
        << "}";

    return result.str();
}

std::string success_result()
{
    return
        "{"
        "\"state\":\"completed\","
        "\"artifacts_written\":true,"
        "\"silent_fallback\":false,"
        "\"input_png\":\"input.png\","
        "\"output_png\":\"output.png\","
        "\"side_by_side_png\":\"side_by_side.png\""
        ",\"diff_png\":\"diff.png\""
        "}";
}

bool read_text_file(
    const fs::path& path,
    std::string* text,
    std::string* error)
{
    if (text == nullptr)
    {
        return fail(
            error,
            "text output is null");
    }

    text->clear();

    std::ifstream input(
        path,
        std::ios::binary);

    if (!input)
    {
        return fail(
            error,
            "could not open file: " +
                path.string());
    }

    std::ostringstream stream;
    stream << input.rdbuf();

    if (!input.eof() && input.fail())
    {
        return fail(
            error,
            "could not read complete file: " +
                path.string());
    }

    *text = stream.str();
    return true;
}

bool read_binary_file(
    const fs::path& path,
    std::vector<uint8_t>* bytes,
    std::string* error)
{
    if (bytes == nullptr)
    {
        return fail(
            error,
            "binary output is null");
    }

    bytes->clear();

    std::ifstream input(
        path,
        std::ios::binary);

    if (!input)
    {
        return fail(
            error,
            "could not open binary file: " +
                path.string());
    }

    input.seekg(0, std::ios::end);

    const std::streamoff end =
        input.tellg();

    if (end < 0)
    {
        return fail(
            error,
            "could not determine binary file size");
    }

    if (static_cast<uintmax_t>(end) >
        static_cast<uintmax_t>(
            std::numeric_limits<size_t>::max()))
    {
        return fail(
            error,
            "binary file is too large");
    }

    input.seekg(0, std::ios::beg);

    try
    {
        bytes->resize(
            static_cast<size_t>(end));
    }
    catch (...)
    {
        return fail(
            error,
            "could not allocate binary file buffer");
    }

    if (!bytes->empty())
    {
        input.read(
            reinterpret_cast<char*>(
                bytes->data()),
            static_cast<std::streamsize>(
                bytes->size()));

        if (!input)
        {
            bytes->clear();

            return fail(
                error,
                "could not read complete binary file");
        }
    }

    return true;
}

bool json_string_equals(
    const std::string& json,
    const std::string& key,
    const std::string& expected)
{
    const std::regex pattern(
        "\"" + key +
        R"json("\s*:\s*"([^"]*)")json");

    std::smatch match;

    return
        std::regex_search(
            json,
            match,
            pattern) &&
        match.size() >= 2u &&
        match[1].str() == expected;
}

bool json_bool_equals(
    const std::string& json,
    const std::string& key,
    bool expected)
{
    const std::regex pattern(
        "\"" + key +
        R"("\s*:\s*(true|false))");

    std::smatch match;

    if (!std::regex_search(
            json,
            match,
            pattern) ||
        match.size() < 2u)
    {
        return false;
    }

    return
        match[1].str() ==
        (expected ? "true" : "false");
}

bool json_nonnegative_int(
    const std::string& json,
    const std::string& key,
    size_t* value)
{
    if (value == nullptr)
        return false;

    const std::regex pattern(
        "\"" + key +
        R"("\s*:\s*([0-9]+))");

    std::smatch match;

    if (!std::regex_search(
            json,
            match,
            pattern) ||
        match.size() < 2u)
    {
        return false;
    }

    try
    {
        const unsigned long long parsed =
            std::stoull(match[1].str());

        if (parsed >
            static_cast<unsigned long long>(
                std::numeric_limits<size_t>::max()))
        {
            return false;
        }

        *value = static_cast<size_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

std::string json_state(
    const std::string& json)
{
    const std::regex pattern(
        R"json("state"\s*:\s*"([^"]+)")json");

    std::smatch match;

    if (!std::regex_search(
            json,
            match,
            pattern) ||
        match.size() < 2u)
    {
        return {};
    }

    return match[1].str();
}

bool looks_like_json_object(
    const std::string& value)
{
    const size_t first =
        value.find_first_not_of(
            " \t\r\n");

    const size_t last =
        value.find_last_not_of(
            " \t\r\n");

    return
        first != std::string::npos &&
        last != std::string::npos &&
        value[first] == '{' &&
        value[last] == '}';
}

std::string incomplete_pipeline_result(
    const std::string& pipeline_result)
{
    std::string state =
        json_state(pipeline_result);

    if (state != "failed" &&
        state != "cancelled")
    {
        state = "failed";
    }

    std::ostringstream result;

    result
        << "{"
        << "\"state\":\""
        << escape_json(state)
        << "\","
        << "\"artifacts_written\":false,"
        << "\"silent_fallback\":false,"
        << "\"pipeline_result\":";

    if (looks_like_json_object(
            pipeline_result))
    {
        result << pipeline_result;
    }
    else
    {
        result
            << "\""
            << escape_json(pipeline_result)
            << "\"";
    }

    result << "}";

    return result.str();
}

ArtifactPathArray make_artifact_paths(
    const fs::path& task_root)
{
    const std::array<const char*, 4> names = {
        "input.png",
        "output.png",
        "side_by_side.png",
        "diff.png",
    };

    ArtifactPathArray paths;

    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        paths[index].final_path =
            task_root / names[index];

        paths[index].pending_path =
            paths[index].final_path;

        paths[index].pending_path +=
            ".pending";

        paths[index].temporary_path =
            paths[index].pending_path;

        paths[index].temporary_path +=
            ".tmp";

        paths[index].backup_path =
            paths[index].final_path;

        paths[index].backup_path +=
            ".phase18d-backup";
    }

    return paths;
}

void remove_path_recursively(
    const fs::path& path)
{
    std::error_code ignored;
    fs::remove_all(path, ignored);
}

void cleanup_staging(
    const ArtifactPathArray& paths)
{
    for (const ArtifactPaths& artifact :
         paths)
    {
        remove_path_recursively(
            artifact.pending_path);

        remove_path_recursively(
            artifact.temporary_path);
    }
}

bool path_exists(
    const fs::path& path,
    bool* exists,
    std::string* error)
{
    if (exists == nullptr)
    {
        return fail(
            error,
            "exists output is null");
    }

    std::error_code exists_error;

    *exists =
        fs::exists(
            path,
            exists_error);

    if (exists_error)
    {
        return fail(
            error,
            "could not inspect path: " +
                path.string() +
                ": " +
                exists_error.message());
    }

    return true;
}

bool move_replace(
    const fs::path& source,
    const fs::path& destination,
    std::string* error)
{
#ifdef _WIN32
    const BOOL moved =
        MoveFileExW(
            source.c_str(),
            destination.c_str(),
            MOVEFILE_REPLACE_EXISTING |
                MOVEFILE_WRITE_THROUGH);

    if (moved == FALSE)
    {
        const std::error_code move_error(
            static_cast<int>(
                GetLastError()),
            std::system_category());

        return fail(
            error,
            "could not move artifact: " +
                source.string() +
                " -> " +
                destination.string() +
                ": " +
                move_error.message());
    }
#else
    std::error_code rename_error;

    fs::rename(
        source,
        destination,
        rename_error);

    if (rename_error)
    {
        return fail(
            error,
            "could not move artifact: " +
                source.string() +
                " -> " +
                destination.string() +
                ": " +
                rename_error.message());
    }
#endif

    return true;
}

bool promote_artifacts(
    const ArtifactPathArray& paths,
    std::string* error)
{
    std::array<bool, 3> backed_up = {
        false,
        false,
        false,
    };

    std::array<bool, 3> promoted = {
        false,
        false,
        false,
    };

    // A stale backup may contain the only surviving copy from an earlier
    // interrupted transaction. Never overwrite or delete it silently.
    for (const ArtifactPaths& artifact :
         paths)
    {
        bool backup_exists = false;

        if (!path_exists(
                artifact.backup_path,
                &backup_exists,
                error))
        {
            return false;
        }

        if (backup_exists)
        {
            return fail(
                error,
                "stale artifact backup exists: " +
                    artifact.backup_path.string());
        }
    }

    // Move existing final files aside only after all three pending images
    // have been generated successfully.
    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        bool final_exists = false;

        if (!path_exists(
                paths[index].final_path,
                &final_exists,
                error))
        {
            goto rollback;
        }

        if (!final_exists)
            continue;

        if (!move_replace(
                paths[index].final_path,
                paths[index].backup_path,
                error))
        {
            goto rollback;
        }

        backed_up[index] = true;
    }

    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        if (!move_replace(
                paths[index].pending_path,
                paths[index].final_path,
                error))
        {
            goto rollback;
        }

        promoted[index] = true;
    }

    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        if (backed_up[index])
            remove_path_recursively(
                paths[index].backup_path);
    }

    return true;

rollback:
    // Remove only final files that were created from this transaction.
    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        if (promoted[index])
            remove_path_recursively(
                paths[index].final_path);
    }

    // Restore prior final artifacts where possible. If restoration itself
    // fails, leave the backup in place instead of deleting the only copy.
    for (size_t index = 0u;
         index < paths.size();
         ++index)
    {
        if (!backed_up[index])
            continue;

        std::string ignored_error;

        if (move_replace(
                paths[index].backup_path,
                paths[index].final_path,
                &ignored_error))
        {
            backed_up[index] = false;
        }
    }

    return false;
}

bool validate_completed_task(
    const fs::path& task_root,
    int width,
    int height,
    std::vector<uint8_t>* output_rgb,
    std::string* error_code,
    std::string* error)
{
    const fs::path status_path =
        task_root / "status.json";

    bool status_exists = false;

    if (!path_exists(
            status_path,
            &status_exists,
            error))
    {
        if (error_code)
            *error_code =
                "TASK_STATUS_INSPECTION_FAILED";

        return false;
    }

    if (!status_exists)
    {
        if (error_code)
            *error_code =
                "TASK_STATUS_MISSING";

        return fail(
            error,
            "task status.json does not exist");
    }

    std::string status;

    if (!read_text_file(
            status_path,
            &status,
            error))
    {
        if (error_code)
            *error_code =
                "TASK_STATUS_READ_FAILED";

        return false;
    }

    if (!json_string_equals(
            status,
            "state",
            "completed"))
    {
        if (error_code)
            *error_code =
                "TASK_NOT_COMPLETED";

        return fail(
            error,
            "task status is not completed");
    }

    if (!json_bool_equals(
            status,
            "silent_fallback",
            false))
    {
        if (error_code)
            *error_code =
                "SILENT_FALLBACK_NOT_FALSE";

        return fail(
            error,
            "task status does not explicitly reject silent fallback");
    }

    size_t tile_count = 0u;
    size_t completed_tiles = 0u;

    if (!json_nonnegative_int(
            status,
            "tile_count",
            &tile_count) ||
        !json_nonnegative_int(
            status,
            "completed_tiles",
            &completed_tiles))
    {
        if (error_code)
            *error_code =
                "TASK_TILE_COUNTS_MISSING";

        return fail(
            error,
            "task tile counts are missing or invalid");
    }

    if (tile_count == 0u ||
        completed_tiles != tile_count)
    {
        if (error_code)
            *error_code =
                "TASK_TILE_COMPLETION_MISMATCH";

        return fail(
            error,
            "completed_tiles does not match tile_count");
    }

    size_t expected_bytes = 0u;

    if (!expected_rgb_bytes(
            width,
            height,
            &expected_bytes,
            error))
    {
        if (error_code)
            *error_code =
                "INVALID_RGB_DIMENSIONS";

        return false;
    }

    const fs::path output_path =
        task_root / "output_full.rgb";

    bool output_exists = false;

    if (!path_exists(
            output_path,
            &output_exists,
            error))
    {
        if (error_code)
            *error_code =
                "OUTPUT_FULL_INSPECTION_FAILED";

        return false;
    }

    if (!output_exists)
    {
        if (error_code)
            *error_code =
                "OUTPUT_FULL_MISSING";

        return fail(
            error,
            "output_full.rgb does not exist");
    }

    std::error_code size_error;

    const uintmax_t actual_size =
        fs::file_size(
            output_path,
            size_error);

    if (size_error)
    {
        if (error_code)
            *error_code =
                "OUTPUT_FULL_SIZE_READ_FAILED";

        return fail(
            error,
            "could not read output_full.rgb size: " +
                size_error.message());
    }

    if (actual_size !=
        static_cast<uintmax_t>(
            expected_bytes))
    {
        if (error_code)
            *error_code =
                "OUTPUT_FULL_SIZE_MISMATCH";

        return fail(
            error,
            "output_full.rgb size does not match width*height*3");
    }

    if (!read_binary_file(
            output_path,
            output_rgb,
            error))
    {
        if (error_code)
            *error_code =
                "OUTPUT_FULL_READ_FAILED";

        return false;
    }

    if (output_rgb->size() !=
        expected_bytes)
    {
        output_rgb->clear();

        if (error_code)
            *error_code =
                "OUTPUT_FULL_SIZE_MISMATCH";

        return fail(
            error,
            "output_full.rgb changed size while being read");
    }

    return true;
}

}  // namespace

std::string write_phase18d_full_photo_artifacts(
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height)
{
    if (task_root.empty())
    {
        return error_result(
            "TASK_ROOT_REQUIRED",
            "task_root must not be empty");
    }

    if (normalized_rgb == nullptr)
    {
        return error_result(
            "INVALID_RGB_INPUT",
            "normalized RGB pointer is null");
    }

    size_t expected_bytes = 0u;
    std::string error;

    if (!expected_rgb_bytes(
            width,
            height,
            &expected_bytes,
            &error))
    {
        return error_result(
            "INVALID_RGB_DIMENSIONS",
            error);
    }

    if (byte_count != expected_bytes)
    {
        return error_result(
            "RGB_BUFFER_SIZE_MISMATCH",
            "normalized RGB byte count does not match width*height*3");
    }

    const fs::path task_path(
        task_root);

    std::vector<uint8_t> output_rgb;
    std::string error_code;

    if (!validate_completed_task(
            task_path,
            width,
            height,
            &output_rgb,
            &error_code,
            &error))
    {
        return error_result(
            error_code,
            error);
    }

    const ArtifactPathArray paths =
        make_artifact_paths(
            task_path);

    if (!write_rgb8_png(
            paths[0].pending_path.string(),
            normalized_rgb,
            byte_count,
            width,
            height,
            &error))
    {
        cleanup_staging(paths);

        return error_result(
            "INPUT_PNG_WRITE_FAILED",
            error);
    }

    if (!write_rgb8_png(
            paths[1].pending_path.string(),
            output_rgb.data(),
            output_rgb.size(),
            width,
            height,
            &error))
    {
        cleanup_staging(paths);

        return error_result(
            "OUTPUT_PNG_WRITE_FAILED",
            error);
    }

    if (!write_side_by_side_png(
            paths[2].pending_path.string(),
            normalized_rgb,
            byte_count,
            output_rgb.data(),
            output_rgb.size(),
            width,
            height,
            &error))
    {
        cleanup_staging(paths);

        return error_result(
            "SIDE_BY_SIDE_PNG_WRITE_FAILED",
            error);
    }

    std::vector<uint8_t> diff_rgb(output_rgb.size());
    for (size_t index = 0u; index < diff_rgb.size(); ++index)
    {
        const int delta = std::abs(
            static_cast<int>(output_rgb[index]) -
            static_cast<int>(normalized_rgb[index]));
        diff_rgb[index] = static_cast<uint8_t>(
            std::min(255, delta * 4));
    }

    if (!write_rgb8_png(
            paths[3].pending_path.string(),
            diff_rgb.data(),
            diff_rgb.size(),
            width,
            height,
            &error))
    {
        cleanup_staging(paths);

        return error_result(
            "DIFF_PNG_WRITE_FAILED",
            error);
    }

    if (!promote_artifacts(
            paths,
            &error))
    {
        cleanup_staging(paths);

        return error_result(
            "ARTIFACT_PROMOTION_FAILED",
            error);
    }

    cleanup_staging(paths);
    return success_result();
}

std::string run_phase18d_full_photo_artifacts_with_hooks(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json,
    const Phase18dFullPhotoHooks& hooks)
{
    const std::string pipeline_result =
        run_phase18d_full_photo_with_hooks(
            model_root,
            task_root,
            normalized_rgb,
            byte_count,
            width,
            height,
            request_json,
            hooks);

    if (!json_string_equals(
            pipeline_result,
            "state",
            "completed"))
    {
        if (hooks.on_progress)
        {
            hooks.on_progress(
                hooks.context,
                json_string_equals(pipeline_result, "state", "cancelled")
                    ? "CANCELLED"
                    : "FAILED",
                0,
                0);
        }
        return incomplete_pipeline_result(
            pipeline_result);
    }

    if (hooks.on_progress)
        hooks.on_progress(hooks.context, "ENCODING_PNG", 0, 0);

    const std::string artifact_result =
        write_phase18d_full_photo_artifacts(
            task_root,
            normalized_rgb,
            byte_count,
            width,
            height);

    if (hooks.on_progress)
    {
        hooks.on_progress(
            hooks.context,
            json_string_equals(artifact_result, "state", "completed")
                ? "COMPLETED"
                : "FAILED",
            0,
            0);
    }
    return artifact_result;
}

std::string run_phase18d_full_photo_artifacts(
    const std::string& model_root,
    const std::string& task_root,
    const uint8_t* normalized_rgb,
    size_t byte_count,
    int width,
    int height,
    const std::string& request_json)
{
    const std::string pipeline_result =
        run_phase18d_full_photo(
            model_root,
            task_root,
            normalized_rgb,
            byte_count,
            width,
            height,
            request_json);

    if (!json_string_equals(
            pipeline_result,
            "state",
            "completed"))
    {
        return incomplete_pipeline_result(
            pipeline_result);
    }

    return write_phase18d_full_photo_artifacts(
        task_root,
        normalized_rgb,
        byte_count,
        width,
        height);
}

}  // namespace seededge::phase18d
