#include "phase18d_full_photo_artifacts.h"

#include "stb_image.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

using seededge::phase18d::PaddedTile;
using seededge::phase18d::Phase18dFullPhotoHooks;
using seededge::phase18d::Phase18dTileExecutionResult;
using seededge::phase18d::TileSpec;
using seededge::phase18d::run_phase18d_full_photo_with_hooks;
using seededge::phase18d::run_phase18d_full_photo_artifacts_with_hooks;
using seededge::phase18d::write_phase18d_full_photo_artifacts;

int failed = 0;

void expect(bool condition, const std::string& name)
{
    if (condition)
    {
        std::cout << "PASS " << name << "\n";
    }
    else
    {
        std::cout << "FAIL " << name << "\n";
        ++failed;
    }
}

bool contains(const std::string& text, const std::string& expected)
{
    return text.find(expected) != std::string::npos;
}

std::string read_text(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};

    std::ostringstream stream;
    stream << input.rdbuf();
    return stream.str();
}

std::vector<uint8_t> read_bytes(const fs::path& path)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return {};

    return std::vector<uint8_t>(
        std::istreambuf_iterator<char>(input),
        std::istreambuf_iterator<char>());
}

fs::path make_task_root(const std::string& task_id)
{
    static int sequence = 0;

    const fs::path base =
        fs::temp_directory_path() /
        (
            "seededge_phase18d_full_photo_artifacts_" +
            std::to_string(++sequence)
        );

    std::error_code ec;
    fs::remove_all(base, ec);

    const fs::path task_root =
        base / "tasks" / task_id;

    fs::create_directories(task_root, ec);
    return task_root;
}

struct FakeContext {
    int manifest_calls = 0;
    int executor_calls = 0;

    int fail_on_executor_call = 0;
    int cancel_after_executor_calls = -1;

    uint8_t output_value = 77;
};

bool fake_validate_manifest(
    void* opaque,
    const std::string& model_root,
    std::string* manifest_sha256,
    std::string* error)
{
    auto* context = static_cast<FakeContext*>(opaque);

    ++context->manifest_calls;

    if (model_root.empty())
    {
        if (error)
            *error = "model_root is empty";
        return false;
    }

    if (manifest_sha256)
        *manifest_sha256 = "fake-manifest-sha256";

    if (error)
        error->clear();

    return true;
}

bool fake_execute_tile(
    void* opaque,
    const std::string& model_root,
    const std::string& tile_root,
    const TileSpec& tile,
    const PaddedTile& padded,
    Phase18dTileExecutionResult* result,
    std::string* error)
{
    auto* context = static_cast<FakeContext*>(opaque);

    ++context->executor_calls;

    if (context->fail_on_executor_call ==
        context->executor_calls)
    {
        if (error)
            *error = "injected tile executor failure";
        return false;
    }

    if (model_root.empty() ||
        tile_root.empty() ||
        result == nullptr)
    {
        if (error)
            *error = "invalid fake executor arguments";
        return false;
    }

    if (padded.rgb.size() != 128u * 128u * 3u)
    {
        if (error)
            *error = "padded tile is not 128x128 RGB8";
        return false;
    }

    const size_t expected_output_bytes =
        static_cast<size_t>(tile.valid_width) *
        static_cast<size_t>(tile.valid_height) *
        3u;

    result->encoder_completed = true;
    result->dit_block_completed = 32;
    result->scheduler_completed = true;
    result->decoder_completed = true;
    result->finite = true;
    result->nan_count = 0u;
    result->inf_count = 0u;

    result->output_valid_rgb.assign(
        expected_output_bytes,
        context->output_value);

    result->pipeline_result_json =
        "{\"model_verified\":true,"
        "\"vae_encoder_executed\":true,"
        "\"dit_blocks_executed\":32,"
        "\"vae_decoder_executed\":true,"
        "\"finite\":true,"
        "\"silent_fallback\":false,"
        "\"error_stage\":null}";

    if (error)
        error->clear();

    return true;
}

bool fake_is_cancelled(void* opaque)
{
    const auto* context =
        static_cast<const FakeContext*>(opaque);

    return
        context->cancel_after_executor_calls >= 0 &&
        context->executor_calls >=
            context->cancel_after_executor_calls;
}

Phase18dFullPhotoHooks make_hooks(FakeContext* context)
{
    Phase18dFullPhotoHooks hooks;
    hooks.context = context;
    hooks.validate_manifest = &fake_validate_manifest;
    hooks.execute_tile = &fake_execute_tile;
    hooks.is_cancelled = &fake_is_cancelled;
    return hooks;
}

std::vector<uint8_t> make_input(
    int width,
    int height,
    uint8_t value)
{
    return std::vector<uint8_t>(
        static_cast<size_t>(width) *
            static_cast<size_t>(height) *
            3u,
        value);
}

struct DecodedPng
{
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

bool decode_png(const fs::path& path, DecodedPng* output)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char* pixels = stbi_load(path.string().c_str(), &width, &height, &channels, 3);
    if (!pixels || output == nullptr)
    {
        stbi_image_free(pixels);
        return false;
    }
    output->width = width;
    output->height = height;
    output->rgb.assign(pixels, pixels + static_cast<size_t>(width) * height * 3u);
    stbi_image_free(pixels);
    return true;
}

bool all_pixels_equal(const DecodedPng& decoded, uint8_t value)
{
    for (uint8_t pixel : decoded.rgb)
        if (pixel != value)
            return false;
    return true;
}

bool region_matches_value(
    const DecodedPng& decoded,
    int region_left,
    int region_width,
    int region_height,
    uint8_t value)
{
    if (region_left < 0 ||
        region_left + region_width > decoded.width ||
        region_height > decoded.height)
    {
        return false;
    }
    for (int y = 0; y < region_height; ++y)
    {
        for (int x = 0; x < region_width; ++x)
        {
            const size_t index =
                (static_cast<size_t>(y) * decoded.width +
                 static_cast<size_t>(region_left) + x) *
                3u;
            if (decoded.rgb[index] != value ||
                decoded.rgb[index + 1] != value ||
                decoded.rgb[index + 2] != value)
            {
                return false;
            }
        }
    }
    return true;
}

bool any_pending_or_tmp_left(const fs::path& task_root)
{
    for (const fs::directory_entry& entry : fs::directory_iterator(task_root))
    {
        const std::string name = entry.path().filename().string();
        if (name.find(".pending") != std::string::npos ||
            name.find(".tmp") != std::string::npos)
        {
            return true;
        }
    }
    return false;
}

void expect_no_final_pngs(
    const fs::path& task_root,
    const std::string& name)
{
    expect(!fs::exists(task_root / "input.png"), name + " leaves no input.png");
    expect(!fs::exists(task_root / "output.png"), name + " leaves no output.png");
    expect(!fs::exists(task_root / "side_by_side.png"), name + " leaves no side_by_side.png");
}

std::string make_request_json(const std::string& task_id)
{
    return "{\"task_id\":\"" + task_id +
           "\",\"model_version\":\"phase18d-host-v2\"}";
}

void test_success_generates_three_pngs()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-success");

    const std::vector<uint8_t> input =
        make_input(width, height, 13);

    FakeContext context;
    context.output_value = 77;

    const std::string result =
        run_phase18d_full_photo_artifacts_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            make_request_json("artifacts-success"),
            make_hooks(&context));

    expect(
        context.manifest_calls == 1,
        "success validates manifest exactly once");

    expect(
        context.executor_calls == 2,
        "success executes exactly two tiles");

    const fs::path input_png = task_root / "input.png";
    const fs::path output_png = task_root / "output.png";
    const fs::path side_by_side_png = task_root / "side_by_side.png";

    expect(fs::exists(input_png), "input.png exists");
    expect(fs::exists(output_png), "output.png exists");
    expect(fs::exists(side_by_side_png), "side_by_side.png exists");

    expect(
        read_bytes(task_root / "output_full.rgb").size() ==
            static_cast<size_t>(width) * height * 3u,
        "output_full.rgb has exact byte count");

    DecodedPng input_decoded;
    DecodedPng output_decoded;
    DecodedPng side_decoded;

    expect(decode_png(input_png, &input_decoded),
           "input.png decodes");
    expect(decode_png(output_png, &output_decoded),
           "output.png decodes");
    expect(decode_png(side_by_side_png, &side_decoded),
           "side_by_side.png decodes");

    expect(
        input_decoded.width == width &&
        input_decoded.height == height,
        "input.png is 160x96");
    expect(
        output_decoded.width == width &&
        output_decoded.height == height,
        "output.png is 160x96");
    expect(
        side_decoded.width == width * 2 &&
        side_decoded.height == height,
        "side_by_side.png is 320x96");

    expect(
        all_pixels_equal(input_decoded, 13),
        "input.png all channels are 13");
    expect(
        all_pixels_equal(output_decoded, 77),
        "output.png all channels are 77");
    expect(
        region_matches_value(side_decoded, 0, width, height, 13),
        "side_by_side left half is 13");
    expect(
        region_matches_value(side_decoded, width, width, height, 77),
        "side_by_side right half is 77");

    expect(
        !any_pending_or_tmp_left(task_root),
        "success leaves no pending or tmp files");

    expect(
        contains(result, "\"state\":\"completed\""),
        "success result reports completed state");
    expect(
        contains(result, "\"artifacts_written\":true"),
        "success result reports artifacts written");
    expect(
        contains(result, "\"silent_fallback\":false"),
        "success result explicitly rejects silent fallback");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_second_tile_failure_generates_no_pngs()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-failure");

    const std::vector<uint8_t> input =
        make_input(width, height, 19);

    FakeContext context;
    context.fail_on_executor_call = 2;

    const std::string result =
        run_phase18d_full_photo_artifacts_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            make_request_json("artifacts-failure"),
            make_hooks(&context));

    expect(
        contains(result, "\"state\":\"failed\""),
        "failure result reports failed state");

    expect(
        contains(result, "\"artifacts_written\":false"),
        "failure result reports no artifacts");

    expect_no_final_pngs(
        task_root,
        "tile failure");

    expect(
        !any_pending_or_tmp_left(task_root),
        "tile failure leaves no pending or tmp files");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_cancellation_generates_no_pngs()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-cancelled");

    const std::vector<uint8_t> input =
        make_input(width, height, 23);

    FakeContext context;
    context.cancel_after_executor_calls = 1;

    const std::string result =
        run_phase18d_full_photo_artifacts_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            make_request_json("artifacts-cancelled"),
            make_hooks(&context));

    expect(
        context.executor_calls == 1,
        "cancellation prevents second tile execution");

    expect(
        contains(result, "\"state\":\"cancelled\""),
        "cancelled result reports cancelled state");

    expect(
        contains(result, "\"artifacts_written\":false"),
        "cancelled result reports no artifacts");

    expect_no_final_pngs(
        task_root,
        "cancellation");

    expect(
        !any_pending_or_tmp_left(task_root),
        "cancellation leaves no pending or tmp files");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_corrupted_output_full_rgb_fails()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-wrong-size");

    const std::vector<uint8_t> input =
        make_input(width, height, 31);

    FakeContext context;
    context.output_value = 77;

    const std::string orchestration =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            make_request_json("artifacts-wrong-size"),
            make_hooks(&context));

    expect(
        contains(orchestration, "\"state\":\"completed\""),
        "orchestration completes before corruption");

    {
        std::ofstream output(
            task_root / "output_full.rgb",
            std::ios::binary | std::ios::trunc);

        const uint8_t invalid[] = {1, 2, 3};
        output.write(
            reinterpret_cast<const char*>(invalid),
            sizeof(invalid));
    }

    const std::string result =
        write_phase18d_full_photo_artifacts(
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height);

    expect(
        contains(result, "\"state\":\"failed\""),
        "corrupted output reports failed state");

    expect(
        contains(result, "OUTPUT_FULL_SIZE_MISMATCH"),
        "corrupted output exposes stable size error code");

    expect_no_final_pngs(
        task_root,
        "corrupted output");

    expect(
        !any_pending_or_tmp_left(task_root),
        "corrupted output leaves no pending or tmp files");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_task_not_completed_fails()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-not-completed");

    const std::vector<uint8_t> input =
        make_input(width, height, 37);

    const std::vector<uint8_t> valid_rgb =
        make_input(width, height, 41);

    {
        std::ofstream status_file(task_root / "status.json");
        status_file << "{"
                    << "\"schema_version\":1,"
                    << "\"mode\":\"phase18d_full_photo\","
                    << "\"state\":\"failed\","
                    << "\"tile_count\":2,"
                    << "\"completed_tiles\":1,"
                    << "\"silent_fallback\":false"
                    << "}";
    }
    {
        std::ofstream output(
            task_root / "output_full.rgb",
            std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(valid_rgb.data()),
            static_cast<std::streamsize>(valid_rgb.size()));
    }

    const std::string result =
        write_phase18d_full_photo_artifacts(
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height);

    expect(
        contains(result, "\"state\":\"failed\""),
        "non-completed task reports failed state");

    expect(
        contains(result, "TASK_NOT_COMPLETED"),
        "non-completed task exposes stable task state error");

    expect_no_final_pngs(
        task_root,
        "non-completed task");

    expect(
        !any_pending_or_tmp_left(task_root),
        "non-completed task leaves no pending or tmp files");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_cancelled_task_fails()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-task-cancelled");

    const std::vector<uint8_t> input =
        make_input(width, height, 43);

    const std::vector<uint8_t> valid_rgb =
        make_input(width, height, 47);

    {
        std::ofstream status_file(task_root / "status.json");
        status_file << "{"
                    << "\"schema_version\":1,"
                    << "\"mode\":\"phase18d_full_photo\","
                    << "\"state\":\"cancelled\","
                    << "\"tile_count\":2,"
                    << "\"completed_tiles\":1,"
                    << "\"silent_fallback\":false"
                    << "}";
    }
    {
        std::ofstream output(
            task_root / "output_full.rgb",
            std::ios::binary);
        output.write(
            reinterpret_cast<const char*>(valid_rgb.data()),
            static_cast<std::streamsize>(valid_rgb.size()));
    }

    const std::string result =
        write_phase18d_full_photo_artifacts(
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height);

    expect(
        contains(result, "\"state\":\"failed\""),
        "cancelled task reports failed state");

    expect(
        contains(result, "TASK_NOT_COMPLETED"),
        "cancelled task exposes stable task state error");

    expect_no_final_pngs(
        task_root,
        "cancelled task");

    expect(
        !any_pending_or_tmp_left(task_root),
        "cancelled task leaves no pending or tmp files");

    std::error_code ec;
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

void test_png_staging_failure_cleans_up()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const fs::path task_root =
        make_task_root("artifacts-staging");

    const std::vector<uint8_t> input =
        make_input(width, height, 53);

    FakeContext context;
    context.output_value = 77;

    const std::string orchestration =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            make_request_json("artifacts-staging"),
            make_hooks(&context));

    expect(
        contains(orchestration, "\"state\":\"completed\""),
        "orchestration completes before staging conflict");

    // A non-empty directory cannot be removed by a plain fs::remove, so the
    // second PNG staging step must fail deterministically.
    const fs::path blocked_tmp =
        task_root / "output.png.pending.tmp";

    std::error_code ec;
    fs::create_directories(
        blocked_tmp / "non_empty",
        ec);

    const std::string result =
        write_phase18d_full_photo_artifacts(
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height);

    expect(
        contains(result, "\"state\":\"failed\""),
        "staging conflict reports failed state");

    expect(
        contains(result, "\"artifacts_written\":false"),
        "staging conflict reports no artifacts");

    expect_no_final_pngs(
        task_root,
        "staging conflict");

    expect(
        !fs::exists(task_root / "input.png.pending"),
        "staging conflict removes staged input.png.pending");

    expect(
        !fs::exists(task_root / "side_by_side.png.pending"),
        "staging conflict leaves no staged side_by_side.png.pending");

    fs::remove_all(blocked_tmp, ec);
    fs::remove_all(task_root.parent_path().parent_path(), ec);
}

}  // namespace

int main()
{
    test_success_generates_three_pngs();
    test_second_tile_failure_generates_no_pngs();
    test_cancellation_generates_no_pngs();
    test_corrupted_output_full_rgb_fails();
    test_task_not_completed_fails();
    test_cancelled_task_fails();
    test_png_staging_failure_cleans_up();

    std::cout << "failed=" << failed << "\n";
    return failed;
}
