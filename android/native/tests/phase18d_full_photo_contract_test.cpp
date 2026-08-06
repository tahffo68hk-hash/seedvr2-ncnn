#include "minimal_video_pipeline.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;


namespace {

using seededge::phase18d::PaddedTile;
using seededge::phase18d::Phase18dFullPhotoHooks;
using seededge::phase18d::Phase18dTileExecutionResult;
using seededge::phase18d::TileSpec;
using seededge::phase18d::run_phase18d_full_photo_with_hooks;

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

bool contains(
    const std::string& text,
    const std::string& expected)
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
            "seededge_phase18d_full_photo_contract_" +
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
    int wrong_size_on_executor_call = 0;
    int cancel_after_executor_calls = -1;

    uint8_t output_value = 77;

    std::vector<TileSpec> executed_tiles;
    std::vector<std::string> tile_roots;
};

bool fake_validate_manifest(
    void* opaque,
    const std::string& model_root,
    std::string* manifest_sha256,
    std::string* error)
{
    auto* context =
        static_cast<FakeContext*>(opaque);

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
    auto* context =
        static_cast<FakeContext*>(opaque);

    ++context->executor_calls;

    context->executed_tiles.push_back(tile);
    context->tile_roots.push_back(tile_root);

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

    const size_t expected_padded_bytes =
        128u * 128u * 3u;

    if (padded.rgb.size() != expected_padded_bytes)
    {
        if (error)
            *error = "padded tile is not 128x128 RGB8";
        return false;
    }

    const size_t expected_output_bytes =
        static_cast<size_t>(tile.valid_width) *
        static_cast<size_t>(tile.valid_height) *
        3u;

    size_t returned_bytes = expected_output_bytes;

    if (context->wrong_size_on_executor_call ==
            context->executor_calls &&
        returned_bytes > 0u)
    {
        --returned_bytes;
    }

    result->encoder_completed = true;
    result->dit_block_completed = 32;
    result->scheduler_completed = true;
    result->decoder_completed = true;
    result->finite = true;
    result->nan_count = 0u;
    result->inf_count = 0u;

    result->output_valid_rgb.assign(
        returned_bytes,
        context->output_value);

    result->pipeline_result_json =
        "{"
        "\"model_verified\":true,"
        "\"vae_encoder_executed\":true,"
        "\"dit_blocks_executed\":32,"
        "\"vae_decoder_executed\":true,"
        "\"finite\":true,"
        "\"silent_fallback\":false,"
        "\"error_stage\":null"
        "}";

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
    hooks.validate_manifest =
        &fake_validate_manifest;
    hooks.execute_tile =
        &fake_execute_tile;
    hooks.is_cancelled =
        &fake_is_cancelled;
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

void expect_no_final_output(
    const fs::path& task_root,
    const std::string& name)
{
    expect(
        !fs::exists(task_root / "output_full.rgb"),
        name);
}

void test_successful_two_tile_orchestration()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const std::string task_id =
        "full-photo-success";

    const fs::path task_root =
        make_task_root(task_id);

    const auto input =
        make_input(width, height, 13);

    FakeContext context;
    context.output_value = 77;

    const std::string result =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            "{"
            "\"task_id\":\"full-photo-success\","
            "\"model_version\":\"phase18d-host-v2\""
            "}",
            make_hooks(&context));

    expect(
        context.manifest_calls == 1,
        "manifest is validated exactly once");

    expect(
        context.executor_calls == 2,
        "160x96 executes exactly two tiles");

    expect(
        context.executed_tiles.size() == 2u,
        "two tile specifications are recorded");

    if (context.executed_tiles.size() == 2u)
    {
        const TileSpec& first =
            context.executed_tiles[0];

        const TileSpec& second =
            context.executed_tiles[1];

        expect(
            first.tile_index == 0 &&
            first.source_left == 0 &&
            first.source_top == 0 &&
            first.source_right == 128 &&
            first.source_bottom == 96,
            "first tile bounds are correct");

        expect(
            second.tile_index == 1 &&
            second.source_left == 32 &&
            second.source_top == 0 &&
            second.source_right == 160 &&
            second.source_bottom == 96,
            "anchored second tile bounds are correct");
    }

    expect(
        context.tile_roots.size() == 2u,
        "two tile roots are passed to executor");

    if (context.tile_roots.size() == 2u)
    {
        expect(
            fs::path(context.tile_roots[0]).filename() ==
                "tile_0000",
            "first tile directory uses zero-padded name");

        expect(
            fs::path(context.tile_roots[1]).filename() ==
                "tile_0001",
            "second tile directory uses zero-padded name");
    }

    const fs::path tile0 =
        task_root / "tiles" / "tile_0000";

    const fs::path tile1 =
        task_root / "tiles" / "tile_0001";

    expect(
        fs::exists(tile0 / "status.json"),
        "first tile status.json exists");

    expect(
        fs::exists(tile1 / "status.json"),
        "second tile status.json exists");

    expect(
        contains(
            read_text(tile0 / "status.json"),
            "\"state\":\"completed\""),
        "first tile status is completed");

    expect(
        contains(
            read_text(tile1 / "status.json"),
            "\"state\":\"completed\""),
        "second tile status is completed");

    const fs::path output_path =
        task_root / "output_full.rgb";

    const std::vector<uint8_t> output =
        read_bytes(output_path);

    expect(
        output.size() ==
            static_cast<size_t>(width) *
            static_cast<size_t>(height) *
            3u,
        "full-resolution RGB output has exact byte count");

    expect(
        !output.empty() &&
        std::all_of(
            output.begin(),
            output.end(),
            [](uint8_t value)
            {
                return value == 77;
            }),
        "full-resolution output comes from tile results");

    expect(
        !output.empty() &&
        output.front() != input.front(),
        "full-resolution output is not silent input fallback");

    const std::string task_status =
        read_text(task_root / "status.json");

    expect(
        contains(task_status, "\"state\":\"completed\""),
        "task status is completed");

    expect(
        contains(task_status, "\"tile_count\":2"),
        "task status records two tiles");

    expect(
        contains(task_status, "\"completed_tiles\":2"),
        "task status records two completed tiles");

    expect(
        contains(task_status, "\"silent_fallback\":false"),
        "task status explicitly rejects silent fallback");

    expect(
        contains(result, "\"state\":\"completed\""),
        "returned result reports completed state");

    std::error_code ec;
    fs::remove_all(
        task_root.parent_path().parent_path(),
        ec);
}

void test_second_tile_failure_stops_fusion()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const std::string task_id =
        "full-photo-failure";

    const fs::path task_root =
        make_task_root(task_id);

    const auto input =
        make_input(width, height, 19);

    FakeContext context;
    context.fail_on_executor_call = 2;

    const std::string result =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            "{"
            "\"task_id\":\"full-photo-failure\","
            "\"model_version\":\"phase18d-host-v2\""
            "}",
            make_hooks(&context));

    expect(
        context.manifest_calls == 1,
        "failure path validates manifest once");

    expect(
        context.executor_calls == 2,
        "failure path reaches second tile");

    expect_no_final_output(
        task_root,
        "tile failure does not produce final output");

    expect(
        contains(
            read_text(task_root / "status.json"),
            "\"state\":\"failed\""),
        "task status is failed");

    expect(
        contains(
            read_text(
                task_root /
                "tiles" /
                "tile_0000" /
                "status.json"),
            "\"state\":\"completed\""),
        "completed first tile evidence is preserved");

    expect(
        contains(
            read_text(
                task_root /
                "tiles" /
                "tile_0001" /
                "status.json"),
            "\"state\":\"failed\""),
        "failed second tile has failed status");

    expect(
        contains(result, "\"state\":\"failed\""),
        "returned result reports failed state");

    expect(
        !contains(result, "\"state\":\"completed\""),
        "failure result is never reported completed");

    std::error_code ec;
    fs::remove_all(
        task_root.parent_path().parent_path(),
        ec);
}

void test_cancellation_after_first_tile()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const std::string task_id =
        "full-photo-cancelled";

    const fs::path task_root =
        make_task_root(task_id);

    const auto input =
        make_input(width, height, 23);

    FakeContext context;
    context.cancel_after_executor_calls = 1;

    const std::string result =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            "{"
            "\"task_id\":\"full-photo-cancelled\","
            "\"model_version\":\"phase18d-host-v2\""
            "}",
            make_hooks(&context));

    expect(
        context.manifest_calls == 1,
        "cancel path validates manifest once");

    expect(
        context.executor_calls == 1,
        "cancellation prevents second tile execution");

    expect(
        fs::exists(
            task_root /
            "tiles" /
            "tile_0000" /
            "status.json"),
        "completed tile evidence remains after cancellation");

    expect(
        !fs::exists(
            task_root /
            "tiles" /
            "tile_0001"),
        "cancelled next tile is never started");

    expect_no_final_output(
        task_root,
        "cancelled task does not produce final output");

    expect(
        contains(
            read_text(task_root / "status.json"),
            "\"state\":\"cancelled\""),
        "task status is cancelled");

    expect(
        contains(result, "\"state\":\"cancelled\""),
        "returned result reports cancelled state");

    std::error_code ec;
    fs::remove_all(
        task_root.parent_path().parent_path(),
        ec);
}

void test_wrong_tile_output_size_fails()
{
    constexpr int width = 160;
    constexpr int height = 96;

    const std::string task_id =
        "full-photo-wrong-size";

    const fs::path task_root =
        make_task_root(task_id);

    const auto input =
        make_input(width, height, 31);

    FakeContext context;
    context.wrong_size_on_executor_call = 1;

    const std::string result =
        run_phase18d_full_photo_with_hooks(
            "fake-model-root",
            task_root.string(),
            input.data(),
            input.size(),
            width,
            height,
            "{"
            "\"task_id\":\"full-photo-wrong-size\","
            "\"model_version\":\"phase18d-host-v2\""
            "}",
            make_hooks(&context));

    expect(
        context.manifest_calls == 1,
        "wrong-size path validates manifest once");

    expect(
        context.executor_calls == 1,
        "wrong-size output fails immediately");

    expect_no_final_output(
        task_root,
        "wrong-size tile output produces no final output");

    expect(
        contains(
            read_text(task_root / "status.json"),
            "\"state\":\"failed\""),
        "wrong-size task status is failed");

    expect(
        contains(result, "\"state\":\"failed\""),
        "wrong-size result reports failed state");

    expect(
        contains(
            result,
            "TILE_OUTPUT_SIZE_MISMATCH"),
        "wrong-size result exposes stable error code");

    std::error_code ec;
    fs::remove_all(
        task_root.parent_path().parent_path(),
        ec);
}

}  // namespace

int main()
{
    test_successful_two_tile_orchestration();
    test_second_tile_failure_stops_fusion();
    test_cancellation_after_first_tile();
    test_wrong_tile_output_size_fails();

    std::cout << "failed=" << failed << "\n";
    return failed;
}
