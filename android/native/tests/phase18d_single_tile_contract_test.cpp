#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include "minimal_video_pipeline.h"

#include <string>
#include <vector>

static int failed = 0;
static void expect(bool condition, const char* name)
{
    if (!condition) { ++failed; std::cerr << "FAIL " << name << "\n"; }
}

int main()
{
    const std::vector<uint8_t> rgb(128u * 128u * 3u, 9u);
    const std::string malformed = run_phase18d_single_photo_tile(
        "missing-model", "task-root", rgb.data(), rgb.size(), 128, 128, "{");
    expect(malformed.find("INVALID_REQUEST_JSON") != std::string::npos,
           "malformed request rejected before model load");
    const std::string wrong_size = run_phase18d_single_photo_tile(
        "missing-model", "task-root", rgb.data(), rgb.size() - 1u, 128, 128, "{}");
    expect(wrong_size.find("RGB_BUFFER_SIZE_MISMATCH") != std::string::npos,
           "undersized RGB rejected before model load");
    const std::filesystem::path task_root = std::filesystem::temp_directory_path() /
                                            "seedvr2-phase18d-contract-root" / "tasks" / "contract-task";
    const std::string missing_model = run_phase18d_single_photo_tile(
        "missing-model", task_root.string(), rgb.data(), rgb.size(), 128, 128,
        R"({"task_id":"contract-task","model_version":"test-model-v1"})");
    const std::filesystem::path tile_root = task_root / "tiles" / "tile_0000";
    expect(missing_model.find("model_manifest") != std::string::npos,
           "valid task reaches full pipeline model validation");
    expect(std::filesystem::exists(tile_root / "tile_input.rgb"),
           "padded tile input is task-local");
    expect(std::filesystem::exists(tile_root / "input_norm.f32"),
           "five-frame FP32 model input is task-local");
    expect(std::filesystem::exists(tile_root / "status.json"),
           "task-local status is persisted after model failure");
    std::ifstream status_file(tile_root / "status.json");
    const std::string status((std::istreambuf_iterator<char>(status_file)), {});
    expect(status.find("\"checkpoint_root\":null") != std::string::npos,
           "single tile never imports a legacy decoder checkpoint");
    expect(status.find("\"encoder_completed\":false") != std::string::npos &&
           status.find("\"dit_block_completed\":0") != std::string::npos &&
           status.find("\"decoder_completed\":false") != std::string::npos,
           "failed model validation cannot claim execution stages");

    // A photo with blank edges must not anchor the single tile on an empty
    // corner: the selected tile_0000 has to contain actual image content.
    {
        const int w = 400, h = 300, border = 64;
        std::vector<uint8_t> bordered(static_cast<size_t>(w) * h * 3u, 0u);
        for (int y = border; y < h - border; ++y)
            for (int x = border; x < w - border; ++x)
            {
                uint8_t* pxl = &bordered[(static_cast<size_t>(y) * w + x) * 3u];
                pxl[0] = static_cast<uint8_t>((x * 7) % 256);
                pxl[1] = static_cast<uint8_t>((y * 11) % 256);
                pxl[2] = static_cast<uint8_t>((x * 13 + y * 17) % 256);
            }
        const std::filesystem::path select_root = task_root.parent_path() / "contract-tile-select";
        const std::string selected = run_phase18d_single_photo_tile(
            "missing-model", select_root.string(), bordered.data(), bordered.size(), w, h,
            R"({"task_id":"contract-tile-select","model_version":"test-model-v1"})");
        expect(selected.find("model_manifest") != std::string::npos,
               "bordered image still reaches model validation");
        const std::filesystem::path select_tile = select_root / "tiles" / "tile_0000";
        std::ifstream input(select_tile / "tile_input.rgb", std::ios::binary);
        std::vector<uint8_t> tile_rgb((std::istreambuf_iterator<char>(input)), {});
        expect(tile_rgb.size() == 128u * 128u * 3u, "bordered tile keeps 128x128 padded input");
        size_t nonzero = 0;
        for (uint8_t value : tile_rgb)
            if (value != 0u) ++nonzero;
        expect(nonzero > 0u, "single tile avoids a blank photo corner");
    }
    std::cout << "phase18d_single_tile_contract_test failed=" << failed << "\n";
    return failed ? 1 : 0;
}
