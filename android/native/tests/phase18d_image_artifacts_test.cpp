#include "phase18d_image_artifacts.h"

#include "stb_image.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace
{

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

fs::path make_artifact_root()
{
    static int sequence = 0;
    const fs::path base = fs::temp_directory_path() /
        ("seededge_phase18d_image_artifacts_" + std::to_string(++sequence));
    std::error_code ec;
    fs::remove_all(base, ec);
    fs::create_directories(base, ec);
    return base;
}

bool pixel_region_matches(
    const DecodedPng& decoded,
    int region_left,
    const std::vector<uint8_t>& source,
    int width,
    int height)
{
    for (int y = 0; y < height; ++y)
    {
        for (int x = 0; x < width; ++x)
        {
            const size_t source_index = (static_cast<size_t>(y) * width + x) * 3u;
            const size_t decoded_index =
                (static_cast<size_t>(y) * decoded.width + region_left + x) * 3u;
            for (int c = 0; c < 3; ++c)
            {
                if (decoded.rgb[decoded_index + c] != source[source_index + c])
                    return false;
            }
        }
    }
    return true;
}

void test_exact_pixel_roundtrip()
{
    constexpr int width = 3;
    constexpr int height = 2;
    // Row 0: red, green, blue.  Row 1: white, black, gray.
    const std::vector<uint8_t> expected = {
        255, 0, 0,       0, 255, 0,       0, 0, 255,
        255, 255, 255,   0, 0, 0,         128, 128, 128,
    };
    const fs::path root = make_artifact_root();
    const fs::path png_path = root / "output.png";
    std::string error;
    const bool written = seededge::phase18d::write_rgb8_png(
        png_path.string(), expected.data(), expected.size(), width, height, &error);
    expect(written, "3x2 RGB PNG write succeeds");
    DecodedPng decoded;
    const bool decoded_ok = decode_png(png_path, &decoded);
    expect(decoded_ok, "3x2 PNG decodes");
    if (decoded_ok)
    {
        expect(decoded.width == width, "decoded PNG width is 3");
        expect(decoded.height == height, "decoded PNG height is 2");
        expect(decoded.rgb == expected, "decoded PNG pixels exactly match input");
    }
    std::error_code ec;
    fs::remove_all(root, ec);
}

void test_side_by_side_layout()
{
    constexpr int width = 3;
    constexpr int height = 2;
    const std::vector<uint8_t> left = {
        10, 20, 30,     40, 50, 60,     70, 80, 90,
        100, 110, 120,  130, 140, 150,  160, 170, 180,
    };
    const std::vector<uint8_t> right = {
        200, 210, 220,  230, 240, 250,  1, 2, 3,
        4, 5, 6,        7, 8, 9,        11, 12, 13,
    };
    const fs::path root = make_artifact_root();
    const fs::path png_path = root / "side_by_side.png";
    std::string error;
    const bool written = seededge::phase18d::write_side_by_side_png(
        png_path.string(), left.data(), left.size(), right.data(), right.size(),
        width, height, &error);
    expect(written, "side-by-side PNG write succeeds");
    DecodedPng decoded;
    const bool decoded_ok = decode_png(png_path, &decoded);
    expect(decoded_ok, "side-by-side PNG decodes");
    if (decoded_ok)
    {
        expect(decoded.width == width * 2, "side-by-side PNG width is 6");
        expect(decoded.height == height, "side-by-side PNG height is 2");
        expect(pixel_region_matches(decoded, 0, left, width, height),
               "left half exactly matches left image");
        expect(pixel_region_matches(decoded, width, right, width, height),
               "right half exactly matches right image");
    }
    std::error_code ec;
    fs::remove_all(root, ec);
}

void test_byte_count_mismatch_fails()
{
    constexpr int width = 3;
    constexpr int height = 2;
    const std::vector<uint8_t> rgb = {
        255, 0, 0,       0, 255, 0,       0, 0, 255,
        255, 255, 255,   0, 0, 0,         128, 128, 128,
    };
    const fs::path root = make_artifact_root();
    const fs::path png_path = root / "output.png";
    std::string error;
    const bool undersized = seededge::phase18d::write_rgb8_png(
        png_path.string(), rgb.data(), rgb.size() - 1u, width, height, &error);
    expect(!undersized, "undersized byte_count is rejected");
    expect(!error.empty(), "undersized byte_count reports an error message");
    expect(!fs::exists(png_path), "undersized write leaves no output file");
    const bool oversized = seededge::phase18d::write_rgb8_png(
        png_path.string(), rgb.data(), rgb.size() + 1u, width, height, &error);
    expect(!oversized, "oversized byte_count is rejected");
    expect(!fs::exists(png_path), "oversized write leaves no output file");
    std::error_code ec;
    fs::remove_all(root, ec);
}

void test_invalid_arguments_fail()
{
    const std::vector<uint8_t> rgb(3u * 2u * 3u, 7u);
    const fs::path root = make_artifact_root();
    const fs::path png_path = root / "output.png";
    std::string error;
    expect(!seededge::phase18d::write_rgb8_png(
               png_path.string(), rgb.data(), rgb.size(), 0, 2, &error),
           "zero width is rejected");
    expect(!seededge::phase18d::write_rgb8_png(
               png_path.string(), rgb.data(), rgb.size(), 3, 0, &error),
           "zero height is rejected");
    expect(!seededge::phase18d::write_rgb8_png(
               png_path.string(), rgb.data(), rgb.size(), -1, 2, &error),
           "negative width is rejected");
    expect(!seededge::phase18d::write_rgb8_png(
               png_path.string(), nullptr, rgb.size(), 3, 2, &error),
           "null RGB pointer is rejected");
    expect(!fs::exists(png_path), "rejected writes leave no output file");
    expect(seededge::phase18d::write_rgb8_png(
               png_path.string(), rgb.data(), rgb.size(), 3, 2, nullptr),
           "nullptr error output is tolerated on success");
    std::error_code ec;
    fs::remove_all(root, ec);
}

void test_atomic_write_and_cleanup()
{
    constexpr int width = 3;
    constexpr int height = 2;
    const std::vector<uint8_t> rgb = {
        255, 0, 0,       0, 255, 0,       0, 0, 255,
        255, 255, 255,   0, 0, 0,         128, 128, 128,
    };
    const fs::path root = make_artifact_root();
    std::string error;

    // Failed validation must not touch a pre-existing valid target file.
    const fs::path target = root / "output.png";
    {
        std::ofstream existing(target, std::ios::binary);
        const char old_content[] = "OLD-CONTENT";
        existing.write(old_content, static_cast<std::streamsize>(sizeof(old_content) - 1u));
    }
    expect(!seededge::phase18d::write_rgb8_png(
               target.string(), rgb.data(), rgb.size() - 1u, width, height, &error),
           "invalid call fails");
    expect(!fs::exists(root / "output.png.tmp"),
           "failed validation leaves no temporary file");
    std::ifstream existing_read(target, std::ios::binary);
    const std::string current((std::istreambuf_iterator<char>(existing_read)), {});
    expect(current == "OLD-CONTENT", "pre-existing target is not modified on failure");

    // Encoding failure (temporary file cannot be created) must not create the
    // final output and must not leave a temporary file behind.
    const fs::path blocker = root / "blocker";
    {
        std::ofstream blocker_file(blocker, std::ios::binary);
        blocker_file.write("BLOCK", 5);
    }
    const fs::path blocked_path = blocker / "output.png";
    expect(!seededge::phase18d::write_rgb8_png(
               blocked_path.string(), rgb.data(), rgb.size(), width, height, &error),
           "unwritable output path fails");
    expect(fs::is_regular_file(blocker), "blocking file is left untouched");
    expect(!fs::exists(root / "blocker" / "output.png.tmp"),
           "failed encode leaves no temporary file");

    // Renaming onto an existing directory must fail and clean the temporary.
    const fs::path dir_target = root / "dir_target.png";
    fs::create_directories(dir_target);
    expect(!seededge::phase18d::write_rgb8_png(
               dir_target.string(), rgb.data(), rgb.size(), width, height, &error),
           "rename onto existing directory fails");
    expect(fs::is_directory(dir_target), "existing directory is not replaced");
    expect(!fs::exists(root / "dir_target.png.tmp"),
           "failed rename leaves no temporary file");

    // Success path leaves only the final PNG: no temporary file remains.
    const fs::path success_path = root / "success.png";
    expect(seededge::phase18d::write_rgb8_png(
               success_path.string(), rgb.data(), rgb.size(), width, height, &error),
           "valid write succeeds");
    expect(!fs::exists(root / "success.png.tmp"),
           "successful write removes its temporary file");
    DecodedPng decoded;
    expect(decode_png(success_path, &decoded) &&
               decoded.width == width && decoded.height == height &&
               decoded.rgb == rgb,
           "successful write yields a complete decodable PNG");

    std::error_code ec;
    fs::remove_all(root, ec);
}

void test_side_by_side_argument_contract()
{
    constexpr int width = 3;
    constexpr int height = 2;
    const std::vector<uint8_t> left(3u * 2u * 3u, 11u);
    const std::vector<uint8_t> right(3u * 2u * 3u, 22u);
    const fs::path root = make_artifact_root();
    const fs::path png_path = root / "side.png";
    std::string error;
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size() - 1u, right.data(), right.size(),
               width, height, &error),
           "left undersized byte_count is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size(), right.data(), right.size() - 1u,
               width, height, &error),
           "right undersized byte_count is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size(), right.data(), right.size() + 1u,
               width, height, &error),
           "right oversized byte_count is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), nullptr, 0u, right.data(), right.size(), width, height, &error),
           "null left RGB is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size(), nullptr, 0u, width, height, &error),
           "null right RGB is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size(), right.data(), right.size(),
               0, height, &error),
           "zero width is rejected");
    expect(!seededge::phase18d::write_side_by_side_png(
               png_path.string(), left.data(), left.size(), right.data(), right.size(),
               width, 0, &error),
           "zero height is rejected");
    expect(!fs::exists(png_path), "rejected side-by-side write leaves no output file");
    std::error_code ec;
    fs::remove_all(root, ec);
}

}  // namespace

int main()
{
    test_exact_pixel_roundtrip();
    test_side_by_side_layout();
    test_byte_count_mismatch_fails();
    test_invalid_arguments_fail();
    test_atomic_write_and_cleanup();
    test_side_by_side_argument_contract();

    std::cout << "failed=" << failed << "\n";
    return failed;
}