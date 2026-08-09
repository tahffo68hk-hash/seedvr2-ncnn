#include "model_manifest.h"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>

namespace fs = std::filesystem;

namespace
{
int passed = 0;
int failed = 0;

void expect_contains(const std::string& name, const std::string& got,
                     const std::string& token)
{
    if (got.find(token) != std::string::npos)
        ++passed;
    else
    {
        ++failed;
        std::cerr << "FAIL " << name << ": expected " << token << " in " << got << '\n';
    }
}

void expect_true(const std::string& name, bool value)
{
    if (value)
        ++passed;
    else
    {
        ++failed;
        std::cerr << "FAIL " << name << '\n';
    }
}

void write_file(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << text;
}

std::string valid_manifest(const std::string& hash =
    "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08")
{
    return "{\"schema_version\":1,\"model_set\":\"phase18c_minimal_video_demo\","
           "\"files\":[{\"relative_path\":\"x\",\"size_bytes\":4,\"sha256\":\"" +
           hash + "\"}]}";
}

std::string manifest_with_model_set(const std::string& model_set)
{
    return "{\"schema_version\":1,\"model_set\":\"" + model_set +
           "\",\"files\":[{\"relative_path\":\"x\",\"size_bytes\":4,"
           "\"sha256\":\"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08\"}]}";
}
}

int main()
{
    const fs::path root = fs::temp_directory_path() / "seededge_manifest_test";
    fs::remove_all(root);
    fs::create_directories(root);

    expect_contains("missing", validate_model_manifest(root.string()), "manifest missing");
    write_file(root / "manifest.json", "bad");
    expect_contains("json", validate_model_manifest(root.string()), "invalid JSON");
    write_file(root / "manifest.json", R"({"schema_version":2,"model_set":"phase18c_minimal_video_demo","files":[]})");
    expect_contains("version", validate_model_manifest(root.string()), "schema_version must be 1");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[]})");
    expect_contains("empty", validate_model_manifest(root.string()), "files must not be empty");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"../x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");
    expect_contains("traversal", validate_model_manifest(root.string()), "path traversal");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"C:\\x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");
    expect_contains("absolute", validate_model_manifest(root.string()), "absolute path");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":0,"sha256":"GG"}]})");
    expect_contains("hash", validate_model_manifest(root.string()), "sha256 must be 64 lowercase hex");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");
    expect_contains("file", validate_model_manifest(root.string()), "model file missing");
    write_file(root / "x", "test");
    write_file(root / "manifest.json", R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":3,"sha256":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}]})");
    expect_contains("size", validate_model_manifest(root.string()), "size mismatch");
    write_file(root / "manifest.json", valid_manifest("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    expect_contains("digest", validate_model_manifest(root.string()), "sha256 mismatch");

    // Break caught: a verified model still rereads every model byte on every request.
    write_file(root / "manifest.json", valid_manifest());
    const std::string first = validate_model_manifest(root.string());
    expect_contains("first validation valid", first, R"("valid":true)");
    expect_contains("first validation performs full SHA", first,
                    R"("validation_mode":"full_sha256")");
    expect_contains("first validation reports cache miss", first, R"("cache_hit":false)");
    expect_true("verified identity cache persisted",
                fs::is_regular_file(root / ".seededge_manifest_verified_v1"));

    const std::string second = validate_model_manifest(root.string());
    expect_contains("unchanged second validation valid", second, R"("valid":true)");
    expect_contains("unchanged second validation uses identity cache", second,
                    R"("validation_mode":"identity_cache")");
    expect_contains("unchanged second validation reports cache hit", second,
                    R"("cache_hit":true)");
    expect_contains("cache hit hashes zero model files", second,
                    R"("full_sha256_files":0)");

    const fs::file_time_type original_mtime = fs::last_write_time(root / "x");
    fs::last_write_time(root / "x", original_mtime + std::chrono::seconds(2));
    const std::string mtime_changed = validate_model_manifest(root.string());
    expect_contains("mtime change remains valid after full verification", mtime_changed,
                    R"("valid":true)");
    expect_contains("mtime change invalidates identity cache", mtime_changed,
                    R"("validation_mode":"full_sha256")");

    write_file(root / "x", "tests");
    expect_contains("model size change is rejected",
                    validate_model_manifest(root.string()), "size mismatch");
    write_file(root / "x", "test");
    write_file(root / "manifest.json", valid_manifest());
    expect_contains("size fixture restore verifies", validate_model_manifest(root.string()),
                    R"("valid":true)");

    write_file(root / "manifest.json", manifest_with_model_set("unexpected_model_set"));
    expect_contains("unexpected model set is rejected before cache reuse",
                    validate_model_manifest(root.string()), "model_set mismatch");
    write_file(root / "manifest.json", valid_manifest());
    expect_contains("rejected model set does not poison trusted cache",
                    validate_model_manifest(root.string()),
                    R"("validation_mode":"identity_cache")");

    // Break caught: same-size model replacement is accepted by a size-only cache.
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
    write_file(root / "x", "tEst");
    expect_contains("same-size changed model invalidates cache",
                    validate_model_manifest(root.string()), "sha256 mismatch");

    // Restore a trusted cache, then prove a changed manifest cannot reuse it.
    write_file(root / "x", "test");
    write_file(root / "manifest.json", valid_manifest());
    expect_contains("restored full validation", validate_model_manifest(root.string()),
                    R"("valid":true)");
    write_file(root / "manifest.json", valid_manifest(
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"));
    expect_contains("manifest digest change invalidates cache",
                    validate_model_manifest(root.string()), "sha256 mismatch");

    // Break caught: malformed cache disables integrity checking instead of falling back.
    write_file(root / "manifest.json", valid_manifest());
    write_file(root / ".seededge_manifest_verified_v1", "corrupt\n");
    const std::string malformed = validate_model_manifest(root.string());
    expect_contains("malformed cache falls back safely", malformed, R"("valid":true)");
    expect_contains("malformed cache forces full SHA", malformed,
                    R"("validation_mode":"full_sha256")");

    // Break caught: a cache from a future or incompatible schema is reused.
    write_file(root / ".seededge_manifest_verified_v1",
               "seededge_verified_manifest_v999\n");
    const std::string incompatible_schema = validate_model_manifest(root.string());
    expect_contains("incompatible cache schema falls back safely", incompatible_schema,
                    R"("valid":true)");
    expect_contains("incompatible cache schema forces full SHA", incompatible_schema,
                    R"("validation_mode":"full_sha256")");

    // Break caught: a cache copied from another model root is trusted.
    const fs::path other = fs::temp_directory_path() / "seededge_manifest_test_other";
    fs::remove_all(other);
    fs::create_directories(other);
    write_file(other / "x", "test");
    write_file(other / "manifest.json", valid_manifest());
    fs::copy_file(root / ".seededge_manifest_verified_v1",
                  other / ".seededge_manifest_verified_v1",
                  fs::copy_options::overwrite_existing);
    const std::string copied = validate_model_manifest(other.string());
    expect_contains("foreign cache validation valid", copied, R"("valid":true)");
    expect_contains("foreign cache rejected and full SHA performed", copied,
                    R"("validation_mode":"full_sha256")");
    fs::remove_all(root);
    fs::remove_all(other);
    std::cout << "manifest_tests passed=" << passed << " failed=" << failed << '\n';
    return failed ? 1 : 0;
}
