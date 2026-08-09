#include "minimal_video_pipeline.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

namespace fs = std::filesystem;
namespace p18d = seededge::phase18d;

// Minimal stand-in for the Android JNI bridge: distinct sentinel values so any
// overwrite by the hooks context (the Phase 18D SIGSEGV root cause) is visible.
struct FakeJniBridge
{
    long long env = 0x1111111111111111LL;
    long long bridge = 0x2222222222222222LL;
    long long on_progress_method = 0x3333333333333333LL;
    long long is_cancelled_method = 0x4444444444444444LL;
};

static int passed = 0;
static int failed = 0;

static void expect(bool condition, const std::string& name)
{
    if (condition) ++passed;
    else { ++failed; std::cerr << "FAIL: " << name << "\n"; }
}

static void write_file(const fs::path& path, const std::string& text)
{
    fs::create_directories(path.parent_path());
    std::ofstream(path) << text;
}

int main()
{
    const fs::path root = fs::temp_directory_path() / "seededge_real_hooks_test";
    fs::remove_all(root);
    fs::create_directories(root);
    write_file(root / "x", "test");
    write_file(root / "manifest.json",
               R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":4,"sha256":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}]})");

    // The real manifest validator must store into a Phase18dRealContext and
    // must never write into the caller's JNI bridge (regression for the
    // FullPhotoJniBridge/Phase18dRealContext type confusion that crashed the
    // Pad in nativeRunFullPhotoWithProgress after the model check).
    p18d::Phase18dRealContext real;
    FakeJniBridge bridge;
    real.jni_bridge = &bridge;

    std::string manifest_sha;
    std::string error;
    const bool valid = p18d::phase18d_real_validate_manifest(
        &real, root.string(), &manifest_sha, &error);
    expect(valid, "real manifest hook returns true");
    expect(real.manifest.find("\"valid\":true") != std::string::npos,
           "real manifest hook stores manifest into Phase18dRealContext");
    expect(!manifest_sha.empty(), "real manifest hook fills manifest_sha256");
    expect(bridge.env == 0x1111111111111111LL, "bridge env untouched");
    expect(bridge.bridge == 0x2222222222222222LL, "bridge object untouched");
    expect(bridge.on_progress_method == 0x3333333333333333LL,
           "bridge onProgress method untouched");
    expect(bridge.is_cancelled_method == 0x4444444444444444LL,
           "bridge isCancelled method untouched");

    // make_real_full_photo_hooks must wire the real hooks around the supplied
    // Phase18dRealContext (compile-time type guard against wrong contexts).
    int progress_calls = 0;
    auto progress = [](void* opaque, const char*, int, int) {
        ++*static_cast<int*>(opaque);
    };
    p18d::Phase18dFullPhotoHooks hooks =
        p18d::make_real_full_photo_hooks(&real, nullptr, progress);
    expect(hooks.context == &real, "hooks.context is the Phase18dRealContext");
    expect(hooks.validate_manifest == &p18d::phase18d_real_validate_manifest,
           "hooks.validate_manifest is the real validator");
    expect(hooks.execute_tile == &p18d::phase18d_real_execute_tile,
           "hooks.execute_tile is the real tile executor");
    expect(hooks.is_cancelled == nullptr, "hooks.is_cancelled preserved");
    expect(hooks.on_progress == progress, "hooks.on_progress preserved");

    // Break caught: requested stage-specific thread counts never reach the
    // shared Host/Android real runtime context.
    p18d::configure_real_context_threads(
        &real,
        R"({"encoder_threads":4,"dit_threads":5,"decoder_threads":6})");
    expect(real.encoder_threads == 4, "encoder thread request parsed");
    expect(real.dit_threads == 5, "DiT thread request parsed");
    expect(real.decoder_threads == 6, "decoder thread request parsed");
    p18d::configure_real_context_threads(
        &real,
        R"({"encoder_threads":0,"dit_threads":99,"decoder_threads":-1})");
    expect(real.encoder_threads > 0 && real.encoder_threads <= 16,
           "invalid encoder threads safely defaulted");
    expect(real.dit_threads > 0 && real.dit_threads <= 16,
           "invalid DiT threads safely defaulted");
    expect(real.decoder_threads > 0 && real.decoder_threads <= 16,
           "invalid decoder threads safely defaulted");

    fs::remove_all(root);
    std::cout << "real_hooks_tests passed=" << passed << " failed=" << failed << "\n";
    return failed ? 1 : 0;
}
