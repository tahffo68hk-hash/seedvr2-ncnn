#include <filesystem>
#include <iostream>
#include "artifact_registry.h"
#include "model_cache.h"
static void require(bool value, const char* message) { if (!value) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); } }
int main() {
    using namespace attention_dispatcher;
    ArtifactRegistry registry; std::string error; double load_ms = 0.0;
    const auto root = std::filesystem::path(__FILE__).parent_path() / "artifacts" / "attention_core";
    require(registry.scan(root, error), "registry scan");
    NcnnModelCache cache;
    const CoreInfo* a = registry.lookup({4,4,4,1,4,4}); const CoreInfo* c = registry.lookup({4,4,4,1,5,5});
    require(a && c && cache.size() == 0, "empty cache");
    ncnn::Net* first = cache.get_or_load(*a, load_ms, error); require(first && cache.size() == 1 && load_ms >= 0.0, "first load");
    ncnn::Net* again = cache.get_or_load(*a, load_ms, error); require(again == first && cache.size() == 1 && load_ms == 0.0, "same core reuse");
    require(cache.get_or_load(*c, load_ms, error) && cache.size() == 2, "different core");
    // Break caught: a stage-specific DiT thread count is ignored and every
    // loaded ncnn Net silently remains single-threaded.
    NcnnModelCache threaded_cache(5);
    ncnn::Net* threaded = threaded_cache.get_or_load(*a, load_ms, error);
    require(threaded && threaded->opt.num_threads == 5,
            "configured thread count reaches loaded ncnn Net");
    CoreInfo bad_param = *a; std::get<SingleModelArtifact>(bad_param.artifact).model.param_path = root / "missing.param"; require(!cache.get_or_load(bad_param, load_ms, error) && cache.size() == 2, "missing param not cached");
    CoreInfo bad_bin = *a; std::get<SingleModelArtifact>(bad_bin.artifact).model.bin_path = root / "missing.bin"; require(!cache.get_or_load(bad_bin, load_ms, error) && cache.size() == 2, "missing bin not cached");
    const auto fidelity = std::filesystem::path(__FILE__).parent_path().parent_path() / "seedvr2_attention_fidelity" / "artifacts" / "attention_core_fidelity"; ArtifactRegistry hybrids; require(hybrids.scan(fidelity, error), "hybrid registry scan"); const CoreInfo* hybrid = hybrids.lookup({4,4,4,1,2,2,"fidelity_ln_rope3d_hybrid"}); require(hybrid && !cache.get_or_load(*hybrid, load_ms, error) && error.find("staged_host_head_merge") != std::string::npos && error.find("fidelity_ln_rope3d_hybrid") != std::string::npos && cache.size() == 2, "hybrid rejected by single-model cache");
    cache.clear(); require(cache.size() == 0, "clear");
    std::cout << "model_cache_tests=PASS\n";
}
