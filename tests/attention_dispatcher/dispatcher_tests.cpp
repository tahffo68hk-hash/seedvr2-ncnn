#include <cassert>
#include <iostream>

#include "attention_dispatcher.h"
#include "artifact_registry.h"

int main()
{
    using namespace attention_dispatcher;

    const Metadata a = dispatch_shape({1, 4, 16, 16, 32});
    assert(a.window_t == 4 && a.grid_h == 4 && a.grid_w == 4);
    assert(a.pad_t == 0 && a.pad_h == 0 && a.pad_w == 0);
    assert(a.num_windows == 16 && a.tokens_per_window == 64);

    const Metadata b = dispatch_shape({1, 3, 16, 16, 32});
    assert(b.window_t == 3 && b.grid_t == 1 && b.num_windows == 16 && b.tokens_per_window == 48);

    const Metadata c = dispatch_shape({1, 4, 17, 17, 32});
    assert(c.window_t == 4 && c.pad_h == 3 && c.pad_w == 3);
    assert(c.grid_h == 5 && c.grid_w == 5 && c.num_windows == 25);

    ArtifactRegistry artifact_registry;
    std::string error;
    const std::filesystem::path artifact_root = std::filesystem::path(__FILE__).parent_path() / "artifacts" / "attention_core";
    assert(artifact_registry.scan(artifact_root, error));
    struct Expected { AttentionRequest request; const char* name; };
    const Expected expected[] = {
        {{1, 4, 16, 16, 32, "fp32"}, "core_444_grid_1x4x4"},
        {{1, 3, 16, 16, 32, "fp32"}, "core_344_grid_1x4x4"},
        {{1, 4, 17, 17, 32, "fp32"}, "core_444_grid_1x5x5"},
        {{1, 8, 32, 32, 32, "fp32"}, "core_444_grid_2x8x8"},
        {{1, 5, 19, 21, 32, "fp32"}, "core_444_grid_2x5x6"},
    };
    for (const Expected& item : expected)
    {
        Metadata selected_metadata{};
        const CoreInfo* selected_core = nullptr;
        assert(dispatch_request(item.request, artifact_registry, selected_metadata, selected_core, error));
        assert(selected_core && selected_core->name == item.name);
    }
    Metadata missing_metadata{};
    const CoreInfo* missing_core = nullptr;
    assert(!dispatch_request({1, 4, 99, 99, 32, "fp32"}, artifact_registry, missing_metadata, missing_core, error));
    assert(!error.empty() && missing_core == nullptr);
    assert(!dispatch_request({2, 4, 16, 16, 32, "fp32"}, artifact_registry, missing_metadata, missing_core, error));
    assert(error == "request batch must be 1");
    assert(!dispatch_request({1, 4, 16, 16, 16, "fp32"}, artifact_registry, missing_metadata, missing_core, error));
    assert(error == "request channels must be 32");
    assert(!dispatch_request({1, 4, 16, 16, 32, "fp16"}, artifact_registry, missing_metadata, missing_core, error));
    assert(error == "request dtype must be fp32");

    CoreRegistry registry = make_default_registry();
    assert(registry.find(a).name == "core_444_grid_1x4x4");
    assert(registry.find(b).name == "core_344_grid_1x4x4");
    assert(registry.find(c).name == "core_444_grid_1x5x5");
    const CoreEntry selected = select_default_core(c);
    assert(selected.name == "core_444_grid_1x5x5" && selected.available);

    const std::vector<float> mask = make_additive_key_mask(c);
    assert(mask.size() == static_cast<std::size_t>(25 * 64 * 64));
    const std::size_t last_window = 24 * 64 * 64;
    assert(mask[last_window + 0 * 64 + 0] == 0.f);
    assert(mask[last_window + 0 * 64 + 1] == -10000.f);
    assert(mask[last_window + 1 * 64 + 0] == 0.f);

    std::cout << "dispatcher_tests=PASS\n";
    return 0;
}
