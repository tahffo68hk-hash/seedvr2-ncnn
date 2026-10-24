#include "hybrid_attention_runtime.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

namespace fs = std::filesystem;
using namespace attention_dispatcher;

static bool read_input(const fs::path& path, std::vector<float>& values)
{
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return false;
    const auto bytes = stream.tellg();
    if (bytes != static_cast<std::streamoff>(8192 * sizeof(float))) return false;
    stream.seekg(0);
    values.resize(8192);
    stream.read(reinterpret_cast<char*>(values.data()), bytes);
    return static_cast<bool>(stream);
}

int main()
{
    const fs::path repository = fs::path(__FILE__).parent_path().parent_path().parent_path();
    const fs::path baseline = repository / "tests/attention_dispatcher/artifacts/attention_core";
    const fs::path fidelity = repository / "tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity";
    const fs::path reference = fidelity / "hybrid_fidelity_rms_rope444_grid_1x2x2/reference/input.bin";

    ArtifactRegistry registry;
    std::string error;
    if (!registry.scan(std::vector<fs::path>{baseline, fidelity}, error))
    {
        std::cerr << "registry: " << error << '\n';
        return 1;
    }
    std::vector<float> input;
    if (!read_input(reference, input)) return 2;
    const AttentionRequest request{1, 4, 8, 8, 32, "fp32", "fidelity_rms_rope3d_hybrid"};

    for (const float bad : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()})
    {
        NcnnModelCache cache;
        auto invalid = input;
        invalid[317] = bad;
        const auto result = run_hybrid_attention(request, invalid, registry, cache);
        if (result.success || result.error_message != "input contains non-finite value at index=317" || cache.size() != 0)
        {
            std::cerr << "non-finite input was not rejected before model loading: error='"
                      << result.error_message << "' cache_size=" << cache.size() << '\n';
            return 3;
        }
    }
    std::cout << "phase8_2_runtime_contract_tests=PASS\n";
    return 0;
}
