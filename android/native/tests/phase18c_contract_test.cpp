#include "phase18c_contract.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

static int failed = 0;
static void expect(bool condition, const char* name)
{
    if (!condition)
    {
        ++failed;
        std::cerr << "FAIL " << name << "\n";
    }
}

int main()
{
    const Phase18cFixedConfig c = phase18c_fixed_config();
    expect(c.frame_count == 4, "four frames");
    expect(c.width == 128 && c.height == 128, "minimum Phase 17 profile A shape");
    expect(c.fps == 4, "four fps");
    expect(c.latent_t == 2 && c.latent_h == 16 && c.latent_w == 16 && c.latent_c == 16,
           "latent layout");
    expect(c.dit_blocks == 32 && c.max_resident_blocks == 1, "strict block streaming");

    const std::vector<float> noise{1.f, -2.f, 3.f};
    const std::vector<float> prediction{0.25f, 0.5f, -1.f};
    const std::vector<float> latent = phase18c_euler_endpoint(noise, prediction);
    expect(latent == std::vector<float>({0.75f, -2.5f, 4.f}), "one-step Euler endpoint");
    expect(phase18c_all_finite(latent), "finite tensor accepted");
    expect(!phase18c_all_finite({0.f, std::numeric_limits<float>::infinity()}), "infinity rejected");
    expect(!phase18c_all_finite({std::numeric_limits<float>::quiet_NaN()}), "nan rejected");
    expect(phase18c_euler_endpoint({1.f}, {1.f, 2.f}).empty(), "shape mismatch rejected");

    // Two channels, one temporal plane and a 2x2 spatial patch. The patch
    // vector order must match einops: (T t) (H h) (W w) c -> T H W (t h w c).
    const std::vector<float> grid{0,1, 2,3, 4,5, 6,7};
    const std::vector<float> patched = phase18c_patchify_thwc(grid, 1, 2, 2, 2, 2, 2);
    expect(patched == grid, "patchify channel-fast order");
    expect(phase18c_unpatchify_thwc(patched, 1, 1, 1, 2, 2, 2) == grid,
           "unpatchify inverse");
    expect(phase18c_patchify_thwc(grid, 1, 2, 3, 2, 2, 2).empty(),
           "illegal patch shape rejected");

    std::cout << "phase18c_contract_test failed=" << failed << "\n";
    return failed ? 1 : 0;
}
