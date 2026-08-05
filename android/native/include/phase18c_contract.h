#pragma once

#include <vector>

struct Phase18cFixedConfig
{
    int frame_count;
    int width;
    int height;
    int fps;
    int latent_t;
    int latent_h;
    int latent_w;
    int latent_c;
    int dit_blocks;
    int max_resident_blocks;
};

Phase18cFixedConfig phase18c_fixed_config();
std::vector<float> phase18c_euler_endpoint(const std::vector<float>& noise,
                                           const std::vector<float>& prediction);
bool phase18c_all_finite(const std::vector<float>& values);
std::vector<float> phase18c_patchify_thwc(const std::vector<float>& input,
                                          int t, int h, int w, int channels,
                                          int patch_h, int patch_w);
std::vector<float> phase18c_unpatchify_thwc(const std::vector<float>& input,
                                            int t, int h, int w, int channels,
                                            int patch_h, int patch_w);
