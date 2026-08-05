#include "phase18c_contract.h"

#include <cmath>

Phase18cFixedConfig phase18c_fixed_config()
{
    return {4, 128, 128, 4, 2, 16, 16, 16, 32, 1};
}

std::vector<float> phase18c_euler_endpoint(const std::vector<float>& noise,
                                           const std::vector<float>& prediction)
{
    if (noise.size() != prediction.size()) return {};
    std::vector<float> latent(noise.size());
    for (size_t i = 0; i < noise.size(); ++i) latent[i] = noise[i] - prediction[i];
    return latent;
}

bool phase18c_all_finite(const std::vector<float>& values)
{
    for (float value : values)
        if (!std::isfinite(value)) return false;
    return true;
}

std::vector<float> phase18c_patchify_thwc(const std::vector<float>& input,
                                          int t, int h, int w, int channels,
                                          int patch_h, int patch_w)
{
    if (t <= 0 || h <= 0 || w <= 0 || channels <= 0 || patch_h <= 0 || patch_w <= 0 ||
        h % patch_h || w % patch_w ||
        input.size() != static_cast<size_t>(t) * h * w * channels) return {};
    std::vector<float> output;
    output.reserve(input.size());
    for (int ti = 0; ti < t; ++ti)
        for (int ho = 0; ho < h; ho += patch_h)
            for (int wo = 0; wo < w; wo += patch_w)
                for (int hi = 0; hi < patch_h; ++hi)
                    for (int wi = 0; wi < patch_w; ++wi)
                        for (int c = 0; c < channels; ++c)
                            output.push_back(input[(((ti * h + ho + hi) * w + wo + wi) * channels) + c]);
    return output;
}

std::vector<float> phase18c_unpatchify_thwc(const std::vector<float>& input,
                                            int t, int h, int w, int channels,
                                            int patch_h, int patch_w)
{
    if (t <= 0 || h <= 0 || w <= 0 || channels <= 0 || patch_h <= 0 || patch_w <= 0 ||
        input.size() != static_cast<size_t>(t) * h * w * patch_h * patch_w * channels) return {};
    const int out_h = h * patch_h, out_w = w * patch_w;
    std::vector<float> output(input.size());
    size_t source = 0;
    for (int ti = 0; ti < t; ++ti)
        for (int ho = 0; ho < h; ++ho)
            for (int wo = 0; wo < w; ++wo)
                for (int hi = 0; hi < patch_h; ++hi)
                    for (int wi = 0; wi < patch_w; ++wi)
                        for (int c = 0; c < channels; ++c)
                            output[(((ti * out_h + ho * patch_h + hi) * out_w + wo * patch_w + wi) * channels) + c] = input[source++];
    return output;
}
