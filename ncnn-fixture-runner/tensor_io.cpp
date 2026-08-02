#include "tensor_io.h"

#include <fstream>

bool read_float32_file(const std::string& path, std::vector<float>& values, std::size_t count)
{
    std::ifstream input(path, std::ios::binary);
    if (!input)
        return false;
    values.resize(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(count * sizeof(float)));
    return input.good() || input.gcount() == static_cast<std::streamsize>(count * sizeof(float));
}

bool write_float32_file(const std::string& path, const std::vector<float>& values)
{
    std::ofstream output(path, std::ios::binary);
    if (!output)
        return false;
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(values.size() * sizeof(float)));
    return output.good();
}

ncnn::Mat bthwc_to_ncnn_mat(const std::vector<float>& bthwc, int t, int h, int w, int channels)
{
    // PyTorch contiguous [B=1,T,H,W,C] uses C as the fastest-changing index.
    // ncnn 4D Mat(w=C,h=W,d=H,c=T) has the same physical order: t,h,w,C.
    ncnn::Mat mat(channels, w, h, t);
    std::memcpy(mat.data, bthwc.data(), bthwc.size() * sizeof(float));
    return mat;
}

void ncnn_mat_to_bthwc(const ncnn::Mat& mat, std::vector<float>& bthwc, int t, int h, int w, int channels)
{
    bthwc.resize(static_cast<std::size_t>(t) * h * w * channels);
    std::memcpy(bthwc.data(), mat.data, bthwc.size() * sizeof(float));
}

ncnn::Mat partition_fixed_windows(const ncnn::Mat& bthwc, int t, int h, int w, int channels, int window_t, int window_hw)
{
    const int nt = t / window_t;
    const int nh = h / window_hw;
    const int nw = w / window_hw;
    const int num_windows = nt * nh * nw;
    const int tokens = window_t * window_hw * window_hw;

    // Window tensor [num_windows,tokens,C] maps to ncnn Mat(w=C,h=tokens,c=num_windows).
    ncnn::Mat windows(channels, tokens, num_windows);
    int window_index = 0;
    for (int bt = 0; bt < nt; ++bt)
        for (int bh = 0; bh < nh; ++bh)
            for (int bw = 0; bw < nw; ++bw, ++window_index)
                for (int lt = 0; lt < window_t; ++lt)
                    for (int lh = 0; lh < window_hw; ++lh)
                        for (int lw = 0; lw < window_hw; ++lw)
                        {
                            const int token = (lt * window_hw + lh) * window_hw + lw;
                            const float* source = bthwc.channel(bt * window_t + lt).depth(bh * window_hw + lh).row(bw * window_hw + lw);
                            float* target = windows.channel(window_index).row(token);
                            std::memcpy(target, source, static_cast<std::size_t>(channels) * sizeof(float));
                        }
    return windows;
}

void reverse_fixed_windows(const ncnn::Mat& windows, ncnn::Mat& bthwc, int t, int h, int w, int channels, int window_t, int window_hw)
{
    const int nt = t / window_t;
    const int nh = h / window_hw;
    const int nw = w / window_hw;
    int window_index = 0;
    for (int bt = 0; bt < nt; ++bt)
        for (int bh = 0; bh < nh; ++bh)
            for (int bw = 0; bw < nw; ++bw, ++window_index)
                for (int lt = 0; lt < window_t; ++lt)
                    for (int lh = 0; lh < window_hw; ++lh)
                        for (int lw = 0; lw < window_hw; ++lw)
                        {
                            const int token = (lt * window_hw + lh) * window_hw + lw;
                            const float* source = windows.channel(window_index).row(token);
                            float* target = bthwc.channel(bt * window_t + lt).depth(bh * window_hw + lh).row(bw * window_hw + lw);
                            std::memcpy(target, source, static_cast<std::size_t>(channels) * sizeof(float));
                        }
}
