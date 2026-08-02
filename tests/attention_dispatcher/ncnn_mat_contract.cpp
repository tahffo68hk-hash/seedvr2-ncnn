#include "ncnn_mat_contract.h"

#include <cstring>

namespace attention_dispatcher
{
bool make_rope_mat(const std::vector<float>& values, int tokens, int pairs, ncnn::Mat& output, std::string& error)
{
    output = ncnn::Mat();
    if (tokens <= 0 || pairs <= 0 || values.size() != static_cast<size_t>(tokens) * pairs) { error = "rope Mat shape/count mismatch"; return false; }
    output = ncnn::Mat(pairs, tokens);
    std::memcpy(output.data, values.data(), values.size() * sizeof(float));
    return true;
}

bool make_mask_mat(const std::vector<float>& values, int windows, int tokens, ncnn::Mat& output, std::string& error)
{
    output = ncnn::Mat();
    if (windows <= 0 || tokens <= 0 || values.size() != static_cast<size_t>(windows) * tokens * tokens) { error = "mask Mat shape/count mismatch"; return false; }
    output = ncnn::Mat(tokens, tokens, windows);
    std::memcpy(output.data, values.data(), values.size() * sizeof(float));
    return true;
}
}
