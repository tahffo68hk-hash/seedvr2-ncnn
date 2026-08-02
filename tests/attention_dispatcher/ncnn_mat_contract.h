#pragma once

#include <string>
#include <vector>
#include <net.h>

namespace attention_dispatcher
{
bool make_rope_mat(const std::vector<float>& values, int tokens, int pairs, ncnn::Mat& output, std::string& error);
bool make_mask_mat(const std::vector<float>& values, int windows, int tokens, ncnn::Mat& output, std::string& error);
}
