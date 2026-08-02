#pragma once

#include <string>
#include <vector>

namespace seedvr2
{
bool merge_attention_heads(const std::vector<std::vector<float>>& heads,
                           int windows,
                           int tokens,
                           int head_dim,
                           std::vector<float>& merged,
                           std::string& error_message);
}
