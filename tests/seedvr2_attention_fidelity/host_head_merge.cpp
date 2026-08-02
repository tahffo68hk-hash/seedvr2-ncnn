#include "host_head_merge.h"

#include <cmath>

namespace seedvr2
{
bool merge_attention_heads(const std::vector<std::vector<float>>& heads,
                           int windows,
                           int tokens,
                           int head_dim,
                           std::vector<float>& merged,
                           std::string& error_message)
{
    if (heads.size() != 4 || windows <= 0 || tokens <= 0 || head_dim <= 0)
    {
        error_message = "expected four heads and positive W/N/D";
        return false;
    }

    const std::size_t head_size = static_cast<std::size_t>(windows) * tokens * head_dim;
    for (std::size_t head = 0; head < heads.size(); ++head)
    {
        if (heads[head].size() != head_size)
        {
            error_message = "head " + std::to_string(head) + " does not match W*N*D";
            return false;
        }
        for (float value : heads[head])
            if (!std::isfinite(value))
            {
                error_message = "head " + std::to_string(head) + " contains a non-finite value";
                return false;
            }
    }

    std::vector<float> candidate(head_size * heads.size());
    for (int window = 0; window < windows; ++window)
        for (int token = 0; token < tokens; ++token)
            for (int head = 0; head < 4; ++head)
                for (int channel = 0; channel < head_dim; ++channel)
                {
                    const std::size_t source = (static_cast<std::size_t>(window) * tokens + token) * head_dim + channel;
                    const std::size_t destination = (static_cast<std::size_t>(window) * tokens + token) * (4 * head_dim) + head * head_dim + channel;
                    candidate[destination] = heads[head][source];
                }
    merged = std::move(candidate);
    error_message.clear();
    return true;
}
}
