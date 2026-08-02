#pragma once

#include <cstddef>
#include <string>

namespace attention_dispatcher
{
struct DynamicAttentionContract
{
    int batch = 0;
    int time = 0;
    int height = 0;
    int width = 0;
    int channels = 0;
    int window_t = 0;
    int window_h = 0;
    int window_w = 0;
    int grid_t = 0;
    int grid_h = 0;
    int grid_w = 0;
    int pad_t = 0;
    int pad_h = 0;
    int pad_w = 0;
    int num_windows = 0;
    int tokens_per_window = 0;
    int heads = 0;
    int head_dim = 0;
    int rope_dim = 0;
};

bool validate_dynamic_attention_contract(const DynamicAttentionContract& contract, std::string& error_message);
bool checked_dynamic_element_count(const DynamicAttentionContract& contract, std::size_t& count, std::string& error_message);
}
