#pragma once

#include "dynamic_attention_inputs.h"

#include <string>
#include <vector>

namespace attention_dispatcher
{
bool generate_padding_validity(const DynamicAttentionContract& contract, std::vector<unsigned char>& validity, std::string& error_message);
int count_valid_tokens(const std::vector<unsigned char>& validity, int window, int tokens_per_window);
bool generate_additive_key_mask(const std::vector<unsigned char>& validity, int num_windows, int tokens_per_window, std::vector<float>& mask, std::string& error_message);
}
