#include "dynamic_padding_mask.h"

#include <cassert>
#include <iostream>

using namespace attention_dispatcher;

int main()
{
    DynamicAttentionContract contract{};
    contract.batch = 1;
    contract.time = 4;
    contract.height = 9;
    contract.width = 10;
    contract.channels = 32;
    contract.window_t = 4;
    contract.window_h = 4;
    contract.window_w = 4;
    contract.grid_t = 1;
    contract.grid_h = 3;
    contract.grid_w = 3;
    contract.pad_t = 0;
    contract.pad_h = 3;
    contract.pad_w = 2;
    contract.num_windows = 9;
    contract.tokens_per_window = 64;
    contract.heads = 4;
    contract.head_dim = 8;
    contract.rope_dim = 8;

    std::vector<unsigned char> validity;
    std::string error;
    assert(generate_padding_validity(contract, validity, error));
    assert(validity.size() == 9 * 64);
    assert(validity[0] == 1);
    assert(validity[8 * 64 + 0] == 1);
    assert(validity[8 * 64 + 2] == 0);
    assert(validity[8 * 64 + 8] == 0);
    assert(count_valid_tokens(validity, 0, 64) == 64);
    assert(count_valid_tokens(validity, 8, 64) == 8);

    std::vector<float> mask;
    assert(generate_additive_key_mask(validity, contract.num_windows, contract.tokens_per_window, mask, error));
    assert(mask.size() == 9 * 64 * 64);
    assert(mask[(8 * 64 + 0) * 64 + 0] == 0.f);
    assert(mask[(8 * 64 + 0) * 64 + 2] == -10000.f);
    assert(mask[(8 * 64 + 63) * 64 + 2] == -10000.f);
    std::cout << "dynamic_padding_mask_tests=PASS\n";
}
