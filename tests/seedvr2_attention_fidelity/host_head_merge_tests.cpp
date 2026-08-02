#include "host_head_merge.h"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace
{
bool expect(bool condition, const char* name)
{
    if (!condition)
        std::cerr << "FAIL " << name << "\n";
    return condition;
}
}

int main()
{
    bool ok = true;
    std::vector<std::vector<float>> heads(4, std::vector<float>(2 * 3 * 2));
    for (int head = 0; head < 4; ++head)
        for (int i = 0; i < 12; ++i)
            heads[head][i] = static_cast<float>((head + 1) * 1000 + i);

    std::vector<float> merged;
    std::string error;
    ok &= expect(seedvr2::merge_attention_heads(heads, 2, 3, 2, merged, error), "tagged merge succeeds");
    ok &= expect(merged.size() == 48, "merged size");
    for (int window = 0; window < 2; ++window)
        for (int token = 0; token < 3; ++token)
            for (int head = 0; head < 4; ++head)
                for (int channel = 0; channel < 2; ++channel)
                {
                    const int input = (window * 3 + token) * 2 + channel;
                    const int output = (window * 3 + token) * 8 + head * 2 + channel;
                    ok &= expect(merged[output] == heads[head][input], "head-major channel order");
                }

    heads[2].pop_back();
    ok &= expect(!seedvr2::merge_attention_heads(heads, 2, 3, 2, merged, error), "mismatched shape rejected");
    heads[2].push_back(0.f);
    heads[1][0] = std::numeric_limits<float>::quiet_NaN();
    ok &= expect(!seedvr2::merge_attention_heads(heads, 2, 3, 2, merged, error), "non-finite rejected");

    std::cout << "host_head_merge_tests=" << (ok ? "PASS" : "FAIL") << "\n";
    return ok ? 0 : 1;
}
