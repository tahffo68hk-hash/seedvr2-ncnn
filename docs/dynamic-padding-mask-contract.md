# Dynamic padding-mask contract

Dynamic A has logical mask shape `[num_windows,tokens,tokens] = [4,64,64]`.
For its natural no-padding input, validity has shape `[4,64]`, every entry is
valid, `valid_tokens_min=valid_tokens_max=64`, `invalid_tokens=0`, and the
generated additive key mask has `masked_key_count=0`.

The ncnn boundary is checked by `ncnn_mat_contract_tests`: the mask is stored
as `ncnn::Mat(64,64,4)` and every flat vector element is copied unchanged.
