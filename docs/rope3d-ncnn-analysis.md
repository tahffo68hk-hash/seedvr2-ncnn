# 3-D RoPE → ncnn analysis

`NaMMRotaryEmbedding3d.get_freqs` is responsible for frequency generation;
`forward` only applies the resulting frequencies to Q/K. This split is practical
for ncnn: generate phase/cos/sin host tensors from dispatcher metadata, then
apply them in the model.

The Phase 7 `rope_apply.pt` receives `[tokens,heads,head_dim]` plus host phases.
pnnx produced native `UnaryOp(cos/sin)`, `Slice`, `Reshape`, `BinaryOp`, and
`Concat`; no custom layer was required. Release Extractor error: max
`2.38419e-07`, mean `4.27463e-09`.

| Item | Classification | Result |
|---|---|---|
| Apply host phases / cos-sin | A native graph | pnnx+ncnn PASS |
| Frequency/position generation | C host tensor | required until exact rotary package/config is available |
| rotate-half equivalent | B rewrite if converter cannot lower it | current interleaved slice/mul/sub/add/concat graph succeeds |
| Runtime trig in ncnn | C host tensor preferred | not needed for the selected path |
| Custom RoPE layer | D custom | not needed for this fixture |

The exact `rotary_embedding_torch` pair layout and rotated-width contract remain
unverified because this workspace deliberately has no installed copy of that
dependency and no download was performed.
