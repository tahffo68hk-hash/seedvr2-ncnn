# Dynamic Window Analysis

## Scope

This fixture models only BTHWC local-window bookkeeping. It does not import
SeedVR2, use weights, run GPU/Vulkan, or implement a custom layer.

Base window is `(T,H,W) = (4,4,4)`. The policy preserves the configured spatial
tile and uses `min(T, 4)` for the temporal tile. Right-edge padding brings each
dimension to a multiple of its selected tile.

## Metadata Contract

`dynamic_window_partition()` returns `[num_windows, tokens_per_window, C]` and:

```text
{
  original_shape, padded_shape,
  window: (window_t, window_h, window_w),
  window_t, window_h, window_w,
  grid: (grid_t, grid_h, grid_w),
  num_windows, tokens_per_window,
  padding: (pad_t, pad_h, pad_w)
}
```

Window order is `B -> temporal grid -> height grid -> width grid`; token order
within a window is `t -> h -> w`. Reverse uses the same grid, reconstructs the
padded BTHWC buffer, then crops the right-edge padding to `original_shape`.

## Verified Cases

| Case | Input BTHWC | Selected window | Padding T/H/W | Windows | Tokens/window | Round-trip |
| --- | --- | --- | --- | ---: | ---: | --- |
| A: divisible | `[1,4,16,16,32]` | `[4,4,4]` | `[0,0,0]` | 16 | 64 | PASS |
| B: short time | `[1,3,16,16,32]` | `[3,4,4]` | `[0,0,0]` | 16 | 48 | PASS |
| C: spatial remainder | `[1,4,17,17,32]` | `[4,4,4]` | `[0,3,3]` | 25 | 64 | PASS |

All tests assert exact equality, not tolerance: `dynamic_window_reverse(*dynamic_window_partition(x)) == x`.

## ncnn Export Boundary

| Step | Classification | Reason |
| --- | --- | --- |
| Fixed-shape reshape/permute partition | A: TorchScript/pnnx candidate | All target dimensions and grid counts can be constants in a fixed export. The pre-partitioned attention core already follows this pattern. |
| Fixed-shape reverse/crop | A: TorchScript/pnnx candidate | It can be represented by constant reshapes/permutations and a fixed crop, although host-side layout handling is simpler. |
| `window_t = min(T, base_t)` | B: graph rewrite / model-variant selection | TorchScript tracing specializes Python shape control flow. pnnx receives one concrete temporal size, not a reusable shape decision. |
| Grid/count/token metadata | B: graph rewrite / host metadata | `num_windows` and tokens are shape-derived values consumed by reshape and attention. They should be supplied by a shape-specialized wrapper or a graph rewrite pass. |
| Runtime right-edge padding | C: custom layer for a general model | The pad amount, allocation, partition, attention mask semantics, reverse, and crop all depend on runtime dimensions. A pnnx static graph cannot express the general metadata-driven control path portably. |
| Attention mask for padded tokens | C: custom layer for a general model | A real model must prevent padded tokens from changing attention results; the current round-trip fixture has no attention mask. |

## Implication for SeedVR2 Migration

For a bounded set of video shapes, select a fixed-window variant before ncnn
execution, perform host-side partition/reverse, and reuse the FP32 QKV
flatten-Gemm-restore attention core. This covers the current fixture without a
custom layer.

For arbitrary SeedVR2 inputs, `DynamicWindowAttentionLayer` needs to own runtime
window selection, pad/mask generation, partition/reverse, and crop. The existing
fixed attention core remains the numerical reference for its per-window compute;
no Vulkan implementation is implied by this analysis.
