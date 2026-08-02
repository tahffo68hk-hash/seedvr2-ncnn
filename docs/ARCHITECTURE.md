# Architecture

```text
AttentionRequest -> dispatch_shape -> CoreKey(window, grid, variant) -> ArtifactRegistry -> CoreInfo
  -> SingleModelArtifact | StagedHeadMergeArtifact -> NcnnModelCache -> executor
  -> baseline single-model | hybrid four-head -> HostHeadMerge -> output projection -> reverse/crop -> BTHWC
```

`MaskMode`: ArtifactReference | PaddingValidity. `RopeMode`: ArtifactReference | DynamicMMRoPE3D. Cache keys use param/bin paths.
