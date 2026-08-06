# SeedVR2 ncnn 自适应窗口原型

[English](README.md) | [简体中文](README.zh-CN.md)

面向 ncnn 的、由元数据驱动的自适应窗口注意力（Adaptive Window Attention）运行时原型。

## 当前状态

Phase 7.5-B 保持 READY：五个由 manifest 驱动的 FP32 CPU 基线 profile 以及 LayerNorm 分阶段混合（staged-hybrid）夹具均通过。Phase 8 数学版 RMSNorm 通过本夹具完整的 ncnn 注意力头精度门限，状态为 READY。Phase 9.5 动态多形状 RMS 混合运行时 READY。Phase 10 head_dim=128 READY：精确锁定并端到端验证了 SeedVR2 MM-RoPE3D 契约（rotary-embedding-torch==0.5.3，配置 128 中有效旋转维度 126、2 维恒等尾部，axis_dims=[42,42,42]）（真实契约 S/M/P），Phase 10 合成夹具作为独立的历史回归保留。Phase 11 真实单个 DiT 块的接口/形状/执行契约 READY（后被 Phase 12 分阶段 ncnn 保真度取代）：在隔离 conda 环境 `seedvr2-block-probe`（torch 2.4.0+cpu、einops 0.7.0、diffusers 0.33.1、rotary-embedding-torch 0.5.3）中以完整 2560 宽度（profile A/B/C/D）执行了 `NaMMSRTransformerBlock`（SeedVR2 `e4de8c2`）的真实随机权重 forward，并带有运行时 hook 证据：`[vid_win; txt]` 打包、`vid_freqs[l:l+f]` 偏移、恒等尾部 2、六段调制、SwiGLU、残差顺序、末层 vid_only、窗口重组，负例 17/17；仅 flash_attn/apex 融合 CUDA 库被测试夹具的 CPU 兼容实现替换（parity < 1e-6；CUDA 保真度未验证）。参见 [seedvr2-single-dit-block-contract-report.md](D:/AIProjects/seedvr2-ncnn/docs/seedvr2-single-dit-block-contract-report.md)。Phase 12 合成单 DiT 块分阶段 ncnn 保真度 READY：真实 `NaMMSRTransformerBlock` 拓扑以缩放规模 C=512/H=4/HD=128 运行（`HybridDiTBlockRuntime`，seed 20260802，MLP hidden 1536 来自真实 SwiGLUMLP 公式）；profile A/B/C/D 全部 PASS，最终 vid/txt < 1e-4，A->B->C->D->A 逐位一致，末层 vid_only PASS，registry/cache 19，负例 24+7+registry，Phase 9.5/10/11 回归 PASS（真实 `repeat_concat_idx` 的 batch>1 txt 索引怪癖被精确复现）。参见 [synthetic-seedvr2-dit-block-ncnn-report.md](D:/AIProjects/seedvr2-ncnn/docs/synthetic-seedvr2-dit-block-ncnn-report.md)。Phase 13 真实宽度单块可行性 + 权重映射 READY：真实 `NaMMSRTransformerBlock`（C=2560/H=20/HD=128，MLP 6912，seed 20260803，随机权重）导出为 19 个 FP32 ncnn 阶段模型（`pnnx fp16=0`），并在现有 `HybridDiTBlockRuntime` 上端到端通过最小 RW-A profile（最终 vid/txt ~1e-5，逐位确定性，cache 19，峰值工作集 ~685 MB，缓存后 forward ~31 ms），同时提交了完整的 state_dict->ncnn 权重映射契约（29/29，AdaSingle 仅运行时，qk-norm 全 1 被 pnnx 折叠，末层复用全部 19 个模型）。参见 [realwidth-seedvr2-block-feasibility-report.md](D:/AIProjects/seedvr2-ncnn/docs/realwidth-seedvr2-block-feasibility-report.md) 与 [seedvr2-block-weight-mapping-report.md](D:/AIProjects/seedvr2-ncnn/docs/seedvr2-block-weight-mapping-report.md)。Phase 14 真实权重单块混合视觉里程碑 **READY**：官方 `ByteDance-Seed/SeedVR2-3B` checkpoint（5/5 sha256 校验，仅官方源）安全加载（`weights_only=True`），schema 校验（32 块，block 16 = floor(32/2) 中间块，635 个 fp32 key），q/k RMSNorm 强制真实仿射（非全 1；禁止 Phase 13 的 0 字节复用），通过 15-key state_dict->ncnn 契约映射（mapped_numel == source_numel，QKV/SwiGLU 逐位一致），导出为 19 个 FP32 阶段（`pnnx fp16=0`，git-ignored `.phase14_local_artifacts/`）；带真实权重的 ncnn `HybridDiTBlockRuntime` 通过 RW-A O(1) 夹具（最终 vid/txt < 2e-4）与官方上下文 block-16 parity（rel < 1e-3），并达成混合视觉里程碑：官方 4 帧 128x128 流水线中单块由 ncnn 替换，像素 PSNR 52.5 dB / SSIM 0.9975，adapter 证据（ncnn=1、original=0、无静默回退），负例 29/29，回归 22/22。参见 [real-weight-single-block-report.md](D:/AIProjects/seedvr2-ncnn/docs/real-weight-single-block-report.md) 与 [hybrid-visual-milestone-report.md](D:/AIProjects/seedvr2-ncnn/docs/hybrid-visual-milestone-report.md)。Phase 15 真实权重多块流式混合 **READY**：官方 `ByteDance-Seed/SeedVR2-3B` checkpoint 的块 15/16/17/18 以严格流式模式（每块 `LOAD -> EXECUTE -> UNLOAD`；运行时销毁、模型缓存清零、常驻字节 0、文件句柄释放；`max_resident_weighted_blocks=1` 恒定）通过 ncnn 运行。链式 parity（streaming == resident，全新干净构建 `build/phase15-multiblock-release`）PASS：chain_1 rel 8.872e-9（限 1e-3）、chain_2 2.835e-6（2e-3）、chain_4 3.777e-6（5e-3），NaN/Inf=0；官方流水线 1/2/4 块替换视觉：像素 PSNR 53.65/52.72/52.90 dB（>=38），SSIM 0.9978/0.9978/0.9977（>=0.98），单块 parity 1.8e-8..3.8e-6（限 1e-3），每块 adapter 证据（ncnn=1 / original=0 / silent_fallback=false）；内存画像：流式峰值平稳 474/496/476 MB，对比常驻线性 446/839/1530 MB（逐块流式是 32 块场景的必需策略）；负例 48/48；回归 27/27；git 保护 PASS。参见 [multiblock-streaming-hybrid-report.md](D:/AIProjects/seedvr2-ncnn/docs/multiblock-streaming-hybrid-report.md) 与 [multiblock-error-growth-report.md](D:/AIProjects/seedvr2-ncnn/docs/multiblock-error-growth-report.md)。

Phase 16 完整 SeedVR2 DiT 流式 **READY**：全部 32 个真实块 0..31（含唯一末层 31）以严格逐块流式模式通过 ncnn 运行（`LOAD -> EXECUTE -> UNLOAD`，`max_resident_weighted_blocks=1`；每块运行时销毁、模型缓存清零、常驻字节 0，工件文件改名探针通过）。块 0..9 使用拆分 29-key 布局，块 10..31 使用共享 15-key `.all.` 布局——均从 checkpoint 推导，绝不硬编码。32/32 工件全部校验（`reports/phase16_artifact_inventory.json`，逻辑 13.32 GB 且物理去重；块 15-18 从 Phase 15 硬链接复用）。链式 parity（streaming == FP32 参考，全新构建 `build/phase16-full-dit-release`）：chain_8 rel 5.6465e-4（限 1e-2）、chain_16 2.0992e-3（2e-2）、chain_full 5.0234e-3（5e-2），NaN/Inf=0，3 次逐位一致；末层独立 parity 组合 rel 4.05e-6（限 1e-3）。官方流水线 8/16/all-32 块替换为单次严格 ncnn adapter 调用：像素 PSNR 51.336/52.551/51.050 dB（>=34），SSIM 0.9974/0.9976/0.9972（>=0.95），adapter 证据 ncnn=8/16/32、original=0、silent_fallback=false、`max_resident_weighted_blocks=1`；负例 80/80；回归 27/27；git 保护 PASS。参见 [full-dit-streaming-report.md](D:/AIProjects/seedvr2-ncnn/docs/full-dit-streaming-report.md) 与 [full-dit-error-growth-report.md](D:/AIProjects/seedvr2-ncnn/docs/full-dit-error-growth-report.md)。

Phase 17 真实权重 SeedVR2 VAE-in-ncnn + 完整视频修复 **READY**：官方 `ema_vae.pth`（sha256 `c7df8a67...`，250 个 fp32 张量，250,648,227 numel，Phase 14 下载 5/5 校验）导出为 14 个 FP32 ncnn 阶段链（7 编码器 + 7 解码器，`pnnx fp16=0`，profile A/B）。阶段 parity 14/14（rel max ~2.1e-6）；编码器 parity A 2.166e-6 / B 1.897e-6；解码器 parity A/B rel 2.08e-5 / 1.73e-5，PNG PSNR 117.9 / 119.3 dB、SSIM 1.0；完整 VAE 往返 A/B 像素 PSNR ~116.7 / 117.8 dB、SSIM 1.0，ncnn 编码器 7 / 解码器 14 次调用，无静默回退；权重映射 250/250（mapped_numel == source_numel）；tiling / 流式生命周期 / 延迟 / 内存 / IO 契约 PASS；负例 70/70；回归 PASS。官方流水线完整视频修复（4 帧 128x128，seed 20260803，steps 1，cfg 1.0，scheduler/宿主 conditioning 仍为 PyTorch）：`full_ncnn_core`（ncnn VAE 编码 + 32 块 ncnn DiT + ncnn VAE 解码）像素 PSNR 51.195 dB（>=40）/ SSIM 0.9990（>=0.98），latent RMSE 0.0206，NaN/Inf=0；`dit_ncnn_vae_pytorch` 51.998 dB / 0.9989；`dit_ncnn_decoder_ncnn` 52.105 dB / 0.9990；adapter 证据 ncnn 编码器=1 / 解码器=1 / original=0 / silent_fallback=false / max_resident_weighted_dit_blocks=1。参见 [vae-ncnn-full-video-restoration-report.md](D:/AIProjects/seedvr2-ncnn/docs/vae-ncnn-full-video-restoration-report.md)。

这是一个 SeedVR2 风格注意力子系统，具备单块、小规模多块（1/2/4）以及完整 32 块（8/16/all）流式 ncnn 运行时，并非完整的 SeedVR2 移植。

```mermaid
flowchart LR
  R[AttentionRequest] --> D[Host dispatcher]
  D --> K[CoreKey]
  K --> G[ArtifactRegistry]
  G --> C[NcnnModelCache]
  C --> P[Partition and mask]
  P --> N[Static ncnn core]
  N --> O[Reverse crop BTHWC]
```

支持的 profile：`[1,4,16,16,32]`、`[1,3,16,16,32]`、`[1,4,17,17,32]`、`[1,8,32,32,32]`、`[1,5,19,21,32]`、动态 A/B/C `[1,4,8,8,32]` / `[1,4,9,10,32]` / `[1,5,9,10,32]`，以及合成/真实契约 HD128 S/M/P `[1,4,4,4,512]` / `[1,4,8,8,512]` / `[1,5,9,10,512]`。

## 构建与验证

配置 `tests/attention_dispatcher` 的 Release Ninja 构建，然后构建 `model_cache_tests` 与 `runtime_validation`。运行：

```text
runtime_validation tests/attention_dispatcher/artifacts/attention_core
```

它会写入 [runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\runtime_validation.json) 与 [runtime-validation-report.md](D:\AIProjects\seedvr2-ncnn\docs\runtime-validation-report.md)。

Phase 10 合成 HD128 验证：

```text
synthetic_hd128_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

它会写入 [synthetic_hd128_runtime_validation.json](D:\AIProjects\seedvr2-ncnn\reports\synthetic_hd128_runtime_validation.json)，文档见 [synthetic-hd128-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\synthetic-hd128-runtime-report.md)。

Phase 10.1 真实 SeedVR2 MM-RoPE3D 契约验证（全新干净构建 `build/phase10_1-real-mmrope-release`）：

```text
real_mmrope_contract_validation tests/seedvr2_attention_fidelity/artifacts/attention_core_fidelity reports/phase10_regression_inputs.txt
```

它会写入 [real_seedvr2_mmrope_contract_validation.json](D:\AIProjects\seedvr2-ncnn\reports\real_seedvr2_mmrope_contract_validation.json)，文档见 [real-seedvr2-mmrope-contract-report.md](D:\AIProjects\seedvr2-ncnn\docs\real-seedvr2-mmrope-contract-report.md)。

Phase 11 真实单 DiT 块契约：提取（source/AST，无需 ncnn 构建）与 Phase 11.1 真实随机权重 forward 执行：

```text
D:\AIProjects\seedvr2-ncnn\.conda-cuda\python.exe tests\seedvr2_dit_block_contract\extract_seedvr2_block_contract.py
conda run -n seedvr2-block-probe python tests\seedvr2_dit_block_contract\probe_real_block.py
conda run -n seedvr2-block-probe python run_phase11_1_probe.py
```

`probe_real_block.py` 在进程内运行三个套件并写入 `tests/seedvr2_dit_block_contract/probe_result.json`（PASS）。运行器 `run_phase11_1_probe.py` 以子进程方式执行三个测试文件并写入 [seedvr2_real_block_forward.json](D:\AIProjects\seedvr2-ncnn\reports\seedvr2_real_block_forward.json)。提取会写入 [seedvr2_single_dit_block_contract.json](D:\AIProjects\seedvr2-ncnn\reports\seedvr2_single_dit_block_contract.json) 与 `tests/seedvr2_dit_block_contract/seedvr2_block_contract.json`。文档见 [seedvr2-single-dit-block-contract-report.md](D:\AIProjects\seedvr2-ncnn\docs\seedvr2-single-dit-block-contract-report.md)。

## 限制

仅支持已注册的 profile；任意分辨率需要匹配的静态核心。混合保真度夹具提供工件自带的 mask 与 RoPE 数据，文档见 [hybrid-runtime-integration-report.md](D:\AIProjects\seedvr2-ncnn\docs\hybrid-runtime-integration-report.md)。宿主代码负责动态控制流，ncnn 负责执行注意力数学运算。Vulkan、SeedVR2 权重加载、泛化运行时验证以及 Apex `fusedrms` 直接依赖 parity 仍在范围外或被阻塞，详见 [rms-hybrid-runtime-report.md](D:\AIProjects\seedvr2-ncnn\docs\rms-hybrid-runtime-report.md)。

## Phase 9.4 状态

`Phase 9.4 metadata/runtime routing = READY`。固定 LayerNorm 与 RMSNorm 使用工件自带的 mask/RoPE 输入；Dynamic A 使用生成的 padding 有效性 mask 与动态 MM-RoPE3D，自然 `valid_tokens=64`。任何公开 C++ 类型布局变更都需要新的干净构建目录。在 Dynamic B/C 实现之前，`Phase 9 multi-shape validation = PARTIAL`。

## 当前结果

这是一个面向 ncnn 的、元数据驱动的 SeedVR2 风格自适应窗口注意力子系统，而非完整的 SeedVR2 推理。它包含：基线单模型执行；分阶段四头 Hybrid 执行；Q/K LayerNorm 与数学版 RMSNorm；局部窗口 3D RoPE；动态 padding 有效性 mask；宿主头合并；ncnn 投影；进程级模型缓存；variant 感知工件；Dynamic A/B/C FP32 CPU parity 约 `1e-7`；以及在文档化合成契约与真实 SeedVR2 MM-RoPE3D 契约（rotary-embedding-torch 0.5.3）下均成立的 head_dim=128 夹具（S/M/P）：RoPE parity 低于 `1e-6`、阶段 parity 低于 `1e-4`、有效性与 mask 完全一致。

所有 FP32 parity/参考工件必须使用 `pnnx fp16=0`。在公开 C++ 类型布局变更（例如 `CoreInfo` 或 `std::variant` payload）后，请使用新的干净构建目录并重建所有依赖目标。

Phase 7.5-B LayerNorm Hybrid Runtime = READY；Phase 8 RMSNorm Hybrid Runtime = READY；Phase 8.1 FP32 转换精度修复 = READY；Phase 9.1 DynamicAttentionContract = READY；Phase 9.2 Dynamic padding mask = READY；Phase 9.3 Dynamic MM-RoPE3D = READY；Phase 9.4 Typed auxiliary metadata/runtime routing = READY；Phase 9.5 Dynamic multi-shape RMS Hybrid Runtime = READY；Phase 10 head_dim=128 = READY（从 rotary-embedding-torch==0.5.3 确认并验证了真实 SeedVR2 MM-RoPE3D 契约；合成契约作为历史回归保留）。

Phase 14 = READY（真实 SeedVR2 3B checkpoint，真实权重单块混合视觉里程碑；见上文报告）。

Phase 13 真实宽度验证（干净构建 `build/phase13-real-width-release`）：

```text
build/phase13-real-width-release/realwidth_dit_block_validation.exe .phase13_local_artifacts reports/realwidth_seedvr2_block_feasibility.json
build/phase13-real-width-release/realwidth_artifact_inventory_tests.exe .phase13_local_artifacts reports/realwidth_artifact_inventory.json
conda run -n seedvr2-block-probe python tests/seedvr2_weight_mapping/aggregate_realwidth_feasibility.py
```

重型真实宽度工件只放在 git-ignored 的 `.phase13_local_artifacts/` 目录；只提交 manifest、映射契约、脚本、测试与报告（见 `tests/seedvr2_weight_mapping/check_no_large_tracked_artifacts.py`）。

TODO（Phase 18，尚未开始——需要明确授权）：Android/Vulkan 目标（Pad 7 Pro 未验证）、直接融合内核 parity（flash_attn/apex）以及 full_pure_ncnn_application（scheduler/timestep/conditioning 移入 ncnn）。Phase 17 已提交证据：真实权重 VAE-in-ncnn 完整视频修复——full_ncnn_core 像素 PSNR 51.195 dB / SSIM 0.9990（>=40 / >=0.98），latent RMSE 0.0206，阶段 parity 14/14 rel <=2.1e-6，编码器/解码器 parity rel 1.9e-6 / 1.7e-5，负例 70/70，回归 PASS，git 保护 PASS。

## Phase 18A Android ARM64 脚手架

Phase 18A 位于 `phase18a-android-arm64-scaffold`，状态为 **READY_FOR_DEVICE**；完整 Phase 18 **NOT READY**。免设备 [`android/`](android/) 工程可构建仅含 AArch64 `libseededge.so` 的 Debug 与未签名 Release APK，包含固定版本的 ncnn/Vulkan 源码集成、五个 JSON JNI 端点、一个确定性合成 CPU 测试以及外部模型 manifest 契约。未进行任何小米 Pad、Android 运行时、Vulkan 推理、VAE/DiT 或完整修复执行。参见 [`docs/android-phase18a-build-and-scaffold-report.md`](docs/android-phase18a-build-and-scaffold-report.md)。

## Phase 18B Find X8 运行时更新

临时 PKB110（Android 16/API 36，arm64-v8a）运行时验证 PASS：CPU FP32 合成 20/20，真实 ncnn Vulkan 上传/推理/下载 10/10（Mali-G925-Immortalis MC12），CPU/Vulkan 最大误差 0，无回退，最终 FATAL/原生崩溃数 0。这不替代小米 Pad 7 Pro 的最终验证；完整 Phase 18 仍为 **NOT READY**。参见 `docs/android-phase18b-findx8-runtime-validation.md`。