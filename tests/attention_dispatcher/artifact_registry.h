#pragma once
#include <array>
#include <filesystem>
#include <map>
#include <string>
#include <variant>
#include <vector>
namespace attention_dispatcher {
struct CoreKey { int window_t,window_h,window_w,grid_t,grid_h,grid_w; std::string variant="baseline"; bool operator==(const CoreKey&)const noexcept; };
enum class ExecutionType { SingleModel, StagedHostHeadMerge, StagedDiTBlock };
enum class MaskMode { ArtifactReference, PaddingValidity };
enum class RopeMode { ArtifactReference, DynamicMMRoPE3D };
struct ArtifactReferenceMaskConfig { std::filesystem::path path; };
struct PaddingValidityMaskConfig { float invalid_key_value = -10000.f; };
using MaskConfig = std::variant<ArtifactReferenceMaskConfig, PaddingValidityMaskConfig>;
struct ArtifactReferenceRopeConfig { std::filesystem::path cos_path, sin_path; };
struct DynamicMMRoPE3DConfig { int rope_dim = 0; double theta = 0.; std::array<int,3> axis_dims{}; std::array<double,3> axis_scales{}; std::string freq_mode = "global_pairs"; };
using RopeConfig = std::variant<ArtifactReferenceRopeConfig, DynamicMMRoPE3DConfig>;
struct AuxiliaryInputConfig { MaskMode mask_mode = MaskMode::ArtifactReference; MaskConfig mask; RopeMode rope_mode = RopeMode::ArtifactReference; RopeConfig rope; };
struct NcnnModelFiles { std::string name; std::filesystem::path param_path,bin_path; };
struct SingleModelArtifact { NcnnModelFiles model; };
struct StagedHeadMergeArtifact { std::array<NcnnModelFiles,4> heads; NcnnModelFiles output_projection; };
struct DiTBlockConfig {
    int hidden_dim=0,txt_dim=0,heads=0,head_dim=0;
    int configured_rope_dim=0,effective_rotary_dim=0,identity_tail_dim=0;
    int mlp_hidden_dim=0,q_max=0,head_pairs=0;
    double norm_eps=0.;
    std::string mlp_type;
    bool is_last_layer=false;
    bool operator==(const DiTBlockConfig&)const noexcept;
};
struct DiTBlockArtifact {
    DiTBlockConfig block;
    std::map<std::string,NcnnModelFiles> models;
    std::filesystem::path params_dir;
    const NcnnModelFiles* model(const std::string&)const noexcept;
};
struct CoreInfo { std::string name; CoreKey key; int tokens_per_window=0; std::vector<int> input_shape; std::string dtype,padding_rule; std::filesystem::path directory; ExecutionType execution_type=ExecutionType::SingleModel; std::variant<SingleModelArtifact,StagedHeadMergeArtifact,DiTBlockArtifact> artifact; std::variant<std::monostate, AuxiliaryInputConfig> auxiliary_inputs; const SingleModelArtifact* single_model()const noexcept; const StagedHeadMergeArtifact* staged_head_merge()const noexcept; const DiTBlockArtifact* dit_block()const noexcept; const AuxiliaryInputConfig* auxiliary()const noexcept; };
class ArtifactRegistry { public: bool scan(const std::filesystem::path&,std::string&); bool scan(const std::vector<std::filesystem::path>&,std::string&); const CoreInfo* lookup(const CoreKey&)const noexcept; std::vector<CoreInfo> list()const; std::size_t size()const noexcept; private:std::vector<CoreInfo> cores_;};
}
