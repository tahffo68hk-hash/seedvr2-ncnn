#pragma once
#include <array>
#include <filesystem>
#include <string>
#include <variant>
#include <vector>
namespace attention_dispatcher {
struct CoreKey { int window_t,window_h,window_w,grid_t,grid_h,grid_w; std::string variant="baseline"; bool operator==(const CoreKey&)const noexcept; };
enum class ExecutionType { SingleModel, StagedHostHeadMerge };
struct NcnnModelFiles { std::string name; std::filesystem::path param_path,bin_path; };
struct SingleModelArtifact { NcnnModelFiles model; };
struct StagedHeadMergeArtifact { std::array<NcnnModelFiles,4> heads; NcnnModelFiles output_projection; };
struct CoreInfo { std::string name; CoreKey key; int tokens_per_window=0; std::vector<int> input_shape; std::string dtype,padding_rule; std::filesystem::path directory; ExecutionType execution_type=ExecutionType::SingleModel; std::variant<SingleModelArtifact,StagedHeadMergeArtifact> artifact; const SingleModelArtifact* single_model()const noexcept; const StagedHeadMergeArtifact* staged_head_merge()const noexcept; };
class ArtifactRegistry { public: bool scan(const std::filesystem::path&,std::string&); bool scan(const std::vector<std::filesystem::path>&,std::string&); const CoreInfo* lookup(const CoreKey&)const noexcept; std::vector<CoreInfo> list()const; std::size_t size()const noexcept; private:std::vector<CoreInfo> cores_;};
}
