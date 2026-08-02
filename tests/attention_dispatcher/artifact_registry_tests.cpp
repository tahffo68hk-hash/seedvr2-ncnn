#include "artifact_registry.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace attention_dispatcher;
namespace fs = std::filesystem;

static std::string base() { return R"({"name":"hybrid","variant":"fidelity_ln_rope3d_hybrid","execution":"staged_host_head_merge","window":[1,2,3],"grid":[1,1,1],"tokens_per_window":6,"dtype":"fp32","features":{"masked_attention":true,"qk_norm":"layernorm","rope3d":"host_tensor","head_merge":"host","output_projection":"ncnn_2d_gemm"},"models":{"heads":[{"name":"head0","ncnn_param":"h0p","ncnn_bin":"h0b"},{"name":"head1","ncnn_param":"h1p","ncnn_bin":"h1b"},{"name":"head2","ncnn_param":"h2p","ncnn_bin":"h2b"},{"name":"head3","ncnn_param":"h3p","ncnn_bin":"h3b"}],"output_projection":{"name":"output_projection","ncnn_param":"op","ncnn_bin":"ob"}}})"; }
static std::string replace(std::string s, const std::string& from, const std::string& to) { auto p=s.find(from); assert(p!=std::string::npos); s.replace(p,from.size(),to); return s; }
static bool rejects(const fs::path& root, const std::string& json) { fs::remove_all(root); fs::create_directories(root/"core"); std::ofstream(root/"core"/"metadata.json")<<json; for(const char* x:{"h0p","h0b","h1p","h1b","h2p","h2b","h3p","h3b","op","ob"}) std::ofstream(root/"core"/x)<<'x'; ArtifactRegistry r; std::string e; return !r.scan(root,e)&&!e.empty(); }

int main() {
    std::string e; ArtifactRegistry baseline;
    const fs::path real=fs::path(__FILE__).parent_path()/"artifacts/attention_core";
    assert(baseline.scan(real,e)&&baseline.size()==5);
    for(const auto& c:baseline.list()) assert(c.key.variant=="baseline"&&c.execution_type==ExecutionType::SingleModel&&c.single_model()&&!c.staged_head_merge());
    const fs::path fidelity=fs::path(__FILE__).parent_path().parent_path()/"seedvr2_attention_fidelity/artifacts/attention_core_fidelity";
    ArtifactRegistry hybrids; assert(hybrids.scan(fidelity,e));
    const CoreInfo* h=hybrids.lookup({4,4,4,1,2,2,"fidelity_ln_rope3d_hybrid"}); assert(h&&h->staged_head_merge()&&!h->single_model());
    assert(h->staged_head_merge()->heads[0].name=="head0"&&h->staged_head_merge()->heads[3].name=="head3"&&h->staged_head_merge()->output_projection.name=="output_projection");
    const fs::path tmp=fs::temp_directory_path()/"seedvr-registry-tests"; const auto b=base();
    assert(rejects(tmp,replace(b,R"("masked_attention":true)",R"("masked_attention":"true")")));
    assert(rejects(tmp,replace(b,R"("masked_attention":true)",R"("masked_attention":True)")));
    assert(rejects(tmp,replace(b,R"(,"variant":"fidelity_ln_rope3d_hybrid")","")));
    assert(rejects(tmp,replace(b,"fidelity_ln_rope3d_hybrid","baseline")));
    assert(rejects(tmp,replace(b,"staged_host_head_merge","unsupported")));
    assert(rejects(tmp,replace(b,"\"h0p\"","\"missing\"")));
    assert(rejects(tmp,replace(b,"\"h0b\"","\"missing\"")));
    assert(rejects(tmp,replace(b,"\"op\"","\"missing\"")));
    assert(rejects(tmp,replace(b,"\"ob\"","\"missing\"")));
    assert(rejects(tmp,replace(b,"\"head3\"","\"head0\"")));
    assert(rejects(tmp,replace(b,"\"head0\"","\"head4\"")));
    assert(rejects(tmp,replace(b,"\"tokens_per_window\":6","\"tokens_per_window\":7")));
    assert(rejects(tmp,replace(b,"\"h0p\"","\"../h0p\"")));
    assert(rejects(tmp,replace(b,"\"models\":{","\"models\":[]")));
    assert(rejects(tmp,replace(b,"\"output_projection\":{","\"output_projection\":[]")));
    fs::remove_all(tmp);
    std::cout<<"artifact_registry_tests=PASS cases=31 baseline_count="<<baseline.size()<<" fidelity_count="<<hybrids.size()<<'\n';
}
