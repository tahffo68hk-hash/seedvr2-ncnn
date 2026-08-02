#include "artifact_registry.h"
#include <cassert>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <initializer_list>
using namespace attention_dispatcher;
namespace fs = std::filesystem;

static std::string base() { return R"({"name":"hybrid","variant":"fidelity_ln_rope3d_hybrid","execution":"staged_host_head_merge","window":[1,2,3],"grid":[1,1,1],"tokens_per_window":6,"dtype":"fp32","features":{"masked_attention":true,"qk_norm":"layernorm","rope3d":"host_tensor","head_merge":"host","output_projection":"ncnn_2d_gemm"},"models":{"heads":[{"name":"head0","ncnn_param":"h0p","ncnn_bin":"h0b"},{"name":"head1","ncnn_param":"h1p","ncnn_bin":"h1b"},{"name":"head2","ncnn_param":"h2p","ncnn_bin":"h2b"},{"name":"head3","ncnn_param":"h3p","ncnn_bin":"h3b"}],"output_projection":{"name":"output_projection","ncnn_param":"op","ncnn_bin":"ob"}},"auxiliary_inputs":{"mask":{"mode":"padding_validity","invalid_key_value":-10000},"rope":{"mode":"dynamic_mmrope3d","rope_dim":6,"theta":10000,"axis_dims":[2,2,2],"axis_scales":[1,0.01,0.0001]}}})"; }
static std::string replace(std::string s, const std::string& from, const std::string& to) { auto p=s.find(from); assert(p!=std::string::npos); s.replace(p,from.size(),to); return s; }
static void prepare(const fs::path& root, const std::string& json) { fs::remove_all(root); fs::create_directories(root/"core/reference"); std::ofstream(root/"core"/"metadata.json")<<json; for(const char* x:{"h0p","h0b","h1p","h1b","h2p","h2b","h3p","h3b","op","ob","reference/mask.bin","reference/rope_cos.bin","reference/rope_sin.bin"}) std::ofstream(root/"core"/x)<<'x'; }
static std::string scan_error(const fs::path& root, const std::string& json) { prepare(root,json); ArtifactRegistry r; std::string e; return r.scan(root,e) ? std::string{} : e; }
static bool rejects(const fs::path& root, const std::string& json) { return !scan_error(root,json).empty(); }
static bool accepts(const fs::path& root, const std::string& json) { fs::remove_all(root); fs::create_directories(root/"core"); std::ofstream(root/"core"/"metadata.json")<<json; for(const char* x:{"h0p","h0b","h1p","h1b","h2p","h2b","h3p","h3b","op","ob"}) std::ofstream(root/"core"/x)<<'x'; ArtifactRegistry r; std::string e; return r.scan(root,e)&&r.size()==1; }
static void check_error_contains(const std::string& error, std::initializer_list<const char*> parts) { assert(!error.empty()); for (const char* part : parts) if(error.find(part)==std::string::npos){std::cerr<<"missing='"<<part<<"' error='"<<error<<"'\n";assert(false);} }
static std::string reference_base() { auto value=base(); value=replace(value,R"("mode":"padding_validity","invalid_key_value":-10000)",R"("mode":"artifact_reference","path":"reference/mask.bin")"); return replace(value,R"("mode":"dynamic_mmrope3d","rope_dim":6,"theta":10000,"axis_dims":[2,2,2],"axis_scales":[1,0.01,0.0001])",R"("mode":"artifact_reference","cos":"reference/rope_cos.bin","sin":"reference/rope_sin.bin")"); }

int main() {
    std::string e; ArtifactRegistry baseline;
    const fs::path real=fs::path(__FILE__).parent_path()/"artifacts/attention_core";
    assert(baseline.scan(real,e)&&baseline.size()==5);
    for(const auto& c:baseline.list()) assert(c.key.variant=="baseline"&&c.execution_type==ExecutionType::SingleModel&&c.single_model()&&!c.staged_head_merge());
    const fs::path fidelity=fs::path(__FILE__).parent_path().parent_path()/"seedvr2_attention_fidelity/artifacts/attention_core_fidelity";
    ArtifactRegistry hybrids; assert(hybrids.scan(fidelity,e)); assert(hybrids.size()==7);
    const CoreInfo* h=hybrids.lookup({4,4,4,1,2,2,"fidelity_ln_rope3d_hybrid"}); assert(h&&h->staged_head_merge()&&!h->single_model());
    const CoreInfo* rms=hybrids.lookup({4,4,4,1,2,2,"fidelity_rms_rope3d_hybrid"}); assert(rms&&rms->staged_head_merge()&&!rms->single_model());
    assert(rms->name=="hybrid_fidelity_rms_rope444_grid_1x2x2");
    const CoreInfo* dynamic=hybrids.lookup({4,4,4,1,2,2,"dynamic_rms_rope3d_hybrid"});
    assert(dynamic && dynamic->auxiliary() && dynamic->auxiliary()->mask_mode==MaskMode::PaddingValidity && dynamic->auxiliary()->rope_mode==RopeMode::DynamicMMRoPE3D);
    assert(hybrids.lookup({4,4,4,1,3,3,"dynamic_rms_rope3d_hybrid"}));
    assert(hybrids.lookup({4,4,4,2,3,3,"dynamic_rms_rope3d_hybrid"}));
    assert(!hybrids.lookup({4,4,4,1,9,9,"dynamic_rms_rope3d_hybrid"}));
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
    assert(rejects(tmp,replace(b,"\"invalid_key_value\":-10000","\"invalid_key_value\":-10000,\"typo_field\":1")));
    for (const char* value : {"0","-0","1","-1","10000","10000.0","0.01","-0.01","1e4","1E4","1e-4","-1.25E+3"}) {
        const auto json = std::string(value[0]=='-' ? replace(b,"\"invalid_key_value\":-10000",std::string("\"invalid_key_value\":")+value) : replace(b,"\"theta\":10000",std::string("\"theta\":")+value));
        const bool semantic_valid = std::string(value)!="0" && std::string(value)!="-0";
        assert(semantic_valid ? accepts(tmp,json) : rejects(tmp,json));
    }
    for (const char* value : {"+","-","+1","01","-01","1.",".5","-.5","1e","1e+","1e-","1E+","--1","1..0","1e1.5","0x10","NaN","nan","Infinity","-Infinity","inf","1e9999","1abc"})
        assert(rejects(tmp,replace(b,"\"theta\":10000",std::string("\"theta\":")+value)));
    assert(rejects(tmp,replace(b,"\"rope_dim\":6","\"rope_dim\":4.5")));
    assert(rejects(tmp,replace(b,"\"rope_dim\":6","\"rope_dim\":999999999999999999999999")));
    assert(rejects(tmp,replace(b,"\"axis_dims\":[2,2,2]","\"axis_dims\":[2,3,2]")));
    assert(rejects(tmp,replace(b,"\"axis_scales\":[1,0.01,0.0001]","\"axis_scales\":[1,0.01,0]")));
    assert(rejects(tmp,replace(b,"\"mode\":\"padding_validity\"","\"mode\":\"unknown\"")));
    assert(rejects(tmp,replace(b,"\"invalid_key_value\":-10000","\"invalid_key_value\":0")));
    assert(rejects(tmp,replace(b,"\"invalid_key_value\":-10000","\"invalid_key_value\":1e9999")));
    assert(rejects(tmp,replace(b,"\"rope\":{","\"rope\":[]")));
    assert(rejects(tmp,replace(b,"\"mode\":\"dynamic_mmrope3d\"","\"mode\":\"unknown\"")));
    assert(rejects(tmp,replace(b,"\"theta\":10000","\"theta\":0")));
    assert(rejects(tmp,replace(b,"\"axis_dims\":[2,2,2]","\"axis_dims\":[2,2]")));
    assert(rejects(tmp,replace(b,"\"axis_scales\":[1,0.01,0.0001]","\"axis_scales\":[1,0.01]")));
    assert(rejects(tmp,replace(b,"\"theta\":10000","\"theta\":10000,\"typo_field\":1")));

    ArtifactRegistry atomic; assert(atomic.scan(real,e)); const auto before=atomic.list();
    fs::remove_all(tmp); fs::create_directories(tmp/"broken"); std::ofstream(tmp/"broken"/"metadata.json")<<replace(b,"\"theta\":10000","\"theta\":-");
    assert(!atomic.scan(std::vector<fs::path>{real,tmp},e) && atomic.size()==before.size() && atomic.list().size()==before.size());
    for (const auto& core : before) assert(atomic.lookup(core.key) && atomic.lookup(core.key)->name==core.name);
    assert(!atomic.lookup({4,4,4,1,2,2,""}) && !atomic.lookup({4,4,4,1,2,2,"unknown"}));

    const auto rb=reference_base();
    for (const auto& json : {replace(rb,R"("mask":{)",R"("mask":[])") , replace(rb,R"("mode":"artifact_reference","path":"reference/mask.bin")",R"("path":"reference/mask.bin")"), replace(rb,R"("path":"reference/mask.bin")",R"("path":1)"), replace(rb,R"("path":"reference/mask.bin")",R"("path":"")"), replace(rb,"reference/mask.bin","../mask.bin"), replace(rb,"reference/mask.bin","a/../../mask.bin"), replace(rb,"reference/mask.bin","C:\\\\temp\\\\mask.bin"), replace(rb,"reference/mask.bin","//server/share/mask.bin"), replace(rb,"reference/mask.bin","reference"), replace(rb,"reference/mask.bin","missing.bin"), replace(rb,R"("path":"reference/mask.bin")",R"("path":"reference/mask.bin","invalid_key_value":-1)")}) check_error_contains(scan_error(tmp,json),{"metadata.json"});
    check_error_contains(scan_error(tmp,replace(rb,"reference/mask.bin","../mask.bin")),{"metadata.json","auxiliary_inputs.mask.path","unsafe path"});
    for (const auto& json : {replace(rb,R"("rope":{)",R"("rope":[])") , replace(rb,R"("mode":"artifact_reference","cos":"reference/rope_cos.bin","sin":"reference/rope_sin.bin")",R"("cos":"reference/rope_cos.bin","sin":"reference/rope_sin.bin")"), replace(rb,R"("cos":"reference/rope_cos.bin")",R"("cos":1)"), replace(rb,R"("sin":"reference/rope_sin.bin")",R"("sin":1)"), replace(rb,"reference/rope_cos.bin","../rope_cos.bin"), replace(rb,"reference/rope_sin.bin","../rope_sin.bin"), replace(rb,"reference/rope_cos.bin","missing.bin"), replace(rb,"reference/rope_sin.bin","missing.bin"), replace(rb,"reference/rope_cos.bin","reference"), replace(rb,R"("sin":"reference/rope_sin.bin")",R"("sin":"reference/rope_sin.bin","rope_dim":6)")}) check_error_contains(scan_error(tmp,json),{"metadata.json"});
    check_error_contains(scan_error(tmp,replace(rb,"reference/rope_sin.bin","../rope_sin.bin")),{"metadata.json","auxiliary_inputs.rope.sin","unsafe path"});
    for (const char* value : {"+1","01","1.",".5","1e","1e+","1e-","1..0","NaN","Infinity","1e9999"}) check_error_contains(scan_error(tmp,replace(b,R"("invalid_key_value":-10000)",std::string(R"("invalid_key_value":)")+value)),{"metadata.json","auxiliary_inputs.mask.invalid_key_value","character"});
    for (const auto& invalid : {replace(rb,"reference/mask.bin","../mask.bin"),replace(rb,"reference/rope_sin.bin","../rope_sin.bin")}) { ArtifactRegistry rollback; assert(rollback.scan(std::vector<fs::path>{real,fidelity},e)); const auto before_reference_failure=rollback.list(); prepare(tmp,invalid); assert(!rollback.scan(std::vector<fs::path>{real,fidelity,tmp},e)); assert(rollback.size()==before_reference_failure.size()); assert(rollback.lookup({4,4,4,1,2,2,"fidelity_ln_rope3d_hybrid"})); assert(rollback.lookup({4,4,4,1,2,2,"fidelity_rms_rope3d_hybrid"})); assert(rollback.lookup({4,4,4,1,2,2,"dynamic_rms_rope3d_hybrid"})); }
    fs::remove_all(tmp);
    std::cout<<"artifact_registry_tests=PASS json_number_valid_cases=12 json_number_invalid_cases=23 mask_reference_negative_cases=11 rope_reference_negative_cases=10 dynamic_metadata_negative_cases=14 atomic_scan_rollback_cases=3 baseline_count="<<baseline.size()<<" fidelity_count="<<hybrids.size()<<'\n';
}
