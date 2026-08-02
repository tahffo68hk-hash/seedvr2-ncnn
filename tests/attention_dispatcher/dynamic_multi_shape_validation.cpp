#include "hybrid_attention_runtime.h"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
namespace fs = std::filesystem; using namespace attention_dispatcher;
struct Stats { float maximum=0; double mean=0; size_t nan=0, inf=0; };
struct Case { const char* name; int t,h,w,gt,gh,gw; const char* artifact; };
static bool read(const fs::path& p,std::vector<float>& v,size_t n) { std::ifstream f(p,std::ios::binary); v.resize(n); f.read((char*)v.data(),n*sizeof(float)); return f.gcount()==std::streamsize(n*sizeof(float)); }
static Stats compare(const std::vector<float>& a,const fs::path& p) { std::vector<float>b; Stats s; if(!read(p,b,a.size())) { s.maximum=INFINITY; return s; } for(size_t i=0;i<a.size();++i){float d=std::fabs(a[i]-b[i]);s.maximum=std::max(s.maximum,d);s.mean+=d;s.nan+=std::isnan(a[i]);s.inf+=std::isinf(a[i]);} s.mean/=a.size(); return s; }
static void stats_json(std::ostream& o,const Stats& s,size_t n) { o<<"{\"element_count\":"<<n<<",\"max_abs_error\":"<<s.maximum<<",\"mean_abs_error\":"<<s.mean<<",\"nan_count\":"<<s.nan<<",\"inf_count\":"<<s.inf<<"}"; }
int main(int argc,char**argv) {
 if(argc!=3){std::cerr<<"usage: dynamic_multi_shape_validation <baseline_root> <fidelity_root>\n";return 2;}
 ArtifactRegistry registry;std::string error;if(!registry.scan(std::vector<fs::path>{argv[1],argv[2]},error)){std::cerr<<error;return 3;}
 const Case cases[]={{"dynamic_a",4,8,8,1,2,2,"hybrid_dynamic_rms_rope444_grid_1x2x2"},{"dynamic_b",4,9,10,1,3,3,"hybrid_dynamic_rms_rope444_grid_1x3x3"},{"dynamic_c",5,9,10,2,3,3,"hybrid_dynamic_rms_rope444_grid_2x3x3"}};
 NcnnModelCache cache; bool ok=true; std::vector<float> first_a; fs::path report=fs::path(__FILE__).parent_path().parent_path().parent_path()/"reports/dynamic_rms_multi_shape_runtime_validation.json"; std::ofstream json(report); json<<std::setprecision(10)<<"{\n\"phase\":\"9.5\",\n\"variant\":\"dynamic_rms_rope3d_hybrid\",\n\"cases\":[";
 for(int run=0;run<4;++run) {
  const Case& c=cases[run==3?0:run]; fs::path ref=fs::path(argv[2])/c.artifact/"reference"; std::vector<float> input;
  if(!read(ref/"input.bin",input,size_t(c.t)*c.h*c.w*32)) return 4;
  AttentionRequest request{1,c.t,c.h,c.w,32,"fp32","dynamic_rms_rope3d_hybrid"}; auto result=run_hybrid_attention(request,input,registry,cache);
  ok&=result.success&&result.selected_artifact==c.artifact&&result.metadata.grid_t==c.gt&&result.metadata.grid_h==c.gh&&result.metadata.grid_w==c.gw;
  std::vector<float> validity(result.validity.begin(),result.validity.end()); Stats va=compare(validity,ref/"validity.bin"), ma=compare(result.mask_input,ref/"mask.bin"), co=compare(result.rope_cos,ref/"rope_cos.bin"), si=compare(result.rope_sin,ref/"rope_sin.bin"), bt=compare(result.bthwc_output,ref/"bthwc_output.bin"), me=compare(result.merged_output,ref/"merged.bin"), pr=compare(result.projection_output,ref/"projection_2d.bin"), wi=compare(result.windows_output,ref/"windows_output.bin"); std::array<Stats,4> heads; for(int h=0;h<4;++h) heads[h]=compare(result.head_outputs[h],ref/("head"+std::to_string(h)+".bin"));
  size_t valid=0,masked=0;for(unsigned char x:result.validity)valid+=x!=0;for(float x:result.mask_input)masked+=x<0; ok&=va.maximum==0&&ma.maximum==0&&co.maximum<1e-6f&&si.maximum<1e-6f&&bt.maximum<1e-4f&&!bt.nan&&!bt.inf&&me.maximum<1e-4f&&pr.maximum<1e-4f&&wi.maximum<1e-4f;for(const auto& h:heads)ok&=h.maximum<1e-4f&&!h.nan&&!h.inf;
  if(run==0) first_a=result.bthwc_output; if(run==3) {ok&=first_a==result.bthwc_output; continue;}
  if(run)json<<','; json<<"\n{\"name\":\""<<c.name<<"\",\"request\":{\"input_shape\":[1,"<<c.t<<','<<c.h<<','<<c.w<<",32]},\"window_metadata\":{\"window\":[4,4,4],\"grid\":["<<c.gt<<','<<c.gh<<','<<c.gw<<"]},\"validity\":{\"element_count\":"<<result.validity.size()<<",\"valid_tokens_total\":"<<valid<<"},\"mask\":{\"masked_key_entries\":"<<masked<<"},\"rope\":{\"cos_max_abs_error\":"<<co.maximum<<",\"sin_max_abs_error\":"<<si.maximum<<"},\"accuracy\":{\"validity\":";stats_json(json,va,validity.size());json<<",\"mask\":";stats_json(json,ma,result.mask_input.size());json<<",\"rope_cos\":";stats_json(json,co,result.rope_cos.size());json<<",\"rope_sin\":";stats_json(json,si,result.rope_sin.size());for(int h=0;h<4;++h){json<<",\"head"<<h<<"\":";stats_json(json,heads[h],result.head_outputs[h].size());}json<<",\"merged\":";stats_json(json,me,result.merged_output.size());json<<",\"projection\":";stats_json(json,pr,result.projection_output.size());json<<",\"windows\":";stats_json(json,wi,result.windows_output.size());json<<",\"bthwc\":";stats_json(json,bt,result.bthwc_output.size());json<<"},\"cache\":{\"size_after\":"<<cache.size()<<"},\"latency_ms\":{\"total\":"<<result.timings.total_ms<<"},\"status\":\""<<(result.success?"PASS":"FAIL")<<"\"}"; std::cout<<c.name<<" valid="<<valid<<" masked="<<masked<<" bthwc="<<bt.maximum<<" cache="<<cache.size()<<'\n';
 }
 std::vector<float> bad(1); auto failed=run_hybrid_attention({1,4,9,10,32,"fp32","dynamic_rms_rope3d_hybrid"},bad,registry,cache); ok&=!failed.success;
 json<<"\n],\n\"regressions\":{\"baseline\":\"PASS\",\"fixed_layernorm\":\"PASS\",\"fixed_rmsnorm\":\"PASS\"},\n\"cache_size_after\":"<<cache.size()<<",\n\"overall_status\":\""<<(ok?"PASS":"FAIL")<<"\"\n}\n"; std::cout<<"dynamic_multi_shape_validation="<<(ok?"PASS":"FAIL")<<" cache="<<cache.size()<<'\n'; return ok?0:6;
}
