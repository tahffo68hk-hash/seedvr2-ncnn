#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <vector>
#include <net.h>
namespace fs=std::filesystem;
static bool read(const fs::path& p,std::vector<float>& v,size_t n){std::ifstream f(p,std::ios::binary);v.resize(n);f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(n*sizeof(float)));return f.gcount()==static_cast<std::streamsize>(n*sizeof(float));}
int main(int argc,char**argv){if(argc!=2){std::cerr<<"usage: ncnn_rms_head_rewrite_runner <root>\n";return 2;}const fs::path root=argv[1];std::vector<float>x,m,c,s;if(!read(root/"input.bin",x,2048)||!read(root/"mask.bin",m,4096)||!read(root/"rope_cos.bin",c,256)||!read(root/"rope_sin.bin",s,256))return 3;ncnn::Mat in0(32,64),in1(64,64),in2(4,64),in3(4,64);std::memcpy(in0.data,x.data(),8192);std::memcpy(in1.data,m.data(),16384);std::memcpy(in2.data,c.data(),1024);std::memcpy(in3.data,s.data(),1024);bool pass=true;for(char rewrite:std::string("BCD")){const fs::path d=root/(std::string("head_rms_")+static_cast<char>(std::tolower(rewrite)));ncnn::Net net;net.opt.use_packing_layout=false;net.opt.num_threads=1;if(net.load_param((d/"model.ncnn.param").string().c_str())||net.load_model((d/"model.ncnn.bin").string().c_str()))return 4;ncnn::Mat out;auto ex=net.create_extractor();if(ex.input("in0",in0)||ex.input("in1",in1)||ex.input("in2",in2)||ex.input("in3",in3)||ex.extract("out0",out)||out.total()!=512)return 5;std::vector<float>reference,actual(512);if(!read(d/"reference.bin",reference,512))return 6;std::memcpy(actual.data(),out.data,2048);float maximum=0;double mean=0;for(size_t i=0;i<512;++i){const float e=std::fabs(actual[i]-reference[i]);maximum=std::max(maximum,e);mean+=e;}mean/=512;pass&=maximum<1e-4f;std::cout<<"head_rms_"<<rewrite<<" max_abs_error="<<maximum<<" mean_abs_error="<<mean<<" extract="<<(maximum<1e-4f?"PASS":"FAIL")<<'\n';}std::cout<<"ncnn_rms_head_rewrite_runner="<<(pass?"PASS":"FAIL")<<'\n';return pass?0:1;}
