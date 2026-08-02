#include "hybrid_window_adapter.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
using namespace attention_dispatcher;namespace fs=std::filesystem; static bool read(const fs::path&p,std::vector<float>&v){std::ifstream f(p,std::ios::binary);f.seekg(0,std::ios::end);auto n=f.tellg()/4;f.seekg(0);v.resize((size_t)n);f.read((char*)v.data(),n*4);return !!f;}
int main(int argc,char**argv){assert(argc==2);fs::path r=fs::path(argv[1])/"hybrid_fidelity_ln_rope444_grid_1x2x2/reference";std::vector<float>x,g,w,u;assert(read(r/"input.bin",x)&&read(r/"windows.bin",g));HybridWindowContract c{1,4,8,8,32,4,4,4,1,2,2,0,0,0};std::string e;assert(pack_hybrid_windows(x,c,w,e)&&w.size()==g.size());float mx=0;double mean=0;size_t first=w.size();for(size_t i=0;i<w.size();++i){float d=std::fabs(w[i]-g[i]);if(d&&first==w.size())first=i;mx=std::max(mx,d);mean+=d;}assert(unpack_hybrid_windows(w,c,u,e)&&u==x);std::cout<<"hybrid_window_adapter max_abs_error="<<mx<<" mean_abs_error="<<mean/w.size()<<" first_difference="<<(first==w.size()?"none":std::to_string(first))<<"\n";return mx<1e-7?0:1;}
