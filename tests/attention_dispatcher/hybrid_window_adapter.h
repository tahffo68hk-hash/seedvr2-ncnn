#pragma once
#include <string>
#include <vector>
namespace attention_dispatcher { struct HybridWindowContract {int batch,time,height,width,channels,window_t,window_h,window_w,grid_t,grid_h,grid_w,pad_t,pad_h,pad_w;}; bool pack_hybrid_windows(const std::vector<float>&,const HybridWindowContract&,std::vector<float>&,std::string&); bool unpack_hybrid_windows(const std::vector<float>&,const HybridWindowContract&,std::vector<float>&,std::string&); }
