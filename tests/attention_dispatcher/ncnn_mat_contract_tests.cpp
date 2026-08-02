#include "ncnn_mat_contract.h"
#include <cassert>
#include <iostream>
using namespace attention_dispatcher;

int main()
{
    std::string error;
    std::vector<float> rope(64 * 4); for (size_t i=0;i<rope.size();++i) rope[i]=static_cast<float>(i);
    ncnn::Mat rope_mat; assert(make_rope_mat(rope,64,4,rope_mat,error));
    assert(rope_mat.w==4 && rope_mat.h==64 && rope_mat.total()==256);
    for (size_t i=0;i<rope.size();++i) assert(static_cast<const float*>(rope_mat.data)[i]==rope[i]);
    ncnn::Mat failed; assert(!make_rope_mat(std::vector<float>(255),64,4,failed,error) && failed.empty());
    std::vector<float> mask(4 * 64 * 64); for (size_t i=0;i<mask.size();++i) mask[i]=static_cast<float>(i);
    ncnn::Mat mask_mat; assert(make_mask_mat(mask,4,64,mask_mat,error));
    assert(mask_mat.w==64 && mask_mat.h==64 && mask_mat.c==4 && mask_mat.total()==mask.size());
    for (size_t i=0;i<mask.size();++i) assert(static_cast<const float*>(mask_mat.data)[i]==mask[i]);
    assert(!make_mask_mat(std::vector<float>(mask.size()-1),4,64,failed,error) && failed.empty());
    assert(make_rope_mat(rope,64,4,rope_mat,error));
    // Phase 10 HD128 contract: rope_dim=128 -> Mat(w=64, h=64).
    std::vector<float> rope128(64 * 64); for (size_t i=0;i<rope128.size();++i) rope128[i]=static_cast<float>(i);
    ncnn::Mat rope128_mat; assert(make_rope_mat(rope128,64,64,rope128_mat,error));
    assert(rope128_mat.w==64 && rope128_mat.h==64 && rope128_mat.total()==4096);
    for (size_t i=0;i<rope128.size();++i) assert(static_cast<const float*>(rope128_mat.data)[i]==rope128[i]);
    assert(!make_rope_mat(std::vector<float>(4095),64,64,failed,error) && failed.empty());
    assert(!make_rope_mat(std::vector<float>(4097),64,64,failed,error) && failed.empty());
    std::cout << "ncnn_mat_contract_tests=PASS\n";
}
