#pragma once

#include <layer.h>

#include <vector>

namespace seededge
{

// Exact FP32 implementation for the SeedVR2 VAE Conv3D contract.  It
// decomposes the temporal kernel into optimized ncnn CPU Conv2D operations.
class FastConvolution3D final : public ncnn::Layer
{
public:
    FastConvolution3D();

    int load_param(const ncnn::ParamDict& pd) override;
    int load_model(const ncnn::ModelBin& mb) override;
    int create_pipeline(const ncnn::Option& opt) override;
    int destroy_pipeline(const ncnn::Option& opt) override;
    int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                const ncnn::Option& opt) const override;

private:
    int num_output_ = 0;
    int kernel_w_ = 0;
    int kernel_h_ = 0;
    int kernel_d_ = 0;
    int dilation_w_ = 1;
    int dilation_h_ = 1;
    int dilation_d_ = 1;
    int stride_w_ = 1;
    int stride_h_ = 1;
    int stride_d_ = 1;
    int pad_left_ = 0;
    int pad_right_ = 0;
    int pad_top_ = 0;
    int pad_bottom_ = 0;
    int pad_front_ = 0;
    int pad_behind_ = 0;
    float pad_value_ = 0.f;
    int bias_term_ = 0;
    int weight_data_size_ = 0;
    int activation_type_ = 0;
    ncnn::Mat activation_params_;
    ncnn::Mat weight_data_;
    ncnn::Mat bias_data_;
    std::vector<ncnn::Layer*> conv2d_;
};

ncnn::Layer* fast_convolution3d_creator(void* userdata);
void fast_convolution3d_destroyer(ncnn::Layer* layer, void* userdata);

// Small deterministic reference comparison used by the Android RED/GREEN test.
std::string fast_convolution3d_test_json();
void reset_fast_convolution3d_profile();
std::vector<long long> fast_convolution3d_profile_ms();

} // namespace seededge
