#include "fast_convolution3d.h"

#include <cpu.h>
#include <modelbin.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <iomanip>
#include <sstream>
#include <mutex>

namespace seededge
{
namespace
{
constexpr int kUnsupportedContract = -233;
std::mutex g_profile_mutex;
std::vector<long long> g_profile_ms;

void configure_fp32(ncnn::Option& opt, int threads)
{
    opt.num_threads = std::max(1, threads);
    opt.use_vulkan_compute = false;
    opt.use_packing_layout = true;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.use_bf16_storage = false;
}
} // namespace

FastConvolution3D::FastConvolution3D()
{
    one_blob_only = true;
    support_inplace = false;
    support_packing = false;
}

int FastConvolution3D::load_param(const ncnn::ParamDict& pd)
{
    num_output_ = pd.get(0, 0);
    kernel_w_ = pd.get(1, 0);
    kernel_h_ = pd.get(11, kernel_w_);
    kernel_d_ = pd.get(21, kernel_w_);
    dilation_w_ = pd.get(2, 1);
    dilation_h_ = pd.get(12, dilation_w_);
    dilation_d_ = pd.get(22, dilation_w_);
    stride_w_ = pd.get(3, 1);
    stride_h_ = pd.get(13, stride_w_);
    stride_d_ = pd.get(23, stride_w_);
    pad_left_ = pd.get(4, 0);
    pad_right_ = pd.get(15, pad_left_);
    pad_top_ = pd.get(14, pad_left_);
    pad_bottom_ = pd.get(16, pad_top_);
    pad_front_ = pd.get(24, pad_left_);
    pad_behind_ = pd.get(17, pad_front_);
    pad_value_ = pd.get(18, 0.f);
    bias_term_ = pd.get(5, 0);
    weight_data_size_ = pd.get(6, 0);
    activation_type_ = pd.get(9, 0);
    activation_params_ = pd.get(10, ncnn::Mat());

    // Phase 18C decoder models use explicit temporal padding before Conv3D,
    // zero pad values, and no fused activation. Reject everything else.
    if (num_output_ <= 0 || kernel_w_ <= 0 || kernel_h_ <= 0 || kernel_d_ <= 0 ||
        pad_left_ < 0 || pad_right_ < 0 || pad_top_ < 0 || pad_bottom_ < 0 ||
        pad_front_ < 0 || pad_behind_ < 0 || pad_value_ != 0.f ||
        activation_type_ != 0)
        return kUnsupportedContract;
    return 0;
}

int FastConvolution3D::load_model(const ncnn::ModelBin& mb)
{
    weight_data_ = mb.load(weight_data_size_, 0);
    if (weight_data_.empty()) return -100;
    if (bias_term_)
    {
        bias_data_ = mb.load(num_output_, 1);
        if (bias_data_.empty()) return -100;
    }
    return 0;
}

int FastConvolution3D::create_pipeline(const ncnn::Option& opt)
{
    const int spatial_kernel = kernel_w_ * kernel_h_;
    const int full_kernel = spatial_kernel * kernel_d_;
    if (full_kernel <= 0 || weight_data_size_ % (num_output_ * full_kernel) != 0)
        return kUnsupportedContract;
    const int channels = weight_data_size_ / (num_output_ * full_kernel);
    if (channels <= 0) return kUnsupportedContract;

    destroy_pipeline(opt);
    conv2d_.resize(kernel_d_, nullptr);
    const float* source = static_cast<const float*>(weight_data_.data);
    for (int kz = 0; kz < kernel_d_; ++kz)
    {
        ncnn::Mat plane_weights(spatial_kernel * channels * num_output_);
        float* destination = static_cast<float*>(plane_weights.data);
        for (int out = 0; out < num_output_; ++out)
            for (int in = 0; in < channels; ++in)
                for (int k = 0; k < spatial_kernel; ++k)
                    destination[(out * channels + in) * spatial_kernel + k] =
                        source[((out * channels + in) * kernel_d_ + kz) * spatial_kernel + k];

        ncnn::Layer* convolution = ncnn::create_layer_cpu("Convolution");
        if (!convolution) return -100;
        ncnn::ParamDict pd;
        pd.set(0, num_output_);
        pd.set(1, kernel_w_);
        pd.set(11, kernel_h_);
        pd.set(2, dilation_w_);
        pd.set(12, dilation_h_);
        pd.set(3, stride_w_);
        pd.set(13, stride_h_);
        pd.set(4, pad_left_);
        pd.set(15, pad_right_);
        pd.set(14, pad_top_);
        pd.set(16, pad_bottom_);
        pd.set(5, bias_term_ && kz == 0 ? 1 : 0);
        pd.set(6, static_cast<int>(plane_weights.total()));
        pd.set(9, 0);
        int rc = convolution->load_param(pd);
        ncnn::Mat weights[2];
        weights[0] = plane_weights;
        if (bias_term_ && kz == 0) weights[1] = bias_data_;
        if (!rc) rc = convolution->load_model(ncnn::ModelBinFromMatArray(weights));
        if (!rc) rc = convolution->create_pipeline(opt);
        if (rc)
        {
            delete convolution;
            return rc;
        }
        conv2d_[kz] = convolution;
    }
    return 0;
}

int FastConvolution3D::destroy_pipeline(const ncnn::Option& opt)
{
    for (ncnn::Layer* convolution : conv2d_)
    {
        if (!convolution) continue;
        convolution->destroy_pipeline(opt);
        delete convolution;
    }
    conv2d_.clear();
    return 0;
}

int FastConvolution3D::forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob,
                               const ncnn::Option& opt) const
{
    const auto profile_started = std::chrono::steady_clock::now();
    if (bottom_blob.dims != 4 || bottom_blob.elempack != 1 ||
        static_cast<int>(conv2d_.size()) != kernel_d_)
        return kUnsupportedContract;

    const int kernel_extent_d = dilation_d_ * (kernel_d_ - 1) + 1;
    const int padded_d = bottom_blob.d + pad_front_ + pad_behind_;
    const int outd = (padded_d - kernel_extent_d) / stride_d_ + 1;
    if (outd <= 0) return kUnsupportedContract;

    ncnn::Option conv_opt = opt;
    ncnn::Mat first_slice(bottom_blob.w, bottom_blob.h, bottom_blob.c,
                          static_cast<size_t>(4u), 1, opt.workspace_allocator);
    if (first_slice.empty()) return -100;
    for (int q = 0; q < bottom_blob.c; ++q)
        std::memcpy(first_slice.channel(q), bottom_blob.channel(q).depth(0),
                    static_cast<size_t>(bottom_blob.w) * bottom_blob.h * sizeof(float));
    ncnn::Mat shape_probe;
    int rc = conv2d_[0]->forward(first_slice, shape_probe, conv_opt);
    if (rc) return rc;
    ncnn::Mat shape_probe_unpacked;
    ncnn::convert_packing(shape_probe, shape_probe_unpacked, 1, conv_opt);
    if (shape_probe_unpacked.empty()) return -100;
    top_blob.create(shape_probe_unpacked.w, shape_probe_unpacked.h, outd, num_output_,
                    static_cast<size_t>(4u), 1, opt.blob_allocator);
    if (top_blob.empty()) return -100;

    for (int oz = 0; oz < outd; ++oz)
    {
        bool initialized = false;
        for (int kz = 0; kz < kernel_d_; ++kz)
        {
            const int iz = oz * stride_d_ + kz * dilation_d_ - pad_front_;
            if (iz < 0 || iz >= bottom_blob.d) continue;
            ncnn::Mat slice(bottom_blob.w, bottom_blob.h, bottom_blob.c,
                            static_cast<size_t>(4u), 1, opt.workspace_allocator);
            if (slice.empty()) return -100;
            for (int q = 0; q < bottom_blob.c; ++q)
                std::memcpy(slice.channel(q), bottom_blob.channel(q).depth(iz),
                            static_cast<size_t>(bottom_blob.w) * bottom_blob.h * sizeof(float));
            ncnn::Mat partial, unpacked;
            rc = conv2d_[kz]->forward(slice, partial, conv_opt);
            if (rc) return rc;
            ncnn::convert_packing(partial, unpacked, 1, conv_opt);
            if (unpacked.empty()) return -100;
            const size_t plane = static_cast<size_t>(unpacked.w) * unpacked.h;
            for (int out = 0; out < num_output_; ++out)
            {
                float* destination = top_blob.channel(out).depth(oz);
                const float* source = unpacked.channel(out);
                if (!initialized)
                    std::memcpy(destination, source, plane * sizeof(float));
                else
                    for (size_t i = 0; i < plane; ++i) destination[i] += source[i];
            }
            initialized = true;
        }
        if (!initialized) return kUnsupportedContract;
    }
    const long long elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - profile_started).count();
    {
        std::lock_guard<std::mutex> lock(g_profile_mutex);
        g_profile_ms.push_back(elapsed);
    }
    return 0;
}

ncnn::Layer* fast_convolution3d_creator(void*) { return new FastConvolution3D; }
void fast_convolution3d_destroyer(ncnn::Layer* layer, void*) { delete layer; }

void reset_fast_convolution3d_profile()
{
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    g_profile_ms.clear();
}

std::vector<long long> fast_convolution3d_profile_ms()
{
    std::lock_guard<std::mutex> lock(g_profile_mutex);
    return g_profile_ms;
}

std::string fast_convolution3d_test_json()
{
    ncnn::ParamDict pd;
    pd.set(0, 3); pd.set(1, 3); pd.set(11, 3); pd.set(21, 3);
    pd.set(2, 1); pd.set(12, 1); pd.set(22, 1);
    pd.set(3, 1); pd.set(13, 1); pd.set(23, 1);
    pd.set(4, 1); pd.set(15, 1); pd.set(14, 1); pd.set(16, 1);
    pd.set(24, 0); pd.set(17, 0); pd.set(5, 1); pd.set(6, 3 * 2 * 27);
    ncnn::Mat input(6, 5, 4, 2), weights(3 * 2 * 27), bias(3);
    for (size_t i = 0; i < input.total(); ++i) static_cast<float*>(input.data)[i] = std::sin(i * .17f) * .25f;
    for (size_t i = 0; i < weights.total(); ++i) static_cast<float*>(weights.data)[i] = std::cos(i * .11f) * .03f;
    for (int i = 0; i < 3; ++i) static_cast<float*>(bias.data)[i] = (i - 1) * .02f;
    ncnn::Mat model[2] = {weights, bias};
    ncnn::Option opt;
    configure_fp32(opt, std::max(1, ncnn::get_big_cpu_count()));
    ncnn::Layer* reference = ncnn::create_layer_cpu("Convolution3D");
    FastConvolution3D fast;
    int reference_rc = reference ? reference->load_param(pd) : -100;
    int fast_rc = fast.load_param(pd);
    if (!reference_rc) reference_rc = reference->load_model(ncnn::ModelBinFromMatArray(model));
    if (!fast_rc) fast_rc = fast.load_model(ncnn::ModelBinFromMatArray(model));
    if (!reference_rc) reference_rc = reference->create_pipeline(opt);
    if (!fast_rc) fast_rc = fast.create_pipeline(opt);
    ncnn::Mat expected, actual;
    if (!reference_rc) reference_rc = reference->forward(input, expected, opt);
    if (!fast_rc) fast_rc = fast.forward(input, actual, opt);
    bool finite = reference_rc == 0 && fast_rc == 0 && expected.total() == actual.total();
    float max_abs_error = 0.f;
    if (finite)
    {
        const float* a = static_cast<const float*>(expected.data);
        const float* b = static_cast<const float*>(actual.data);
        for (size_t i = 0; i < expected.total(); ++i)
        {
            finite = finite && std::isfinite(a[i]) && std::isfinite(b[i]);
            max_abs_error = std::max(max_abs_error, std::fabs(a[i] - b[i]));
        }
    }
    fast.destroy_pipeline(opt);
    if (reference)
    {
        reference->destroy_pipeline(opt);
        delete reference;
    }
    const bool pass = finite && max_abs_error <= 1e-4f;
    std::ostringstream json;
    json << std::setprecision(9)
         << "{\"reference_rc\":" << reference_rc << ",\"fast_rc\":" << fast_rc
         << ",\"finite\":" << (finite ? "true" : "false")
         << ",\"max_abs_error\":" << max_abs_error
         << ",\"overall_pass\":" << (pass ? "true" : "false") << "}";
    return json.str();
}

} // namespace seededge
