#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <net.h>

bool read_float32_file(const std::string& path, std::vector<float>& values, std::size_t count);
bool write_float32_file(const std::string& path, const std::vector<float>& values);

ncnn::Mat bthwc_to_ncnn_mat(const std::vector<float>& bthwc, int t, int h, int w, int channels);
void ncnn_mat_to_bthwc(const ncnn::Mat& mat, std::vector<float>& bthwc, int t, int h, int w, int channels);

ncnn::Mat partition_fixed_windows(const ncnn::Mat& bthwc, int t, int h, int w, int channels, int window_t, int window_hw);
void reverse_fixed_windows(const ncnn::Mat& windows, ncnn::Mat& bthwc, int t, int h, int w, int channels, int window_t, int window_hw);
