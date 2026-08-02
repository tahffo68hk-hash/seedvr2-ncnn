#pragma once

namespace attention_dispatcher
{
struct Shape
{
    int batch;
    int time;
    int height;
    int width;
    int channels;
};

struct Metadata
{
    Shape input;
    int window_t;
    int window_h;
    int window_w;
    int grid_t;
    int grid_h;
    int grid_w;
    int pad_t;
    int pad_h;
    int pad_w;
    int num_windows;
    int tokens_per_window;
};
}
