#pragma once
#include <string>

namespace seededge_progress {
void reset(const std::string& mode, const std::string& output_root);
void update(const std::string& phase, const std::string& stage, const std::string& message,
            int file_done = 0, int file_total = 0, long long bytes_done = 0, long long bytes_total = 0,
            int dit_block = 0, int decoder_stage = 0, int decoder_total = 0,
            int frame_index = 0, int output_frames = 0, const std::string& current_file = "");
void finish(const std::string& state, const std::string& error_code = "", const std::string& error_message = "");
void cancel();
bool cancelled();
std::string json();
}
