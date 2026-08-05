#include "runtime_progress.h"
#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <mutex>
#include <sstream>

namespace fs = std::filesystem;
namespace {
std::mutex g_mutex; std::atomic<bool> g_cancel{false}; std::string g_json;
std::string esc(const std::string& s){std::string o; for(char c:s){if(c=='\\'||c=='"'){o+='\\';o+=c;}else if(c=='\n')o+="\\n";else if((unsigned char)c>=0x20)o+=c;}return o;}
long long now_ms(){return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();}
std::string root;
}
namespace seededge_progress {
void reset(const std::string& mode, const std::string& output_root){ {std::lock_guard<std::mutex> l(g_mutex); root=output_root; g_cancel=false; g_json="{\"schema_version\":1,\"run_id\":\""+std::to_string(now_ms())+"\",\"mode\":\""+esc(mode)+"\",\"state\":\"running\",\"phase\":\"checking_model\"}";} update("checking_model","","starting");}
void update(const std::string& phase,const std::string& stage,const std::string& message,int fd,int ft,long long bd,long long bt,int block,int ds,int dt,int frame,int out,const std::string& file){std::lock_guard<std::mutex> l(g_mutex); std::ostringstream o; o << "{\"schema_version\":1,\"state\":\"running\",\"phase\":\"" << esc(phase) << "\",\"stage_name\":\"" << esc(stage) << "\",\"message\":\"" << esc(message) << "\",\"current_file\":\"" << esc(file) << "\",\"model_files_done\":" << fd << ",\"model_files_total\":" << ft << ",\"model_bytes_done\":" << bd << ",\"model_bytes_total\":" << bt << ",\"frame_index\":" << frame << ",\"frame_count\":4,\"dit_block_index\":" << block << ",\"dit_block_total\":32,\"decoder_stage_index\":" << ds << ",\"decoder_stage_total\":" << dt << ",\"elapsed_ms\":0,\"stage_elapsed_ms\":0,\"last_stage_duration_ms\":null,\"estimated_remaining_ms\":null,\"rss_mb\":null,\"swap_mb\":null,\"battery_temperature_c\":null,\"thermal_status\":null,\"backend\":\"cpu_fp32\",\"finite\":null,\"cancellable\":true,\"output_frame_count\":" << out << ",\"output_video_path\":null,\"error_code\":null,\"error_message\":null}"; g_json=o.str(); if(!root.empty()){try{fs::create_directories(root);auto tmp=fs::path(root)/"runtime_progress.json.tmp", dst=fs::path(root)/"runtime_progress.json";std::ofstream f(tmp);f<<g_json;f.close();fs::rename(tmp,dst);}catch(...){}}}
void finish(const std::string& state,const std::string& code,const std::string& message){std::lock_guard<std::mutex> l(g_mutex); if(g_json.empty()) reset("unknown",root); auto p=g_json.find("\"state\":\"running\""); if(p!=std::string::npos)g_json.replace(p,24,"\"state\":\""+esc(state)+"\""); if(!code.empty()){g_json.replace(g_json.size()-1,1,",\"error_code\":\""+esc(code)+"\",\"error_message\":\""+esc(message)+"\"}");} if(!root.empty()){try{std::ofstream f(fs::path(root)/"runtime_progress.json");f<<g_json;}catch(...){}}}
void cancel(){g_cancel=true;}
bool cancelled(){return g_cancel.load();}
std::string json(){std::lock_guard<std::mutex> l(g_mutex);return g_json.empty()?"{\"schema_version\":1,\"state\":\"idle\"}":g_json;}
}
