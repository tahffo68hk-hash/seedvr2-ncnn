#include "model_manifest.h"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
namespace fs=std::filesystem;
static int passed=0,failed=0;
static void expect(const std::string& name,const std::string& got,const std::string& token){if(got.find(token)!=std::string::npos){passed++;}else{failed++;std::cerr<<name<<" got "<<got<<"\n";}}
static void write(const fs::path&p,const std::string&s){fs::create_directories(p.parent_path());std::ofstream(p)<<s;}
int main(){fs::path r=fs::temp_directory_path()/"seededge_manifest_test";fs::remove_all(r);fs::create_directories(r);
 expect("missing",validate_model_manifest(r.string()),"manifest missing");
 write(r/"manifest.json","bad");expect("json",validate_model_manifest(r.string()),"invalid JSON");
 write(r/"manifest.json",R"({"schema_version":2,"model_set":"phase18c_minimal_video_demo","files":[]})");expect("version",validate_model_manifest(r.string()),"schema_version must be 1");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[]})");expect("empty",validate_model_manifest(r.string()),"files must not be empty");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"../x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");expect("traversal",validate_model_manifest(r.string()),"path traversal");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"C:\\x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");expect("absolute",validate_model_manifest(r.string()),"absolute path");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":0,"sha256":"GG"}]})");expect("hash",validate_model_manifest(r.string()),"sha256 must be 64 lowercase hex");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":0,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");expect("file",validate_model_manifest(r.string()),"model file missing");
 write(r/"x","test");write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":3,"sha256":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}]})");expect("size",validate_model_manifest(r.string()),"size mismatch");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":4,"sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}]})");expect("digest",validate_model_manifest(r.string()),"sha256 mismatch");
 write(r/"manifest.json",R"({"schema_version":1,"model_set":"phase18c_minimal_video_demo","files":[{"relative_path":"x","size_bytes":4,"sha256":"9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"}]})");expect("valid",validate_model_manifest(r.string()),R"("valid":true)");
 fs::remove_all(r);std::cout<<"manifest_tests passed="<<passed<<" failed="<<failed<<"\n";return failed?1:0;}
