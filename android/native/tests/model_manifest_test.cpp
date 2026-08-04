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
 write(r/"manifest.json",R"({"format_version":2,"files":[]})");expect("version",validate_model_manifest(r.string()),"format_version must be 1");
 write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"../x","size":0,"sha256":""}]})");expect("traversal",validate_model_manifest(r.string()),"path traversal");
 write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"C:\\x","size":0,"sha256":""}]})");expect("absolute",validate_model_manifest(r.string()),"absolute path");
 write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"x","size":0,"sha256":"GG"}]})");expect("hash",validate_model_manifest(r.string()),"sha256 must be 64 lowercase hex");
 write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"x","size":0,"sha256":""}]})");expect("file",validate_model_manifest(r.string()),"model file missing");
 write(r/"x","test");write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"x","size":3,"sha256":""}]})");expect("size",validate_model_manifest(r.string()),"size mismatch");
 write(r/"manifest.json",R"({"format_version":1,"files":[{"path":"x","size":4,"sha256":""}]})");expect("valid",validate_model_manifest(r.string()),R"("valid":true)");
 fs::remove_all(r);std::cout<<"manifest_tests passed="<<passed<<" failed="<<failed<<"\n";return failed?1:0;}
