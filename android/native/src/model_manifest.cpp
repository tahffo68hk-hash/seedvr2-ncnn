#include "model_manifest.h"
#include <filesystem>
#include <fstream>
#include <regex>
#include <sstream>
namespace fs=std::filesystem;
static std::string fail(const char* e){return std::string("{\"valid\":false,\"error\":\"")+e+"\",\"silent_fallback\":false}";}
std::string validate_model_manifest(const std::string& root){
    fs::path base(root), path=base/"manifest.json"; if(!fs::exists(path)) return fail("manifest missing");
    std::ifstream f(path);std::stringstream b;b<<f.rdbuf();std::string s=b.str();
    if(s.empty()||s.front()!='{')return fail("invalid JSON");
    if(!std::regex_search(s,std::regex(R"("format_version"\s*:\s*1)")))return fail("format_version must be 1");
    std::regex entry(R"xx("path"\s*:\s*"([^"]+)"\s*,\s*"size"\s*:\s*([0-9]+)\s*,\s*"sha256"\s*:\s*"([^"]*)")xx");
    for(auto i=std::sregex_iterator(s.begin(),s.end(),entry);i!=std::sregex_iterator();++i){std::string rel=(*i)[1],hash=(*i)[3];uintmax_t size=std::stoull((*i)[2]);
        if(rel.find("..")!=std::string::npos)return fail("path traversal"); fs::path rp(rel);if(rp.is_absolute()||rel.find(':')!=std::string::npos||(!rel.empty()&&(rel[0]=='/'||rel[0]=='\\')))return fail("absolute path");
        if(!hash.empty()&&!std::regex_match(hash,std::regex("[0-9a-f]{64}")))return fail("sha256 must be 64 lowercase hex");fs::path target=base/rp;if(!fs::exists(target))return fail("model file missing");if(fs::file_size(target)!=size)return fail("size mismatch");}
    return R"({"valid":true,"format_version":1,"silent_fallback":false})";
}
