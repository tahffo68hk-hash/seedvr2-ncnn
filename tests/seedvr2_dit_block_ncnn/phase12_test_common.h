#pragma once
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>


// Shared Phase 12 test helpers: minimal JSON reader for the generated
// reference manifests plus bin loading / comparison utilities.
namespace phase12_test
{
struct JVal
{
    enum Kind { Null, Bool, Number, String, Array, Object };
    Kind kind = Null;
    bool b = false;
    double num = 0;
    std::string str;
    std::vector<JVal> arr;
    std::map<std::string, JVal> obj;
};

namespace json_detail
{
inline void skip_ws(const std::string& s, std::size_t& i)
{
    while (i < s.size() && std::isspace(static_cast<unsigned char>(s[i]))) ++i;
}
inline JVal parse_value(const std::string& s, std::size_t& i);
inline std::string parse_string(const std::string& s, std::size_t& i)
{
    ++i; // opening quote
    std::string out;
    while (i < s.size())
    {
        const char c = s[i++];
        if (c == '"') return out;
        if (c == '\\')
        {
            const char e = i < s.size() ? s[i++] : 0;
            switch (e)
            {
            case 'n': out += '\n'; break;
            case 't': out += '\t'; break;
            case 'r': out += '\r'; break;
            case 'b': out += '\b'; break;
            case 'f': out += '\f'; break;
            case 'u':
                if (i + 4 <= s.size()) { out += static_cast<char>(std::strtol(s.substr(i, 4).c_str(), nullptr, 16)); i += 4; }
                break;
            default: out += e; break;
            }
        }
        else out += c;
    }
    return out;
}
inline JVal parse_number(const std::string& s, std::size_t& i)
{
    const std::size_t begin = i;
    if (i < s.size() && (s[i] == '-' || s[i] == '+')) ++i;
    while (i < s.size() && (std::isdigit(static_cast<unsigned char>(s[i])) || s[i] == '.' || s[i] == 'e' || s[i] == 'E' || s[i] == '-' || s[i] == '+')) ++i;
    JVal v; v.kind = JVal::Number; v.num = std::strtod(s.substr(begin, i - begin).c_str(), nullptr); return v;
}
inline JVal parse_array(const std::string& s, std::size_t& i)
{
    ++i; // [
    JVal v; v.kind = JVal::Array;
    for (;;)
    {
        skip_ws(s, i);
        if (i < s.size() && s[i] == ']') { ++i; return v; }
        v.arr.push_back(parse_value(s, i));
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
}
inline JVal parse_object(const std::string& s, std::size_t& i)
{
    ++i; // {
    JVal v; v.kind = JVal::Object;
    for (;;)
    {
        skip_ws(s, i);
        if (i < s.size() && s[i] == '}') { ++i; return v; }
        skip_ws(s, i);
        const std::string key = parse_string(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ':') ++i;
        v.obj[key] = parse_value(s, i);
        skip_ws(s, i);
        if (i < s.size() && s[i] == ',') ++i;
    }
}
inline JVal parse_value(const std::string& s, std::size_t& i)
{
    skip_ws(s, i);
    if (i >= s.size()) return JVal{};
    const char c = s[i];
    if (c == '{') return parse_object(s, i);
    if (c == '[') return parse_array(s, i);
    if (c == '"') { JVal v; v.kind = JVal::String; v.str = parse_string(s, i); return v; }
    if (s.compare(i, 4, "true") == 0) { i += 4; JVal v; v.kind = JVal::Bool; v.b = true; return v; }
    if (s.compare(i, 5, "false") == 0) { i += 5; JVal v; v.kind = JVal::Bool; v.b = false; return v; }
    if (s.compare(i, 4, "null") == 0) { i += 4; return JVal{}; }
    return parse_number(s, i);
}
} // namespace json_detail

inline JVal parse_json(const std::string& text)
{
    std::size_t i = 0;
    return json_detail::parse_value(text, i);
}

inline bool read_text(const std::filesystem::path& path, std::string& out)
{
    std::ifstream file(path, std::ios::binary);
    if (!file) return false;
    std::stringstream ss;
    ss << file.rdbuf();
    out = ss.str();
    return true;
}

inline bool read_bin(const std::filesystem::path& path, std::vector<float>& out)
{
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) return false;
    const auto bytes = file.tellg();
    if (bytes % static_cast<std::streamoff>(sizeof(float)) != 0) return false;
    out.resize(static_cast<std::size_t>(bytes / sizeof(float)));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(out.data()), bytes);
    return bool(file);
}

inline std::vector<int> int_array(const JVal& v)
{
    std::vector<int> result;
    for (const auto& item : v.arr) result.push_back(static_cast<int>(item.num));
    return result;
}

inline std::vector<std::vector<int>> nested_int_array(const JVal& v)
{
    std::vector<std::vector<int>> result;
    for (const auto& item : v.arr) result.push_back(int_array(item));
    return result;
}

struct ErrorStats { float maximum = 0; double mean = 0; std::size_t nan = 0, inf = 0; };

inline ErrorStats compare(const std::vector<float>& actual, const std::vector<float>& reference)
{
    ErrorStats stats;
    if (actual.size() != reference.size()) { stats.maximum = INFINITY; return stats; }
    for (std::size_t i = 0; i < actual.size(); ++i)
    {
        const float d = std::fabs(actual[i] - reference[i]);
        stats.maximum = std::max(stats.maximum, d);
        stats.mean += d;
        stats.nan += std::isnan(actual[i]);
        stats.inf += std::isinf(actual[i]);
    }
    stats.mean = actual.empty() ? 0.0 : stats.mean / actual.size();
    return stats;
}

inline bool emit(const std::string& name, const ErrorStats& stats, float threshold)
{
    std::cout << "  " << name << " max_abs_error=" << stats.maximum
              << " mean_abs_error=" << stats.mean << " nan=" << stats.nan << " inf=" << stats.inf
              << " threshold=" << threshold << (stats.maximum < threshold && !stats.nan && !stats.inf ? " PASS" : " FAIL") << '\n';
    return stats.maximum < threshold && !stats.nan && !stats.inf;
}

inline bool emit_bool(const std::string& name, bool ok)
{
    std::cout << "  " << name << " " << (ok ? "PASS" : "FAIL") << '\n';
    return ok;
}

inline std::string join(const std::vector<int>& v)
{
    std::ostringstream os;
    for (std::size_t i = 0; i < v.size(); ++i) os << (i ? "," : "") << v[i];
    return os.str();
}


inline void append_json(std::string& out, const JVal& v)
{
    switch (v.kind)
    {
    case JVal::Null: out += "null"; break;
    case JVal::Bool: out += v.b ? "true" : "false"; break;
    case JVal::Number:
    {
        char buf[64];
        std::snprintf(buf, sizeof buf, "%.9g", v.num);
        out += buf;
        break;
    }
    case JVal::String:
    {
        out += '"';
        for (char c : v.str)
        {
            if (c == '"' || c == '\\') { out += '\\'; out += c; }
            else if (c == '\n') out += "\\n";
            else if (c == '\t') out += "\\t";
            else if (c == '\r') out += "\\r";
            else if (static_cast<unsigned char>(c) < 0x20)
            {
                char b[8];
                std::snprintf(b, sizeof b, "\\u%04x", static_cast<unsigned char>(c));
                out += b;
            }
            else out += c;
        }
        out += '"';
        break;
    }
    case JVal::Array:
    {
        out += '[';
        for (std::size_t i = 0; i < v.arr.size(); ++i)
        {
            if (i) out += ',';
            append_json(out, v.arr[i]);
        }
        out += ']';
        break;
    }
    case JVal::Object:
    {
        out += '{';
        bool first = true;
        for (const auto& kv : v.obj)
        {
            if (!first) out += ',';
            first = false;
            append_json(out, JVal{JVal::String, false, 0, kv.first, {}, {}});
            out += ':';
            append_json(out, kv.second);
        }
        out += '}';
        break;
    }
    }
}

inline std::string to_json(const JVal& v)
{
    std::string out;
    append_json(out, v);
    return out;
}
} // namespace phase12_test
