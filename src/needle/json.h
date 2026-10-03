#pragma once
// Minimal JSON value, parser and compact serializer for tool schemas.
// Object key order is preserved; numbers keep their original text. dump() matches Python's
//   json.dumps(v, separators=(",", ":"), ensure_ascii=False)
// for the inputs Needle sees (strings escape only quotes, backslashes and control characters).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace needle {
namespace json {

struct Value {
    enum Type { Null, Bool, Number, String, Array, Object } type = Null;
    bool b = false;
    std::string s;                                          // string content, or the number's text
    std::vector<Value> arr;
    std::vector<std::pair<std::string, Value>> obj;

    const Value* get(const std::string& key) const {
        for (const auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
    Value* get(const std::string& key) {
        for (auto& kv : obj) if (kv.first == key) return &kv.second;
        return nullptr;
    }
    void erase(const std::string& key) {
        for (size_t i = 0; i < obj.size(); i++)
            if (obj[i].first == key) { obj.erase(obj.begin() + i); return; }
    }
};

class Parser {
public:
    explicit Parser(const std::string& t) : t_(t) {}
    Value parse() {
        Value v = value();
        ws();
        if (i_ != t_.size()) fail("trailing characters");
        return v;
    }

private:
    [[noreturn]] void fail(const char* m) const {
        throw std::runtime_error(std::string("JSON error at byte ") + std::to_string(i_) + ": " + m);
    }
    void ws() { while (i_ < t_.size() && (t_[i_] == ' ' || t_[i_] == '\t' || t_[i_] == '\n' || t_[i_] == '\r')) i_++; }
    bool lit(const char* w) {
        size_t n = 0; while (w[n]) n++;
        if (t_.compare(i_, n, w) == 0) { i_ += n; return true; }
        return false;
    }
    static void utf8(std::string& out, uint32_t c) {
        if (c < 0x80) out += (char)c;
        else if (c < 0x800) { out += (char)(0xC0 | (c >> 6)); out += (char)(0x80 | (c & 0x3F)); }
        else if (c < 0x10000) { out += (char)(0xE0 | (c >> 12)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
        else { out += (char)(0xF0 | (c >> 18)); out += (char)(0x80 | ((c >> 12) & 0x3F)); out += (char)(0x80 | ((c >> 6) & 0x3F)); out += (char)(0x80 | (c & 0x3F)); }
    }
    uint32_t hex4() {
        if (i_ + 4 > t_.size()) fail("bad \\u escape");
        uint32_t v = (uint32_t)std::strtoul(t_.substr(i_, 4).c_str(), nullptr, 16);
        i_ += 4;
        return v;
    }
    std::string str() {
        if (t_[i_] != '"') fail("expected string");
        i_++;
        std::string out;
        while (i_ < t_.size() && t_[i_] != '"') {
            char c = t_[i_++];
            if (c != '\\') { out += c; continue; }
            if (i_ >= t_.size()) fail("bad escape");
            char e = t_[i_++];
            switch (e) {
                case '"': out += '"'; break;   case '\\': out += '\\'; break;
                case '/': out += '/'; break;   case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;  case 'n': out += '\n'; break;
                case 'r': out += '\r'; break;  case 't': out += '\t'; break;
                case 'u': {
                    uint32_t cp = hex4();
                    if (cp >= 0xD800 && cp < 0xDC00 && t_.compare(i_, 2, "\\u") == 0) {
                        i_ += 2;
                        uint32_t lo = hex4();
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                    }
                    utf8(out, cp);
                    break;
                }
                default: fail("bad escape");
            }
        }
        if (i_ >= t_.size()) fail("unterminated string");
        i_++;
        return out;
    }
    Value value() {
        ws();
        if (i_ >= t_.size()) fail("unexpected end");
        Value v;
        const char c = t_[i_];
        if (c == '{') {
            v.type = Value::Object; i_++; ws();
            if (t_[i_] == '}') { i_++; return v; }
            while (true) {
                ws();
                std::string k = str();
                ws();
                if (t_[i_++] != ':') fail("expected ':'");
                v.obj.emplace_back(std::move(k), value());
                ws();
                if (t_[i_] == ',') { i_++; continue; }
                if (t_[i_] == '}') { i_++; break; }
                fail("expected ',' or '}'");
            }
        } else if (c == '[') {
            v.type = Value::Array; i_++; ws();
            if (t_[i_] == ']') { i_++; return v; }
            while (true) {
                v.arr.push_back(value());
                ws();
                if (t_[i_] == ',') { i_++; continue; }
                if (t_[i_] == ']') { i_++; break; }
                fail("expected ',' or ']'");
            }
        } else if (c == '"') {
            v.type = Value::String; v.s = str();
        } else if (lit("true")) { v.type = Value::Bool; v.b = true; }
        else if (lit("false")) { v.type = Value::Bool; v.b = false; }
        else if (lit("null")) { v.type = Value::Null; }
        else {
            size_t st = i_;
            while (i_ < t_.size() && (std::string("+-0123456789.eE").find(t_[i_]) != std::string::npos)) i_++;
            if (st == i_) fail("unexpected character");
            v.type = Value::Number; v.s = t_.substr(st, i_ - st);
        }
        return v;
    }
    const std::string& t_;
    size_t i_ = 0;
};

inline Value parse(const std::string& text) { return Parser(text).parse(); }

inline void dumpString(std::string& out, const std::string& s) {
    out += '"';
    for (unsigned char c : s) {
        switch (c) {
            case '"': out += "\\\""; break;  case '\\': out += "\\\\"; break;
            case '\n': out += "\\n"; break;  case '\r': out += "\\r"; break;
            case '\t': out += "\\t"; break;  case '\b': out += "\\b"; break;
            case '\f': out += "\\f"; break;
            default:
                if (c < 0x20) { char b[8]; std::snprintf(b, sizeof(b), "\\u%04x", c); out += b; }
                else out += (char)c;
        }
    }
    out += '"';
}

inline void dump(std::string& out, const Value& v) {
    switch (v.type) {
        case Value::Null: out += "null"; break;
        case Value::Bool: out += v.b ? "true" : "false"; break;
        case Value::Number: out += v.s; break;
        case Value::String: dumpString(out, v.s); break;
        case Value::Array:
            out += '[';
            for (size_t i = 0; i < v.arr.size(); i++) { if (i) out += ','; dump(out, v.arr[i]); }
            out += ']';
            break;
        case Value::Object:
            out += '{';
            for (size_t i = 0; i < v.obj.size(); i++) {
                if (i) out += ',';
                dumpString(out, v.obj[i].first);
                out += ':';
                dump(out, v.obj[i].second);
            }
            out += '}';
            break;
    }
}

inline std::string dump(const Value& v) { std::string s; dump(s, v); return s; }

} // namespace json
} // namespace needle
