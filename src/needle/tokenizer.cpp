#include "tokenizer.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>

namespace needle {

static const char* kMetaSpace = "\xE2\x96\x81";   // U+2581
enum { TK_NORMAL = 0, TK_UNKNOWN = 1, TK_CONTROL = 2, TK_USER_DEFINED = 3, TK_BYTE = 4 };

bool Tokenizer::load(const std::string& blob) {
    const size_t hdr = 4 * 5 + 1 + 1 + 2;                    // <IIIIIBBH
    if (blob.size() < hdr) { error_ = "tokenizer blob too small"; return false; }
    const uint8_t* p = (const uint8_t*)blob.data();
    uint32_t n, pad, eos, bos, unk;
    std::memcpy(&n, p, 4); std::memcpy(&pad, p + 4, 4); std::memcpy(&eos, p + 8, 4);
    std::memcpy(&bos, p + 12, 4); std::memcpy(&unk, p + 16, 4);
    addDummy_ = p[20] != 0;
    byteFallback_ = p[21] != 0;
    unk_ = (int)unk;

    pieces_.resize(n); scores_.resize(n); types_.resize(n);
    size_t off = hdr;
    for (uint32_t i = 0; i < n; i++) {
        if (off + 7 > blob.size()) { error_ = "truncated tokenizer record"; return false; }
        float score; std::memcpy(&score, p + off, 4);
        const uint8_t type = p[off + 4];
        uint16_t len; std::memcpy(&len, p + off + 5, 2);
        off += 7;
        if (off + len > blob.size()) { error_ = "truncated tokenizer piece"; return false; }
        pieces_[i].assign((const char*)p + off, len);
        scores_[i] = score; types_[i] = type;
        off += len;
    }

    p2id_.clear();
    for (uint32_t i = 0; i < n; i++) p2id_[pieces_[i]] = (int)i;       // later duplicates win, as in Python
    for (int b = 0; b < 256; b++) byteId_[b] = -1;
    for (uint32_t i = 0; i < n; i++) {
        if (types_[i] == TK_BYTE && pieces_[i].size() >= 5) {          // "<0xNN>"
            const int b = (int)std::strtol(pieces_[i].substr(3, 2).c_str(), nullptr, 16);
            byteId_[b & 255] = (int)i;
        }
    }
    markers_.clear();
    for (uint32_t i = 0; i < n; i++) if (types_[i] == TK_USER_DEFINED) markers_.push_back(pieces_[i]);
    std::stable_sort(markers_.begin(), markers_.end(),
                     [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
    return true;
}

// Split UTF-8 text into code points (each as its own string).
static std::vector<std::string> codepoints(const std::string& s) {
    std::vector<std::string> out;
    for (size_t i = 0; i < s.size();) {
        const uint8_t c = (uint8_t)s[i];
        size_t len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
        if (i + len > s.size()) len = 1;
        out.emplace_back(s.substr(i, len));
        i += len;
    }
    return out;
}

std::vector<int> Tokenizer::bpe(const std::string& segment) const {
    std::vector<std::string> syms = codepoints(segment);
    while (syms.size() > 1) {
        float bestScore = 0.f;
        int bestJ = -1;
        for (size_t j = 0; j + 1 < syms.size(); j++) {
            auto it = p2id_.find(syms[j] + syms[j + 1]);
            if (it != p2id_.end() && (bestJ < 0 || scores_[it->second] > bestScore)) {
                bestScore = scores_[it->second];
                bestJ = (int)j;
            }
        }
        if (bestJ < 0) break;
        syms[bestJ] += syms[bestJ + 1];
        syms.erase(syms.begin() + bestJ + 1);
    }
    std::vector<int> ids;
    for (const std::string& s : syms) {
        auto it = p2id_.find(s);
        if (it != p2id_.end()) ids.push_back(it->second);
        else if (byteFallback_) {
            for (unsigned char b : s) ids.push_back(byteId_[b]);
        } else ids.push_back(unk_);
    }
    return ids;
}

std::vector<int> Tokenizer::encode(const std::string& text) const {
    std::vector<int> ids;
    if (text.empty()) return ids;
    std::string esc;
    esc.reserve(text.size() + 8);
    if (addDummy_) esc += kMetaSpace;
    for (char c : text) {
        if (c == ' ') esc += kMetaSpace;
        else esc += c;
    }
    std::string buf;
    size_t i = 0;
    while (i < esc.size()) {
        const std::string* marker = nullptr;
        for (const std::string& m : markers_) {
            if (esc.compare(i, m.size(), m) == 0) { marker = &m; break; }
        }
        if (marker) {
            if (!buf.empty()) { auto b = bpe(buf); ids.insert(ids.end(), b.begin(), b.end()); buf.clear(); }
            ids.push_back(p2id_.at(*marker));
            i += marker->size();
        } else {
            buf += esc[i++];
        }
    }
    if (!buf.empty()) { auto b = bpe(buf); ids.insert(ids.end(), b.begin(), b.end()); }
    return ids;
}

bool Tokenizer::tokenBytes(int id, std::string& out) const {
    out.clear();
    if (id < 0 || id >= (int)pieces_.size()) return false;
    const uint8_t t = types_[id];
    if (t == TK_BYTE) {
        out.push_back((char)std::strtol(pieces_[id].substr(3, 2).c_str(), nullptr, 16));
        return true;
    }
    if (t != TK_NORMAL) return false;
    const std::string& p = pieces_[id];
    for (size_t i = 0; i < p.size();) {
        if (p.compare(i, 3, kMetaSpace) == 0) { out += ' '; i += 3; }
        else out += p[i++];
    }
    return !out.empty();
}

std::string Tokenizer::decode(const std::vector<int>& ids) const {
    std::string bytes;
    for (int id : ids) {
        if (id < 0 || id >= (int)pieces_.size()) continue;
        const uint8_t t = types_[id];
        if (t == TK_BYTE) bytes.push_back((char)std::strtol(pieces_[id].substr(3, 2).c_str(), nullptr, 16));
        else if (t == TK_CONTROL || t == TK_UNKNOWN) continue;
        else bytes += pieces_[id];
    }
    std::string text;
    for (size_t i = 0; i < bytes.size();) {
        if (bytes.compare(i, 3, kMetaSpace) == 0) { text += ' '; i += 3; }
        else text += bytes[i++];
    }
    if (addDummy_ && !text.empty() && text[0] == ' ') text.erase(0, 1);
    return text;
}

} // namespace needle
