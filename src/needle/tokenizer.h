#pragma once
// SentencePiece BPE tokenizer loaded from the RAW attachment of a .cact archive.
// Mirrors RefTokenizer in needle/model/export.py (the reference encoder/decoder):
//   - spaces become U+2581 (the "meta space"); a dummy meta space is prefixed when add_dummy_prefix
//   - chat markers (USER_DEFINED pieces such as <|im_start|>) are atomic and split the text
//   - between markers, characters are merged greedily by the highest piece score
//   - leftovers fall back to byte pieces <0xNN>
#include <string>
#include <unordered_map>
#include <vector>

namespace needle {

class Tokenizer {
public:
    // Blob layout: u32 n_pieces, u32 pad/eos/bos/unk ids, u8 add_dummy_prefix, u8 byte_fallback,
    // u16 pad; then n_pieces records {f32 score, u8 type, u16 len, len bytes}.
    bool load(const std::string& blob);
    const std::string& error() const { return error_; }

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& ids) const;

    // The bytes a token adds to the output text. False for control / unknown / chat-marker tokens
    // (their text is not part of a JSON value) and for empty pieces.
    bool tokenBytes(int id, std::string& out) const;

    int size() const { return (int)pieces_.size(); }
    const std::string& piece(int id) const { return pieces_[id]; }
    int unkId() const { return unk_; }

private:
    std::vector<int> bpe(const std::string& segment) const;

    std::string error_;
    std::vector<std::string> pieces_;
    std::vector<float> scores_;
    std::vector<uint8_t> types_;
    std::unordered_map<std::string, int> p2id_;
    int byteId_[256];
    std::vector<std::string> markers_;     // USER_DEFINED pieces, longest first
    bool addDummy_ = false, byteFallback_ = false;
    int unk_ = 3;
};

} // namespace needle
