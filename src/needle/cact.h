#pragma once
// Reader for Cactus .cact archives (Needle 3 format, tag 0x05E12A84).
//
// Layout (little-endian): 196-byte header of 49 fields, the shared Lloyd-Max codebooks
// (cb2[4] | cb3[8] | cb4[16] floats), a nameless 44-byte-per-record tensor directory, then
// 64-byte-aligned tensor blobs. CQ matrices are stored [out, in] in groups of 128 along the
// input axis: packed LSB-first indices, then fp16 per-group norms. A group reconstructs as
//   w = (codebook[idx] * norm) @ H     (H = normalised Walsh-Hadamard matrix, symmetric)
#include <cstdint>
#include <string>
#include <vector>

namespace needle {

struct CactHeader {
    uint32_t tag = 0, numTensors = 0, codebookLen = 0, kvWindow = 0, kvBits = 0;
    uint32_t vocab = 0, outVocab = 0, dModel = 0, numHeads = 0, numKvHeads = 0, numLayers = 0;
    uint32_t qkHeadDim = 0, vHeadDim = 0, maxSeqLen = 0, hadaN = 0, mhcLanes = 0;
    uint32_t slidingWindow = 0, globalMaskLo = 0, globalMaskHi = 0, qkvConvTaps = 0;
    uint32_t engramSlots = 0, engramSubDim = 0, numEngramTables = 0, engramConvTaps = 0;
    uint32_t engramConvDilation = 0, engramSeedHeads = 0, numEngramOrders = 0;
    uint32_t engramOrders[4] = {0, 0, 0, 0};
    uint32_t numEngramSites = 0;
    uint32_t engramSites[16] = {0};
    float ropeTheta = 0.f;
};

enum TensorDtype : uint8_t { DT_FP16 = 1, DT_FP32 = 2, DT_CQ = 3, DT_RAW = 4 };

struct TensorRecord {
    uint8_t dtype = 0, ndim = 0;
    uint32_t shape[4] = {0, 0, 0, 0};
    uint64_t offset = 0, nbytes = 0;
    uint32_t group = 0, bits = 0;
    size_t elements() const {
        size_t n = 1;
        for (int i = 0; i < ndim; i++) n *= shape[i];
        return ndim ? n : 0;
    }
};

// A CQ matrix kept in its packed form: LSB-first indices per row, plus per-group norms (as float).
struct CqMatrix {
    int out = 0, in = 0, group = 128, bits = 2;
    std::vector<uint8_t> packed;     // out * (in * bits / 8) bytes
    std::vector<float> norms;        // out * (in / group)
};

class Cact {
public:
    // Reads the whole archive into memory. Returns false and sets error() on failure.
    bool load(const std::string& path);
    const std::string& error() const { return error_; }

    const CactHeader& header() const { return hdr_; }
    size_t numTensors() const { return records_.size(); }
    const TensorRecord& record(size_t i) const { return records_[i]; }

    // Dequantised / converted float32 copy of tensor i (FP16, FP32 or CQ). Row-major.
    std::vector<float> floats(size_t i) const;
    // Packed copy of a CQ tensor (needs in % group == 0). False if tensor i is not such a matrix.
    bool cqMatrix(size_t i, CqMatrix& m) const;
    // The codebook for a CQ width (2, 3 or 4 bits): 4, 8 or 16 floats, or nullptr.
    const float* codebook(int bits) const;
    // Raw bytes of a RAW tensor (the tokenizer blob).
    std::string raw(size_t i) const;

    bool isGlobalLayer(int layer) const;
    // Drop the raw file image once the model has copied what it needs (header and records stay valid).
    void release() { std::vector<uint8_t>().swap(data_); }

private:
    std::vector<float> dequantCQ(const TensorRecord& r) const;

    std::string error_;
    std::vector<uint8_t> data_;
    CactHeader hdr_;
    std::vector<float> codebook_;
    std::vector<TensorRecord> records_;
};

// IEEE half -> float
float halfToFloat(uint16_t h);

} // namespace needle
