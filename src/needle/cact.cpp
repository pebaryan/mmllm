#include "cact.h"
#include <cmath>
#include <cstdio>
#include <cstring>

namespace needle {

static const uint32_t kTag = 0x05E12A84;
static const size_t kHeaderBytes = 49 * 4;   // 48 u32 fields + rope_theta (f32)
static const size_t kRecordBytes = 44;

float halfToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t man = h & 0x3FF;
    uint32_t bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;                                   // +-0
        } else {                                           // subnormal: normalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) { man <<= 1; exp--; }
            man &= 0x3FF;
            bits = sign | (exp << 23) | (man << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7F800000u | (man << 13);           // inf / nan
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

static inline uint32_t rd32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
static inline uint64_t rd64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

bool Cact::load(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) { error_ = "cannot open " + path; return false; }
    std::fseek(f, 0, SEEK_END);
    const long size = std::ftell(f);
    std::fseek(f, 0, SEEK_SET);
    data_.resize((size_t)size);
    if (std::fread(data_.data(), 1, data_.size(), f) != data_.size()) {
        std::fclose(f);
        error_ = "short read of " + path;
        return false;
    }
    std::fclose(f);

    if (data_.size() < kHeaderBytes) { error_ = "file too small for a .cact header"; return false; }
    const uint8_t* p = data_.data();
    uint32_t w[49];
    for (int i = 0; i < 49; i++) w[i] = rd32(p + 4 * i);

    CactHeader& h = hdr_;
    h.tag = w[0];
    if (h.tag != kTag) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "not a Needle 3 archive (tag 0x%08X, expected 0x%08X)", h.tag, kTag);
        error_ = buf;
        return false;
    }
    h.numTensors = w[1]; h.codebookLen = w[2]; h.kvWindow = w[3]; h.kvBits = w[4];
    h.vocab = w[5]; h.outVocab = w[6]; h.dModel = w[7]; h.numHeads = w[8]; h.numKvHeads = w[9];
    h.numLayers = w[10]; h.qkHeadDim = w[11]; h.vHeadDim = w[12]; h.maxSeqLen = w[13];
    h.hadaN = w[14]; h.mhcLanes = w[15]; h.slidingWindow = w[16];
    h.globalMaskLo = w[17]; h.globalMaskHi = w[18]; h.qkvConvTaps = w[19];
    h.engramSlots = w[20]; h.engramSubDim = w[21]; h.numEngramTables = w[22];
    h.engramConvTaps = w[23]; h.engramConvDilation = w[24]; h.engramSeedHeads = w[25];
    h.numEngramOrders = w[26];
    for (int i = 0; i < 4; i++) h.engramOrders[i] = w[27 + i];
    h.numEngramSites = w[31];
    for (int i = 0; i < 16; i++) h.engramSites[i] = w[32 + i];
    std::memcpy(&h.ropeTheta, &w[48], 4);

    size_t off = kHeaderBytes;
    if (off + (size_t)h.codebookLen * 4 > data_.size()) { error_ = "truncated codebook"; return false; }
    codebook_.resize(h.codebookLen);
    std::memcpy(codebook_.data(), p + off, (size_t)h.codebookLen * 4);
    off += (size_t)h.codebookLen * 4;

    if (off + (size_t)h.numTensors * kRecordBytes > data_.size()) { error_ = "truncated directory"; return false; }
    records_.resize(h.numTensors);
    for (uint32_t i = 0; i < h.numTensors; i++, off += kRecordBytes) {
        TensorRecord& r = records_[i];
        r.dtype = p[off];
        r.ndim = p[off + 1];
        for (int k = 0; k < 4; k++) r.shape[k] = rd32(p + off + 4 + 4 * k);
        r.offset = rd64(p + off + 20);
        r.nbytes = rd64(p + off + 28);
        r.group = rd32(p + off + 36);
        r.bits = rd32(p + off + 40);
        if (r.offset + r.nbytes > data_.size()) { error_ = "tensor extends past end of file"; return false; }
    }
    return true;
}

bool Cact::isGlobalLayer(int layer) const {
    if (layer < 32) return (hdr_.globalMaskLo >> layer) & 1u;
    return (hdr_.globalMaskHi >> (layer - 32)) & 1u;
}

std::string Cact::raw(size_t i) const {
    const TensorRecord& r = records_[i];
    return std::string((const char*)data_.data() + r.offset, (size_t)r.nbytes);
}

// In-place normalised fast Walsh-Hadamard transform on n (power of two) values.
static void fwht(float* v, int n) {
    for (int len = 1; len < n; len <<= 1)
        for (int i = 0; i < n; i += len << 1)
            for (int j = i; j < i + len; j++) {
                const float a = v[j], b = v[j + len];
                v[j] = a + b;
                v[j + len] = a - b;
            }
    const float s = 1.0f / std::sqrt((float)n);
    for (int i = 0; i < n; i++) v[i] *= s;
}

std::vector<float> Cact::dequantCQ(const TensorRecord& r) const {
    const size_t out = r.shape[0], in = r.shape[1];
    const int group = (int)r.group, bits = (int)r.bits;
    const size_t inPad = (in + group - 1) / group * group;
    const size_t rowBytes = inPad * bits / 8;
    const size_t nGroups = inPad / group;

    // codebook for this width: cb2[4] | cb3[8] | cb4[16] concatenated
    const float* cb;
    if (bits == 2) cb = codebook_.data();
    else if (bits == 3) cb = codebook_.data() + 4;
    else if (bits == 4) cb = codebook_.data() + 12;
    else return {};  // ternary/binary not used by shipped archives

    const uint8_t* packed = data_.data() + r.offset;
    const uint8_t* norms = packed + out * rowBytes;

    std::vector<float> w(out * in);
    std::vector<float> grp(group);
    for (size_t o = 0; o < out; o++) {
        const uint8_t* row = packed + o * rowBytes;
        for (size_t g = 0; g < nGroups; g++) {
            uint16_t nh;
            std::memcpy(&nh, norms + (o * nGroups + g) * 2, 2);
            const float norm = halfToFloat(nh);
            for (int j = 0; j < group; j++) {
                const size_t k = g * group + j;            // index within the (padded) row
                uint32_t idx;
                if (bits == 2)      idx = (row[k >> 2] >> ((k & 3) * 2)) & 3;
                else if (bits == 4) idx = (row[k >> 1] >> ((k & 1) * 4)) & 15;
                else {                                      // 3 bits: continuous LSB-first stream
                    const size_t bp = k * 3;
                    uint32_t win = row[bp >> 3];
                    if ((bp >> 3) + 1 < rowBytes) win |= (uint32_t)row[(bp >> 3) + 1] << 8;
                    idx = (win >> (bp & 7)) & 7;
                }
                grp[j] = cb[idx] * norm;
            }
            fwht(grp.data(), group);
            for (int j = 0; j < group; j++) {
                const size_t col = g * group + j;
                if (col < in) w[o * in + col] = grp[j];
            }
        }
    }
    return w;
}

const float* Cact::codebook(int bits) const {
    if (bits == 2) return codebook_.data();
    if (bits == 3) return codebook_.data() + 4;
    if (bits == 4) return codebook_.data() + 12;
    return nullptr;
}

bool Cact::cqMatrix(size_t i, CqMatrix& m) const {
    const TensorRecord& r = records_[i];
    if (r.dtype != DT_CQ || r.ndim != 2 || r.group == 0 || (r.shape[1] % r.group) != 0) return false;
    if (r.bits != 2 && r.bits != 3 && r.bits != 4) return false;
    m.out = (int)r.shape[0]; m.in = (int)r.shape[1]; m.group = (int)r.group; m.bits = (int)r.bits;
    const size_t rowBytes = (size_t)m.in * m.bits / 8, nGroups = (size_t)m.in / m.group;
    const uint8_t* p = data_.data() + r.offset;
    m.packed.assign(p, p + (size_t)m.out * rowBytes);
    const uint8_t* nb = p + (size_t)m.out * rowBytes;
    m.norms.resize((size_t)m.out * nGroups);
    for (size_t k = 0; k < m.norms.size(); k++) {
        uint16_t h;
        std::memcpy(&h, nb + 2 * k, 2);
        m.norms[k] = halfToFloat(h);
    }
    return true;
}

std::vector<float> Cact::floats(size_t i) const {
    const TensorRecord& r = records_[i];
    if (r.dtype == DT_FP16) {
        const size_t n = (size_t)(r.nbytes / 2);
        std::vector<float> v(n);
        const uint8_t* p = data_.data() + r.offset;
        for (size_t k = 0; k < n; k++) {
            uint16_t h;
            std::memcpy(&h, p + 2 * k, 2);
            v[k] = halfToFloat(h);
        }
        return v;
    }
    if (r.dtype == DT_FP32) {
        const size_t n = (size_t)(r.nbytes / 4);
        std::vector<float> v(n);
        std::memcpy(v.data(), data_.data() + r.offset, n * 4);
        return v;
    }
    if (r.dtype == DT_CQ) return dequantCQ(r);
    return {};
}

} // namespace needle
