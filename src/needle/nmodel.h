#pragma once
// Needle 3 (SimpleAttentionNetwork) inference, one token at a time with KV caches.
// Direct port of the validated NumPy incremental model (needle_inc.py), which was checked against
// the JAX reference (cosine 1.0, max logit diff 8e-5) and against the shipped engine's answers.
#include "cact.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace needle {

class NGpu;   // GPU backend (ngpu.h) - optional, only touched when --gpu is used

struct Config {
    int D = 768, L = 20, H = 12, KVH = 2, QK = 48, VH = 64;
    int lanes = 4, taps = 3, vocab = 8192, hadaN = 1024;
    int engramSub = 128, engramTables = 6, engramSlots = 18432, engramTaps = 4, engramDil = 3;
    int engramSeedHeads = 0, engramHeads = 3;
    int window = 1024;
    float rope = 100000.f;
    std::vector<int> globalLayers, engramLayers, orders;
    bool isGlobal(int l) const;
    int siteOf(int layer) const;     // engram site index at a layer, or -1
};

// A weight matrix [out, in]. 2-bit CQ matrices stay packed and are multiplied directly (the 12x smaller
// DRAM traffic matters: this CPU is memory-bandwidth bound); anything else is held as float32.
struct Mat {
    int out = 0, in = 0;
    bool quant = false;
    std::vector<float> f;           // float32 row-major (when !quant)
    std::vector<uint8_t> packed;    // 2-bit indices, 4 per byte, LSB first (when quant)
    std::vector<float> norms;       // per row, per 128-group
};

struct Layer {
    Mat qProj, kProj, vProj, gateProj, outProj;
    std::vector<float> normIn, qTaps, kTaps, vTaps, qNorm, kNorm,
        postNorm, preHada, d1, d2, b2, d3, d4, w1a, w1b, w2a, w2b, w3a, w3b, condV, condU;
    float attnGate = 0.f;
};

// The hashed n-gram tables stay in their packed CQ form (one 128-group per row) and a row is expanded when
// it is looked up: 30 rows per token instead of 280 MB of float32.
struct EngramSite {
    std::vector<float> tables, taps;                  // tables: only filled when the archive stores them unpacked
    std::vector<uint8_t> tPacked;                     // packed CQ rows
    std::vector<float> tNorms;                        // one norm per row
    int tBits = 0, tRowBytes = 0;
    Mat keyProj, valueProj;
};

// Per-sequence state: attention KV caches, conv-tap history, engram value history.
struct State {
    struct LayerCache {
        std::vector<float> k, v;                  // [pos][kvHeads][qk|v]
        std::vector<float> prevQ, prevK, prevV;   // raw projections at t-1 then t-2 (zeros before the start)
    };
    std::vector<int> tokens;
    std::vector<LayerCache> layers;
    std::vector<std::vector<float>> engramV;      // per site: value_proj outputs, [pos * D]
    // Online softmax-pooling accumulators of the confidence probe head, one per (layer cell, probe):
    // running max, running denominator and running weighted sum of the cells (see Model::confidenceLogit).
    std::vector<float> probeM, probeS, probeR;
    int position() const { return (int)tokens.size(); }
};

class Model {
public:
    Model();
    ~Model();
    bool load(const Cact& cact, std::string& error);
    const Config& config() const { return cfg_; }

    State newState() const;

    // Process one token at the next position. If logits != nullptr it receives vocab floats.
    // quant = simulate the engine's int8 activations / KV (what the shipped archive was tuned for).
    void step(State& st, int token, bool quant, float* logits);

    // Batched prefill: advance the state by `count` tokens in one go (logits are not produced, so this
    // is only valid where the caller does not need them, e.g. Session::init). Numerically identical to
    // `count` calls of step(..., nullptr); the 2-bit projections run as GEMMs instead of GEMVs.
    // NOTE: measured slower than the per-token path on the 2-core target CPU (see NEEDLE.md), so it is
    // opt-in via --batch-prefill and exists mainly to carry the formulation to wider hardware.
    void prefill(State& st, const int* tokens, int count, bool quant);
    bool batchPrefill = false;

    void setThreads(int n);

    // ---- optional GPU offload of the 2-bit CQ matrix-vector products (--gpu).
    // The weights are uploaded once, in their packed 2-bit form; the rest of the per-token graph
    // (norms, rope, attention, the Hadamard MLP, the tied head) stays on the CPU. See NEEDLE.md:
    // on the target GeForce 9400M this is slower than the CPU, so it is opt-in.
    bool enableGpu(NGpu* gpu);
    bool gpuEnabled() const { return gpu_ != nullptr; }
    // Recomputes every offloaded matrix on both paths and reports the worst discrepancy.
    bool gpuSelfCheck(std::string& report);

    // Confidence head (probe pooling over the hidden cells of every token seen so far).
    bool hasConfidenceHead() const { return !headProj_.empty(); }
    // Raw head logit for the tokens in `st`; apply a sigmoid for the score.
    double confidenceLogit(const State& st) const;

    // Optional per-section timing (NEEDLE_PROFILE=1): seconds accumulated over `tokens` steps.
    struct Profile { double mhc = 0, engram = 0, qkv = 0, attn = 0, gateout = 0, hada = 0, mhc2 = 0, head = 0; long tokens = 0; };
    Profile profile;
    bool profiling = false;

private:
    struct Pool;
    Config cfg_;
    std::vector<float> emb_;
    std::vector<Layer> layers_;
    std::vector<EngramSite> sites_;
    std::vector<float> mhcAPre_, mhcAPost_, mhcARes_, mhcBPre_, mhcBPost_, mhcBRes_;
    std::vector<float> phiPre_, phiPost_, phiRes_;
    std::vector<int> hadaP1_, hadaP2_;
    std::vector<float> finalNorm_;
    std::vector<float> ropeInv_;
    // confidence head: probes [(L+1)*K][D], gain [(L+1)*K], query [Q][D], rowBias [Q][(L+1)*K], proj [Q*D], bias
    std::vector<float> headProbes_, headGain_, headQuery_, headRowBias_, headProj_;
    float headBias_ = 0.f;
    int headK_ = 4, headQ_ = 4;
    void probeUpdate(State& st, int cell, const float* x) const;
    float cbTable_[3][16] = {};                       // codebooks for 2/3/4 bit CQ, indexed bits-2
    void engramRow(const EngramSite& es, size_t row, float* dst) const;
    std::unique_ptr<Pool> pool_;

    // GPU offload (null unless enableGpu() succeeded)
    NGpu* gpu_ = nullptr;
    std::vector<std::pair<const Mat*, int>> gpuMap_;   // packed 2-bit matrix -> GPU handle

    void gemv(const float* W, int out, int in, const float* x, float* y);
    // y = M x. For packed matrices xr must be the group-rotated copy of x (see rotateInput).
    void matvec(const Mat& M, const float* x, const float* xr, float* y);
    // Batched form: Y[m][out] = M * XR[m] for m in [0, count); XR is [count][in] group-rotated.
    void matvecBatch(const Mat& M, const float* XR, int count, float* Y);
    bool loadMat(const Cact& c, size_t index, Mat& m);
    float lut2_[256][4] __attribute__((aligned(16)));   // byte -> 4 codebook values (2-bit)
};

// Helpers shared with tests
void fakeQuantRow(float* x, int n);

} // namespace needle
