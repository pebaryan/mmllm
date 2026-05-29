#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include "matmul.h"
#include "softmax.h"
#include <memory>

// Self-attention layer: Attn(Q, K, V) = softmax(Q*K^T / sqrt(d_head)) * V
// With KV-cache for autoregressive generation.
class SelfAttention {
public:
    SelfAttention();
    ~SelfAttention() = default;

    bool init(int dModel, int nHeads, int maxSeqLen);

    // Forward pass for a single token (with KV cache)
    // inputTex: [1, d_model] — the normalized input (after LayerNorm)
    bool forward(const gl::Texture& inputTex,
                 int seqLen,
                 const gl::Texture* wqkv,
                 const gl::Texture* wo,
                 float attnScale,
                 gl::FBO& outputFBO);

    // Access the KV cache textures (for managing externally if needed)
    const GPUTensor& kCache() const { return *kCache_; }
    const GPUTensor& vCache() const { return *vCache_; }

    // Reset KV cache (for new sequence)
    void resetCache();

private:
    int dModel_ = 0;
    int nHeads_ = 0;
    int dHead_ = 0;
    int maxSeqLen_ = 0;

    // KV cache: [max_seq_len, d_model] each
    std::unique_ptr<GPUTensor> kCache_;
    std::unique_ptr<GPUTensor> vCache_;
    int cachedLen_ = 0;

    // Sub-layers
    std::unique_ptr<MatMul> matmulQ_;      // Q projection
    std::unique_ptr<MatMul> matmulScore_;  // Q * K^T
    std::unique_ptr<Softmax> softmax_;
    std::unique_ptr<MatMul> matmulOut_;    // scores * V
    std::unique_ptr<MatMul> matmulProj_;   // output projection

    // Temporary FBOs
    std::shared_ptr<gl::FBO> qBuf_;
    std::shared_ptr<gl::FBO> scoreBuf_;
    std::shared_ptr<gl::FBO> attnOutBuf_;
    std::shared_ptr<gl::FBO> finalBuf_;

    // Readback buffer for softmax
    std::vector<float> readbackBuf_;
};
