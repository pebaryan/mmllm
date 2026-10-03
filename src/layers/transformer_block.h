#pragma once
#include "../engine/tensor.h"
#include "matmul.h"
#include "layernorm.h"
#include "gelu.h"
#include "elementwise.h"
#include "attention.h"
#include <memory>

// A single pre-LN decoder-only transformer block:
//   x = x + Attn(LN1(x)) [+ bo]
//   x = x + (GELU(LN2(x) * Wg1 [+ bg1]) * Wg2 [+ bg2])
// Every bias is optional (null = no bias). No pass reads and writes the same texture.
class TransformerBlock {
public:
    TransformerBlock();
    ~TransformerBlock() = default;

    bool init(int dModel, int nHeads, int maxSeqLen, int ffnHidden);

    // Forward pass for one token with KV cache
    bool forward(const gl::Texture& inputTex, int seqLen,
                 // Attention weights
                 const gl::Texture* wqkv, const gl::Texture* bqkv,
                 const gl::Texture* wo, const gl::Texture* bo,
                 const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                 // FFN weights
                 const gl::Texture* wg1, const gl::Texture* bg1,
                 const gl::Texture* wg2, const gl::Texture* bg2,
                 const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                 // Config
                 float attnScale, float epsilon,
                 // Output (must not be the texture behind inputTex)
                 gl::FBO& outputFBO);

    // Local attention window for this block (0 = global attention)
    void setLocalWindow(int window) { if (attn_) attn_->setLocalWindow(window); }

    // Reset the attention KV cache (for a new sequence)
    void resetCache() { if (attn_) attn_->resetCache(); }

private:
    // Fused path: LayerNorm stats on the GPU, LN/bias/GELU/residual folded into the matmuls
    bool forwardFused(const gl::Texture& inputTex, int seqLen,
                      const gl::Texture* wqkv, const gl::Texture* bqkv,
                      const gl::Texture* wo, const gl::Texture* bo,
                      const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                      const gl::Texture* wg1, const gl::Texture* bg1,
                      const gl::Texture* wg2, const gl::Texture* bg2,
                      const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                      float attnScale, float epsilon, gl::FBO& outputFBO);

    // out = a + b over a [1, n] packed vector
    bool addVec(const gl::Texture& a, const gl::Texture& b, int n, gl::FBO& out);

    int dModel_ = 0;
    int ffnHidden_ = 0;
    bool fused_ = false;   // MMLLM_FUSE=1 enables (opt-in: slower on the 9400M)
    float epsilon_ = 1e-5f;

    // Sub-layers
    std::unique_ptr<LayerNorm> ln1_;
    std::unique_ptr<SelfAttention> attn_;
    std::unique_ptr<LayerNorm> ln2_;
    std::unique_ptr<MatMul> ffnGate_;
    std::unique_ptr<GELU> gelu_;
    std::unique_ptr<MatMul> ffnDown_;
    std::unique_ptr<ElementwiseAdd> add_;

    // Temporary buffers (each pass writes a buffer it does not read)
    std::shared_ptr<gl::FBO> stats1Buf_;     // fused path: LN1 mean/invStd (1x1)
    std::shared_ptr<gl::FBO> stats2Buf_;     // fused path: LN2 mean/invStd (1x1)
    std::shared_ptr<gl::FBO> normBuf_;       // LN output
    std::shared_ptr<gl::FBO> attnOutBuf_;    // attention projection output
    std::shared_ptr<gl::FBO> attnBiasBuf_;   // attention output + bo
    std::shared_ptr<gl::FBO> residualBuf_;   // x + attention
    std::shared_ptr<gl::FBO> ffnUpBuf_;      // LN2(x) * Wg1
    std::shared_ptr<gl::FBO> ffnUpBiasBuf_;  // + bg1
    std::shared_ptr<gl::FBO> ffnActBuf_;     // GELU output
    std::shared_ptr<gl::FBO> ffnOutBuf_;     // act * Wg2
    std::shared_ptr<gl::FBO> ffnOutBiasBuf_; // + bg2
};
