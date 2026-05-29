#pragma once
#include "../engine/tensor.h"
#include "matmul.h"
#include "layernorm.h"
#include "gelu.h"
#include "elementwise.h"
#include "attention.h"
#include <memory>

// A single decoder-only transformer block:
//   x = LayerNorm(x); attn_out = SelfAttention(x); x = x + attn_out
//   x = LayerNorm(x); ffn_out = GELU(x * Wg1) * Wg2; x = x + ffn_out
class TransformerBlock {
public:
    TransformerBlock();
    ~TransformerBlock() = default;

    bool init(int dModel, int nHeads, int maxSeqLen, int ffnHidden);

    // Forward pass with KV cache
    bool forward(const gl::Texture& inputTex, int seqLen,
                 // Attention weights
                 const gl::Texture* wqkv,
                 const gl::Texture* wo,
                 const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                 // FFN weights
                 const gl::Texture* wg1,
                 const gl::Texture* wg2,
                 const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                 // Optional biases
                 const gl::Texture* bqkv,
                 const gl::Texture* bo,
                 // Config
                 float attnScale, float epsilon,
                 // Output
                 gl::FBO& outputFBO);

private:
    int dModel_ = 0;
    int ffnHidden_ = 0;
    float epsilon_ = 1e-5f;

    // Sub-layers
    std::unique_ptr<LayerNorm> ln1_;
    std::unique_ptr<SelfAttention> attn_;
    std::unique_ptr<LayerNorm> ln2_;
    std::unique_ptr<MatMul> ffnGate_;
    std::unique_ptr<GELU> gelu_;
    std::unique_ptr<MatMul> ffnDown_;
    std::unique_ptr<ElementwiseAdd> residualAdd_;

    // Temporary buffers
    std::shared_ptr<gl::FBO> normBuf_;      // for LN output
    std::shared_ptr<gl::FBO> attnOutBuf_;   // for attention output
    std::shared_ptr<gl::FBO> ffnGateBuf_;   // for GELU output
    std::shared_ptr<gl::FBO> ffnOutBuf_;    // for FFN down output
    std::shared_ptr<gl::FBO> residualBuf_;  // for residual connection
};
