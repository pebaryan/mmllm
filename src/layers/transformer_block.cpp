#include "transformer_block.h"
#include "../gl/fbo.h"
#include <cstdio>

TransformerBlock::TransformerBlock() {}

bool TransformerBlock::init(int dModel, int nHeads, int maxSeqLen, int ffnHidden) {
    dModel_ = dModel;
    ffnHidden_ = ffnHidden;

    // Pre-attention LayerNorm
    ln1_ = std::make_unique<LayerNorm>();
    if (!ln1_->init()) return false;

    // Self-attention
    attn_ = std::make_unique<SelfAttention>();
    if (!attn_->init(dModel, nHeads, maxSeqLen)) return false;

    // Pre-FFN LayerNorm
    ln2_ = std::make_unique<LayerNorm>();
    if (!ln2_->init()) return false;

    // FFN gate (up projection)
    ffnGate_ = std::make_unique<MatMul>();
    if (!ffnGate_->init()) return false;

    // GELU activation
    gelu_ = std::make_unique<GELU>();
    if (!gelu_->init()) return false;

    // FFN down projection
    ffnDown_ = std::make_unique<MatMul>();
    if (!ffnDown_->init()) return false;

    // Residual add
    residualAdd_ = std::make_unique<ElementwiseAdd>();
    if (!residualAdd_->init()) return false;

    // Create temporary buffers
    normBuf_ = createRenderTarget(1, dModel, "block_norm");
    attnOutBuf_ = createRenderTarget(1, dModel, "block_attn");
    ffnGateBuf_ = createRenderTarget(1, ffnHidden, "block_ffn_gate");
    ffnOutBuf_ = createRenderTarget(1, dModel, "block_ffn_out");
    residualBuf_ = createRenderTarget(1, dModel, "block_residual");

    if (!normBuf_ || !attnOutBuf_ || !ffnGateBuf_ || !ffnOutBuf_ || !residualBuf_) {
        return false;
    }

    std::printf("[mmllm] TransformerBlock initialized: d=%d ffn=%d\n", dModel, ffnHidden);
    return true;
}

bool TransformerBlock::forward(const gl::Texture& inputTex, int seqLen,
                               const gl::Texture* wqkv,
                               const gl::Texture* wo,
                               const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                               const gl::Texture* wg1,
                               const gl::Texture* wg2,
                               const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                               const gl::Texture* bqkv,
                               const gl::Texture* bo,
                               float attnScale, float epsilon,
                               gl::FBO& outputFBO)
{
    epsilon_ = epsilon;

    // 1. Pre-attention LayerNorm
    if (!ln1_->forward(inputTex, dModel_,
                       *ln1Gain, *ln1Bias, epsilon, *normBuf_)) {
        return false;
    }

    // 2. Self-attention
    if (!attn_->forward(normBuf_->colorTexture(), seqLen, wqkv, wo, attnScale, *attnOutBuf_)) {
        return false;
    }

    // 3. Residual add: x + attn_out
    int tw = (dModel_ + 3) / 4;
    if (!residualAdd_->forward(inputTex, attnOutBuf_->colorTexture(),
                               tw, 1, dModel_, *residualBuf_)) {
        return false;
    }

    // 4. Pre-FFN LayerNorm
    if (!ln2_->forward(residualBuf_->colorTexture(), dModel_,
                       *ln2Gain, *ln2Bias, epsilon, *normBuf_)) {
        return false;
    }

    // 5. FFN gate: LN(x) * Wg1  →  [1, ffn_hidden]
    if (!ffnGate_->forward(normBuf_->colorTexture(), 1, dModel_,
                           *wg1, dModel_, ffnHidden_, *ffnGateBuf_)) {
        return false;
    }

    // 6. GELU activation
    if (!gelu_->forward(ffnGateBuf_->colorTexture(), ffnHidden_, *ffnGateBuf_)) {
        return false;
    }

    // 7. FFN down: GELU(x) * Wg2  →  [1, d_model]
    if (!ffnDown_->forward(ffnGateBuf_->colorTexture(), 1, ffnHidden_,
                           *wg2, ffnHidden_, dModel_, *ffnOutBuf_)) {
        return false;
    }

    // 8. Residual add: residual + ffn_out
    if (!residualAdd_->forward(residualBuf_->colorTexture(), ffnOutBuf_->colorTexture(),
                               tw, 1, dModel_, outputFBO)) {
        return false;
    }

    return true;
}
