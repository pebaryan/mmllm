#include "transformer_block.h"
#include "../gl/fbo.h"
#include <cstdio>
#include <cstdlib>

TransformerBlock::TransformerBlock() {}

bool TransformerBlock::init(int dModel, int nHeads, int maxSeqLen, int ffnHidden) {
    dModel_ = dModel;
    ffnHidden_ = ffnHidden;

    ln1_ = std::make_unique<LayerNorm>();
    if (!ln1_->init()) return false;

    attn_ = std::make_unique<SelfAttention>();
    if (!attn_->init(dModel, nHeads, maxSeqLen)) return false;

    ln2_ = std::make_unique<LayerNorm>();
    if (!ln2_->init()) return false;

    ffnGate_ = std::make_unique<MatMul>();
    if (!ffnGate_->init()) return false;

    gelu_ = std::make_unique<GELU>();
    if (!gelu_->init()) return false;

    ffnDown_ = std::make_unique<MatMul>();
    if (!ffnDown_->init()) return false;

    add_ = std::make_unique<ElementwiseAdd>();
    if (!add_->init()) return false;

    normBuf_ = createRenderTarget(1, dModel, "block_norm");
    attnOutBuf_ = createRenderTarget(1, dModel, "block_attn");
    attnBiasBuf_ = createRenderTarget(1, dModel, "block_attn_bias");
    residualBuf_ = createRenderTarget(1, dModel, "block_residual");
    ffnUpBuf_ = createRenderTarget(1, ffnHidden, "block_ffn_up");
    ffnUpBiasBuf_ = createRenderTarget(1, ffnHidden, "block_ffn_up_bias");
    ffnActBuf_ = createRenderTarget(1, ffnHidden, "block_ffn_act");
    ffnOutBuf_ = createRenderTarget(1, dModel, "block_ffn_out");
    ffnOutBiasBuf_ = createRenderTarget(1, dModel, "block_ffn_out_bias");

    if (!normBuf_ || !attnOutBuf_ || !attnBiasBuf_ || !residualBuf_ || !ffnUpBuf_ ||
        !ffnUpBiasBuf_ || !ffnActBuf_ || !ffnOutBuf_ || !ffnOutBiasBuf_) {
        return false;
    }

    stats1Buf_ = createRenderTarget(1, 4, "ln1_stats");
    stats2Buf_ = createRenderTarget(1, 4, "ln2_stats");
    // Opt-in: measured slower than the unfused path on the GeForce 9400M
    fused_ = stats1Buf_ && stats2Buf_ && std::getenv("MMLLM_FUSE") &&
             ln1_->usesGpu() && ln2_->usesGpu() && attn_->usesGpu() &&
             ffnGate_->fusedAvailable() && ffnDown_->fusedAvailable();

    std::printf("[mmllm] TransformerBlock initialized: d=%d ffn=%d (%s)\n", dModel, ffnHidden,
                fused_ ? "fused" : "unfused");
    return true;
}

bool TransformerBlock::addVec(const gl::Texture& a, const gl::Texture& b, int n, gl::FBO& out) {
    const int tw = (n + 3) / 4;
    return add_->forward(a, b, tw, 1, n, out);
}

bool TransformerBlock::forward(const gl::Texture& inputTex, int seqLen,
                               const gl::Texture* wqkv, const gl::Texture* bqkv,
                               const gl::Texture* wo, const gl::Texture* bo,
                               const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                               const gl::Texture* wg1, const gl::Texture* bg1,
                               const gl::Texture* wg2, const gl::Texture* bg2,
                               const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                               float attnScale, float epsilon,
                               gl::FBO& outputFBO)
{
    epsilon_ = epsilon;

    if (fused_) {
        return forwardFused(inputTex, seqLen, wqkv, bqkv, wo, bo, ln1Gain, ln1Bias,
                            wg1, bg1, wg2, bg2, ln2Gain, ln2Bias, attnScale, epsilon, outputFBO);
    }

    // 1. Pre-attention LayerNorm
    if (!ln1_->forward(inputTex, dModel_, *ln1Gain, *ln1Bias, epsilon, *normBuf_)) return false;

    // 2. Self-attention (QKV + per-head attention + output projection)
    if (!attn_->forward(normBuf_->colorTexture(), seqLen, wqkv, bqkv, wo, attnScale, *attnOutBuf_)) {
        return false;
    }

    // 2b. Output projection bias
    const gl::Texture* attnResult = &attnOutBuf_->colorTexture();
    if (bo) {
        if (!addVec(*attnResult, *bo, dModel_, *attnBiasBuf_)) return false;
        attnResult = &attnBiasBuf_->colorTexture();
    }

    // 3. Residual: x + attn
    if (!addVec(inputTex, *attnResult, dModel_, *residualBuf_)) return false;

    // 4. Pre-FFN LayerNorm
    if (!ln2_->forward(residualBuf_->colorTexture(), dModel_, *ln2Gain, *ln2Bias, epsilon, *normBuf_)) {
        return false;
    }

    // 5. FFN up projection: LN(x) * Wg1  ->  [1, ffn_hidden]
    if (!ffnGate_->forward(normBuf_->colorTexture(), 1, dModel_,
                           *wg1, dModel_, ffnHidden_, *ffnUpBuf_)) {
        return false;
    }
    const gl::Texture* up = &ffnUpBuf_->colorTexture();
    if (bg1) {
        if (!addVec(*up, *bg1, ffnHidden_, *ffnUpBiasBuf_)) return false;
        up = &ffnUpBiasBuf_->colorTexture();
    }

    // 6. GELU (reads `up`, writes a different buffer)
    if (!gelu_->forward(*up, ffnHidden_, *ffnActBuf_)) return false;

    // 7. FFN down projection: GELU(x) * Wg2  ->  [1, d_model]
    if (!ffnDown_->forward(ffnActBuf_->colorTexture(), 1, ffnHidden_,
                           *wg2, ffnHidden_, dModel_, *ffnOutBuf_)) {
        return false;
    }
    const gl::Texture* down = &ffnOutBuf_->colorTexture();
    if (bg2) {
        if (!addVec(*down, *bg2, dModel_, *ffnOutBiasBuf_)) return false;
        down = &ffnOutBiasBuf_->colorTexture();
    }

    // 8. Residual: attn residual + ffn
    if (!addVec(residualBuf_->colorTexture(), *down, dModel_, outputFBO)) return false;

    return true;
}

bool TransformerBlock::forwardFused(const gl::Texture& inputTex, int seqLen,
                                    const gl::Texture* wqkv, const gl::Texture* bqkv,
                                    const gl::Texture* wo, const gl::Texture* bo,
                                    const gl::Texture* ln1Gain, const gl::Texture* ln1Bias,
                                    const gl::Texture* wg1, const gl::Texture* bg1,
                                    const gl::Texture* wg2, const gl::Texture* bg2,
                                    const gl::Texture* ln2Gain, const gl::Texture* ln2Bias,
                                    float attnScale, float epsilon, gl::FBO& outputFBO)
{
    // 1. LN1 statistics (one tiny pass); the normalisation itself is folded into the QKV matmul
    if (!ln1_->computeStats(inputTex, dModel_, epsilon, *stats1Buf_)) return false;

    // 2. residual = x + Attn(LN1(x)) * Wo + bo    (4 attention passes + 2 fused matmuls)
    if (!attn_->forwardFused(inputTex, seqLen,
                             &stats1Buf_->colorTexture(), ln1Gain, ln1Bias,
                             wqkv, bqkv, wo, bo, attnScale,
                             inputTex, *residualBuf_)) {
        return false;
    }

    // 3. LN2 statistics; normalisation folded into the FFN up-projection
    if (!ln2_->computeStats(residualBuf_->colorTexture(), dModel_, epsilon, *stats2Buf_)) return false;

    // 4. act = GELU(LN2(residual) * Wg1 + bg1)
    MatMul::FusedOps up;
    up.lnStats = &stats2Buf_->colorTexture();
    up.lnGain = ln2Gain;
    up.lnBias = ln2Bias;
    up.bias = bg1;
    up.gelu = true;
    if (!ffnGate_->forwardFused(residualBuf_->colorTexture(), 1, dModel_,
                                *wg1, dModel_, ffnHidden_, *ffnActBuf_, up)) {
        return false;
    }

    // 5. out = residual + act * Wg2 + bg2
    MatMul::FusedOps down;
    down.bias = bg2;
    down.residual = &residualBuf_->colorTexture();
    return ffnDown_->forwardFused(ffnActBuf_->colorTexture(), 1, ffnHidden_,
                                  *wg2, ffnHidden_, dModel_, outputFBO, down);
}
