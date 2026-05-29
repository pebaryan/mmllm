#include "attention.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cmath>
#include <cstdio>
#include <cstring>

SelfAttention::SelfAttention() {}

bool SelfAttention::init(int dModel, int nHeads, int maxSeqLen) {
    dModel_ = dModel;
    nHeads_ = nHeads;
    dHead_ = dModel / nHeads;
    maxSeqLen_ = maxSeqLen;

    // Create KV cache tensors
    kCache_ = std::make_unique<GPUTensor>();
    if (!kCache_->create(maxSeqLen_, dModel, "k_cache")) {
        return false;
    }

    vCache_ = std::make_unique<GPUTensor>();
    if (!vCache_->create(maxSeqLen_, dModel, "v_cache")) {
        return false;
    }

    // Initialize sub-layers
    matmulQ_ = std::make_unique<MatMul>();
    if (!matmulQ_->init()) return false;

    matmulScore_ = std::make_unique<MatMul>();
    if (!matmulScore_->init()) return false;

    softmax_ = std::make_unique<Softmax>();
    if (!softmax_->init()) return false;

    matmulOut_ = std::make_unique<MatMul>();
    if (!matmulOut_->init()) return false;

    matmulProj_ = std::make_unique<MatMul>();
    if (!matmulProj_->init()) return false;

    // Create temporary buffers
    qBuf_ = createRenderTarget(1, 3 * dModel_, "q_buf");
    scoreBuf_ = createRenderTarget(1, maxSeqLen_, "score_buf");
    attnOutBuf_ = createRenderTarget(1, dModel_, "attn_out_buf");
    finalBuf_ = createRenderTarget(1, dModel_, "attn_final_buf");

    if (!qBuf_ || !scoreBuf_ || !attnOutBuf_ || !finalBuf_) {
        return false;
    }

    std::printf("[mmllm] SelfAttention initialized: d=%d heads=%d d_head=%d max_seq=%d\n",
                dModel, nHeads, dHead_, maxSeqLen);
    return true;
}

bool SelfAttention::forward(const gl::Texture& inputTex, int seqLen,
                            const gl::Texture* wqkv, const gl::Texture* wo,
                            float attnScale, gl::FBO& outputFBO)
{
    // 1. QKV = inputTex * Wqkv  →  [1, 3*d_model]
    if (!matmulQ_->forward(inputTex, 1, dModel_,
                           *wqkv, dModel_, 3 * dModel_,
                           *qBuf_)) {
        return false;
    }

    // Extract Q, K, V from the fused QKV buffer (all [1, d_model])
    // For simplicity, the weight matrix wqkv is [d_model, 3*d_model] = [Wq|Wk|Wv]
    // The output is [1, 3*d_model] = [q|k|v]
    // We need to split them. We'll read back and re-upload K and V to the KV cache.

    int qkvElements = qBuf_->colorTexture().elements();
    readbackBuf_.resize(qkvElements);
    qBuf_->colorTexture().download(readbackBuf_.data());

    int qStart = 0;
    int kStart = dModel_;
    int vStart = 2 * dModel_;

    // Update K cache: copy k into cache at position seqLen-1
    // Read current K cache contents
    std::vector<float> kCacheData(maxSeqLen_ * dModel_ * 4, 0.0f);
    kCache_->texture().download(kCacheData.data());

    // Write K at row (seqLen - 1)
    int tw = kCache_->packedCols();
    for (int c = 0; c < dModel_; c++) {
        int tx = c / 4;
        int tc = c % 4;
        kCacheData[((seqLen - 1) * tw + tx) * 4 + tc] = readbackBuf_[kStart + c];
    }
    kCache_->texture().upload(kCacheData.data());

    // Update V cache similarly
    std::vector<float> vCacheData(maxSeqLen_ * dModel_ * 4, 0.0f);
    vCache_->texture().download(vCacheData.data());
    for (int c = 0; c < dModel_; c++) {
        int tx = c / 4;
        int tc = c % 4;
        vCacheData[((seqLen - 1) * tw + tx) * 4 + tc] = readbackBuf_[vStart + c];
    }
    vCache_->texture().upload(vCacheData.data());

    cachedLen_ = seqLen;

    // 2. Attention scores = Q * K^T / sqrt(d_head)
    // Q: [1, d_model] (just the Q portion)
    // K: [seqLen, d_model] (from cache)
    // K^T: [d_model, seqLen] — we need to matmul Q with K^T
    // For simplicity, compute score[i] = sum_j Q[j] * K[i][j] manually on CPU since this is small
    // Then apply softmax on CPU, then matmul with V on GPU.

    float* qPtr = readbackBuf_.data() + qStart;
    std::vector<float> scores(seqLen);

    // Download K cache for the current sequence
    std::vector<float> kSlice(seqLen * dModel_);
    for (int i = 0; i < seqLen; i++) {
        for (int c = 0; c < dModel_; c++) {
            int tx = c / 4;
            int tc = c % 4;
            kSlice[i * dModel_ + c] = kCacheData[(i * tw + tx) * 4 + tc];
        }
    }

    // Compute raw scores
    for (int i = 0; i < seqLen; i++) {
        float sum = 0.0f;
        for (int j = 0; j < dModel_; j++) {
            sum += qPtr[j] * kSlice[i * dModel_ + j];
        }
        scores[i] = sum * attnScale;
    }

    // Softmax over scores
    float maxScore = -1e20f;
    for (int i = 0; i < seqLen; i++) {
        if (scores[i] > maxScore) maxScore = scores[i];
    }
    double sumExp = 0.0;
    for (int i = 0; i < seqLen; i++) {
        sumExp += std::exp((double)(scores[i] - maxScore));
    }
    float invSum = 1.0f / (float)sumExp;
    for (int i = 0; i < seqLen; i++) {
        scores[i] = std::exp(scores[i] - maxScore) * invSum;
    }

    // 3. Attention output = softmax_scores * V
    // scores: [1, seqLen], V: [seqLen, d_model]
    // We compute this on CPU since the attention vectors are small
    std::vector<float> attnOut(dModel_, 0.0f);
    for (int c = 0; c < dModel_; c++) {
        float sum = 0.0f;
        for (int i = 0; i < seqLen; i++) {
            int tx = c / 4;
            int tc = c % 4;
            sum += scores[i] * vCacheData[(i * tw + tx) * 4 + tc];
        }
        attnOut[c] = sum;
    }

    // Upload attention output to GPU for output projection
    std::vector<float> attnPacked(tw * 1 * 4, 0.0f);
    for (int c = 0; c < dModel_; c++) {
        int tx = c / 4;
        int tc = c % 4;
        attnPacked[(0 * tw + tx) * 4 + tc] = attnOut[c];
    }
    attnOutBuf_->colorTexture().upload(attnPacked.data());

    // 4. Output projection: attnOut * Wo  →  [1, d_model]
    if (!matmulProj_->forward(attnOutBuf_->colorTexture(), 1, dModel_,
                              *wo, dModel_, dModel_,
                              outputFBO)) {
        return false;
    }

    return true;
}

void SelfAttention::resetCache() {
    cachedLen_ = 0;
    // Zero out cache textures
    std::vector<float> zeros(maxSeqLen_ * packedWidth(dModel_) * 4, 0.0f);
    if (kCache_ && kCache_->valid())
        kCache_->texture().upload(zeros.data());
    if (vCache_ && vCache_->valid())
        vCache_->texture().upload(zeros.data());
}
