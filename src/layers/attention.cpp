#include "attention.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <GL/glew.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>

// nouveau on NV50-class GPUs (GeForce 9400M) reports GL_MAX_TEXTURE_SIZE = 8192 but returns
// wrong values when a shader samples a texture wider than 4096 texels. Anything we sample
// (or read back) is kept within this safe dimension.
static const int kSafeTexDim = 4096;

SelfAttention::SelfAttention() {}

std::unique_ptr<gl::Program> SelfAttention::makeProgram(
    const char* fragPath, std::vector<std::unique_ptr<gl::Shader>>& keep)
{
    auto vert = std::make_unique<gl::Shader>();
    if (!vert->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) return nullptr;
    auto frag = std::make_unique<gl::Shader>();
    if (!frag->compileFromFile(GL_FRAGMENT_SHADER, fragPath)) return nullptr;

    auto prog = std::make_unique<gl::Program>();
    if (!prog->link(*vert, *frag)) return nullptr;

    keep.push_back(std::move(vert));
    keep.push_back(std::move(frag));
    return prog;
}

bool SelfAttention::initGpuPasses() {
    appendProg_ = makeProgram("src/shaders/kv_append.frag", shaders_);
    scoresProg_ = makeProgram("src/shaders/attn_scores.frag", shaders_);
    statsProg_ = makeProgram("src/shaders/attn_stats.frag", shaders_);
    avProg_ = makeProgram("src/shaders/attn_av.frag", shaders_);
    if (!appendProg_ || !scoresProg_ || !statsProg_ || !avProg_) return false;

    add_ = std::make_unique<ElementwiseAdd>();
    if (!add_->init()) return false;

    const int tw = (dModel_ + 3) / 4;

    // KV cache: one row per position, [k (T texels) | v (T texels)]
    // fp16 halves the largest position-sized GPU buffer (MMLLM_KV_FP32=1 keeps fp32). GPU memory
    // is the scarce resource on the 9400M: a working set beyond its 256 MB spills to system
    // memory and everything slows down several times over.
    kvBuf_ = createRenderTarget(maxSeqLen_, 2 * tw * 4, "kv_cache",
                                std::getenv("MMLLM_KV_FP32") ? gl::TextureFormat::RGBA32F
                                                              : gl::TextureFormat::RGBA16F);
    // Scores: texel (position, head); only .r is used
    scoresBuf_ = createRenderTarget(nHeads_, maxSeqLen_ * 4, "attn_scores");
    // Softmax stats: texel (0, head)
    statsBuf_ = createRenderTarget(nHeads_, 4, "attn_stats");
    qkvBiasBuf_ = createRenderTarget(1, 3 * dModel_, "qkv_bias_buf");

    return kvBuf_ && scoresBuf_ && statsBuf_ && qkvBiasBuf_;
}

bool SelfAttention::init(int dModel, int nHeads, int maxSeqLen) {
    dModel_ = dModel;
    nHeads_ = nHeads;
    dHead_ = dModel / nHeads;

    // Position-indexed GPU textures cannot be taller/wider than the texture size limit
    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    if (maxTex <= 0) maxTex = 8192;
    maxSeqLen_ = std::min(maxSeqLen, std::min((int)maxTex, kSafeTexDim));

    matmulQkv_ = std::make_unique<MatMul>();
    if (!matmulQkv_->init()) return false;

    matmulProj_ = std::make_unique<MatMul>();
    if (!matmulProj_->init()) return false;

    qkvBuf_ = createRenderTarget(1, 3 * dModel_, "qkv_buf");
    attnOutBuf_ = createRenderTarget(1, dModel_, "attn_out_buf");
    if (!qkvBuf_ || !attnOutBuf_) return false;

    const bool forceCpu = std::getenv("MMLLM_CPU_ATTN") != nullptr;
    if (!forceCpu && dHead_ % 4 == 0) {
        gpu_ = initGpuPasses();
        if (!gpu_) {
            std::fprintf(stderr, "[mmllm] GPU attention passes failed to initialize; using CPU attention\n");
        }
    }

    if (!gpu_) {
        kCache_.assign((size_t)maxSeqLen_ * dModel_, 0.0f);
        vCache_.assign((size_t)maxSeqLen_ * dModel_, 0.0f);
        scores_.assign(maxSeqLen_, 0.0f);
    }

    std::printf("[mmllm] SelfAttention initialized: d=%d heads=%d d_head=%d max_seq=%d (%s)\n",
                dModel, nHeads, dHead_, maxSeqLen_, gpu_ ? "GPU" : "CPU");
    return true;
}

bool SelfAttention::forward(const gl::Texture& inputTex, int seqLen,
                            const gl::Texture* wqkv, const gl::Texture* bqkv,
                            const gl::Texture* wo,
                            float attnScale, gl::FBO& outputFBO)
{
    if (seqLen < 1 || seqLen > maxSeqLen_) {
        std::fprintf(stderr, "[mmllm] Attention: sequence length %d out of range (max %d)\n",
                     seqLen, maxSeqLen_);
        return false;
    }
    return gpu_ ? forwardGpu(inputTex, seqLen, wqkv, bqkv, wo, attnScale, outputFBO)
                : forwardCpu(inputTex, seqLen, wqkv, bqkv, wo, attnScale, outputFBO);
}

void SelfAttention::attendGpu(const gl::Texture* qkv, int seqLen, int pos, int first,
                              int headTexels, int tw, float attnScale)
{
    // 2. Append [k | v] to the GPU KV cache at row `pos` (scissor limits the draw to that row)
    dispatchShader(*appendProg_, *kvBuf_, {{0, qkv, "texSrc"}},
        [&](gl::Program& p) {
            p.setInt("srcOffset", tw);
            glEnable(GL_SCISSOR_TEST);
            glScissor(0, pos, 2 * tw, 1);
        });
    glDisable(GL_SCISSOR_TEST);

    // 3. Scores for every (position, head)
    dispatchShader(*scoresProg_, *scoresBuf_,
        {{0, qkv, "texQKV"}, {1, &kvBuf_->colorTexture(), "texKV"}},
        [&](gl::Program& p) {
            p.setInt("seqLen", seqLen);
            p.setInt("first", first);
            p.setInt("headTexels", headTexels);
            p.setFloat("scale", attnScale);
        });

    // 4. Softmax statistics per head (max and 1/sum)
    dispatchShader(*statsProg_, *statsBuf_,
        {{0, &scoresBuf_->colorTexture(), "texScores"}},
        [&](gl::Program& p) {
            p.setInt("seqLen", seqLen);
            p.setInt("first", first);
        });

    // 5. Softmax-weighted sum of V -> [1, d_model]
    dispatchShader(*avProg_, *attnOutBuf_,
        {{0, &scoresBuf_->colorTexture(), "texScores"},
         {1, &statsBuf_->colorTexture(), "texStats"},
         {2, &kvBuf_->colorTexture(), "texKV"}},
        [&](gl::Program& p) {
            p.setInt("seqLen", seqLen);
            p.setInt("first", first);
            p.setInt("headTexels", headTexels);
            p.setInt("vOffset", tw);
        });
}

bool SelfAttention::forwardGpu(const gl::Texture& inputTex, int seqLen,
                               const gl::Texture* wqkv, const gl::Texture* bqkv,
                               const gl::Texture* wo,
                               float attnScale, gl::FBO& outputFBO)
{
    const int tw = (dModel_ + 3) / 4;
    const int headTexels = dHead_ / 4;
    const int pos = seqLen - 1;
    const int first = (localWindow_ > 0 && seqLen > localWindow_) ? seqLen - localWindow_ : 0;

    // 1. QKV = input * Wqkv  ->  [1, 3*d_model] = [q | k | v]
    if (!matmulQkv_->forward(inputTex, 1, dModel_, *wqkv, dModel_, 3 * dModel_, *qkvBuf_)) {
        return false;
    }
    const gl::Texture* qkv = &qkvBuf_->colorTexture();
    if (bqkv) {
        if (!add_->forward(*qkv, *bqkv, (3 * dModel_ + 3) / 4, 1, 3 * dModel_, *qkvBiasBuf_)) {
            return false;
        }
        qkv = &qkvBiasBuf_->colorTexture();
    }

    attendGpu(qkv, seqLen, pos, first, headTexels, tw, attnScale);

    // 6. Output projection: attnOut * Wo -> [1, d_model]
    return matmulProj_->forward(attnOutBuf_->colorTexture(), 1, dModel_,
                                *wo, dModel_, dModel_, outputFBO);
}

bool SelfAttention::forwardCpu(const gl::Texture& inputTex, int seqLen,
                               const gl::Texture* wqkv, const gl::Texture* bqkv,
                               const gl::Texture* wo,
                               float attnScale, gl::FBO& outputFBO)
{
    // 1. QKV = input * Wqkv  ->  [1, 3*d_model] = [q | k | v]
    if (!matmulQkv_->forward(inputTex, 1, dModel_,
                             *wqkv, dModel_, 3 * dModel_,
                             *qkvBuf_)) {
        return false;
    }

    // A single-row RGBA32F texture reads back as one contiguous float array.
    readbackBuf_.resize(qkvBuf_->colorTexture().elements());
    qkvBuf_->colorTexture().download(readbackBuf_.data());

    if (bqkv) {
        biasBuf_.resize(bqkv->width() * bqkv->height() * 4);
        bqkv->download(biasBuf_.data());
        for (int i = 0; i < 3 * dModel_; i++) {
            readbackBuf_[i] += biasBuf_[i];
        }
    }

    const float* q = readbackBuf_.data();
    const float* k = readbackBuf_.data() + dModel_;
    const float* v = readbackBuf_.data() + 2 * dModel_;

    // 2. Append this token's K and V to the cache at position seqLen-1
    const int pos = seqLen - 1;
    std::copy(k, k + dModel_, kCache_.begin() + (size_t)pos * dModel_);
    std::copy(v, v + dModel_, vCache_.begin() + (size_t)pos * dModel_);

    // 3. Per-head attention over the cached positions visible to this token:
    //    global: 0..pos;  local: the last localWindow_ positions ending at pos
    const int first = (localWindow_ > 0 && seqLen > localWindow_) ? seqLen - localWindow_ : 0;
    std::vector<float> attnOut(dModel_, 0.0f);
    for (int h = 0; h < nHeads_; h++) {
        const int off = h * dHead_;

        float maxScore = -1e30f;
        for (int i = first; i < seqLen; i++) {
            const float* ki = &kCache_[(size_t)i * dModel_ + off];
            float dot = 0.0f;
            for (int j = 0; j < dHead_; j++) {
                dot += q[off + j] * ki[j];
            }
            scores_[i] = dot * attnScale;
            if (scores_[i] > maxScore) maxScore = scores_[i];
        }

        double sumExp = 0.0;
        for (int i = first; i < seqLen; i++) {
            scores_[i] = std::exp(scores_[i] - maxScore);
            sumExp += scores_[i];
        }
        const float invSum = 1.0f / (float)sumExp;

        for (int i = first; i < seqLen; i++) {
            const float p = scores_[i] * invSum;
            const float* vi = &vCache_[(size_t)i * dModel_ + off];
            for (int j = 0; j < dHead_; j++) {
                attnOut[off + j] += p * vi[j];
            }
        }
    }

    // 4. Upload the attention result and apply the output projection on the GPU
    const int tw = (dModel_ + 3) / 4;
    std::vector<float> packed(tw * 4, 0.0f);
    std::copy(attnOut.begin(), attnOut.end(), packed.begin());
    attnOutBuf_->colorTexture().upload(packed.data());

    return matmulProj_->forward(attnOutBuf_->colorTexture(), 1, dModel_,
                                *wo, dModel_, dModel_, outputFBO);
}

void SelfAttention::resetCache() {
    // The GPU cache needs no clearing: only positions < seqLen are ever read, and every
    // position is rewritten (prefill) before it becomes visible.
    std::fill(kCache_.begin(), kCache_.end(), 0.0f);
    std::fill(vCache_.begin(), vCache_.end(), 0.0f);
}

bool SelfAttention::forwardFused(const gl::Texture& xTex, int seqLen,
                                 const gl::Texture* lnStats, const gl::Texture* lnGain,
                                 const gl::Texture* lnBias,
                                 const gl::Texture* wqkv, const gl::Texture* bqkv,
                                 const gl::Texture* wo, const gl::Texture* bo,
                                 float attnScale, const gl::Texture& residual,
                                 gl::FBO& outputFBO)
{
    if (!gpu_ || !matmulQkv_->fusedAvailable() || !matmulProj_->fusedAvailable()) return false;
    if (seqLen < 1 || seqLen > maxSeqLen_) {
        std::fprintf(stderr, "[mmllm] Attention: sequence length %d out of range (max %d)\n",
                     seqLen, maxSeqLen_);
        return false;
    }

    const int tw = (dModel_ + 3) / 4;
    const int headTexels = dHead_ / 4;
    const int pos = seqLen - 1;
    const int first = (localWindow_ > 0 && seqLen > localWindow_) ? seqLen - localWindow_ : 0;

    // 1. [q | k | v] = LN(x) * Wqkv + bqkv  (LayerNorm and bias folded in)
    MatMul::FusedOps qkvOps;
    qkvOps.lnStats = lnStats;
    qkvOps.lnGain = lnGain;
    qkvOps.lnBias = lnBias;
    qkvOps.bias = bqkv;
    if (!matmulQkv_->forwardFused(xTex, 1, dModel_, *wqkv, dModel_, 3 * dModel_, *qkvBuf_, qkvOps)) {
        return false;
    }

    // 2. KV append, scores, softmax stats, weighted sum of V
    attendGpu(&qkvBuf_->colorTexture(), seqLen, pos, first, headTexels, tw, attnScale);

    // 3. out = residual + attnOut * Wo + bo
    MatMul::FusedOps outOps;
    outOps.bias = bo;
    outOps.residual = &residual;
    return matmulProj_->forwardFused(attnOutBuf_->colorTexture(), 1, dModel_,
                                     *wo, dModel_, dModel_, outputFBO, outOps);
}
