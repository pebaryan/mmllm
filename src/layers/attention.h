#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include "../gl/shader.h"
#include "matmul.h"
#include "elementwise.h"
#include <memory>
#include <vector>

// Multi-head causal self-attention for one token, using a KV cache.
//   per head h:  scores_i = scale * q_h . k_h[i]  (i visible to the current position)
//                out_h    = softmax(scores) * V_h
//
// GPU path (default): the KV cache lives in a GPU texture and attention runs as five
// fragment-shader passes with no CPU readback:
//   QKV matmul -> (bias) -> append K|V to the cache -> scores -> softmax stats
//   -> softmax-weighted sum of V -> output projection matmul
// It needs d_head to be a multiple of 4. Otherwise (or with MMLLM_CPU_ATTN=1 in the
// environment) the original CPU implementation with a CPU KV cache is used.
class SelfAttention {
public:
    SelfAttention();
    ~SelfAttention() = default;

    bool init(int dModel, int nHeads, int maxSeqLen);

    // Forward pass for a single token at position seqLen-1.
    // inputTex: [1, d_model] - the normalized input (after LayerNorm)
    // bqkv: optional [1, 3*d_model] bias for the fused QKV projection (may be null)
    // The output projection bias is applied by the caller.
    bool forward(const gl::Texture& inputTex,
                 int seqLen,
                 const gl::Texture* wqkv,
                 const gl::Texture* bqkv,
                 const gl::Texture* wo,
                 float attnScale,
                 gl::FBO& outputFBO);

    // Reset KV cache (for a new sequence)
    void resetCache();

    // Local attention: only the last `window` positions are visible (0 = global).
    void setLocalWindow(int window) { localWindow_ = window; }

    bool usesGpu() const { return gpu_; }

    // Fused variant (GPU only): takes the RAW residual stream x and folds the pre-attention
    // LayerNorm (lnStats/lnGain/lnBias), the QKV bias, the output-projection bias and the
    // residual add into the two projection matmuls:
    //     out = residual + Attn(LN(x)) * Wo + bo
    bool forwardFused(const gl::Texture& xTex, int seqLen,
                      const gl::Texture* lnStats, const gl::Texture* lnGain, const gl::Texture* lnBias,
                      const gl::Texture* wqkv, const gl::Texture* bqkv,
                      const gl::Texture* wo, const gl::Texture* bo,
                      float attnScale, const gl::Texture& residual, gl::FBO& outputFBO);

private:
    bool forwardGpu(const gl::Texture& inputTex, int seqLen,
                    const gl::Texture* wqkv, const gl::Texture* bqkv,
                    const gl::Texture* wo, float attnScale, gl::FBO& outputFBO);
    bool forwardCpu(const gl::Texture& inputTex, int seqLen,
                    const gl::Texture* wqkv, const gl::Texture* bqkv,
                    const gl::Texture* wo, float attnScale, gl::FBO& outputFBO);
    bool initGpuPasses();
    // KV append + scores + softmax stats + weighted sum of V, from a QKV row texture
    void attendGpu(const gl::Texture* qkv, int seqLen, int pos, int first,
                   int headTexels, int tw, float attnScale);
    std::unique_ptr<gl::Program> makeProgram(const char* fragPath,
                                             std::vector<std::unique_ptr<gl::Shader>>& keep);

    int dModel_ = 0;
    int nHeads_ = 0;
    int dHead_ = 0;
    int maxSeqLen_ = 0;
    int localWindow_ = 0;
    bool gpu_ = false;

    // CPU KV cache (CPU path only): [max_seq_len, d_model] each, row-major
    std::vector<float> kCache_;
    std::vector<float> vCache_;

    // Sub-layers
    std::unique_ptr<MatMul> matmulQkv_;    // fused Q/K/V projection
    std::unique_ptr<MatMul> matmulProj_;   // output projection
    std::unique_ptr<ElementwiseAdd> add_;  // QKV bias

    // GPU passes
    std::vector<std::unique_ptr<gl::Shader>> shaders_;
    std::unique_ptr<gl::Program> appendProg_;
    std::unique_ptr<gl::Program> scoresProg_;
    std::unique_ptr<gl::Program> statsProg_;
    std::unique_ptr<gl::Program> avProg_;

    // Temporary FBOs
    std::shared_ptr<gl::FBO> qkvBuf_;      // [1, 3*d_model]
    std::shared_ptr<gl::FBO> qkvBiasBuf_;  // [1, 3*d_model] (+ bqkv)
    std::shared_ptr<gl::FBO> attnOutBuf_;  // [1, d_model] attention result (pre-projection)
    std::shared_ptr<gl::FBO> kvBuf_;       // GPU KV cache: width 2*T, row = position
    std::shared_ptr<gl::FBO> scoresBuf_;   // (position, head) scores
    std::shared_ptr<gl::FBO> statsBuf_;    // (0, head) -> max, 1/sum

    // Scratch buffers (CPU path)
    std::vector<float> readbackBuf_;
    std::vector<float> biasBuf_;
    std::vector<float> scores_;
};
