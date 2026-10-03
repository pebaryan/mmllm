#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>

// Layer Normalization: LN(x) = gain * (x - mean) / sqrt(var + eps) + bias
//
// GPU path (default): a stats pass renders mean and 1/sqrt(var+eps) into a 1x1 texture and
// the apply pass reads it, so nothing is read back to the CPU and the GPU pipeline never
// stalls. With MMLLM_CPU_LN=1 in the environment the original path is used: the input is
// read back and mean/variance are computed on the CPU.
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm() = default;

    bool init();

    // Normalize a [1, D] vector
    bool forward(const gl::Texture& texInput, int dModel,
                 const gl::Texture& texGain, const gl::Texture& texBias,
                 float epsilon, gl::FBO& outputFBO);

    bool usesGpu() const { return gpu_; }

    // GPU path only: compute mean and 1/sqrt(var+eps) of texInput into a 1x1 target
    // (.r = mean, .g = invStd). The apply step is then fused into a following matmul.
    bool computeStats(const gl::Texture& texInput, int dModel, float epsilon, gl::FBO& statsFBO);

private:
    bool forwardGpu(const gl::Texture& texInput, int dModel,
                    const gl::Texture& texGain, const gl::Texture& texBias,
                    float epsilon, gl::FBO& outputFBO);

    bool gpu_ = false;
    std::unique_ptr<gl::Program> statsProg_;
    std::unique_ptr<gl::Program> applyProg_;
    std::unique_ptr<gl::Shader> statsFrag_;
    std::unique_ptr<gl::Shader> applyFrag_;
    std::shared_ptr<gl::FBO> statsBuf_;   // 1x1: mean, invStd

    std::unique_ptr<gl::Program> lnProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;

    // CPU-side buffer for reading back the input vector
    std::vector<float> readbackBuf_;
};
