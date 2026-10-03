#include "layernorm.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cmath>
#include <cstdio>
#include <cstdlib>

LayerNorm::LayerNorm() {}

bool LayerNorm::init() {
    vertShader_ = std::make_unique<gl::Shader>();
    if (!vertShader_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) {
        return false;
    }

    fragShader_ = std::make_unique<gl::Shader>();
    if (!fragShader_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/layernorm.frag")) {
        return false;
    }

    lnProg_ = std::make_unique<gl::Program>();
    if (!lnProg_->link(*vertShader_, *fragShader_)) {
        return false;
    }

    // GPU statistics path (unless the CPU path is requested)
    if (!std::getenv("MMLLM_CPU_LN")) {
        statsFrag_ = std::make_unique<gl::Shader>();
        applyFrag_ = std::make_unique<gl::Shader>();
        if (statsFrag_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/layernorm_stats.frag") &&
            applyFrag_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/layernorm_gpu.frag")) {
            statsProg_ = std::make_unique<gl::Program>();
            applyProg_ = std::make_unique<gl::Program>();
            statsBuf_ = createRenderTarget(1, 4, "ln_stats");   // 1x1 texel
            gpu_ = statsProg_->link(*vertShader_, *statsFrag_) &&
                   applyProg_->link(*vertShader_, *applyFrag_) && statsBuf_;
        }
        if (!gpu_) {
            std::fprintf(stderr, "[mmllm] GPU LayerNorm failed to initialize; using CPU statistics\n");
        }
    }

    std::printf("[mmllm] LayerNorm layer initialized (%s statistics)\n", gpu_ ? "GPU" : "CPU");
    return true;
}

bool LayerNorm::forward(const gl::Texture& texInput, int dModel,
                        const gl::Texture& texGain, const gl::Texture& texBias,
                        float epsilon, gl::FBO& outputFBO)
{
    if (gpu_) return forwardGpu(texInput, dModel, texGain, texBias, epsilon, outputFBO);
    if (!lnProg_ || !lnProg_->valid()) return false;

    // Read back the input vector to compute mean and variance
    int texW = texInput.width();
    int texH = texInput.height();
    readbackBuf_.resize(texW * texH * 4);
    texInput.download(readbackBuf_.data());

    // Compute mean
    double sum = 0.0;
    int count = dModel < texW * texH * 4 ? dModel : texW * texH * 4;
    for (int i = 0; i < count; i++) {
        sum += readbackBuf_[i];
    }
    float mean = static_cast<float>(sum / dModel);

    // Compute variance
    double varSum = 0.0;
    for (int i = 0; i < count; i++) {
        float diff = readbackBuf_[i] - mean;
        varSum += diff * diff;
    }
    float variance = static_cast<float>(varSum / dModel);
    float invStd = 1.0f / std::sqrt(variance + epsilon);

    // Dispatch normalization shader
    dispatchShader(*lnProg_, outputFBO,
        {
            {0, &texInput, "texInput"},
            {1, &texGain, "texGain"},
            {2, &texBias, "texBias"}
        },
        [&](gl::Program& prog) {
            prog.setFloat("mean", mean);
            prog.setFloat("invStd", invStd);
            prog.setInt("N", dModel);
        });

    return true;
}

bool LayerNorm::forwardGpu(const gl::Texture& texInput, int dModel,
                           const gl::Texture& texGain, const gl::Texture& texBias,
                           float epsilon, gl::FBO& outputFBO)
{
    // 1. Statistics: one fragment loops over the vector -> 1x1 texture (mean, invStd)
    dispatchShader(*statsProg_, *statsBuf_,
        {{0, &texInput, "texInput"}},
        [&](gl::Program& prog) {
            prog.setInt("N", dModel);
            prog.setFloat("epsilon", epsilon);
        });

    // 2. Normalize using the statistics texture
    dispatchShader(*applyProg_, outputFBO,
        {
            {0, &texInput, "texInput"},
            {1, &texGain, "texGain"},
            {2, &texBias, "texBias"},
            {3, &statsBuf_->colorTexture(), "texStats"}
        },
        [&](gl::Program& prog) {
            prog.setInt("N", dModel);
        });

    return true;
}

bool LayerNorm::computeStats(const gl::Texture& texInput, int dModel, float epsilon,
                             gl::FBO& statsFBO)
{
    if (!gpu_) return false;
    dispatchShader(*statsProg_, statsFBO,
        {{0, &texInput, "texInput"}},
        [&](gl::Program& prog) {
            prog.setInt("N", dModel);
            prog.setFloat("epsilon", epsilon);
        });
    return true;
}
