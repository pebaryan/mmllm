#include "layernorm.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cmath>
#include <cstdio>

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

    std::printf("[mmllm] LayerNorm layer initialized\n");
    return true;
}

bool LayerNorm::forward(const gl::Texture& texInput, int dModel,
                        const gl::Texture& texGain, const gl::Texture& texBias,
                        float epsilon, gl::FBO& outputFBO)
{
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
