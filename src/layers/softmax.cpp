#include "softmax.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cmath>
#include <algorithm>
#include <cstdio>

Softmax::Softmax() {}

bool Softmax::init() {
    vertShader_ = std::make_unique<gl::Shader>();
    if (!vertShader_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) {
        return false;
    }

    fragShader_ = std::make_unique<gl::Shader>();
    if (!fragShader_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/softmax_apply.frag")) {
        return false;
    }

    softmaxProg_ = std::make_unique<gl::Program>();
    if (!softmaxProg_->link(*vertShader_, *fragShader_)) {
        return false;
    }

    std::printf("[mmllm] Softmax layer initialized\n");
    return true;
}

bool Softmax::forward(gl::Texture& texInput, int N, gl::FBO& outputFBO) {
    if (!softmaxProg_ || !softmaxProg_->valid()) return false;

    // Read back input to compute softmax parameters on CPU
    int texW = texInput.width();
    int texH = texInput.height();
    readbackBuf_.resize(texW * texH * 4);
    texInput.download(readbackBuf_.data());

    // Find max
    float maxVal = -1e20f;
    for (int i = 0; i < N; i++) {
        maxVal = std::max(maxVal, readbackBuf_[i]);
    }

    // Compute sum of exp(x - max)
    double sumExp = 0.0;
    for (int i = 0; i < N; i++) {
        sumExp += std::exp((double)(readbackBuf_[i] - maxVal));
    }

    float invSumExp = 1.0f / (float)sumExp;

    // Dispatch normalization shader
    dispatchShader(*softmaxProg_, outputFBO,
        {
            {0, &texInput, "texInput"}
        },
        [&](gl::Program& prog) {
            prog.setFloat("maxVal", maxVal);
            prog.setFloat("invSumExp", invSumExp);
            prog.setInt("N", N);
        });

    return true;
}
