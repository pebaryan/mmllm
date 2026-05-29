#include "gelu.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include <cstdio>

GELU::GELU() {}

bool GELU::init() {
    vertShader_ = std::make_unique<gl::Shader>();
    if (!vertShader_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) {
        return false;
    }

    fragShader_ = std::make_unique<gl::Shader>();
    if (!fragShader_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/gelu.frag")) {
        return false;
    }

    geluProg_ = std::make_unique<gl::Program>();
    if (!geluProg_->link(*vertShader_, *fragShader_)) {
        return false;
    }

    std::printf("[mmllm] GELU layer initialized\n");
    return true;
}

bool GELU::forward(const gl::Texture& texInput, int N, gl::FBO& outputFBO) {
    if (!geluProg_ || !geluProg_->valid()) return false;

    dispatchShader(*geluProg_, outputFBO,
        {
            {0, &texInput, "texInput"}
        },
        [&](gl::Program& prog) {
            prog.setInt("N", N);
        });

    return true;
}
