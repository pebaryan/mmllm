#include "elementwise.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cstdio>

// === ElementwiseAdd ===

ElementwiseAdd::ElementwiseAdd() {}

bool ElementwiseAdd::init() {
    vertShader_ = std::make_unique<gl::Shader>();
    if (!vertShader_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert"))
        return false;

    fragShader_ = std::make_unique<gl::Shader>();
    if (!fragShader_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/add.frag"))
        return false;

    addProg_ = std::make_unique<gl::Program>();
    if (!addProg_->link(*vertShader_, *fragShader_)) return false;

    std::printf("[mmllm] ElementwiseAdd layer initialized\n");
    return true;
}

bool ElementwiseAdd::forward(const gl::Texture& texA, const gl::Texture& texB,
                             int texWidth, int texHeight, int totalElements,
                             gl::FBO& outputFBO) {
    if (!addProg_ || !addProg_->valid()) return false;

    dispatchShader(*addProg_, outputFBO,
        {
            {0, &texA, "texA"},
            {1, &texB, "texB"}
        },
        [&](gl::Program& prog) {
            prog.setVec2("texSize", texWidth, texHeight);
            prog.setInt("N", totalElements);
        });

    return true;
}

// === ElementwiseScale ===

ElementwiseScale::ElementwiseScale() {}

bool ElementwiseScale::init() {
    vertShader_ = std::make_unique<gl::Shader>();
    if (!vertShader_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert"))
        return false;

    fragShader_ = std::make_unique<gl::Shader>();
    if (!fragShader_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/scale.frag"))
        return false;

    scaleProg_ = std::make_unique<gl::Program>();
    if (!scaleProg_->link(*vertShader_, *fragShader_)) return false;

    std::printf("[mmllm] ElementwiseScale layer initialized\n");
    return true;
}

bool ElementwiseScale::forward(const gl::Texture& texInput, float scalar,
                               const float bias[4], int N, gl::FBO& outputFBO) {
    if (!scaleProg_ || !scaleProg_->valid()) return false;

    dispatchShader(*scaleProg_, outputFBO,
        {
            {0, &texInput, "texA"}
        },
        [&](gl::Program& prog) {
            prog.setFloat("scalar", scalar);
            prog.setInt("N", N);
            // Bias as vec4 uniform (no wrapper in Program yet)
            GLint loc = glGetUniformLocation(prog.id(), "bias");
            glUniform4f(loc, bias[0], bias[1], bias[2], bias[3]);
        });

    return true;
}
