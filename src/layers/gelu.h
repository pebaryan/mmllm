#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>

class GELU {
public:
    GELU();
    ~GELU() = default;

    bool init();

    // Apply GELU activation in-place
    bool forward(const gl::Texture& texInput, int N, gl::FBO& outputFBO);

private:
    std::unique_ptr<gl::Program> geluProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;
};
