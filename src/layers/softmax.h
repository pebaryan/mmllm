#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>

// Softmax normalization for a [1, N] vector.
// Reads back from GPU to compute max/sum on CPU, then dispatches normalization pass.
class Softmax {
public:
    Softmax();
    ~Softmax() = default;

    bool init();

    // Compute softmax in-place on a [1, N] vector
    bool forward(gl::Texture& texInput, int N, gl::FBO& outputFBO);

private:
    std::unique_ptr<gl::Program> softmaxProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;
    std::vector<float> readbackBuf_;
};
