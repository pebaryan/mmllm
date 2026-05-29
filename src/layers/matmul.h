#pragma once
#include "../engine/tensor.h"
#include "../engine/model.h"
#include "../gl/program.h"
#include "../gl/texture.h"
#include <memory>

// GPU-accelerated matrix multiplication: C = A * B
// A: [M, K], B: [K, N], C: [M, N]
class MatMul {
public:
    MatMul();
    ~MatMul() = default;

    bool init();

    // Compute C = A * B, writing to an existing FBO or creating a new one
    bool forward(const gl::Texture& texA, int rowsA, int colsA,
                 const gl::Texture& texB, int rowsB, int colsB,
                 gl::FBO& outputFBO);

    // Convenience: compute and return a new tensor
    GPUTensor forward(const GPUTensor& a, const GPUTensor& b);

private:
    std::unique_ptr<gl::Program> matmulProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;
};
