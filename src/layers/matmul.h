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

    // Optional work folded into the same pass (see matmul_fused.frag). All pointers may be null.
    struct FusedOps {
        // Prologue: layer-normalise A on the fly (M must be 1). Needs all three.
        const gl::Texture* lnStats = nullptr;   // [1,1]: mean, invStd
        const gl::Texture* lnGain = nullptr;
        const gl::Texture* lnBias = nullptr;
        // Epilogue, applied in this order: bias, GELU, residual
        const gl::Texture* bias = nullptr;      // [1, N] vector
        bool gelu = false;
        const gl::Texture* residual = nullptr;  // same shape as the output; not the output itself
    };

    // C = epilogue(prologue(A) * B)
    bool forwardFused(const gl::Texture& texA, int rowsA, int colsA,
                      const gl::Texture& texB, int rowsB, int colsB,
                      gl::FBO& outputFBO, const FusedOps& ops);
    bool fusedAvailable() const { return fusedProg_ && fusedProg_->valid(); }

    // Convenience: compute and return a new tensor
    GPUTensor forward(const GPUTensor& a, const GPUTensor& b);

private:
    std::unique_ptr<gl::Program> matmulProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;

    std::unique_ptr<gl::Program> fusedProg_;
    std::unique_ptr<gl::Shader> fusedFrag_;
};
