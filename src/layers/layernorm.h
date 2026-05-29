#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>

// Layer Normalization: LN(x) = gain * (x - mean) / sqrt(var + eps) + bias
// Mean and variance are computed on CPU from a texture readback.
class LayerNorm {
public:
    LayerNorm();
    ~LayerNorm() = default;

    bool init();

    // Normalize a [1, D] vector, reading back to CPU for mean/var computation
    bool forward(const gl::Texture& texInput, int dModel,
                 const gl::Texture& texGain, const gl::Texture& texBias,
                 float epsilon, gl::FBO& outputFBO);

private:
    std::unique_ptr<gl::Program> lnProg_;
    std::unique_ptr<gl::Shader> vertShader_;
    std::unique_ptr<gl::Shader> fragShader_;

    // CPU-side buffer for reading back the input vector
    std::vector<float> readbackBuf_;
};
