#pragma once
// GPU backend for the Needle 3 matrix-vector products.
//
// The 2-bit CQ weights stay packed exactly as the archive stores them (an R8UI texture, one byte per
// texel = 4 two-bit indices) and are decoded in the fragment shader with the same codebook and
// per-128-group norms the CPU path uses, so no weight ever has to be expanded to float32. That keeps
// VRAM use at ~2 bits/weight, which is what makes a 121M-parameter model fit a GeForce 9400M.
//
// The activation is rotated with the normalised Walsh-Hadamard transform on the CPU (identical to the
// reference path) and uploaded per call as a 4-float-per-texel RGBA32F texture; the result is read
// back once per matrix-vector product.
//
// This path is numerically verified against the CPU implementation by Model::gpuSelfCheck(); on the
// target hardware it is SLOWER than the CPU (see NEEDLE.md), so it is opt-in via --gpu.
#include "../gl/fbo.h"
#include "../gl/program.h"
#include "../gl/shader.h"
#include "../gl/texture.h"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace needle {

class NGpu {
public:
    NGpu();
    ~NGpu();

    // Requires a current OpenGL 3.3 context. Loads the shaders and clears the codebook.
    bool init(std::string& err);
    void shutdown();
    bool ok() const { return ok_; }
    const std::string& error() const { return err_; }

    // 2-bit CQ matrix [out, in] (in must be a multiple of 128). `packed` is out*(in/4) bytes,
    // `norms` is out*(in/128) floats. Returns a handle >= 0, or -1 on failure.
    int uploadQuant(const uint8_t* packed, const float* norms, int out, int in);

    // 2-bit codebook (4 values) used to decode the packed indices.
    void setCodebook(const float* cb);

    // y[out] = W x, where x is the group-rotated activation (in floats) for a quantized matrix.
    bool matvec(int handle, const float* x, float* y);

    long long calls = 0;      // number of GPU matrix-vector products performed
    long long uploads = 0;    // number of weight matrices uploaded

private:
    struct Entry {
        int out = 0, in = 0;
        int kg = 0;                    // in / 128
        gl::Texture w;                 // R8UI  [in/4 x out]
        gl::Texture n;                 // R32F  [kg   x out]
    };

    bool beginActivation(int in);
    bool ensurePartials(int kg, int out);
    bool ensureOutput(int out);
    bool loadProgram(gl::Program& prog, const char* frag, const std::string& errPrefix);

    std::vector<Entry> entries_;
    std::unique_ptr<gl::Shader> vert_;
    std::unique_ptr<gl::Program> partial_, reduce_;
    gl::Texture act_;                                  // RGBA32F [in/4 x 1]
    int actIn_ = 0;
    std::unique_ptr<gl::FBO> part_, outFbo_;
    int partKg_ = 0, partRows_ = 0, outRows_ = 0;
    std::vector<float> outBuf_;
    float cb_[4] = {0.f, 0.f, 0.f, 0.f};
    bool ok_ = false;
    std::string err_;
};

} // namespace needle
