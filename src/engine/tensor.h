#pragma once
#include "../gl/texture.h"
#include "../gl/fbo.h"
#include "../gl/program.h"
#include "model_config.h"
#include <vector>
#include <memory>
#include <functional>

// A GPU-resident tensor backed by an FBO for render-to-texture.
// Used for activations during inference.
class GPUTensor {
public:
    GPUTensor() = default;

    // Create a new FBO-backed tensor
    bool create(int rows, int cols, const std::string& debugName = "tensor");

    // Upload packed data directly to the texture
    void upload(const float* data, int rows, int cols);

    // Download texture contents back to CPU (unpacks RGBA -> linear)
    void download(float* data) const;

    // Zero-initialize
    void clear(const gl::Program& clearProg);

    // Access the underlying FBO and texture
    std::shared_ptr<gl::FBO> fbo() const { return fbo_; }
    gl::Texture& texture() { return fbo_->colorTexture(); }
    const gl::Texture& texture() const { return fbo_->colorTexture(); }

    int rows() const { return rows_; }
    int cols() const { return cols_; }

    // Number of padded (packed) columns in texture
    int packedCols() const { return (cols_ + 3) / 4; }

    bool valid() const { return fbo_ != nullptr && fbo_->colorTexture().id() != 0; }

private:
    std::shared_ptr<gl::FBO> fbo_;
    int rows_ = 0;
    int cols_ = 0;
    bool owned_ = false;
    std::string debugName_;
};

// Describes a texture bound to a specific unit with its sampler uniform name.
struct TextureBinding {
    int unit;                // texture unit (0..GL_MAX_COMBINED_TEXTURE_IMAGE_UNITS)
    const gl::Texture* tex;  // texture to bind
    const char* sampler;     // GLSL sampler uniform name (e.g. "texA", "texInput")
};

// Helper: dispatch a full-screen quad shader pass
void dispatchShader(
    gl::Program& program,
    gl::FBO& target,
    const std::vector<TextureBinding>& textures,
    std::function<void(gl::Program&)> setUniforms = {});

// Helper: create a standalone FBO for use as a render target
std::shared_ptr<gl::FBO> createRenderTarget(int rows, int cols,
                                             const std::string& debugName = "target",
                                             gl::TextureFormat fmt = gl::TextureFormat::RGBA32F);
