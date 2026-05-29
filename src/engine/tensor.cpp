#include "tensor.h"
#include "../gl/texture.h"
#include "../gl/fbo.h"
#include "../gl/program.h"
#include "../gl/shader.h"
#include <GL/glew.h>
#include <cstdio>

bool GPUTensor::create(int rows, int cols, const std::string& debugName) {
    rows_ = rows;
    cols_ = cols;
    debugName_ = debugName;

    int tw = packedWidth(cols);
    int th = rows;

    auto fbo = std::make_shared<gl::FBO>();
    if (!fbo->create(tw, th, gl::TextureFormat::RGBA32F)) {
        std::fprintf(stderr, "[mmllm] Failed to create tensor FBO '%s': %dx%d\n",
                     debugName.c_str(), tw, th);
        return false;
    }

    fbo_ = std::move(fbo);
    owned_ = true;
    return true;
}

void GPUTensor::upload(const float* data, int rows, int cols) {
    rows_ = rows;
    cols_ = cols;

    int tw = packedWidth(cols);
    int th = rows;

    // Recreate FBO if dimensions changed
    if (!fbo_ || fbo_->colorTexture().width() != tw || fbo_->colorTexture().height() != th) {
        create(rows, cols, debugName_);
    }

    // Pack data into RGBA32F format (already packed on CPU)
    fbo_->colorTexture().upload(data);
}

void GPUTensor::download(float* data) const {
    if (!fbo_) return;
    fbo_->colorTexture().download(data);
}

void GPUTensor::clear(const gl::Program& clearProg) {
    if (!fbo_) return;
    fbo_->bind();

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    fbo_->unbind();
}

void dispatchShader(
    gl::Program& program,
    gl::FBO& target,
    const std::vector<TextureBinding>& textures,
    std::function<void(gl::Program&)> setUniforms)
{
    target.bind();
    program.use();

    // Bind all textures and set their sampler uniforms
    for (auto& tb : textures) {
        if (tb.tex) {
            tb.tex->bind(tb.unit);
            // Tell the shader which texture unit this sampler uses
            GLint loc = glGetUniformLocation(program.id(), tb.sampler);
            if (loc >= 0) {
                glUniform1i(loc, tb.unit);
            }
        }
    }

    // Set additional uniforms
    if (setUniforms) {
        setUniforms(program);
    }

    // Draw full-screen quad (4 vertices, triangle strip)
    glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);

    // Unbind textures
    for (auto& tb : textures) {
        if (tb.tex) tb.tex->unbind(tb.unit);
    }

    program.unuse();
    target.unbind();
}

std::shared_ptr<gl::FBO> createRenderTarget(int rows, int cols, const std::string& debugName) {
    int tw = packedWidth(cols);
    auto fbo = std::make_shared<gl::FBO>();
    if (!fbo->create(tw, rows, gl::TextureFormat::RGBA32F)) {
        std::fprintf(stderr, "[mmllm] Failed to create render target '%s': %dx%d\n",
                     debugName.c_str(), tw, rows);
        return nullptr;
    }
    return fbo;
}
