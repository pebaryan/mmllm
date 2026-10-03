#include "fbo.h"
#include <GL/glew.h>
#include <cstdio>
#include <utility>

namespace gl {

FBO::~FBO() {
    destroy();
}

bool FBO::create(int width, int height, TextureFormat fmt) {
    destroy();

    width_ = width;
    height_ = height;

    // Create color texture attachment
    if (!colorTex_.create(width_, height_, fmt)) {
        std::fprintf(stderr, "[mmllm] FBO: failed to create color texture\n");
        return false;
    }

    // Create FBO
    glGenFramebuffers(1, &id_);
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0,
                           GL_TEXTURE_2D, colorTex_.id(), 0);

    // Enable the color attachment for drawing (required in core profile)
    GLenum drawBufs[] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers(1, drawBufs);

    // Check completeness
    GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);

    if (status != GL_FRAMEBUFFER_COMPLETE) {
        std::fprintf(stderr, "[mmllm] FBO incomplete: 0x%x\n", status);
        destroy();
        return false;
    }

    if (!quiet) std::printf("[mmllm] Created FBO %u: %dx%d\n", id_, width_, height_);
    return true;
}

void FBO::bind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, id_);
    glViewport(0, 0, width_, height_);
}

void FBO::unbind() const {
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

bool FBO::resize(int width, int height) {
    return create(width, height);
}

void FBO::destroy() {
    if (id_) {
        glDeleteFramebuffers(1, &id_);
        id_ = 0;
    }
    colorTex_.destroy();
    width_ = height_ = 0;
}

FBO::FBO(FBO&& other) noexcept
    : id_(other.id_)
    , colorTex_(std::move(other.colorTex_))
    , width_(other.width_)
    , height_(other.height_)
{
    other.id_ = 0;
    other.width_ = other.height_ = 0;
}

FBO& FBO::operator=(FBO&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        colorTex_ = std::move(other.colorTex_);
        width_ = other.width_;
        height_ = other.height_;
        other.id_ = 0;
        other.width_ = other.height_ = 0;
    }
    return *this;
}

} // namespace gl
