#include "texture.h"
#include <GL/glew.h>
#include <cstdio>
#include <utility>
#include <cstring>

namespace gl {

Texture::~Texture() {
    destroy();
}

bool Texture::create(int width, int height, TextureFormat fmt) {
    destroy();

    width_ = width;
    height_ = height;

    switch (fmt) {
        case TextureFormat::R32F:
            channels_ = 1;
            internalFormat_ = GL_R32F;
            format_ = GL_RED;
            break;
        case TextureFormat::RGBA32F:
            channels_ = 4;
            internalFormat_ = GL_RGBA32F;
            format_ = GL_RGBA;
            break;
    }

    glGenTextures(1, &id_);
    glBindTexture(GL_TEXTURE_2D, id_);

    // Allocate texture storage
    glTexImage2D(GL_TEXTURE_2D, 0, internalFormat_, width_, height_, 0,
                 format_, GL_FLOAT, nullptr);

    // Set sampling parameters — we use texelFetch so no filtering
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

    glBindTexture(GL_TEXTURE_2D, 0);

    std::printf("[mmllm] Created texture %u: %dx%d (%d channels, %s)\n",
        id_, width_, height_, channels_,
        fmt == TextureFormat::RGBA32F ? "RGBA32F" : "R32F");

    return true;
}

void Texture::upload(const float* data) {
    if (!id_) return;
    glBindTexture(GL_TEXTURE_2D, id_);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, width_, height_,
                    format_, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::download(float* data) const {
    if (!id_) return;
    glBindTexture(GL_TEXTURE_2D, id_);
    glGetTexImage(GL_TEXTURE_2D, 0, format_, GL_FLOAT, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::bind(int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, id_);
}

void Texture::unbind(int unit) const {
    glActiveTexture(GL_TEXTURE0 + unit);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void Texture::destroy() {
    if (id_) {
        glDeleteTextures(1, &id_);
        id_ = 0;
    }
    width_ = height_ = 0;
    channels_ = 4;
}

Texture::Texture(Texture&& other) noexcept
    : id_(other.id_)
    , width_(other.width_)
    , height_(other.height_)
    , channels_(other.channels_)
    , internalFormat_(other.internalFormat_)
    , format_(other.format_)
{
    other.id_ = 0;
    other.width_ = other.height_ = 0;
}

Texture& Texture::operator=(Texture&& other) noexcept {
    if (this != &other) {
        destroy();
        id_ = other.id_;
        width_ = other.width_;
        height_ = other.height_;
        channels_ = other.channels_;
        internalFormat_ = other.internalFormat_;
        format_ = other.format_;
        other.id_ = 0;
        other.width_ = other.height_ = 0;
    }
    return *this;
}

} // namespace gl
