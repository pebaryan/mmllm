#pragma once
#include <GL/glew.h>
#include "texture.h"
#include <cstdint>

namespace gl {

class FBO {
public:
    FBO() = default;
    ~FBO();

    FBO(const FBO&) = delete;
    FBO& operator=(const FBO&) = delete;

    FBO(FBO&& other) noexcept;
    FBO& operator=(FBO&& other) noexcept;

    // Create FBO with a color attachment texture of given size
    bool create(int width, int height, TextureFormat fmt = TextureFormat::RGBA32F);

    // Bind as render target
    void bind() const;
    void unbind() const;

    // Access the color attachment
    const Texture& colorTexture() const { return colorTex_; }
    Texture& colorTexture() { return colorTex_; }

    // Resize the FBO (destroys and recreates)
    bool resize(int width, int height);

    void destroy();

private:
    uint32_t id_ = 0;
    Texture colorTex_;
    int width_ = 0;
    int height_ = 0;
};

} // namespace gl
