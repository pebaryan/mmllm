#pragma once
#include <GL/glew.h>
#include <vector>
#include <cstdint>

namespace gl {

enum class TextureFormat {
    R32F,     // Single-channel 32-bit float
    RGBA32F   // Four-channel 32-bit float
};

class Texture {
public:
    Texture() = default;
    ~Texture();

    Texture(const Texture&) = delete;
    Texture& operator=(const Texture&) = delete;

    Texture(Texture&& other) noexcept;
    Texture& operator=(Texture&& other) noexcept;

    // Create a texture with given dimensions and format
    bool create(int width, int height, TextureFormat fmt = TextureFormat::RGBA32F);

    // Upload float data (must match format's channel count)
    void upload(const float* data);

    // Download float data back to CPU
    void download(float* data) const;

    // Get the number of elements (total floats, not texels)
    int elements() const { return width_ * height_ * channels_; }

    int width() const { return width_; }
    int height() const { return height_; }
    int channels() const { return channels_; }
    uint32_t id() const { return id_; }

    // Bind to a texture unit
    void bind(int unit = 0) const;
    void unbind(int unit = 0) const;

    // Destroy the texture
    void destroy();

private:
    uint32_t id_ = 0;
    int width_ = 0;
    int height_ = 0;
    int channels_ = 4;  // 1 for R32F, 4 for RGBA32F
    GLenum internalFormat_ = GL_RGBA32F;
    GLenum format_ = GL_RGBA;
};

// Helper: compute packed texture dimensions for a matrix [rows, cols]
// Returns (tex_width, tex_height) where each texel packs `packing` elements
inline void packedSize(int rows, int cols, int packing, int& outW, int& outH) {
    outW = (cols + packing - 1) / packing;
    outH = rows;
}

} // namespace gl
