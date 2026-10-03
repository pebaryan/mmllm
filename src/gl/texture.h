#pragma once
#include <GL/glew.h>
#include <vector>
#include <cstdint>

namespace gl {

// When true, the GL layer stays silent on stdout. The Needle CLI writes its JSON result to stdout,
// so the GPU path turns this on (its logs would otherwise corrupt the output).
extern bool quiet;

enum class TextureFormat {
    R32F,     // Single-channel 32-bit float
    RGBA32F,  // Four-channel 32-bit float
    RGBA16F,  // Four-channel 16-bit float (half the memory; uploads still take 32-bit floats)
    R8UI,     // Single-channel 8-bit unsigned integer (usampler2D; packed 2-bit weights)
    RGBA8UI   // Four-channel 8-bit unsigned integer (4 bytes/texel: 16 two-bit weights)
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

    // Upload raw bytes for integer formats (R8UI: one byte per texel, row-major)
    void uploadBytes(const uint8_t* data);

    // Download float data back to CPU
    void download(float* data) const;

    // Download raw bytes (R8UI only)
    void downloadBytes(uint8_t* data) const;

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
    int channels_ = 4;  // 1 for R32F, 4 for RGBA32F, 1 for R8UI
    bool integer_ = false;   // true for R8UI (integer sampler, byte uploads)
    GLenum internalFormat_ = GL_RGBA32F;
    GLenum format_ = GL_RGBA;
    GLenum type_ = GL_FLOAT;
};

// Helper: compute packed texture dimensions for a matrix [rows, cols]
// Returns (tex_width, tex_height) where each texel packs `packing` elements
inline void packedSize(int rows, int cols, int packing, int& outW, int& outH) {
    outW = (cols + packing - 1) / packing;
    outH = rows;
}

} // namespace gl
