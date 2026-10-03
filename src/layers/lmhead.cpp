#include "lmhead.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <GL/glew.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// nouveau on NV50-class GPUs (GeForce 9400M) reports GL_MAX_TEXTURE_SIZE = 8192 but returns
// wrong values when a shader samples a texture wider than 4096 texels. Anything we sample
// (or read back) is kept within this safe dimension.
static const int kSafeTexDim = 4096;

// Tallest table texture verified to sample correctly (a 3142 x 3072 table returned NaN/zeros,
// 3142 x 2048 works). Taller tables are split over several textures.
static const int kSafeTexHeight = 2048;

LMHead::LMHead() {}

bool LMHead::init() {
    vert_ = std::make_unique<gl::Shader>();
    if (!vert_->compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) return false;

    frag_ = std::make_unique<gl::Shader>();
    if (!frag_->compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/lmhead.frag")) return false;

    prog_ = std::make_unique<gl::Program>();
    if (!prog_->link(*vert_, *frag_)) return false;

    std::printf("[mmllm] LMHead layer initialized (GPU)\n");
    return true;
}

bool LMHead::uploadTable(const std::vector<float>& packedRows, int vocabSize, int dModel) {
    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    if (maxTex <= 0) maxTex = 8192;

    vocabSize_ = vocabSize;
    hiddenTexels_ = (dModel + 3) / 4;
    rowsPerTile_ = hiddenTexels_ * 4;       // d_model padded to a multiple of 4
    nTexels_ = (vocabSize + 3) / 4;

    if (rowsPerTile_ > kSafeTexHeight) {
        std::fprintf(stderr, "[mmllm] d_model %d is too large for the GPU LM head (max %d)\n",
                     dModel, kSafeTexHeight);
        return false;
    }

    // Cut the vocabulary (nTexels_ texels wide) into the fewest tiles that fit the safe
    // texture width, then pack as many tiles per texture as stay within the safe height.
    int tileCap = std::min<int>(maxTex, kSafeTexDim);
    if (const char* e = std::getenv("MMLLM_LM_TILE_MAX")) tileCap = std::min<int>(tileCap, std::atoi(e));
    const int tiles = (nTexels_ + tileCap - 1) / tileCap;
    tileW_ = (nTexels_ + tiles - 1) / tiles;
    const int tilesPerTex = std::max(1, kSafeTexHeight / rowsPerTile_);
    const int numTex = (tiles + tilesPerTex - 1) / tilesPerTex;

    const long long needed = (long long)vocabSize * hiddenTexels_ * 4;
    if ((long long)packedRows.size() < needed) {
        std::fprintf(stderr, "[mmllm] Token table data too small (%zu floats, need %lld)\n",
                     packedRows.size(), needed);
        return false;
    }

    half_ = std::getenv("MMLLM_LM_FP32") == nullptr;
    const size_t rowStride = (size_t)hiddenTexels_ * 4;
    double totalMB = 0.0;
    parts_.clear();

    for (int t = 0; t < numTex; t++) {
        const int tileBegin = t * tilesPerTex;
        const int tileEnd = std::min(tiles, tileBegin + tilesPerTex);
        const int tilesHere = tileEnd - tileBegin;
        const int height = tilesHere * rowsPerTile_;

        // Transpose: texel (n % tileW, (tile - tileBegin) * rowsPerTile + k) component c
        //            = table[4n + c][k]
        std::vector<float> data((size_t)tileW_ * height * 4, 0.0f);
        const int vBegin = tileBegin * tileW_ * 4;
        const int vEnd = std::min(vocabSize, tileEnd * tileW_ * 4);
        for (int v = vBegin; v < vEnd; v++) {
            const int n = v >> 2;
            const int c = v & 3;
            const int tile = n / tileW_;
            const int tx = n - tile * tileW_;
            const float* src = packedRows.data() + (size_t)v * rowStride;
            for (int k = 0; k < dModel; k++) {
                const size_t row = (size_t)(tile - tileBegin) * rowsPerTile_ + k;
                data[(row * tileW_ + tx) * 4 + c] = src[k];
            }
        }

        TablePart part;
        part.tex = std::make_unique<gl::Texture>();
        if (!part.tex->create(tileW_, height,
                              half_ ? gl::TextureFormat::RGBA16F : gl::TextureFormat::RGBA32F)) {
            return false;
        }
        part.tex->upload(data.data());
        part.tileBase = tileBegin;
        part.nStart = tileBegin * tileW_;
        part.nEnd = std::min(nTexels_, tileEnd * tileW_);
        parts_.push_back(std::move(part));

        totalMB += data.size() * (half_ ? 2 : 4) / (1024.0 * 1024.0);
    }

    std::printf("[mmllm] LM head token table: %d tiles of %d texels in %d texture(s) (%s, %.1f MB) for vocab %d\n",
                tiles, tileW_, numTex, half_ ? "fp16" : "fp32", totalMB, vocabSize);
    return true;
}

bool LMHead::createLogitsTarget(int vocabSize) {
    GLint maxTex = 0;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
    if (maxTex <= 0) maxTex = 8192;

    const int texels = packedWidth(vocabSize);            // 4 logits per texel
    logitsWidth_ = std::min<int>(texels, std::min<int>(maxTex, kSafeTexDim));
    const int rows = (texels + logitsWidth_ - 1) / logitsWidth_;

    logitsBuf_ = createRenderTarget(rows, logitsWidth_ * 4, "logits");
    if (!logitsBuf_) return false;

    std::printf("[mmllm] Logits target: %dx%d texels\n", logitsWidth_, rows);
    return true;
}

bool LMHead::forward(const gl::Texture& hidden) {
    if (!prog_ || !prog_->valid() || !logitsBuf_ || parts_.empty()) return false;

    // One draw per table texture; each writes only its own range of output texels
    for (const TablePart& part : parts_) {
        dispatchShader(*prog_, *logitsBuf_,
            {{0, &hidden, "texH"}, {1, part.tex.get(), "texW"}},
            [&](gl::Program& p) {
                p.setInt("T", hiddenTexels_);
                p.setInt("V", vocabSize_);
                p.setInt("tileW", tileW_);
                p.setInt("rowsPerTile", rowsPerTile_);
                p.setInt("tileBase", part.tileBase);
                p.setInt("nStart", part.nStart);
                p.setInt("nEnd", part.nEnd);
                p.setInt("outW", logitsWidth_);
            });
    }
    return true;
}

void LMHead::readLogits(std::vector<float>& out) {
    const gl::Texture& tex = logitsBuf_->colorTexture();
    readbackBuf_.resize((size_t)tex.width() * tex.height() * 4);
    tex.download(readbackBuf_.data());

    // Texel (x, y) holds logits 4f..4f+3 with f = y * width + x, so the readback is
    // already a contiguous logits array.
    out.assign(readbackBuf_.begin(), readbackBuf_.begin() + vocabSize_);
}
