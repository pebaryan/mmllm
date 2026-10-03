#include "embedding.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cstdio>

Embedding::Embedding() {}

bool Embedding::init() {
    // No shaders needed - embedding lookup is a CPU table read
    std::printf("[mmllm] Embedding layer initialized (CPU-based)\n");
    return true;
}

bool Embedding::forward(int tokenId, int pos,
                        const std::vector<float>* tokenEmbed,
                        const std::vector<float>* posEmbed,
                        int dModel, int vocabSize, int maxSeqLen,
                        gl::FBO& outputFBO)
{
    if (!tokenEmbed || !posEmbed) return false;
    if (tokenId < 0 || tokenId >= vocabSize) {
        std::fprintf(stderr, "[mmllm] Token id %d out of range (vocab %d)\n", tokenId, vocabSize);
        return false;
    }
    if (pos < 0 || pos >= maxSeqLen) {
        std::fprintf(stderr, "[mmllm] Position %d out of range (max %d)\n", pos, maxSeqLen);
        return false;
    }

    // Both tables are packed RGBA32F rows of ceil(d_model/4) texels, so row r starts
    // at r * tw * 4 and the d_model values are contiguous within the row.
    const int tw = (dModel + 3) / 4;
    const float* tok = tokenEmbed->data() + (size_t)tokenId * tw * 4;
    const float* ps = posEmbed->data() + (size_t)pos * tw * 4;

    std::vector<float> packed(tw * 4, 0.0f);
    for (int c = 0; c < dModel; c++) {
        packed[c] = tok[c] + ps[c];
    }

    outputFBO.colorTexture().upload(packed.data());
    return true;
}
