#include "embedding.h"
#include "../gl/shader.h"
#include "../gl/fbo.h"
#include "../gl/texture.h"
#include <cstdio>
#include <cstring>

Embedding::Embedding() {}

bool Embedding::init() {
    // No shaders needed — we do embedding lookup on CPU
    std::printf("[mmllm] Embedding layer initialized (CPU-based)\n");
    return true;
}

bool Embedding::forward(int tokenId, int pos,
                        const gl::Texture* tokenEmbedWeight,
                        const gl::Texture* posEmbedWeight,
                        int dModel, int vocabSize,
                        gl::FBO& outputFBO)
{
    if (!tokenEmbedWeight || !posEmbedWeight) return false;

    // Read token embedding row
    int tw = tokenEmbedWeight->width();
    int th = tokenEmbedWeight->height();

    readbackBuf_.resize(tw * th * 4);
    tokenEmbedWeight->download(readbackBuf_.data());

    std::vector<float> tokenVec(dModel);
    for (int c = 0; c < dModel; c++) {
        int tx = c / 4;
        int tc = c % 4;
        tokenVec[c] = readbackBuf_[(tokenId * tw + tx) * 4 + tc];
    }

    // Read position embedding row
    int ptw = posEmbedWeight->width();
    int pth = posEmbedWeight->height();

    std::vector<float> posBuf(ptw * pth * 4);
    posEmbedWeight->download(posBuf.data());

    std::vector<float> posVec(dModel);
    for (int c = 0; c < dModel; c++) {
        int tx = c / 4;
        int tc = c % 4;
        posVec[c] = posBuf[(pos * ptw + tx) * 4 + tc];
    }

    // Sum token + position embeddings
    std::vector<float> result(dModel);
    for (int i = 0; i < dModel; i++) {
        result[i] = tokenVec[i] + posVec[i];
    }

    // Upload to output FBO as packed RGBA32F
    int outTW = (dModel + 3) / 4;
    std::vector<float> packed(outTW * 1 * 4, 0.0f);
    for (int c = 0; c < dModel; c++) {
        int tx = c / 4;
        int tc = c % 4;
        packed[(0 * outTW + tx) * 4 + tc] = result[c];
    }
    outputFBO.colorTexture().upload(packed.data());

    return true;
}
