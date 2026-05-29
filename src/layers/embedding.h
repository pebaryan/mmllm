#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include "matmul.h"
#include <memory>

// Embedding lookup: maps token IDs to dense vectors.
// For this prototype, we do the lookup on CPU and upload the result.
class Embedding {
public:
    Embedding();
    ~Embedding() = default;

    bool init();

    // Look up token + position embeddings
    // token_id: the input token
    // pos: the position in the sequence
    // token_embed_weight: [vocab_size, d_model] texture
    // pos_embed_weight: [max_seq_len, d_model] texture
    // outputFBO: [1, d_model] output
    bool forward(int tokenId, int pos,
                 const gl::Texture* tokenEmbedWeight,
                 const gl::Texture* posEmbedWeight,
                 int dModel, int vocabSize,
                 gl::FBO& outputFBO);

private:
    // CPU-side storage for embedding weights
    std::vector<float> readbackBuf_;
};
