#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include <memory>
#include <vector>

// Embedding lookup: maps token IDs to dense vectors.
// The lookup runs on the CPU from the raw packed weight arrays (the token table
// is too tall to fit in a GPU texture on old GPUs) and the result is uploaded.
class Embedding {
public:
    Embedding();
    ~Embedding() = default;

    bool init();

    // Look up token + position embeddings.
    // tokenEmbed: packed RGBA32F data of the [vocab_size, d_model] table
    // posEmbed:   packed RGBA32F data of the [max_seq_len, d_model] table
    // outputFBO:  [1, d_model] output
    bool forward(int tokenId, int pos,
                 const std::vector<float>* tokenEmbed,
                 const std::vector<float>* posEmbed,
                 int dModel, int vocabSize, int maxSeqLen,
                 gl::FBO& outputFBO);
};
