#pragma once
#include "../engine/model.h"
#include "../engine/tensor.h"
#include "../layers/transformer_block.h"
#include "../layers/embedding.h"
#include "../layers/layernorm.h"
#include "../layers/matmul.h"
#include "../layers/softmax.h"
#include <memory>
#include <vector>
#include <string>
#include <random>

// Inference engine: orchestrates the full autoregressive generation pipeline.
class InferenceEngine {
public:
    InferenceEngine();
    ~InferenceEngine() = default;

    // Initialize with a loaded model
    bool init(const Model& model);

    // Generate text given a prompt (sequence of token IDs)
    // temperature: >0. Higher = more random. 0 = greedy (argmax). Default 1.0.
    // topK: >0 = keep only top-K logits before softmax. 0 = disabled. Default 0.
    std::vector<int> generate(const std::vector<int>& promptTokens, int maxNewTokens = 50,
                              float temperature = 1.0f, int topK = 0);

    // Access the model
    const Model* model() const { return model_; }

private:
    const Model* model_ = nullptr;

    // Transformer blocks (one per layer)
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;

    // Embedding layer (CPU-based lookup)
    std::unique_ptr<Embedding> embedding_;

    // Final LayerNorm
    std::unique_ptr<LayerNorm> finalLN_;

    // LM head matmul
    std::unique_ptr<MatMul> lmHead_;

    // Softmax for logits
    std::unique_ptr<Softmax> softmax_;

    // Temporary buffers
    std::shared_ptr<gl::FBO> embedBuf_;
    std::shared_ptr<gl::FBO> blockOutBuf_[2];  // ping-pong buffers
    std::shared_ptr<gl::FBO> lnOutBuf_;
    std::shared_ptr<gl::FBO> logitsBuf_;

    // RNG for token sampling
    std::mt19937 rng_;

    // Sample next token from logits with temperature and top-k
    int sampleToken(const float* logits, int vocabSize, float temperature, int topK);

    // Logits FBO tile dimensions (multi-row to fit within GL_MAX_TEXTURE_SIZE)
    int logitTileW_ = 0;
    int logitTileH_ = 0;

    // Readback buffer for logits
    std::vector<float> readbackBuf_;
};
