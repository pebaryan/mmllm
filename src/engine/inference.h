#pragma once
#include "../engine/model.h"
#include "../engine/tensor.h"
#include "../layers/transformer_block.h"
#include "../layers/embedding.h"
#include "../layers/layernorm.h"
#include "../layers/lmhead.h"
#include <memory>
#include <vector>
#include <string>
#include <random>

// Inference engine: orchestrates the full autoregressive generation pipeline.
//
//   token -> embedding (CPU) -> N transformer blocks (GPU + CPU attention)
//         -> final LayerNorm (GPU) -> LM head (GPU, tied to the token table) -> sample (CPU)
//
// The prompt is run through the model one token at a time ("prefill") so the KV
// caches hold every prompt position before the first new token is sampled.
//
// Set MMLLM_DEBUG=1 to print per-stage activations for comparison with a reference.
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
    // Run one token at position `pos` through the whole model, filling the KV caches.
    // If `logits` is non-null, also compute the vocabulary logits.
    bool runToken(int token, int pos, std::vector<float>* logits);

    // Sample next token from logits with temperature and top-k
    int sampleToken(const float* logits, int vocabSize, float temperature, int topK);

    const Model* model_ = nullptr;
    bool debug_ = false;
    bool lmOnCpu_ = false;    // LM head on the CPU (large models, or MMLLM_LM=cpu)
    int maxSeq_ = 0;          // context length actually used (<= model max, MMLLM_MAX_SEQ caps it)

    // Optional per-stage timing (MMLLM_PROFILE=1). Each stage is fenced with glFinish.
    bool profile_ = false;
    double tEmbed_ = 0, tBlocks_ = 0, tFinalLN_ = 0, tLmHead_ = 0, tReadback_ = 0;
    int profTokens_ = 0;

    // Transformer blocks (one per layer)
    std::vector<std::unique_ptr<TransformerBlock>> blocks_;

    // Embedding layer (CPU-based lookup)
    std::unique_ptr<Embedding> embedding_;

    // Final LayerNorm
    std::unique_ptr<LayerNorm> finalLN_;

    // LM head (GPU)
    std::unique_ptr<LMHead> lmHead_;

    // Temporary buffers
    std::shared_ptr<gl::FBO> embedBuf_;
    std::shared_ptr<gl::FBO> blockOutBuf_[2];  // ping-pong buffers
    std::shared_ptr<gl::FBO> lnOutBuf_;

    // RNG for token sampling
    std::mt19937 rng_;

    std::vector<float> readbackBuf_;
};
