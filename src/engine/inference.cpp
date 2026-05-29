#include "inference.h"
#include "../gl/fbo.h"
#include <GL/glew.h>
#include <cmath>
#include <cstdio>
#include <algorithm>
#include <numeric>

InferenceEngine::InferenceEngine()
    : rng_(std::random_device{}())
{
}

bool InferenceEngine::init(const Model& model) {
    model_ = &model;
    const auto& cfg = model.config();

    // Initialize embedding layer (CPU-based)
    embedding_ = std::make_unique<Embedding>();
    if (!embedding_->init()) return false;

    // Initialize transformer blocks
    blocks_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        auto block = std::make_unique<TransformerBlock>();
        if (!block->init(cfg.d_model, cfg.n_heads, cfg.max_seq_len, cfg.ffn_hidden)) {
            std::fprintf(stderr, "[mmllm] Failed to init block %d\n", i);
            return false;
        }
        blocks_.push_back(std::move(block));
    }

    // Final layer norm
    finalLN_ = std::make_unique<LayerNorm>();
    if (!finalLN_->init()) return false;

    // LM head (matmul with token embedding weights)
    lmHead_ = std::make_unique<MatMul>();
    if (!lmHead_->init()) return false;

    // Softmax for sampling distribution
    softmax_ = std::make_unique<Softmax>();
    if (!softmax_->init()) return false;

    // Create temporary buffers (ping-pong between blocks)
    embedBuf_ = createRenderTarget(1, cfg.d_model, "embed");
    blockOutBuf_[0] = createRenderTarget(1, cfg.d_model, "block_ping");
    blockOutBuf_[1] = createRenderTarget(1, cfg.d_model, "block_pong");
    lnOutBuf_ = createRenderTarget(1, cfg.d_model, "ln_out");

    // Logits buffer: needs vocab_size = 50257 values → 12565 texels packed as RGBA32F.
    // Some GPUs (e.g. GeForce 9400M with nouveau/GL 3.3) have GL_MAX_TEXTURE_SIZE = 8192,
    // so a single-row texture would fail. Tile into multiple rows to fit.
    GLint maxTexSize;
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTexSize);
    if (maxTexSize < 64) maxTexSize = 8192;  // safety fallback

    int logitTexW = packedWidth(cfg.vocab_size);
    logitTileW_ = std::min(logitTexW, (int)maxTexSize);
    logitTileH_ = (logitTexW + logitTileW_ - 1) / logitTileW_;

    std::printf("[mmllm] Creating logits FBO: %dx%d (max_tex_size=%d)\n",
                logitTileW_, logitTileH_, (int)maxTexSize);
    logitsBuf_ = createRenderTarget(logitTileH_, logitTileW_ * 4, "logits");
    // Note: cols passed as logitTileW_ * 4 so packedWidth gives logitTileW_

    if (!embedBuf_ || !blockOutBuf_[0] || !blockOutBuf_[1] ||
        !lnOutBuf_ || !logitsBuf_) {
        return false;
    }

    std::printf("[mmllm] Inference engine initialized\n");
    return true;
}

std::vector<int> InferenceEngine::generate(const std::vector<int>& promptTokens, int maxNewTokens,
                                              float temperature, int topK) {
    if (!model_) {
        std::fprintf(stderr, "[mmllm] No model loaded\n");
        return {};
    }

    const auto& cfg = model_->config();
    std::vector<int> outputTokens = promptTokens;
    int totalLen = (int)promptTokens.size();

    std::printf("[mmllm] Generating %d tokens from prompt of length %d...\n",
                maxNewTokens, totalLen);

    for (int step = 0; step < maxNewTokens; step++) {
        int currentPos = totalLen - 1;
        int currentToken = outputTokens[currentPos];
        int seqLen = totalLen;

        // 1. Embedding: token + position (CPU lookup → upload to GPU)
        if (!embedding_->forward(currentToken, currentPos,
                                 model_->weight("token_embed"),
                                 model_->weight("pos_embed"),
                                 cfg.d_model, cfg.vocab_size,
                                 *embedBuf_)) {
            std::fprintf(stderr, "[mmllm] Embedding failed at step %d\n", step);
            break;
        }

        // 2. Run through all transformer blocks (ping-pong between buffer 0 and 1)
        const gl::Texture* layerInput = &embedBuf_->colorTexture();

        for (int layer = 0; layer < cfg.n_layers; layer++) {
            std::string pfx = "layer" + std::to_string(layer) + ".";

            int writeIdx = layer & 1;  // 0, 1, 0, 1, ...
            gl::FBO& targetFBO = *blockOutBuf_[writeIdx];

            if (!blocks_[layer]->forward(
                    *layerInput, seqLen,
                    model_->weight(pfx + "wqkv"),
                    model_->weight(pfx + "wo"),
                    model_->weight(pfx + "ln1_gain"),
                    model_->weight(pfx + "ln1_bias"),
                    model_->weight(pfx + "wg1"),
                    model_->weight(pfx + "wg2"),
                    model_->weight(pfx + "ln2_gain"),
                    model_->weight(pfx + "ln2_bias"),
                    model_->weight(pfx + "bqkv"),
                    model_->weight(pfx + "bo"),
                    cfg.attn_scale(), cfg.epsilon,
                    targetFBO)) {
                std::fprintf(stderr, "[mmllm] Block %d failed at step %d\n", layer, step);
                goto generate_done;
            }

            layerInput = &targetFBO.colorTexture();
        }

        // 3. Final LayerNorm
        if (!finalLN_->forward(*layerInput, cfg.d_model,
                               *model_->weight("ln_final_gain"),
                               *model_->weight("ln_final_bias"),
                               cfg.epsilon, *lnOutBuf_)) {
            std::fprintf(stderr, "[mmllm] Final LN failed\n");
            break;
        }

        // 4. LM head: ln_out * W_token_embed  →  [1, vocab_size]
        const gl::Texture* lmHeadTex = model_->weight("token_embed");
        if (!lmHead_->forward(lnOutBuf_->colorTexture(), 1, cfg.d_model,
                              *lmHeadTex, cfg.d_model, cfg.vocab_size,
                              *logitsBuf_)) {
            std::fprintf(stderr, "[mmllm] LM head failed\n");
            break;
        }

        // 5. Read back logits and sample next token
        {
            int tw = logitsBuf_->colorTexture().width();
            int th = logitsBuf_->colorTexture().height();
            readbackBuf_.resize(tw * th * 4);
            logitsBuf_->colorTexture().download(readbackBuf_.data());

            // Unpack RGBA32F → linear float array from tiled layout
            std::vector<float> logits(cfg.vocab_size, -1e20f);
            for (int i = 0; i < cfg.vocab_size && i < tw * th * 4; i++) {
                int texelIdx = i / 4;           // which texel in the packed layout
                int tc = i % 4;                 // which RGBA component
                int tileX = texelIdx % logitTileW_;
                int tileY = texelIdx / logitTileW_;
                logits[i] = readbackBuf_[(tileY * tw + tileX) * 4 + tc];
            }

            // 6. Sample from logits with temperature and top-k
            int nextToken = sampleToken(logits.data(), cfg.vocab_size, temperature, topK);
            outputTokens.push_back(nextToken);
            totalLen++;

            if ((step + 1) % 10 == 0 || step == 0 || step >= maxNewTokens - 3) {
                std::printf("[mmllm] Step %d: token %d\n", step + 1, nextToken);
            }

            // Stop conditions
            if (nextToken < 2) break;  // EOS or padding
            if (totalLen >= cfg.max_seq_len) {
                std::printf("[mmllm] Reached max sequence length (%d)\n", cfg.max_seq_len);
                break;
            }
        }
    }

generate_done:
    std::printf("[mmllm] Generated %zu tokens total\n", outputTokens.size());
    return outputTokens;
}

int InferenceEngine::sampleToken(const float* logits, int vocabSize, float temperature, int topK) {
    // Apply temperature: 0 = greedy (argmax)
    if (temperature <= 0.0f) {
        int bestIdx = 0;
        float bestVal = -1e20f;
        for (int i = 0; i < vocabSize; i++) {
            if (logits[i] > bestVal) {
                bestVal = logits[i];
                bestIdx = i;
            }
        }
        return bestIdx;
    }

    // Copy logits so we can modify them
    std::vector<float> scaled(vocabSize);
    for (int i = 0; i < vocabSize; i++) {
        scaled[i] = logits[i] / temperature;
    }

    // Apply top-k filtering: keep only the top-k largest logits
    if (topK > 0 && topK < vocabSize) {
        // Find the top-k threshold value
        std::vector<float> sorted(scaled.begin(), scaled.end());
        std::nth_element(sorted.begin(), sorted.begin() + (topK - 1), sorted.end(),
                         std::greater<float>());
        float threshold = sorted[topK - 1];

        // Zero out everything below the threshold
        for (int i = 0; i < vocabSize; i++) {
            if (scaled[i] < threshold) {
                scaled[i] = -1e20f;  // effectively zero after softmax
            }
        }
    }

    // Softmax: compute probabilities
    // Find max for numerical stability
    float maxLogit = -1e20f;
    for (int i = 0; i < vocabSize; i++) {
        if (scaled[i] > maxLogit) maxLogit = scaled[i];
    }

    std::vector<float> probs(vocabSize);
    float sum = 0.0f;
    for (int i = 0; i < vocabSize; i++) {
        probs[i] = std::exp(scaled[i] - maxLogit);
        sum += probs[i];
    }

    // Normalize (should be ~1.0, but handle numerical edge cases)
    if (sum <= 0.0f) {
        // Fallback: uniform distribution
        std::uniform_int_distribution<int> dist(0, vocabSize - 1);
        return dist(rng_);
    }
    float invSum = 1.0f / sum;
    for (int i = 0; i < vocabSize; i++) {
        probs[i] *= invSum;
    }

    // Sample from the probability distribution
    std::uniform_real_distribution<float> uniform(0.0f, 1.0f);
    float r = uniform(rng_);
    float cumulative = 0.0f;
    for (int i = 0; i < vocabSize; i++) {
        cumulative += probs[i];
        if (r < cumulative) {
            return i;
        }
    }

    // Fallback: last token (shouldn't reach here)
    return vocabSize - 1;
}
