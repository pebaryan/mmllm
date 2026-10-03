#include "inference.h"
#include "../gl/fbo.h"
#include <GL/glew.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <thread>

namespace {

// Print the first few values and the L2 norm of a [1, n] packed vector texture.
void debugVec(const char* tag, const gl::Texture& tex, int n) {
    std::vector<float> buf(tex.width() * tex.height() * 4);
    tex.download(buf.data());
    double norm = 0.0;
    for (int i = 0; i < n; i++) norm += (double)buf[i] * buf[i];
    std::printf("[dbg] %-12s norm=%.5f  [%.5f %.5f %.5f %.5f %.5f %.5f]\n", tag, std::sqrt(norm),
                buf[0], buf[1], buf[2], buf[3], buf[4], buf[5]);
}

} // namespace

InferenceEngine::InferenceEngine()
    : rng_(std::random_device{}())
{
}

bool InferenceEngine::init(const Model& model) {
    model_ = &model;
    const auto& cfg = model.config();
    debug_ = std::getenv("MMLLM_DEBUG") != nullptr;
    maxSeq_ = cfg.max_seq_len;
    if (const char* e = std::getenv("MMLLM_MAX_SEQ")) {
        const int cap = std::atoi(e);
        if (cap > 0) maxSeq_ = std::min(maxSeq_, cap);
    }
    profile_ = std::getenv("MMLLM_PROFILE") != nullptr;

    embedding_ = std::make_unique<Embedding>();
    if (!embedding_->init()) return false;

    blocks_.reserve(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; i++) {
        auto block = std::make_unique<TransformerBlock>();
        if (!block->init(cfg.d_model, cfg.n_heads, maxSeq_, cfg.ffn_hidden)) {
            std::fprintf(stderr, "[mmllm] Failed to init block %d\n", i);
            return false;
        }
        if (cfg.layer_is_local(i)) block->setLocalWindow(cfg.local_window);
        blocks_.push_back(std::move(block));
    }

    finalLN_ = std::make_unique<LayerNorm>();
    if (!finalLN_->init()) return false;

    // LM head on the GPU. The token table is far taller than GL_MAX_TEXTURE_SIZE, so
    // it is uploaded as a wide, short texture addressed by flat texel index.
    // The 9400M has 256 MB of its own memory, shared with the desktop. When the working set does
    // not fit, whatever spills to system memory samples several times slower (GPT-2 small: LM head
    // 180 ms on the GPU vs ~35 ms on the CPU). So estimate the footprint and keep the LM table (the
    // largest single item) on the CPU when it would not fit. Override with MMLLM_LM=cpu|gpu and
    // MMLLM_GPU_BUDGET_MB.
    {
        const int tw = (cfg.d_model + 3) / 4;
        const size_t kvBytes = (size_t)cfg.n_layers * maxSeq_ * 2 * tw * (std::getenv("MMLLM_KV_FP32") ? 16 : 8);
        const size_t lmBytes = (size_t)cfg.vocab_size * tw * 8;      // fp16 table
        const size_t total = model.gpuBytes() + kvBytes + lmBytes;
        size_t budget = 200ull << 20;
        if (const char* b = std::getenv("MMLLM_GPU_BUDGET_MB")) budget = (size_t)std::atoll(b) << 20;
        lmOnCpu_ = total > budget;
        if (const char* m = std::getenv("MMLLM_LM")) {
            if (std::string(m) == "cpu") lmOnCpu_ = true;
            else if (std::string(m) == "gpu") lmOnCpu_ = false;
        }
        std::printf("[mmllm] GPU footprint estimate: weights %.0f MB + KV %.0f MB + LM table %.0f MB = %.0f MB (budget %.0f MB) -> LM head on the %s\n",
                    model.gpuBytes() / 1048576.0, kvBytes / 1048576.0, lmBytes / 1048576.0,
                    total / 1048576.0, budget / 1048576.0, lmOnCpu_ ? "CPU" : "GPU");
    }

    if (!model.rawWeights("token_embed")) return false;
    if (!lmOnCpu_) {
        lmHead_ = std::make_unique<LMHead>();
        if (!lmHead_->init()) return false;
        if (!lmHead_->uploadTable(*model.rawWeights("token_embed"), cfg.vocab_size, cfg.d_model)) {
            return false;
        }
        if (!lmHead_->createLogitsTarget(cfg.vocab_size)) return false;
    }

    // Buffers: ping-pong between blocks
    embedBuf_ = createRenderTarget(1, cfg.d_model, "embed");
    blockOutBuf_[0] = createRenderTarget(1, cfg.d_model, "block_ping");
    blockOutBuf_[1] = createRenderTarget(1, cfg.d_model, "block_pong");
    lnOutBuf_ = createRenderTarget(1, cfg.d_model, "ln_out");

    if (!embedBuf_ || !blockOutBuf_[0] || !blockOutBuf_[1] || !lnOutBuf_) {
        return false;
    }

    if (!model.rawWeights("token_embed") || !model.rawWeights("pos_embed")) {
        std::fprintf(stderr, "[mmllm] Model is missing the embedding tables\n");
        return false;
    }

    std::printf("[mmllm] Inference engine initialized\n");
    return true;
}

bool InferenceEngine::runToken(int token, int pos, std::vector<float>* logits) {
    const auto& cfg = model_->config();
    const int seqLen = pos + 1;
    using Clock = std::chrono::steady_clock;
    auto mark = Clock::now();
    auto lap = [&](double& acc) {
        if (!profile_) return;
        glFinish();
        auto now = Clock::now();
        acc += std::chrono::duration<double, std::milli>(now - mark).count();
        mark = now;
    };
    if (profile_) { glFinish(); mark = Clock::now(); }

    // The embedding upload below overwrites embedBuf_ from the CPU. During prompt prefill no
    // readback happens, so the previous token's draws may still be reading it; on a slow
    // GPU relative to the CPU (e.g. the 33M model on the GeForce 9400M) that race corrupts
    // the KV cache and raises a TRAP_TEXTURE fault. Fence first. In the decode loop the
    // logits readback has already synced, so this costs nothing there.
    glFinish();

    // 1. Embedding: token + position (CPU lookup -> upload)
    if (!embedding_->forward(token, pos,
                             model_->rawWeights("token_embed"),
                             model_->rawWeights("pos_embed"),
                             cfg.d_model, cfg.vocab_size, cfg.max_seq_len,
                             *embedBuf_)) {
        std::fprintf(stderr, "[mmllm] Embedding failed at position %d\n", pos);
        return false;
    }
    lap(tEmbed_);
    if (debug_ && logits) debugVec("embed", embedBuf_->colorTexture(), cfg.d_model);

    // 2. Transformer blocks (ping-pong between buffer 0 and 1)
    const gl::Texture* layerInput = &embedBuf_->colorTexture();
    for (int layer = 0; layer < cfg.n_layers; layer++) {
        const std::string pfx = "layer" + std::to_string(layer) + ".";
        gl::FBO& target = *blockOutBuf_[layer & 1];

        const gl::Texture* bqkv = cfg.has_bias ? model_->weight(pfx + "bqkv") : nullptr;
        const gl::Texture* bo = cfg.has_bias ? model_->weight(pfx + "bo") : nullptr;
        const gl::Texture* bg1 = cfg.has_bias ? model_->weight(pfx + "bg1") : nullptr;
        const gl::Texture* bg2 = cfg.has_bias ? model_->weight(pfx + "bg2") : nullptr;

        if (!blocks_[layer]->forward(
                *layerInput, seqLen,
                model_->weight(pfx + "wqkv"), bqkv,
                model_->weight(pfx + "wo"), bo,
                model_->weight(pfx + "ln1_gain"), model_->weight(pfx + "ln1_bias"),
                model_->weight(pfx + "wg1"), bg1,
                model_->weight(pfx + "wg2"), bg2,
                model_->weight(pfx + "ln2_gain"), model_->weight(pfx + "ln2_bias"),
                cfg.attn_scale(), cfg.epsilon,
                target)) {
            std::fprintf(stderr, "[mmllm] Block %d failed at position %d\n", layer, pos);
            return false;
        }
        layerInput = &target.colorTexture();

        if (debug_ && logits) {
            char tag[32];
            std::snprintf(tag, sizeof(tag), "layer%d", layer);
            debugVec(tag, *layerInput, cfg.d_model);
        }
    }

    lap(tBlocks_);
    if (profile_) profTokens_++;

    if (!logits) return true;  // prefill step: only the KV caches matter

    // 3. Final LayerNorm
    if (!finalLN_->forward(*layerInput, cfg.d_model,
                           *model_->weight("ln_final_gain"), *model_->weight("ln_final_bias"),
                           cfg.epsilon, *lnOutBuf_)) {
        std::fprintf(stderr, "[mmllm] Final LN failed\n");
        return false;
    }
    lap(tFinalLN_);
    if (debug_) debugVec("ln_final", lnOutBuf_->colorTexture(), cfg.d_model);

    // 4. LM head: logits[v] = ln_out . token_embed[v]  (weights are tied)
    if (lmOnCpu_) {
        // Hidden vector from the GPU, then a multi-threaded dot product against the raw table
        const int D = cfg.d_model;
        const int V = cfg.vocab_size;
        const int rowStride = ((D + 3) / 4) * 4;
        const gl::Texture& lnTex = lnOutBuf_->colorTexture();
        readbackBuf_.resize((size_t)lnTex.width() * lnTex.height() * 4);
        lnTex.download(readbackBuf_.data());
        const float* h = readbackBuf_.data();
        const float* wte = model_->rawWeights("token_embed")->data();
        logits->assign(V, 0.0f);
        float* out = logits->data();
        auto work = [&](int v0, int v1) {
            for (int v = v0; v < v1; v++) {
                const float* row = wte + (size_t)v * rowStride;
                float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
                int c = 0;
                for (; c + 4 <= D; c += 4) {
                    a0 += h[c] * row[c];
                    a1 += h[c + 1] * row[c + 1];
                    a2 += h[c + 2] * row[c + 2];
                    a3 += h[c + 3] * row[c + 3];
                }
                for (; c < D; c++) a0 += h[c] * row[c];
                out[v] = (a0 + a1) + (a2 + a3);
            }
        };
        unsigned nt = std::max(1u, std::min(4u, std::thread::hardware_concurrency()));
        std::vector<std::thread> pool;
        const int chunk = (V + (int)nt - 1) / (int)nt;
        for (unsigned t = 1; t < nt; t++) {
            const int a = (int)t * chunk, b = std::min(V, a + chunk);
            if (a < b) pool.emplace_back(work, a, b);
        }
        work(0, std::min(V, chunk));
        for (auto& th : pool) th.join();
        lap(tLmHead_);
    } else {
        if (!lmHead_->forward(lnOutBuf_->colorTexture())) {
            std::fprintf(stderr, "[mmllm] LM head failed\n");
            return false;
        }
        lap(tLmHead_);
        lmHead_->readLogits(*logits);
    }
    lap(tReadback_);
    if (debug_) {
        // Cross-check a spread of vocabulary entries against a CPU dot product
        const int D = cfg.d_model, tw = (D + 3) / 4;
        std::vector<float> h(lnOutBuf_->colorTexture().width() * 4);
        lnOutBuf_->colorTexture().download(h.data());
        const float* wte = model_->rawWeights("token_embed")->data();
        const int probe[] = {0, 1, 2, 3, 4, 100, 12564, 12565, 25127, 25128, 25129, 25130, 40000, 50255, 50256};
        for (int v : probe) {
            if (v >= cfg.vocab_size) continue;
            double cpu = 0.0;
            for (int c = 0; c < D; c++) cpu += (double)h[c] * wte[(size_t)v * tw * 4 + c];
            std::printf("[dbg] logit[%5d] gpu=%10.4f cpu=%10.4f %s\n", v, (*logits)[v], cpu,
                        std::fabs((*logits)[v] - cpu) < 1e-2 ? "" : "  <-- MISMATCH");
        }
    }
    return true;
}

std::vector<int> InferenceEngine::generate(const std::vector<int>& promptTokens, int maxNewTokens,
                                              float temperature, int topK) {
    if (!model_) {
        std::fprintf(stderr, "[mmllm] No model loaded\n");
        return {};
    }
    if (promptTokens.empty()) {
        std::fprintf(stderr, "[mmllm] Empty prompt\n");
        return {};
    }

    const auto& cfg = model_->config();
    std::vector<int> outputTokens = promptTokens;
    const int promptLen = (int)promptTokens.size();

    if (promptLen > maxSeq_) {
        std::fprintf(stderr, "[mmllm] Prompt (%d tokens) exceeds max sequence length (%d)\n",
                     promptLen, maxSeq_);
        return outputTokens;
    }

    for (auto& b : blocks_) b->resetCache();

    std::printf("[mmllm] Generating up to %d tokens from a prompt of %d tokens...\n",
                maxNewTokens, promptLen);

    // Prefill: run prompt tokens 0..n-2 so every position is in the KV caches.
    for (int pos = 0; pos < promptLen - 1; pos++) {
        if (!runToken(promptTokens[pos], pos, nullptr)) return outputTokens;
    }

    std::vector<float> logits;
    for (int step = 0; step < maxNewTokens; step++) {
        const int pos = (int)outputTokens.size() - 1;
        if (pos >= maxSeq_) {
            std::printf("[mmllm] Reached max sequence length (%d)\n", maxSeq_);
            break;
        }

        if (!runToken(outputTokens[pos], pos, &logits)) break;

        if (debug_) {
            std::vector<int> idx(logits.size());
            std::iota(idx.begin(), idx.end(), 0);
            std::partial_sort(idx.begin(), idx.begin() + 5, idx.end(),
                              [&](int a, int b) { return logits[a] > logits[b]; });
            std::printf("[dbg] top5 logits:");
            for (int i = 0; i < 5; i++) std::printf("  %d:%.4f", idx[i], logits[idx[i]]);
            std::printf("\n");
        }

        const int nextToken = sampleToken(logits.data(), cfg.vocab_size, temperature, topK);
        outputTokens.push_back(nextToken);

        if ((step + 1) % 10 == 0 || step == 0 || step >= maxNewTokens - 3) {
            std::printf("[mmllm] Step %d: token %d\n", step + 1, nextToken);
        }

        if (cfg.eos_token >= 0 && nextToken == cfg.eos_token) {
            std::printf("[mmllm] Hit end-of-text token (%d)\n", cfg.eos_token);
            break;
        }
    }

    std::printf("[mmllm] Generated %zu new tokens (%zu total)\n",
                outputTokens.size() - promptTokens.size(), outputTokens.size());
    if (profile_ && profTokens_ > 0) {
        const double n = profTokens_;
        std::printf("[prof] per token (ms): embed %.2f  blocks %.2f  finalLN %.2f  lmhead-draw %.2f  logit-readback %.2f\n",
                    tEmbed_ / n, tBlocks_ / n, tFinalLN_ / n, tLmHead_ / n, tReadback_ / n);
    }
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
        std::vector<float> sorted(scaled.begin(), scaled.end());
        std::nth_element(sorted.begin(), sorted.begin() + (topK - 1), sorted.end(),
                         std::greater<float>());
        float threshold = sorted[topK - 1];

        for (int i = 0; i < vocabSize; i++) {
            if (scaled[i] < threshold) {
                scaled[i] = -1e20f;  // effectively zero after softmax
            }
        }
    }

    // Softmax: compute probabilities (subtract max for numerical stability)
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

    if (sum <= 0.0f) {
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

    return vocabSize - 1;
}
