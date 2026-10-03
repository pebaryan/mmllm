#include "model.h"
#include <GL/glew.h>
#include <fstream>
#include <cstring>
#include <algorithm>
#include <cstdlib>

bool Model::load(const std::string& filepath) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open()) {
        std::fprintf(stderr, "[mmllm] Failed to open model file: %s\n", filepath.c_str());
        return false;
    }

    // Read and validate header
    ModelHeader header;
    file.read(reinterpret_cast<char*>(&header), sizeof(header));
    if (!file) {
        std::fprintf(stderr, "[mmllm] Failed to read model header\n");
        return false;
    }

    if (std::memcmp(header.magic, "MMLM", 4) != 0) {
        std::fprintf(stderr, "[mmllm] Invalid model magic (expected 'MMLM')\n");
        return false;
    }

    // Populate config
    config_.vocab_size   = header.vocab_size;
    config_.d_model      = header.d_model;
    config_.n_layers     = header.n_layers;
    config_.n_heads      = header.n_heads;
    config_.d_head       = header.d_head;
    config_.ffn_hidden   = header.ffn_hidden;
    config_.max_seq_len  = header.max_seq_len;
    config_.has_bias     = header.has_bias != 0;
    config_.weight_tying = header.weight_tying != 0;
    config_.attn_unscaled = (header.reserved[0] & 1) != 0;
    config_.eos_token    = header.reserved[1] > 0 ? header.reserved[1] - 1 : -1;
    config_.local_window = header.reserved[2] > 0 ? header.reserved[2] : 0;
    config_.local_mask   = static_cast<uint32_t>(header.reserved[3]);

    if (!config_.valid()) {
        std::fprintf(stderr, "[mmllm] Invalid model config\n");
        return false;
    }

    config_.print();

    const int D = config_.d_model;
    const int H = config_.ffn_hidden;
    const int V = config_.vocab_size;
    const int L = config_.n_layers;
    const int T = config_.max_seq_len;

    // Helper lambda: read a raw float array
    auto readTensor = [&](const std::string& name, int rows, int cols) -> bool {
        int count = rows * cols;
        std::vector<float> buf(count);
        file.read(reinterpret_cast<char*>(buf.data()), count * sizeof(float));
        if (!file) {
            std::fprintf(stderr, "[mmllm] Failed to read tensor '%s'\n", name.c_str());
            return false;
        }

        // Store as packed RGBA32F texture data
        int tw = texWidth(cols);
        int th = rows;
        std::vector<float> packed(tw * th * 4, 0.0f);

        for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
                int tx = c / 4;
                int tc = c % 4;
                int packedIdx = (r * tw + tx) * 4 + tc;
                packed[packedIdx] = buf[r * cols + c];
            }
        }

        Weight w;
        w.name = name;
        w.data = std::move(packed);

        // Tensors larger than GL_MAX_TEXTURE_SIZE (such as the vocab x d_model token
        // table on a GPU limited to 8192) cannot live in a texture. Keep them on the
        // CPU only; callers use rawWeights() for those.
        GLint maxTex = 0;
        glGetIntegerv(GL_MAX_TEXTURE_SIZE, &maxTex);
        if (maxTex <= 0) maxTex = 8192;

        // The embedding tables are only ever read on the CPU (embedding lookup and the LM head
        // table is built from the raw data), so they get no GPU texture at all.
        const bool cpuOnly = (name == "token_embed" || name == "pos_embed");

        // The big matmul weights are stored as fp16 textures: half the GPU memory (the 9400M has
        // only 256 MB of its own; spilling to system memory is very slow and eventually fails
        // with ENOMEM) and half the bandwidth. MMLLM_W_FP32=1 keeps them fp32.
        auto endsWith = [&](const char* suf) {
            const std::string s(suf);
            return name.size() >= s.size() && name.compare(name.size() - s.size(), s.size(), s) == 0;
        };
        const bool bigWeight = endsWith("wqkv") || endsWith("wo") || endsWith("wg1") || endsWith("wg2");
        const bool useHalf = bigWeight && std::getenv("MMLLM_W_FP32") == nullptr;

        if (cpuOnly) {
            // no texture
        } else if (tw > maxTex || th > maxTex) {
            std::printf("[mmllm] '%s' is %dx%d texels (> max texture %d): kept on CPU only\n",
                        name.c_str(), tw, th, (int)maxTex);
        } else {
            if (!w.texture.create(tw, th, useHalf ? gl::TextureFormat::RGBA16F : gl::TextureFormat::RGBA32F)) {
                std::fprintf(stderr, "[mmllm] Failed to create texture for '%s'\n", name.c_str());
                return false;
            }
            w.texture.upload(w.data.data());
            gpuBytes_ += (size_t)tw * th * (useHalf ? 8 : 16);
        }
        weights_.push_back(std::move(w));
        return true;
    };

    // Read all model weights
    // Token embedding: [V, D]
    if (!readTensor("token_embed", V, D)) return false;

    // Position embedding: [T, D]
    if (!readTensor("pos_embed", T, D)) return false;

    // Per-layer weights
    for (int i = 0; i < L; i++) {
        std::string prefix = "layer" + std::to_string(i) + ".";

        // QKV projection: [D, 3*D] (fused Q, K, V)
        if (!readTensor(prefix + "wqkv", D, 3 * D)) return false;

        // Attention output: [D, D]
        if (!readTensor(prefix + "wo", D, D)) return false;

        // FFN gate (up projection): [D, H]
        if (!readTensor(prefix + "wg1", D, H)) return false;

        // FFN down projection: [H, D]
        if (!readTensor(prefix + "wg2", H, D)) return false;

        // Layer norm 1 (pre-attention): gain [D], bias [D]
        if (!readTensor(prefix + "ln1_gain", 1, D)) return false;
        if (!readTensor(prefix + "ln1_bias", 1, D)) return false;

        // Layer norm 2 (pre-FFN): gain [D], bias [D]
        if (!readTensor(prefix + "ln2_gain", 1, D)) return false;
        if (!readTensor(prefix + "ln2_bias", 1, D)) return false;

        // Optional biases for QKV and output
        if (config_.has_bias) {
            if (!readTensor(prefix + "bqkv", 1, 3 * D)) return false;
            if (!readTensor(prefix + "bo", 1, D)) return false;
        }
    }

    // Final layer norm: gain [D], bias [D]
    if (!readTensor("ln_final_gain", 1, D)) return false;
    if (!readTensor("ln_final_bias", 1, D)) return false;

    // LM head: [D, V] (may be skipped if weight tying)
    if (!config_.weight_tying) {
        if (!readTensor("lm_head", D, V)) return false;
    }

    // Read any remaining biases
    for (int i = 0; i < L; i++) {
        std::string prefix = "layer" + std::to_string(i) + ".";
        if (config_.has_bias) {
            if (!readTensor(prefix + "bg1", 1, H)) return false;
            if (!readTensor(prefix + "bg2", 1, D)) return false;
        }
    }

    loaded_ = true;
    printSummary();
    return true;
}

Model::Weight* Model::findWeight(const std::string& name) {
    for (auto& w : weights_) {
        if (w.name == name) return &w;
    }
    return nullptr;
}

const Model::Weight* Model::findWeight(const std::string& name) const {
    for (const auto& w : weights_) {
        if (w.name == name) return &w;
    }
    return nullptr;
}

const gl::Texture* Model::weight(const std::string& name) const {
    const auto* w = findWeight(name);
    return w ? &w->texture : nullptr;
}

const std::vector<float>* Model::rawWeights(const std::string& name) const {
    const auto* w = findWeight(name);
    return w ? &w->data : nullptr;
}

void Model::printSummary() const {
    std::printf("[mmllm] Model loaded: %d weights, %.1f MB total\n",
                (int)weights_.size(),
                loaded_ ? 0.0f : 0.0f); // Will compute below
    if (!loaded_) return;

    size_t totalBytes = 0;
    for (const auto& w : weights_) {
        totalBytes += w.data.size() * sizeof(float);
        std::printf("  %-30s %dx%d (%d texels, %zu KB)\n",
                    w.name.c_str(),
                    w.texture.width(), w.texture.height(),
                    w.texture.width() * w.texture.height(),
                    w.data.size() * sizeof(float) / 1024);
    }
    std::printf("  Total weights: %.2f MB\n", totalBytes / (1024.0 * 1024.0));
}
