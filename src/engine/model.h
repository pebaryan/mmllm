#pragma once
#include "model_config.h"
#include "../gl/texture.h"
#include <vector>
#include <string>
#include <cstdint>
#include <cstdio>

// The Model class stores all transformer weights as GPU textures.
// Weights are packed in RGBA32F format, 4 elements per texel.
class Model {
public:
    Model() = default;

    bool load(const std::string& filepath);

    const ModelConfig& config() const { return config_; }
    bool loaded() const { return loaded_; }

    // Access a weight texture by name
    const gl::Texture* weight(const std::string& name) const;

    // Access raw CPU data
    const std::vector<float>* rawWeights(const std::string& name) const;

    // Helpers for texture dimensions
    static int texWidth(int cols) { return packedWidth(cols, 4); }
    static int texHeight(int rows) { return rows; }

    void printSummary() const;

    struct Weight {
        std::string name;
        std::vector<float> data;  // packed RGBA32F data
        gl::Texture texture;

        Weight() = default;
        Weight(Weight&&) = default;
        Weight& operator=(Weight&&) = default;
        Weight(const Weight&) = delete;
        Weight& operator=(const Weight&) = delete;
    };

    const std::vector<Weight>& weights() const { return weights_; }

    // Bytes of GPU texture memory used by the weight textures
    size_t gpuBytes() const { return gpuBytes_; }

private:
    Weight* findWeight(const std::string& name);
    const Weight* findWeight(const std::string& name) const;

    ModelConfig config_;
    bool loaded_ = false;
    size_t gpuBytes_ = 0;
    std::vector<Weight> weights_;
};
