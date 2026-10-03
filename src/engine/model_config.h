#pragma once
#include <cstdint>
#include <cstddef>
#include <cstdio>
#include <string>
#include <vector>
#include <cmath>

// Model configuration for TinyStories decoder-only transformer.
// All dimensions can be configured at load time.
struct ModelConfig {
    int vocab_size      = 1000;
    int d_model         = 128;     // embedding / hidden dimension
    int n_layers        = 4;       // number of transformer blocks
    int n_heads         = 4;       // number of attention heads
    int d_head          = 32;      // head dimension (should be d_model / n_heads)
    int ffn_hidden      = 512;     // feed-forward hidden dimension (typically 4x d_model)
    int max_seq_len     = 512;     // maximum sequence length
    bool has_bias       = true;    // whether QKV/output/FFN projections have bias terms
    bool weight_tying   = true;    // whether lm_head shares weights with token embedding
    float epsilon       = 1e-5f;   // layer norm epsilon
    bool attn_unscaled  = false;   // GPT-Neo does not scale Q*K^T by 1/sqrt(d_head)
    int eos_token       = -1;      // stop generating on this token (-1 = none)
    int local_window    = 0;       // GPT-Neo local-attention window (0 = none)
    uint32_t local_mask = 0;       // bit i set = layer i uses local attention

    bool layer_is_local(int layer) const {
        return local_window > 0 && layer >= 0 && layer < 32 && ((local_mask >> layer) & 1u);
    }

    // Derived helpers
    int d_head_divisor() const { return d_head > 0 ? d_head : 1; }
    float attn_scale() const {
        return attn_unscaled ? 1.0f : 1.0f / std::sqrt((float)d_head);
    }

    void print() const {
        std::printf("[mmllm] Model config:\n");
        std::printf("  vocab=%d  d_model=%d  layers=%d  heads=%d  d_head=%d\n",
                    vocab_size, d_model, n_layers, n_heads, d_head);
        std::printf("  ffn=%d  max_seq=%d  bias=%d  tie=%d  attn_scale=%s  eos=%d\n",
                    ffn_hidden, max_seq_len, has_bias, weight_tying,
                    attn_unscaled ? "none(GPT-Neo)" : "1/sqrt(d_head)", eos_token);
        if (local_window > 0) {
            std::printf("  local attention: window=%d on layers mask=0x%x\n", local_window, local_mask);
        }
    }

    // Validate configuration
    bool valid() const {
        if (vocab_size <= 0 || d_model <= 0 || n_layers <= 0) return false;
        if (n_heads <= 0 || d_head <= 0) return false;
        if (d_model != n_heads * d_head) return false;
        if (ffn_hidden <= 0 || max_seq_len <= 0) return false;
        return true;
    }
};

// Binary model file header (little-endian)
// Packed to ensure consistent size across platforms/compilers
//
// reserved[0]: flag bits. bit 0 = attention scores are NOT scaled (GPT-Neo)
// reserved[1]: EOS token id + 1 (0 = no EOS token)
// reserved[2]: local-attention window size (0 = none)
// reserved[3]: bit mask of layers that use local attention (bit i = layer i)
#pragma pack(push, 1)
struct ModelHeader {
    char     magic[4] = {'M', 'M', 'L', 'M'};
    uint32_t version  = 1;
    // Config
    int32_t  vocab_size;
    int32_t  d_model;
    int32_t  n_layers;
    int32_t  n_heads;
    int32_t  d_head;
    int32_t  ffn_hidden;
    int32_t  max_seq_len;
    int32_t  has_bias;
    int32_t  weight_tying;
    // Flags and padding for future use
    int32_t  reserved[8];
};
#pragma pack(pop)

static_assert(sizeof(ModelHeader) == 76, "ModelHeader must be 76 bytes (packed)");

// Packed sizes for texture storage
inline int packedWidth(int cols, int packing = 4) {
    return (cols + packing - 1) / packing;
}

inline int packedFloats(int rows, int cols, int packing = 4) {
    return ((cols + packing - 1) / packing) * rows * 4;
}
