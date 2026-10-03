#pragma once
#include "../engine/tensor.h"
#include "../gl/program.h"
#include "../gl/shader.h"
#include <memory>
#include <vector>

// LM head with weights tied to the token embedding table, computed on the GPU:
//   logits[v] = dot(hidden, token_table[v])
//
// The [vocab, d_model] table is repacked transposed: texel (n, k) holds table[4n..4n+3][k], so
// neighbouring fragments read neighbouring texels. The vocabulary is wider than the GPU samples
// reliably, so it is cut into column tiles stacked vertically; when the stack would get too tall
// it is spread over several textures, each drawn separately (see uploadTable / lmhead.frag).
class LMHead {
public:
    LMHead();
    ~LMHead() = default;

    bool init();

    // Build the transposed table texture(s) from packed RGBA32F rows (row v starts at
    // v * ceil(dModel/4) * 4 floats). Returns false if it cannot fit.
    bool uploadTable(const std::vector<float>& packedRows, int vocabSize, int dModel);

    // Create the render target that receives the logits (4 vocabulary entries per texel).
    bool createLogitsTarget(int vocabSize);

    // hidden: [1, d_model] packed vector. Fills the logits target.
    bool forward(const gl::Texture& hidden);

    // Read the logits back into out[0..vocabSize)
    void readLogits(std::vector<float>& out);

private:
    // One texture of the table: a run of whole vocabulary tiles
    struct TablePart {
        std::unique_ptr<gl::Texture> tex;
        int tileBase = 0;    // first tile stored here
        int nStart = 0;      // first output texel (vocab texel index) covered
        int nEnd = 0;        // one past the last
    };

    std::unique_ptr<gl::Shader> vert_;
    std::unique_ptr<gl::Shader> frag_;
    std::unique_ptr<gl::Program> prog_;

    std::vector<TablePart> parts_;
    std::shared_ptr<gl::FBO> logitsBuf_;
    std::vector<float> readbackBuf_;

    int vocabSize_ = 0;
    int hiddenTexels_ = 0;   // T = ceil(d_model / 4)
    int nTexels_ = 0;        // ceil(vocab / 4)
    int tileW_ = 0;          // texels per column tile (= table texture width)
    int rowsPerTile_ = 0;    // 4 * T
    int logitsWidth_ = 0;
    bool half_ = true;       // store the table as RGBA16F (MMLLM_LM_FP32=1 disables)
};
