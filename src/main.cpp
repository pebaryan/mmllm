/**
 * mmllm - Minimal GLSL-based LLM Inference Engine
 *
 * Uses OpenGL 3.3 fragment shaders (GLSL 3.30) to accelerate transformer
 * inference on older GPUs via the classic "render-to-texture" GPGPU approach.
 *
 * Designed for Macmini3,1 (GeForce 9400M, nouveau driver, OpenGL 3.3).
 * Target model: TinyStories-1M (~4 MB FP32).
 *
 * Build:
 *   mkdir build && cd build
 *   cmake .. && make -j$(nproc)
 *
 * Run:
 *   ./mmllm models/tinystories-1m.mlm
 */

#include "gl/context.h"
#include "gl/program.h"
#include "gl/shader.h"
#include "gl/texture.h"
#include "gl/fbo.h"
#include "engine/model.h"
#include "engine/inference.h"
#include "needle/needle_main.h"
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <string>

// Forward declaration of self-test
bool runSelfTest(gl::Context& ctx);

int main(int argc, char** argv) {
    // Needle 3 mode (CPU, no OpenGL context): mmllm --needle model.cact --tools t.json --prompt "..."
    for (int i = 1; i < argc; i++)
        if (std::string(argv[i]) == "--needle") return needle::needleMain(argc, argv);

    std::printf("========================================\n");
    std::printf("  mmllm — Minimal GLSL LLM Engine v1.0\n");
    std::printf("  OpenGL 3.3 / GLSL 3.30\n");
    std::printf("========================================\n\n");

    // Parse arguments
    std::string modelPath;
    int maxTokens = 50;
    float temperature = 1.0f;
    int topK = 0;
    std::vector<int> promptTokens = {42};  // default prompt (just a start token)

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            modelPath = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            maxTokens = std::atoi(argv[++i]);
        } else if (arg == "--prompt" && i + 1 < argc) {
            // Parse comma-separated token IDs
            std::string prompt = argv[++i];
            promptTokens.clear();
            size_t pos = 0;
            while (pos < prompt.size()) {
                size_t comma = prompt.find(',', pos);
                if (comma == std::string::npos) comma = prompt.size();
                int tok = std::atoi(prompt.substr(pos, comma - pos).c_str());
                promptTokens.push_back(tok);
                pos = comma + 1;
            }
        } else if (arg == "--temperature" && i + 1 < argc) {
            temperature = std::atof(argv[++i]);
        } else if (arg == "--top-k" && i + 1 < argc) {
            topK = std::atoi(argv[++i]);
            if (topK < 0) topK = 0;
        } else if (arg == "--help" || arg == "-h") {
            std::printf("Usage: mmllm [options]\n");
            std::printf("  --model <path>    Path to .mlm model file\n");
            std::printf("  --tokens <n>      Max tokens to generate (default: 50)\n");
            std::printf("  --prompt <ids>    Prompt token IDs (comma-separated, default: 42)\n");
            std::printf("  --temperature <f> Sampling temperature (default: 1.0). 0 = greedy\n");
            std::printf("  --top-k <n>       Top-k sampling (default: 0 = disabled)\n");
            std::printf("  --self-test       Run GPU self-test and exit\n");
            std::printf("  --help            Show this help\n");
            return 0;
        } else if (arg == "--self-test") {
            // Will handle after init
        }
    }

    // Initialize GL context (hidden window)
    gl::Context ctx;
    if (!ctx.init(256, 256, "mmllm")) {
        std::fprintf(stderr, "FATAL: Failed to initialize OpenGL 3.3 context.\n");
        std::fprintf(stderr, "Make sure Mesa/nouveau drivers support OpenGL 3.3.\n");
        return 1;
    }

    // Check for --self-test
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--self-test") {
            bool passed = runSelfTest(ctx);
            return passed ? 0 : 1;
        }
    }

    // If no model specified, look for default
    if (modelPath.empty()) {
        // Check common locations
        const char* candidates[] = {
            "models/tinystories-1m.mlm",
            "./tinystories-1m.mlm",
            "../models/tinystories-1m.mlm",
            nullptr
        };
        for (int i = 0; candidates[i]; i++) {
            FILE* f = std::fopen(candidates[i], "rb");
            if (f) {
                std::fclose(f);
                modelPath = candidates[i];
                std::printf("[mmllm] Found model: %s\n", modelPath.c_str());
                break;
            }
        }
    }

    if (modelPath.empty()) {
        std::fprintf(stderr, "No model file found.\n");
        std::fprintf(stderr, "Run: python3 tools/export_model.py to create a model.\n");
        std::fprintf(stderr, "Or specify: ./mmllm --model <path>\n\n");
        std::fprintf(stderr, "Run --self-test to verify GPU functionality.\n");
        return 1;
    }

    // Load model
    Model model;
    if (!model.load(modelPath)) {
        std::fprintf(stderr, "FATAL: Failed to load model: %s\n", modelPath.c_str());
        return 1;
    }

    // Initialize inference engine
    InferenceEngine engine;
    if (!engine.init(model)) {
        std::fprintf(stderr, "FATAL: Failed to initialize inference engine\n");
        return 1;
    }

    // Generate text
    std::printf("\n[mmllm] Generating with prompt of %zu tokens...\n\n", promptTokens.size());
    std::printf("[mmllm] Sampling: temperature=%.2f", temperature);
    if (topK > 0) std::printf(", top-k=%d", topK);
    std::printf("\n\n");

    auto startTime = std::chrono::steady_clock::now();

    std::vector<int> output = engine.generate(promptTokens, maxTokens, temperature, topK);

    auto endTime = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(endTime - startTime).count();

    // Print results
    const size_t newTokens = output.size() - promptTokens.size();
    std::printf("\n[mmllm] Generated %zu tokens in %.2f seconds (%.2f tok/s)\n",
                newTokens, elapsed, newTokens / elapsed);

    std::printf("[mmllm] Output token IDs:\n  ");
    for (size_t i = promptTokens.size(); i < output.size(); i++) {
        std::printf("%d ", output[i]);
    }
    std::printf("\n");

    // Print model info
    std::printf("\n[mmllm] Done. Peak GPU textures ~%d MB\n",
                (model.weights().size() * 64) / 1024);  // rough estimate

    std::printf("[mmllm] For token-to-text conversion, use tools/decode.py\n");

    return 0;
}

bool runSelfTest(gl::Context& ctx) {
    std::printf("\n=== GPU Self-Test ===\n\n");

    bool allPassed = true;

    // Test 1: Texture creation
    std::printf("Test 1: Texture creation...\n");
    {
        gl::Texture tex;
        if (!tex.create(4, 4, gl::TextureFormat::RGBA32F)) {
            std::printf("  FAILED: Could not create RGBA32F texture\n");
            allPassed = false;
        } else {
            std::printf("  PASSED: Created 4x4 RGBA32F texture (ID %u)\n", tex.id());
        }
    }

    // Test 2: Shader compilation
    std::printf("Test 2: Shader compilation...\n");
    {
        gl::Shader vert;
        if (!vert.compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert")) {
            std::printf("  FAILED: Vertex shader compilation\n");
            allPassed = false;
        } else {
            std::printf("  PASSED: Vertex shader compiled\n");
        }

        gl::Shader frag;
        if (!frag.compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/matmul.frag")) {
            std::printf("  FAILED: MatMul fragment shader compilation\n");
            allPassed = false;
        } else {
            std::printf("  PASSED: MatMul fragment shader compiled\n");
        }
    }

    // Test 3: MatMul correctness
    std::printf("Test 3: MatMul correctness...\n");
    {
        // Create a simple 2x2 * 2x2 multiplication
        // A = [1 2; 3 4], B = [5 6; 7 8]
        // C = [19 22; 43 50]

        gl::Shader vert;
        vert.compileFromFile(GL_VERTEX_SHADER, "src/shaders/passthrough.vert");
        gl::Shader frag;
        frag.compileFromFile(GL_FRAGMENT_SHADER, "src/shaders/matmul.frag");
        gl::Program prog;
        prog.link(vert, frag);

        // Matrix A: [2, 2], packed: [ceil(2/4)=1, 2] RGBA32F
        float dataA[2*4] = {
            1.0f, 2.0f, 0.0f, 0.0f,  // row 0: A[0][0]=1, A[0][1]=2
            3.0f, 4.0f, 0.0f, 0.0f   // row 1: A[1][0]=3, A[1][1]=4
        };

        // Matrix B: [2, 2], packed: [1, 2] RGBA32F
        float dataB[2*4] = {
            5.0f, 6.0f, 0.0f, 0.0f,  // row 0: B[0][0]=5, B[0][1]=6
            7.0f, 8.0f, 0.0f, 0.0f   // row 1: B[1][0]=7, B[1][1]=8
        };

        gl::Texture texA, texB;
        texA.create(1, 2, gl::TextureFormat::RGBA32F);
        texA.upload(dataA);
        texB.create(1, 2, gl::TextureFormat::RGBA32F);
        texB.upload(dataB);

        gl::FBO outputFBO;
        outputFBO.create(1, 2, gl::TextureFormat::RGBA32F);

        dispatchShader(prog, outputFBO,
            {{
                {0, &texA, "texA"},
                {1, &texB, "texB"}
            }},
            [&](gl::Program& p) {
                p.setInt("M", 2);
                p.setInt("K", 2);
                p.setInt("N", 2);
                p.setInt("outputTexWidth", 1);
            });

        // Read back result
        // Texture is 1x2 RGBA32F, so download gives 8 floats:
        //   texel (0,0) = [C[0][0], C[0][1], pad, pad]
        //   texel (0,1) = [C[1][0], C[1][1], pad, pad]
        float result[2*4] = {};
        outputFBO.colorTexture().download(result);

        // Extract actual matrix values (ignoring padding)
        float actualC[4] = {result[0], result[1], result[4], result[5]};
        float expectedC[] = {19.0f, 22.0f, 43.0f, 50.0f};
        bool match = true;
        for (int i = 0; i < 4; i++) {
            if (std::abs(actualC[i] - expectedC[i]) > 1e-4f) {
                match = false;
                break;
            }
        }

        if (match) {
            std::printf("  PASSED: C = [%.0f %.0f; %.0f %.0f]\n",
                        result[0], result[1], result[4], result[5]);
        } else {
            std::printf("  FAILED: Got [%.2f %.2f; %.2f %.2f], expected [19 22; 43 50]\n",
                        result[0], result[1], result[4], result[5]);
            allPassed = false;
        }
    }

    std::printf("\n=== Self-test %s ===\n", allPassed ? "PASSED" : "FAILED");
    return allPassed;
}
