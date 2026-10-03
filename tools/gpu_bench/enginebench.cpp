// Times the engine's own validated MatMul (src/shaders/matmul.frag) at Needle-sized shapes.
// Build (from ~/mmllm): see tools in session notes; run from ~/mmllm so src/shaders is found.
#include "gl/context.h"
#include "gl/fbo.h"
#include "gl/texture.h"
#include "layers/matmul.h"
#include <GL/glew.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

static void run(MatMul& mm, int M, int K, int N, bool half, int chain) {
    const int K4 = K / 4, N4 = N / 4;
    std::vector<float> A((size_t)M * K), B((size_t)K * N);
    for (size_t i = 0; i < A.size(); i++) A[i] = (float)(int)((i * 40503u >> 4) % 2001) / 1000.f - 1.f;
    for (size_t i = 0; i < B.size(); i++) B[i] = (float)(int)((i * 2654435761u >> 8) % 2001) / 1000.f - 1.f;
    gl::Texture ta, tb;
    ta.create(K4, M, gl::TextureFormat::RGBA32F);
    ta.upload(A.data());
    tb.create(N4, K, half ? gl::TextureFormat::RGBA16F : gl::TextureFormat::RGBA32F);
    tb.upload(B.data());
    gl::FBO fbo;
    fbo.create(N4, M, gl::TextureFormat::RGBA32F);
    glFinish();
    mm.forward(ta, M, K, tb, K, N, fbo);
    glFinish();
    std::vector<float> out((size_t)N4 * M * 4);
    fbo.colorTexture().download(out.data());
    double err = 0;
    for (int m = 0; m < std::min(M, 2); m++)
        for (int n = 0; n < std::min(N, 64); n++) {
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)A[(size_t)m * K + k] * B[(size_t)k * N + n];
            err = std::max(err, std::fabs(ref - out[(size_t)m * N + n]));
        }
    const int reps = 30;
    double t0 = now();
    for (int i = 0; i < reps; i++) {
        mm.forward(ta, M, K, tb, K, N, fbo);
        glFinish();
    }
    const double each = (now() - t0) / reps * 1000;
    t0 = now();
    for (int i = 0; i < reps; i++) {
        for (int c = 0; c < chain; c++) mm.forward(ta, M, K, tb, K, N, fbo);
        glFinish();
    }
    const double chained = (now() - t0) / reps / chain * 1000;
    std::printf("M=%3d K=%5d N=%5d %s | finish each %7.3f ms | chained x%d %7.3f ms/call (%.2f GMAC/s) | max|err| %.4f\n", M, K, N,
                half ? "fp16" : "fp32", each, chain, chained, (double)M * K * N / (chained / 1000) / 1e9, err);
}

int main() {
    gl::Context ctx;
    if (!ctx.init()) return 1;
    glewExperimental = GL_TRUE;
    glewInit();
    MatMul mm;
    if (!mm.init()) return 1;
    struct S { int M, K, N; } shapes[] = {
        {1, 768, 96}, {1, 768, 576}, {1, 768, 768}, {1, 768, 1536}, {1, 768, 2048},
        {8, 768, 768}, {32, 768, 768},
    };
    for (bool half : {false, true})
        for (const S& s : shapes) run(mm, s.M, s.K, s.N, half, 8);
    return 0;
}
