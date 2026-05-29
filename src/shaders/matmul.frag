#version 330 core

// General matrix multiply: C[M, N] = A[M, K] * B[K, N]
// All matrices are stored as RGBA32F textures with 4 columns packed per texel.
//
// Layout:
//   A: texture size [ceil(K/4), M]  — texel (kx, m) = A[m][kx*4 .. kx*4+3]
//   B: texture size [ceil(N/4), K]  — texel (nx, k) = B[k][nx*4 .. nx*4+3]
//   C: render target [outputTexWidth, outputTexHeight] — tiled across rows
//      to stay within GL_MAX_TEXTURE_SIZE on GPUs with limited texture dimensions.
//      Fragment at (texelX, texelY) computes C[m][nx*4 .. nx*4+3] where:
//        flatIdx = texelY * outputTexWidth + texelX
//        m       = flatIdx / N_blocks
//        nx      = flatIdx % N_blocks

uniform sampler2D texA;
uniform sampler2D texB;
uniform int M;                  // output rows (logical)
uniform int N;                  // output columns (logical)
uniform int K;                  // inner dimension
uniform int outputTexWidth;     // physical FBO texel width

out vec4 fragColor;

void main() {
    // Convert 2D texel position to flat texel index, then to logical (m, nx)
    int texelX = int(gl_FragCoord.x);
    int texelY = int(gl_FragCoord.y);
    int flatIdx = texelY * outputTexWidth + texelX;

    int N_blocks = (N + 3) / 4;    // number of 4-element blocks in N

    int m = flatIdx / N_blocks;     // output row
    int nx = flatIdx % N_blocks;    // output column block

    // Guard: skip fragments beyond logical output dimensions
    // Handles cases where the physical FBO is larger than M * N_blocks texels
    if (nx >= N_blocks || m >= M) {
        fragColor = vec4(0.0);
        return;
    }

    int K_blocks = (K + 3) / 4;    // number of 4-element blocks in K

    vec4 sum = vec4(0.0);

    for (int kx = 0; kx < K_blocks; kx++) {
        // Fetch A[m][kx*4 .. kx*4+3]
        vec4 a = texelFetch(texA, ivec2(kx, m), 0);

        // Fetch B[kx*4 .. kx*4+3][nx*4 .. nx*4+3]
        // Each texel read gives 4 consecutive column values from a single row of B
        vec4 b0 = texelFetch(texB, ivec2(nx, kx * 4 + 0), 0);
        vec4 b1 = texelFetch(texB, ivec2(nx, kx * 4 + 1), 0);
        vec4 b2 = texelFetch(texB, ivec2(nx, kx * 4 + 2), 0);
        vec4 b3 = (kx * 4 + 3 < K)
            ? texelFetch(texB, ivec2(nx, kx * 4 + 3), 0)
            : vec4(0.0);

        // 4-wide dot product: sum_q A[m][kx*4+q] * B[kx*4+q][nx*4+c]
        sum.r += a.r * b0.r + a.g * b1.r + a.b * b2.r + a.a * b3.r;
        sum.g += a.r * b0.g + a.g * b1.g + a.b * b2.g + a.a * b3.g;
        sum.b += a.r * b0.b + a.g * b1.b + a.b * b2.b + a.a * b3.b;
        sum.a += a.r * b0.a + a.g * b1.a + a.b * b2.a + a.a * b3.a;
    }

    fragColor = sum;
}
