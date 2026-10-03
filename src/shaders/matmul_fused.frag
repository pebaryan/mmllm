#version 330 core

// Fused matrix multiply: C[M, N] = epilogue( prologue(A)[M, K] * B[K, N] )
//
// Same layout and tiling as matmul.frag (RGBA32F, 4 columns per texel), plus optional
// work folded into the same pass to save draw calls:
//
//   prologue (hasLN):       A is the raw residual stream; each element is layer-normalised
//                           on the fly: (a - mean) * invStd * gain + bias.
//                           Only valid for M == 1 (the statistics texture is per vector).
//   epilogue:
//     hasBias:     C += bias            (bias is a [1, N] vector texture)
//     doGelu:      C  = gelu(C)         (tanh approximation, as GPT-Neo's gelu_new)
//     hasResidual: C += residual        (same shape as the output)
//
// Samplers that a given call does not use are bound to a dummy texture and never read.
//
// NOTE: do not unroll the K loop in this shader. A 4x-unrolled version (16 B fetches in
// flight) made nouveau on the GeForce 9400M fault the GPU (PGRAPH TRAP_MP_EXEC timeout +
// bad write) and hang the whole machine. The plain loop below is the tested form.

uniform sampler2D texA;
uniform sampler2D texB;
uniform sampler2D texBias;
uniform sampler2D texRes;
uniform sampler2D texLnStats;   // [1, 1]: .r = mean, .g = 1 / sqrt(var + eps)
uniform sampler2D texLnGain;    // [ceil(K/4), 1]
uniform sampler2D texLnBias;    // [ceil(K/4), 1]

uniform int M;
uniform int N;
uniform int K;
uniform int outputTexWidth;

uniform int hasLN;
uniform int hasBias;
uniform int hasResidual;
uniform int doGelu;

out vec4 fragColor;

void main() {
    int texelX = int(gl_FragCoord.x);
    int texelY = int(gl_FragCoord.y);
    int flatIdx = texelY * outputTexWidth + texelX;

    int N_blocks = (N + 3) / 4;
    int m = flatIdx / N_blocks;
    int nx = flatIdx % N_blocks;

    if (nx >= N_blocks || m >= M) {
        fragColor = vec4(0.0);
        return;
    }

    int K_blocks = (K + 3) / 4;

    vec2 st = vec2(0.0, 1.0);
    if (hasLN != 0) {
        st = texelFetch(texLnStats, ivec2(0, 0), 0).rg;
    }

    vec4 sum = vec4(0.0);
    for (int kx = 0; kx < K_blocks; kx++) {
        vec4 a = texelFetch(texA, ivec2(kx, m), 0);

        if (hasLN != 0) {
            vec4 g = texelFetch(texLnGain, ivec2(kx, 0), 0);
            vec4 b = texelFetch(texLnBias, ivec2(kx, 0), 0);
            a = (a - st.x) * st.y * g + b;
            int base = kx * 4;
            if (base + 0 >= K) a.x = 0.0;
            if (base + 1 >= K) a.y = 0.0;
            if (base + 2 >= K) a.z = 0.0;
            if (base + 3 >= K) a.w = 0.0;
        }

        vec4 b0 = texelFetch(texB, ivec2(nx, kx * 4 + 0), 0);
        vec4 b1 = texelFetch(texB, ivec2(nx, kx * 4 + 1), 0);
        vec4 b2 = texelFetch(texB, ivec2(nx, kx * 4 + 2), 0);
        vec4 b3 = (kx * 4 + 3 < K)
            ? texelFetch(texB, ivec2(nx, kx * 4 + 3), 0)
            : vec4(0.0);

        sum.r += a.r * b0.r + a.g * b1.r + a.b * b2.r + a.a * b3.r;
        sum.g += a.r * b0.g + a.g * b1.g + a.b * b2.g + a.a * b3.g;
        sum.b += a.r * b0.b + a.g * b1.b + a.b * b2.b + a.a * b3.b;
        sum.a += a.r * b0.a + a.g * b1.a + a.b * b2.a + a.a * b3.a;
    }

    if (hasBias != 0) {
        sum += texelFetch(texBias, ivec2(nx, 0), 0);
    }

    if (doGelu != 0) {
        const float sqrt2pi = 0.7978845608028654;
        const float coeff = 0.044715;
        vec4 inner = sqrt2pi * (sum + coeff * sum * sum * sum);
        sum = 0.5 * sum * (1.0 + tanh(inner));
    }

    if (hasResidual != 0) {
        sum += texelFetch(texRes, ivec2(texelX, texelY), 0);
    }

    fragColor = sum;
}
