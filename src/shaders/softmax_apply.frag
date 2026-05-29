#version 330 core

// Apply softmax normalization to a vector.
// Computes: out[i] = exp(in[i] - max_val) / sum_exp
//
// Max and sum_exp are precomputed on CPU and passed as uniforms.
// This shader just does the final element-wise normalization.

uniform sampler2D texInput;
uniform float maxVal;
uniform float invSumExp;  // 1.0 / sum(exp(x_i - maxVal))
uniform int N;

out vec4 fragColor;

void main() {
    int nx = int(gl_FragCoord.x);
    vec4 x = texelFetch(texInput, ivec2(nx, 0), 0);

    vec4 result = exp(x - maxVal) * invSumExp;

    int base = nx * 4;
    if (base + 0 >= N) result.r = 0.0;
    if (base + 1 >= N) result.g = 0.0;
    if (base + 2 >= N) result.b = 0.0;
    if (base + 3 >= N) result.a = 0.0;

    fragColor = result;
}
