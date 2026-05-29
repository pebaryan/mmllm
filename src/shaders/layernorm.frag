#version 330 core

// Layer Normalization on a single vector [1, N].
// Computes: out[i] = gain[i] * (in[i] - mean) / sqrt(var + eps) + bias[i]
//
// Input: texture [ceil(N/4), 1] — the vector to normalize
// Output: same size
//
// Mean and variance are passed as uniforms (computed on CPU from a readback).

uniform sampler2D texInput;
uniform sampler2D texGain;     // [ceil(N/4), 1]
uniform sampler2D texBias;     // [ceil(N/4), 1]
uniform float mean;
uniform float invStd;          // 1.0 / sqrt(variance + epsilon)
uniform int N;

out vec4 fragColor;

void main() {
    int nx = int(gl_FragCoord.x);
    vec4 x = texelFetch(texInput, ivec2(nx, 0), 0);
    vec4 g = texelFetch(texGain, ivec2(nx, 0), 0);
    vec4 b = texelFetch(texBias, ivec2(nx, 0), 0);

    vec4 result = g * (x - mean) * invStd + b;

    // Zero out padding elements (when N is not multiple of 4)
    int base = nx * 4;
    if (base + 0 >= N) result.r = 0.0;
    if (base + 1 >= N) result.g = 0.0;
    if (base + 2 >= N) result.b = 0.0;
    if (base + 3 >= N) result.a = 0.0;

    fragColor = result;
}
