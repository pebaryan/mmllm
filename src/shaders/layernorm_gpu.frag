#version 330 core

// Layer Normalization apply pass, with the statistics supplied as a texture
// (produced by layernorm_stats.frag) instead of CPU-computed uniforms.
// out[i] = gain[i] * (in[i] - mean) * invStd + bias[i]

uniform sampler2D texInput;    // [ceil(N/4), 1]
uniform sampler2D texGain;     // [ceil(N/4), 1]
uniform sampler2D texBias;     // [ceil(N/4), 1]
uniform sampler2D texStats;    // [1, 1]: .r = mean, .g = 1 / sqrt(var + eps)
uniform int N;

out vec4 fragColor;

void main() {
    int nx = int(gl_FragCoord.x);
    vec4 x = texelFetch(texInput, ivec2(nx, 0), 0);
    vec4 g = texelFetch(texGain, ivec2(nx, 0), 0);
    vec4 b = texelFetch(texBias, ivec2(nx, 0), 0);
    vec2 st = texelFetch(texStats, ivec2(0, 0), 0).rg;

    vec4 result = g * (x - st.x) * st.y + b;

    // Zero out padding elements (when N is not multiple of 4)
    int base = nx * 4;
    if (base + 0 >= N) result.r = 0.0;
    if (base + 1 >= N) result.g = 0.0;
    if (base + 2 >= N) result.b = 0.0;
    if (base + 3 >= N) result.a = 0.0;

    fragColor = result;
}
