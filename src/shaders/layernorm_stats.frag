#version 330 core

// LayerNorm statistics for a single vector, computed entirely on the GPU.
// Draw into a 1x1 target. Output: .r = mean, .g = 1 / sqrt(variance + epsilon).
//
// Two passes over the data (mean, then centred sum of squares) for numerical accuracy.
// Only the first N elements count; any padding in the last texel is ignored.

uniform sampler2D texInput;   // [T, 1], T = ceil(N / 4)
uniform int N;
uniform float epsilon;

out vec4 fragColor;

void main() {
    int T = (N + 3) / 4;

    // Per-component masks for the last (possibly partial) texel
    vec4 sum = vec4(0.0);
    for (int i = 0; i < T; i++) {
        vec4 x = texelFetch(texInput, ivec2(i, 0), 0);
        int base = i * 4;
        if (base + 0 >= N) x.x = 0.0;
        if (base + 1 >= N) x.y = 0.0;
        if (base + 2 >= N) x.z = 0.0;
        if (base + 3 >= N) x.w = 0.0;
        sum += x;
    }
    float mean = (sum.x + sum.y + sum.z + sum.w) / float(N);

    vec4 sq = vec4(0.0);
    for (int i = 0; i < T; i++) {
        vec4 x = texelFetch(texInput, ivec2(i, 0), 0);
        vec4 d = x - mean;
        int base = i * 4;
        if (base + 0 >= N) d.x = 0.0;
        if (base + 1 >= N) d.y = 0.0;
        if (base + 2 >= N) d.z = 0.0;
        if (base + 3 >= N) d.w = 0.0;
        sq += d * d;
    }
    float variance = (sq.x + sq.y + sq.z + sq.w) / float(N);

    fragColor = vec4(mean, 1.0 / sqrt(variance + epsilon), 0.0, 0.0);
}
