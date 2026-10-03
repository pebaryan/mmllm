#version 330 core

// Attention scores for one query token against all cached keys, every head.
//
// Output texel (i, h) holds, in .r:  scale * dot(q_h, k_h[i])
// for sequence position i in [first, seqLen); everything else holds -1e30.
//
// texQKV: fused QKV row of the current token; q occupies texels [0, T)
// texKV:  KV cache, row = position, k occupies texels [0, T) of each row
// head h covers texels [h * headTexels, (h + 1) * headTexels), headTexels = d_head / 4

uniform sampler2D texQKV;
uniform sampler2D texKV;
uniform int seqLen;
uniform int first;           // first visible position (local attention window)
uniform int headTexels;
uniform float scale;

out vec4 fragColor;

void main() {
    int i = int(gl_FragCoord.x);
    int h = int(gl_FragCoord.y);

    if (i < first || i >= seqLen) {
        fragColor = vec4(-1e30, 0.0, 0.0, 0.0);
        return;
    }

    int base = h * headTexels;
    float s = 0.0;
    for (int t = 0; t < headTexels; t++) {
        vec4 q = texelFetch(texQKV, ivec2(base + t, 0), 0);
        vec4 k = texelFetch(texKV, ivec2(base + t, i), 0);
        s += dot(q, k);
    }

    fragColor = vec4(s * scale, 0.0, 0.0, 0.0);
}
