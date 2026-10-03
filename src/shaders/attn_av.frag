#version 330 core

// Softmax-weighted sum of the cached values: out[h*d_head + c] = sum_i p[i] * v_h[i][c]
// where p[i] = exp(score[i] - max) / sum, applied on the fly from the stats texture.
//
// One fragment per output texel (4 consecutive elements, all inside one head
// because d_head is a multiple of 4). Output is [T, 1], ready for the output projection.

uniform sampler2D texScores;   // (position, head) -> .r = score
uniform sampler2D texStats;    // (0, head) -> .r = max, .g = 1 / sum
uniform sampler2D texKV;       // KV cache, v occupies texels [vOffset, vOffset + T) of each row
uniform int seqLen;
uniform int first;
uniform int headTexels;        // d_head / 4
uniform int vOffset;           // T

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    int h = x / headTexels;

    vec2 st = texelFetch(texStats, ivec2(0, h), 0).rg;

    vec4 acc = vec4(0.0);
    for (int i = first; i < seqLen; i++) {
        float p = exp(texelFetch(texScores, ivec2(i, h), 0).r - st.x) * st.y;
        acc += p * texelFetch(texKV, ivec2(vOffset + x, i), 0);
    }

    fragColor = acc;
}
