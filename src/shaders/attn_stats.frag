#version 330 core

// Softmax statistics per head: one fragment per head.
// Output texel (0, h):  .r = max score,  .g = 1 / sum(exp(score - max))
// over the visible positions [first, seqLen).

uniform sampler2D texScores;   // (position, head) -> .r = score
uniform int seqLen;
uniform int first;

out vec4 fragColor;

void main() {
    int h = int(gl_FragCoord.y);

    float m = -1e30;
    for (int i = first; i < seqLen; i++) {
        m = max(m, texelFetch(texScores, ivec2(i, h), 0).r);
    }

    float sum = 0.0;
    for (int i = first; i < seqLen; i++) {
        sum += exp(texelFetch(texScores, ivec2(i, h), 0).r - m);
    }

    fragColor = vec4(m, 1.0 / sum, 0.0, 0.0);
}
