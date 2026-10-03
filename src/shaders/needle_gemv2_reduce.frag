#version 330 core

// Needle 3 CQ matvec, split-K pass 2: one fragment per output row, summing the per-group partials
// produced by needle_gemv2_partial.frag.
//
//   texP : RGBA32F [in/128 x out]   per-group partials (only .r is used)
//
// Writes y[r] to component .r.

uniform sampler2D texP;
uniform int kg;            // number of 128-groups (= in / 128)

out vec4 fragColor;

void main() {
    int r = int(gl_FragCoord.y);
    float t = 0.0;
    for (int g = 0; g < kg; g++) t += texelFetch(texP, ivec2(g, r), 0).r;
    fragColor = vec4(t, 0.0, 0.0, 0.0);
}
