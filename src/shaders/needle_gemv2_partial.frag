#version 330 core

// Needle 3 CQ matvec, split-K pass 1: one fragment per (128-group g, output row r).
// The weight matrix is a 2-bit CQ matrix: `in/4` packed bytes per row, 4 LSB-first 2-bit indices
// per byte, and a per-group norm. Reconstructing w = (codebook[idx] * norm) @ H and rotating the
// activation with the same normalised Walsh-Hadamard matrix H (symmetric, orthogonal) gives
//   w . x = norm * sum_k codebook[idx_k] * (H x)_k
// so the rotated activation xr can be used directly and the 2-bit weights never need unpacking.
//
//   texW : R8UI  [in/4      x out]   packed 2-bit indices (byte b at (g*32+j, r))
//   texX : RGBA32F [in/4    x 1  ]   rotated activation, 4 floats per texel
//   texN : R32F  [in/128    x out]   per-group norms (row r holds its in/128 norms)
//
// Writes the group's contribution (already scaled by the norm) to partial[g][r].

uniform usampler2D texW;
uniform sampler2D  texX;
uniform sampler2D  texN;
uniform vec4       cb;

out vec4 fragColor;

float cbv(int i) {
    return i == 0 ? cb.x : (i == 1 ? cb.y : (i == 2 ? cb.z : cb.w));
}

void main() {
    int g = int(gl_FragCoord.x);   // 128-group
    int r = int(gl_FragCoord.y);   // output row

    float s = 0.0;
    for (int j = 0; j < 32; j++) {
        uint b = texelFetch(texW, ivec2(g * 32 + j, r), 0).r;
        vec4 x4 = texelFetch(texX, ivec2(g * 32 + j, 0), 0);
        vec4 w = vec4(cbv(int(b & 3u)),
                      cbv(int((b >> 2u) & 3u)),
                      cbv(int((b >> 4u) & 3u)),
                      cbv(int((b >> 6u) & 3u)));
        s += dot(w, x4);
    }

    fragColor = vec4(texelFetch(texN, ivec2(g, r), 0).r * s);
}
