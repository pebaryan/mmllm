#version 330 core

// LM head with tied weights: logits[v] = dot(hidden, token_table[v]).
//
// The token table is stored TRANSPOSED so that neighbouring fragments read neighbouring
// texels (coalesced), like the weight matrix of the matmul shader:
//   texel (n, k) of a tile holds table[4n + 0..3][k]   (4 vocabulary entries, one hidden index)
// The vocabulary is 4*N entries wide, which can exceed what the GPU samples reliably, so it
// is cut into column tiles stacked vertically inside a texture: row = localTile * rowsPerTile + k.
// A large table needs several textures; one draw runs per texture, each handling only the
// output texels [nStart, nEnd) whose tiles live in that texture (other fragments are discarded,
// leaving whatever the other draws wrote).
//
// Output: 4 vocabulary entries per texel (4f .. 4f+3 for flat texel index f = n),
// tiled across rows of width outW. Entries past the vocabulary hold -1e30.

uniform sampler2D texH;       // final-LayerNorm output [T, 1], T = ceil(d_model / 4)
uniform sampler2D texW;       // this draw's part of the transposed token table
uniform int T;                // hidden texels
uniform int V;                // vocabulary size
uniform int tileW;            // texels per tile (width of texW)
uniform int rowsPerTile;      // 4 * T (d_model padded to a multiple of 4)
uniform int tileBase;         // index of the first tile stored in texW
uniform int nStart;           // first output texel handled by this draw
uniform int nEnd;             // one past the last output texel handled by this draw
uniform int outW;             // width of the logits render target

out vec4 fragColor;

void main() {
    int f = int(gl_FragCoord.y) * outW + int(gl_FragCoord.x);
    if (f < nStart || f >= nEnd) {
        discard;
    }

    int tile = f / tileW;
    int tx = f - tile * tileW;
    int row = (tile - tileBase) * rowsPerTile;

    vec4 acc = vec4(0.0);
    for (int kx = 0; kx < T; kx++) {
        vec4 h = texelFetch(texH, ivec2(kx, 0), 0);
        int k = row + kx * 4;
        acc += h.x * texelFetch(texW, ivec2(tx, k), 0);
        acc += h.y * texelFetch(texW, ivec2(tx, k + 1), 0);
        acc += h.z * texelFetch(texW, ivec2(tx, k + 2), 0);
        acc += h.w * texelFetch(texW, ivec2(tx, k + 3), 0);
    }

    int v0 = f * 4;
    fragColor = vec4(
        v0     < V ? acc.x : -1e30,
        v0 + 1 < V ? acc.y : -1e30,
        v0 + 2 < V ? acc.z : -1e30,
        v0 + 3 < V ? acc.w : -1e30);
}
