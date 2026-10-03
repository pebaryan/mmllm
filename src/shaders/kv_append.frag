#version 330 core

// Copy the K and V halves of the fused QKV row into one row of the KV cache.
//
// Source (texSrc):   fused QKV output, one row of 3*T texels: [ q (T) | k (T) | v (T) ]
// Target (KV cache): width 2*T, one row per sequence position: [ k (T) | v (T) ]
// The caller scissors the draw to the single row for the current position, so
// texel x of the target row comes from texel x + T of the source row.

uniform sampler2D texSrc;
uniform int srcOffset;       // T = ceil(d_model / 4)

out vec4 fragColor;

void main() {
    int x = int(gl_FragCoord.x);
    fragColor = texelFetch(texSrc, ivec2(x + srcOffset, 0), 0);
}
