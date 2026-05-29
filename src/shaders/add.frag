#version 330 core

// Element-wise addition: out = a + b
// Both inputs and output have the same texture dimensions.

uniform sampler2D texA;
uniform sampler2D texB;
uniform ivec2 texSize;  // size of the texture in texels
uniform int N;          // total number of elements (for padding check)

out vec4 fragColor;

void main() {
    ivec2 pos = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y));

    // Only process valid texels
    if (pos.x < texSize.x && pos.y < texSize.y) {
        vec4 a = texelFetch(texA, pos, 0);
        vec4 b = texelFetch(texB, pos, 0);

        vec4 result = a + b;

        // Zero padding elements
        int elem = (pos.y * texSize.x + pos.x) * 4;
        if (elem + 0 >= N) result.r = 0.0;
        if (elem + 1 >= N) result.g = 0.0;
        if (elem + 2 >= N) result.b = 0.0;
        if (elem + 3 >= N) result.a = 0.0;

        fragColor = result;
    } else {
        discard;
    }
}
