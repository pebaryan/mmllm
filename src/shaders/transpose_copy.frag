#version 330 core

// Copy with optional transpose or data rearrangement.
// Reads from texInput at position (x, y), writes to output at (y, x) if transpose is set.
// Used for transposing K cache for attention score computation.

uniform sampler2D texInput;
uniform bool doTranspose;
uniform ivec2 inputSize;
uniform int N;

out vec4 fragColor;

void main() {
    ivec2 outPos = ivec2(int(gl_FragCoord.x), int(gl_FragCoord.y));

    ivec2 inPos;
    if (doTranspose) {
        inPos = ivec2(outPos.y, outPos.x);
    } else {
        inPos = outPos;
    }

    if (inPos.x < inputSize.x && inPos.y < inputSize.y &&
        outPos.x < inputSize.y && outPos.y < inputSize.x) {
        vec4 val = texelFetch(texInput, inPos, 0);
        fragColor = val;
    } else {
        discard;
    }
}
