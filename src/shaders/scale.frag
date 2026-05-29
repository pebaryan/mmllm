#version 330 core

// Element-wise: out = a * scalar + b

uniform sampler2D texA;
uniform float scalar;
uniform vec4 bias;
uniform int N;

out vec4 fragColor;

void main() {
    int nx = int(gl_FragCoord.x);

    vec4 a = texelFetch(texA, ivec2(nx, 0), 0);
    vec4 result = a * scalar + bias;

    int base = nx * 4;
    if (base + 0 >= N) result.r = 0.0;
    if (base + 1 >= N) result.g = 0.0;
    if (base + 2 >= N) result.b = 0.0;
    if (base + 3 >= N) result.a = 0.0;

    fragColor = result;
}
