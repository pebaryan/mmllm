#version 330 core

// GELU activation: GELU(x) ≈ 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Accurate approximation using the tanh formulation.

uniform sampler2D texInput;
uniform int N;

out vec4 fragColor;

void main() {
    int nx = int(gl_FragCoord.x);

    vec4 x = texelFetch(texInput, ivec2(nx, 0), 0);

    const float sqrt2pi = 0.7978845608028654;  // sqrt(2.0 / 3.141592653589793)
    const float coeff = 0.044715;

    vec4 x3 = x * x * x;
    vec4 inner = sqrt2pi * (x + coeff * x3);
    vec4 gelu = 0.5 * x * (1.0 + tanh(inner));

    int base = nx * 4;
    if (base + 0 >= N) gelu.r = 0.0;
    if (base + 1 >= N) gelu.g = 0.0;
    if (base + 2 >= N) gelu.b = 0.0;
    if (base + 3 >= N) gelu.a = 0.0;

    fragColor = gelu;
}
