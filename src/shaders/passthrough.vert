#version 330 core

// Generates a full-screen quad from gl_VertexID (0,1,2,3).
// No VBOs or VAOs needed — just glDrawArrays(GL_TRIANGLE_STRIP, 0, 4).
void main() {
    float x = float((gl_VertexID & 1) << 2) - 1.0;  // -1, 1, -1, 1
    float y = float((gl_VertexID & 2) << 1) - 1.0;  // -1, -1, 1, 1
    gl_Position = vec4(x, y, 0.0, 1.0);
}
