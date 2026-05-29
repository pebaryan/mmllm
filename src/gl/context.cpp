#include "context.h"
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cstdio>
#include <stdexcept>
#include <utility>

namespace gl {

Context::Context() {}

Context::~Context() {
    destroy();
}

bool Context::init(int width, int height, const std::string& title) {
    if (!glfwInit()) {
        std::fprintf(stderr, "[mmllm] Failed to initialize GLFW\n");
        return false;
    }

    // Request OpenGL 3.3 core profile (matches what nouveau can provide on NV50)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);

    // Off-screen mode: don't show the window, but still need a context
    glfwWindowHint(GLFW_VISIBLE, GLFW_FALSE);

    window_ = glfwCreateWindow(width, height, title.c_str(), nullptr, nullptr);
    if (!window_) {
        std::fprintf(stderr, "[mmllm] Failed to create GLFW window (GL 3.3 may not be supported)\n");
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window_);

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        std::fprintf(stderr, "[mmllm] GLEW init failed: %s\n", glewGetErrorString(err));
        glfwDestroyWindow(window_);
        window_ = nullptr;
        glfwTerminate();
        return false;
    }

    // Log OpenGL info
    std::printf("[mmllm] OpenGL %s  (%s, %s)\n",
        glGetString(GL_VERSION),
        glGetString(GL_RENDERER),
        glGetString(GL_VENDOR));

    // Verify ARB_texture_float is available (needed for RGBA32F)
    if (!GLEW_ARB_texture_float) {
        std::fprintf(stderr, "[mmllm] WARNING: ARB_texture_float not supported!\n");
        std::fprintf(stderr, "          Will attempt to continue anyway.\n");
    }

    // Create and bind a default VAO (required by OpenGL 3.3 core profile)
    GLuint defaultVAO;
    glGenVertexArrays(1, &defaultVAO);
    glBindVertexArray(defaultVAO);

    return true;
}

void Context::destroy() {
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

void Context::swapBuffers() {
    glfwSwapBuffers(window_);
}

void Context::pollEvents() {
    glfwPollEvents();
}

bool Context::shouldClose() const {
    return window_ ? glfwWindowShouldClose(window_) : true;
}

Context::Context(Context&& other) noexcept
    : window_(other.window_)
{
    other.window_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        destroy();
        window_ = other.window_;
        other.window_ = nullptr;
    }
    return *this;
}

} // namespace gl
