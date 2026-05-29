#pragma once
#include <GL/glew.h>
#include <string>

struct GLFWwindow;

namespace gl {

class Context {
public:
    Context();
    ~Context();

    bool init(int width = 256, int height = 256, const std::string& title = "mmllm");
    void destroy();
    void swapBuffers();
    void pollEvents();
    bool shouldClose() const;
    GLFWwindow* window() const { return window_; }

    // Prevent copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;

private:
    GLFWwindow* window_ = nullptr;
};

} // namespace gl
