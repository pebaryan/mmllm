#pragma once
#include <GL/glew.h>
#include <string>

namespace gl {

// Headless OpenGL 3.3 core context created through EGL (surfaceless).
// No X server, window system or DISPLAY/XAUTHORITY is needed: it talks to
// the GPU directly through the DRM render node.
class Context {
public:
    Context();
    ~Context();

    // width/height/title are accepted for API compatibility and ignored.
    bool init(int width = 256, int height = 256, const std::string& title = "mmllm");
    void destroy();

    // No window: kept so existing callers keep compiling.
    void swapBuffers() {}
    void pollEvents() {}
    bool shouldClose() const { return false; }

    // Prevent copy
    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    Context(Context&& other) noexcept;
    Context& operator=(Context&& other) noexcept;

private:
    void* display_ = nullptr;  // EGLDisplay
    void* context_ = nullptr;  // EGLContext
};

} // namespace gl
