#include "context.h"
#include "texture.h"     // gl::quiet
#include <GL/glew.h>
#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <cstdio>
#include <cstring>
#include <utility>

namespace gl {

Context::Context() {}

Context::~Context() {
    destroy();
}

// Get an EGLDisplay without any window system: prefer the Mesa surfaceless
// platform, fall back to the first EGL device (EGL_EXT_platform_device).
static EGLDisplay openHeadlessDisplay() {
    auto getPlatformDisplay = reinterpret_cast<PFNEGLGETPLATFORMDISPLAYEXTPROC>(
        eglGetProcAddress("eglGetPlatformDisplayEXT"));
    if (!getPlatformDisplay) {
        return EGL_NO_DISPLAY;
    }

    EGLDisplay dpy = getPlatformDisplay(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, nullptr);
    if (dpy != EGL_NO_DISPLAY) {
        return dpy;
    }

    auto queryDevices = reinterpret_cast<PFNEGLQUERYDEVICESEXTPROC>(
        eglGetProcAddress("eglQueryDevicesEXT"));
    if (queryDevices) {
        EGLDeviceEXT devices[8];
        EGLint n = 0;
        if (queryDevices(8, devices, &n) && n > 0) {
            return getPlatformDisplay(EGL_PLATFORM_DEVICE_EXT, devices[0], nullptr);
        }
    }
    return EGL_NO_DISPLAY;
}

bool Context::init(int /*width*/, int /*height*/, const std::string& /*title*/) {
    EGLDisplay dpy = openHeadlessDisplay();
    if (dpy == EGL_NO_DISPLAY) {
        std::fprintf(stderr, "[mmllm] No headless EGL display available (egl error 0x%x)\n", eglGetError());
        return false;
    }

    EGLint major = 0, minor = 0;
    if (!eglInitialize(dpy, &major, &minor)) {
        std::fprintf(stderr, "[mmllm] eglInitialize failed (0x%x)\n", eglGetError());
        return false;
    }

    const char* exts = eglQueryString(dpy, EGL_EXTENSIONS);
    if (!exts || !std::strstr(exts, "EGL_KHR_surfaceless_context")) {
        std::fprintf(stderr, "[mmllm] EGL_KHR_surfaceless_context not supported\n");
        eglTerminate(dpy);
        return false;
    }

    if (!eglBindAPI(EGL_OPENGL_API)) {
        std::fprintf(stderr, "[mmllm] eglBindAPI(OpenGL) failed (0x%x)\n", eglGetError());
        eglTerminate(dpy);
        return false;
    }

    // Request OpenGL 3.3 core profile (matches what nouveau can provide on NV50)
    const EGLint ctxAttribs[] = {
        EGL_CONTEXT_MAJOR_VERSION, 3,
        EGL_CONTEXT_MINOR_VERSION, 3,
        EGL_CONTEXT_OPENGL_PROFILE_MASK, EGL_CONTEXT_OPENGL_CORE_PROFILE_BIT,
        EGL_NONE
    };
    EGLContext ctx = eglCreateContext(dpy, EGL_NO_CONFIG_KHR, EGL_NO_CONTEXT, ctxAttribs);
    if (ctx == EGL_NO_CONTEXT) {
        std::fprintf(stderr, "[mmllm] Failed to create EGL context (GL 3.3 may not be supported, 0x%x)\n", eglGetError());
        eglTerminate(dpy);
        return false;
    }

    if (!eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, ctx)) {
        std::fprintf(stderr, "[mmllm] eglMakeCurrent failed (0x%x)\n", eglGetError());
        eglDestroyContext(dpy, ctx);
        eglTerminate(dpy);
        return false;
    }
    display_ = dpy;
    context_ = ctx;

    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    // libGLEW is built for GLX: after loading all the GL entry points it also
    // looks for a GLX display and reports GLEW_ERROR_NO_GLX_DISPLAY when there
    // is none. That is expected without X; the GL functions are already loaded.
    if (err != GLEW_OK && err != GLEW_ERROR_NO_GLX_DISPLAY) {
        std::fprintf(stderr, "[mmllm] GLEW init failed: %s\n", glewGetErrorString(err));
        destroy();
        return false;
    }

    // Log OpenGL info
    if (!quiet) std::printf("[mmllm] OpenGL %s  (%s, %s)  [EGL %d.%d headless]\n",
        glGetString(GL_VERSION),
        glGetString(GL_RENDERER),
        glGetString(GL_VENDOR),
        major, minor);

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
    if (display_) {
        EGLDisplay dpy = static_cast<EGLDisplay>(display_);
        eglMakeCurrent(dpy, EGL_NO_SURFACE, EGL_NO_SURFACE, EGL_NO_CONTEXT);
        if (context_) {
            eglDestroyContext(dpy, static_cast<EGLContext>(context_));
        }
        eglTerminate(dpy);
    }
    display_ = nullptr;
    context_ = nullptr;
}

Context::Context(Context&& other) noexcept
    : display_(other.display_), context_(other.context_)
{
    other.display_ = nullptr;
    other.context_ = nullptr;
}

Context& Context::operator=(Context&& other) noexcept {
    if (this != &other) {
        destroy();
        display_ = other.display_;
        context_ = other.context_;
        other.display_ = nullptr;
        other.context_ = nullptr;
    }
    return *this;
}

} // namespace gl
