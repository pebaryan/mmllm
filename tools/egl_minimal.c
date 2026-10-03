#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdio.h>

int main() {
    EGLDisplay dpy;
    EGLint major, minor;

    printf("Try 1: eglGetDisplay(DEFAULT)\n");
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    printf("  display=%p\n", (void*)dpy);
    if (dpy != EGL_NO_DISPLAY) {
        printf("  Calling eglInitialize...\n");
        if (eglInitialize(dpy, &major, &minor)) {
            printf("  OK: %d.%d\n", major, minor);
            printf("  Vendor: %s\n", eglQueryString(dpy, EGL_VENDOR));
            eglTerminate(dpy);
        } else {
            printf("  FAILED, error=0x%x\n", eglGetError());
        }
    }

    printf("\nTry 2: eglGetPlatformDisplay(SURFACELESS)\n");
    /* Use the native EGL 1.5 function if available */
    PFNEGLGETPLATFORMDISPLAYPROC gpd_native =
        (PFNEGLGETPLATFORMDISPLAYPROC)eglGetProcAddress("eglGetPlatformDisplay");
    if (!gpd_native)
        gpd_native = (PFNEGLGETPLATFORMDISPLAYPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (gpd_native) {
        EGLAttrib attrs[] = { EGL_NONE };
        dpy = gpd_native(0x31D9 /* EGL_PLATFORM_SURFACELESS_MESA */,
                         EGL_DEFAULT_DISPLAY, attrs);
        printf("  display=%p\n", (void*)dpy);
        if (dpy != EGL_NO_DISPLAY) {
            if (eglInitialize(dpy, &major, &minor)) {
                printf("  OK: %d.%d\n", major, minor);
                eglTerminate(dpy);
            } else {
                printf("  FAILED, error=0x%x\n", eglGetError());
            }
        } else {
            printf("  FAILED, error=0x%x\n", eglGetError());
        }
    } else {
        printf("  eglGetPlatformDisplay not available\n");
    }

    printf("\nDone.\n");
    return 0;
}
