#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>

int main() {
    EGLDisplay dpy;
    EGLint major, minor;

    /* Try EGL_DEFAULT_DISPLAY (X11 platform) */
    printf("=== EGL_DEFAULT_DISPLAY ===\n");
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    printf("eglGetDisplay: %s\n", dpy != EGL_NO_DISPLAY ? "OK" : "NO_DISPLAY");
    if (dpy != EGL_NO_DISPLAY) {
        if (eglInitialize(dpy, &major, &minor)) {
            printf("Init OK: %d.%d\n", major, minor);
            printf("Vendor: %s\n", eglQueryString(dpy, EGL_VENDOR));
            eglTerminate(dpy);
        } else {
            printf("Init FAILED, error: 0x%x\n", eglGetError());
        }
    }

    /* Try SURFACELESS_MESA */
    printf("\n=== SURFACELESS_MESA ===\n");
    PFNEGLGETPLATFORMDISPLAYEXTPROC gpd =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");
    if (gpd) {
        dpy = gpd(EGL_PLATFORM_SURFACELESS_MESA, EGL_DEFAULT_DISPLAY, NULL);
        if (dpy != EGL_NO_DISPLAY) {
            if (eglInitialize(dpy, &major, &minor)) {
                printf("Surfaceless OK: %d.%d (%s)\n", major, minor, eglQueryString(dpy, EGL_VENDOR));
                eglTerminate(dpy);
            } else {
                printf("Surfaceless init FAILED, error: 0x%x\n", eglGetError());
            }
        } else {
            printf("Surfaceless display FAILED, error: 0x%x\n", eglGetError());
        }
    } else {
        printf("eglGetPlatformDisplayEXT not available\n");
    }

    /* Try DEVICE_EXT */
    printf("\n=== DEVICE_EXT ===\n");
    if (gpd) {
        dpy = gpd(EGL_PLATFORM_DEVICE_EXT, EGL_DEFAULT_DISPLAY, NULL);
        if (dpy != EGL_NO_DISPLAY) {
            if (eglInitialize(dpy, &major, &minor)) {
                printf("Device OK: %d.%d (%s)\n", major, minor, eglQueryString(dpy, EGL_VENDOR));
                eglTerminate(dpy);
            } else {
                printf("Device init FAILED, error: 0x%x\n", eglGetError());
            }
        } else {
            printf("Device display FAILED, error: 0x%x\n", eglGetError());
        }
    }

    /* Try DRM/GBM */
    printf("\n=== DRM/GBM ===\n");
    if (gpd) {
        int drm_fd = open("/dev/dri/card0", O_RDWR);
        if (drm_fd < 0) drm_fd = open("/dev/dri/card1", O_RDWR);
        if (drm_fd >= 0) {
            dpy = gpd(EGL_PLATFORM_GBM_MESA, (void*)(intptr_t)drm_fd, NULL);
            if (dpy != EGL_NO_DISPLAY) {
                if (eglInitialize(dpy, &major, &minor)) {
                    printf("DRM/GBM OK: %d.%d (%s)\n", major, minor, eglQueryString(dpy, EGL_VENDOR));
                    eglTerminate(dpy);
                } else {
                    printf("DRM/GBM init FAILED, error: 0x%x\n", eglGetError());
                }
            } else {
                printf("DRM/GBM display FAILED, error: 0x%x\n", eglGetError());
            }
            close(drm_fd);
        } else {
            printf("Cannot open /dev/dri/card0 or card1\n");
        }
    }

    /* Try with LIBGL_ALWAYS_SOFTWARE hint */
    printf("\n=== SOFTWARE RENDERER (via MESA env) ===\n");
    setenv("LIBGL_ALWAYS_SOFTWARE", "1", 1);
    dpy = eglGetDisplay(EGL_DEFAULT_DISPLAY);
    if (dpy != EGL_NO_DISPLAY && eglInitialize(dpy, &major, &minor)) {
        printf("Software EGL OK: %d.%d (%s)\n", major, minor, eglQueryString(dpy, EGL_VENDOR));
        eglTerminate(dpy);
    } else {
        printf("Software EGL FAILED\n");
    }

    return 0;
}
