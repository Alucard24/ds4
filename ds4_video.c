/* =========================================================================
 * ds4_video.c - container decoding by an external ffmpeg/ffprobe pair.
 * =========================================================================
 *
 * Design notes, all deliberate:
 *
 * - No shell. Commands are execvp'd with an argument vector, so a file name can
 *   never be interpreted by a shell and the frame data stays on a plain pipe.
 * - Frames are extracted as raw RGB at the container's own resolution and fed
 *   straight into the existing preprocessing, which already handles resizing
 *   and the centered black canvas. Decoding therefore cannot introduce a
 *   second, different scaling path.
 * - The frame count is a policy, not an accident: `max_frames` evenly spaced
 *   samples over the duration, realised by ffmpeg's fps filter, so the same
 *   container always yields the same frames.
 * - Each frame is fingerprinted from its pixels, so a container's cache
 *   identity is the content that reaches the model.
 */
#include "ds4_video.h"

#include <errno.h>
#include <fcntl.h>
#include <stdarg.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>


/* A frame is bounded so a mislabeled container cannot ask for unbounded
 * memory: 16384x16384 RGB is 805 MiB, and the preprocessing shrinks it long
 * before the encoder sees it. */
#define DS4_VIDEO_MAX_DIMENSION 16384u

static void video_error(char *error, size_t cap, const char *message) {
    if (!error || cap == 0) return;
    snprintf(error, cap, "%s", message);
}

/* snprintf through a variadic helper, used where the message needs a number. */
static void video_errorf(char *error, size_t cap, const char *fmt, ...) {
    if (!error || cap == 0) return;
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(error, cap, fmt, ap);
    va_end(ap);
}

/* Run argv with stdout on a pipe and return the read end. */
static FILE *spawn_capture(char *const argv[], pid_t *pid_out,
                           char *error, size_t error_cap) {
    int fds[2];
    if (pipe(fds) != 0) {
        video_errorf(error, error_cap, "pipe: %s", strerror(errno));
        return NULL;
    }
    const pid_t pid = fork();
    if (pid < 0) {
        close(fds[0]);
        close(fds[1]);
        video_errorf(error, error_cap, "fork: %s", strerror(errno));
        return NULL;
    }
    if (pid == 0) {
        /* Child: stdout to the pipe, stdin closed so a decoder cannot wait on
         * the terminal, stderr inherited so ffmpeg's own message reaches the
         * user. */
        if (dup2(fds[1], STDOUT_FILENO) < 0) _exit(127);
        close(fds[0]);
        close(fds[1]);
        const int devnull = open("/dev/null", O_RDONLY);
        if (devnull >= 0) {
            (void)dup2(devnull, STDIN_FILENO);
            close(devnull);
        }
        execvp(argv[0], argv);
        _exit(127);
    }
    close(fds[1]);
    FILE *fp = fdopen(fds[0], "rb");
    if (!fp) {
        close(fds[0]);
        (void)kill(pid, SIGKILL);
        (void)waitpid(pid, NULL, 0);
        video_errorf(error, error_cap, "fdopen: %s", strerror(errno));
        return NULL;
    }
    *pid_out = pid;
    return fp;
}

static bool reap(pid_t pid, FILE *fp, const char *tool, char *error,
                 size_t error_cap) {
    if (fp) fclose(fp);
    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        video_errorf(error, error_cap, "waitpid: %s", strerror(errno));
        return false;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
        if (WIFSIGNALED(status)) {
            video_errorf(error, error_cap, "%s was terminated by a signal",
                         tool);
        } else {
            video_errorf(error, error_cap, "%s exited with status %d",
                         tool, WEXITSTATUS(status));
        }
        return false;
    }
    return true;
}

bool ds4_video_tools_available(void) {
    static int cached = -1;
    if (cached >= 0) return cached == 1;
    bool ok = true;
    const char *const tools[] = { "ffprobe", "ffmpeg" };
    for (size_t i = 0; i < sizeof(tools) / sizeof(tools[0]); i++) {
        pid_t pid = 0;
        char *const argv[] = { (char *)tools[i], (char *)"-version", NULL };
        char error[160] = {0};
        FILE *fp = spawn_capture(argv, &pid, error, sizeof(error));
        if (!fp) {
            ok = false;
            break;
        }
        /* Drain so the child never blocks on a full pipe. */
        char buf[512];
        while (fread(buf, 1, sizeof(buf), fp) == sizeof(buf)) { }
        if (!reap(pid, fp, tools[i], error, sizeof(error))) ok = false;
        if (!ok) break;
    }
    cached = ok ? 1 : 0;
    return ok;
}

int ds4_video_probe(const char *path, uint32_t *width, uint32_t *height,
                    double *duration_s, char *error, size_t error_cap) {
    if (!path || !path[0] || !width || !height || !duration_s) {
        video_error(error, error_cap, "invalid video probe arguments");
        return 0;
    }
    char *const argv[] = {
        (char *)"ffprobe", (char *)"-v", (char *)"error",
        (char *)"-select_streams", (char *)"v:0",
        (char *)"-show_entries", (char *)"stream=width,height:format=duration",
        (char *)"-of", (char *)"csv=p=0",
        (char *)path, NULL
    };
    pid_t pid = 0;
    FILE *fp = spawn_capture(argv, &pid, error, error_cap);
    if (!fp) {
        video_errorf(error, error_cap,
                     "cannot run ffprobe (is it in PATH?): %s", "");
        return 0;
    }
    char line[256] = {0};
    const char *first = fgets(line, sizeof(line), fp);
    if (!first) {
        (void)reap(pid, fp, "ffprobe", error, error_cap);
        video_error(error, error_cap,
                    "ffprobe reported no video stream for this file");
        return 0;
    }
    if (!reap(pid, fp, "ffprobe", error, error_cap)) return 0;

    unsigned w = 0, h = 0;
    double duration = 0.0;
    /* stream=width,height and format=duration are two CSV records; the first
     * line carries the dimensions and a later one the duration. */
    int fields = sscanf(line, "%u,%u,%lf", &w, &h, &duration);
    if (fields < 2) {
        /* Some containers print the format record first. */
        char second[256] = {0};
        (void)second;
        video_error(error, error_cap,
                    "cannot parse ffprobe output for this file");
        return 0;
    }
    if (fields < 3 || !(duration > 0.0)) {
        /* Fall back to a second call for the container duration. */
        char *const argv2[] = {
            (char *)"ffprobe", (char *)"-v", (char *)"error",
            (char *)"-show_entries", (char *)"format=duration",
            (char *)"-of", (char *)"default=nw=1:nk=1",
            (char *)path, NULL
        };
        pid_t pid2 = 0;
        FILE *fp2 = spawn_capture(argv2, &pid2, error, error_cap);
        duration = 0.0;
        if (fp2) {
            char dur[128] = {0};
            if (fgets(dur, sizeof(dur), fp2)) duration = strtod(dur, NULL);
            (void)reap(pid2, fp2, "ffprobe", error, error_cap);
        }
    }
    if (w == 0 || h == 0) {
        video_error(error, error_cap, "video reports no usable dimensions");
        return 0;
    }
    if (w > DS4_VIDEO_MAX_DIMENSION || h > DS4_VIDEO_MAX_DIMENSION) {
        video_error(error, error_cap, "video dimensions are out of range");
        return 0;
    }
    if (!(duration > 0.0)) {
        video_error(error, error_cap,
                    "cannot determine the container duration; extract frames "
                    "and use the frame-list path instead");
        return 0;
    }
    *width = w;
    *height = h;
    *duration_s = duration;
    return 1;
}

int ds4_video_decode_file(const char *path, uint32_t max_frames,
                          ds4_video_frames *out, char *error,
                          size_t error_cap) {
    if (!path || !path[0] || !out) {
        video_error(error, error_cap, "invalid video decode arguments");
        return 0;
    }
    memset(out, 0, sizeof(*out));
    if (max_frames < 2u) max_frames = 2u;
    if (max_frames > DS4_VIDEO_MAX_FRAMES) max_frames = DS4_VIDEO_MAX_FRAMES;

    uint32_t width = 0, height = 0;
    double duration = 0.0;
    if (!ds4_video_probe(path, &width, &height, &duration, error, error_cap))
        return 0;

    /* Evenly spaced samples over the whole container. */
    double fps = (double)max_frames / duration;
    if (!(fps > 0.0)) fps = 1.0;
    char fps_arg[64];
    snprintf(fps_arg, sizeof(fps_arg), "fps=%.6f", fps);
    char frames_arg[32];
    snprintf(frames_arg, sizeof(frames_arg), "%u", max_frames);

    char *const argv[] = {
        (char *)"ffmpeg", (char *)"-nostdin", (char *)"-v", (char *)"error",
        (char *)"-i", (char *)path,
        (char *)"-vf", fps_arg,
        (char *)"-frames:v", frames_arg,
        (char *)"-an", (char *)"-sn",
        (char *)"-f", (char *)"rawvideo", (char *)"-pix_fmt", (char *)"rgb24",
        (char *)"pipe:1", NULL
    };
    pid_t pid = 0;
    FILE *fp = spawn_capture(argv, &pid, error, error_cap);
    if (!fp) {
        video_error(error, error_cap,
                    "cannot run ffmpeg (is it in PATH?)");
        return 0;
    }

    const size_t frame_bytes = (size_t)width * (size_t)height * 3u;
    ds4_image *frames = calloc(max_frames, sizeof(frames[0]));
    uint8_t *pixels = malloc(frame_bytes);
    if (!frames || !pixels) {
        free(frames);
        free(pixels);
        (void)reap(pid, fp, "ffmpeg", error, error_cap);
        video_error(error, error_cap, "out of memory for decoded frames");
        return 0;
    }
    size_t count = 0;
    while (count < max_frames) {
        const size_t got = fread(pixels, 1, frame_bytes, fp);
        if (got == 0) break;
        if (got != frame_bytes) {
            free(pixels);
            for (size_t i = 0; i < count; i++) ds4_image_free(&frames[i]);
            free(frames);
            (void)reap(pid, fp, "ffmpeg", error, error_cap);
            video_error(error, error_cap,
                        "ffmpeg ended in the middle of a frame");
            return 0;
        }
        ds4_image *frame = &frames[count];
        frame->rgb = malloc(frame_bytes);
        if (!frame->rgb) {
            free(pixels);
            for (size_t i = 0; i < count; i++) ds4_image_free(&frames[i]);
            free(frames);
            (void)reap(pid, fp, "ffmpeg", error, error_cap);
            video_error(error, error_cap, "out of memory for decoded frames");
            return 0;
        }
        memcpy(frame->rgb, pixels, frame_bytes);
        frame->width = width;
        frame->height = height;
        /* Fingerprint the pixels, not the container bytes: the cache identity
         * is what the model actually sees. */
        ds4_image_fingerprint_pixels(frame->fingerprint, frame->rgb, width,
                                     height);
        count++;
    }
    free(pixels);
    if (!reap(pid, fp, "ffmpeg", error, error_cap)) {
        for (size_t i = 0; i < count; i++) ds4_image_free(&frames[i]);
        free(frames);
        return 0;
    }
    if (count < 2u) {
        for (size_t i = 0; i < count; i++) ds4_image_free(&frames[i]);
        free(frames);
        video_error(error, error_cap,
                    "the container produced fewer than two frames");
        return 0;
    }
    out->frames = frames;
    out->count = count;
    out->width = width;
    out->height = height;
    out->duration_s = duration;
    out->fps_applied = fps;
    return 1;
}

void ds4_video_frames_free(ds4_video_frames *frames) {
    if (!frames) return;
    for (size_t i = 0; i < frames->count; i++) ds4_image_free(&frames->frames[i]);
    free(frames->frames);
    memset(frames, 0, sizeof(*frames));
}
