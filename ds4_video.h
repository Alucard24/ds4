/* =========================================================================
 * ds4_video.h - video container decoding for the vision front ends.
 * =========================================================================
 *
 * The engine consumes ordered RGB frames, not containers. Reading an .mp4 or
 * .mkv therefore means handing the container to an external decoder and
 * keeping everything after that identical to the frame-list path: same pixel
 * buffers, same preprocessing, same embeddings, same cache identity.
 *
 * Decoding is delegated to ffprobe/ffmpeg from PATH, exactly as llama.cpp's
 * mtmd does. There is no link-time dependency: when the tools are missing the
 * call fails with a message that names them, and the frame-list path keeps
 * working.
 */
#ifndef DS4_VIDEO_H
#define DS4_VIDEO_H

#include "ds4_image.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* Frames one container may contribute. Large enough for the model's temporal
 * budget (2..128 frames after the caller's own cap) and small enough that a
 * full decode stays bounded in memory. */
#define DS4_VIDEO_MAX_FRAMES 128u
#define DS4_VIDEO_DEFAULT_FRAMES 32u

typedef struct {
    ds4_image *frames;
    size_t     count;
    uint32_t   width;
    uint32_t   height;
    double     duration_s;
    double     fps_applied;
} ds4_video_frames;

/* Probe dimensions and duration. Returns 0 and fills `error` when ffprobe is
 * missing, the file is not readable, or the duration is unknown. */
int ds4_video_probe(const char *path, uint32_t *width, uint32_t *height,
                    double *duration_s, char *error, size_t error_cap);

/* Decode at most `max_frames` evenly spaced frames into RGB buffers. The frame
 * count is realised with ffmpeg's fps filter, so a container of any length maps
 * to a deterministic, bounded sample of it. */
int ds4_video_decode_file(const char *path, uint32_t max_frames,
                          ds4_video_frames *out, char *error,
                          size_t error_cap);

void ds4_video_frames_free(ds4_video_frames *frames);

/* True when both tools answer in PATH, so callers can report the real remedy
 * before the user waits for a decode that cannot work. */
bool ds4_video_tools_available(void);

#endif
