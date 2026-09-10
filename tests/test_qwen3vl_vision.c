#include "ds4.h"
#include "ds4_image.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int write_f32(const char *path, const float *data, size_t count) {
    FILE *fp = fopen(path, "wb");
    if (!fp) {
        fprintf(stderr, "%s: %s\n", path, strerror(errno));
        return 0;
    }
    const int ok = fwrite(data, sizeof(float), count, fp) == count &&
                   fclose(fp) == 0;
    if (!ok) fprintf(stderr, "cannot write %s\n", path);
    return ok;
}

int main(int argc, char **argv) {
    if (argc != 4 && argc != 5) {
        fprintf(stderr,
                "usage: %s MAIN.gguf MMPROJ.gguf IMAGE [OUTPUT.f32]\n",
                argv[0]);
        return 2;
    }
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.vision_path = argv[2];
    options.backend = DS4_BACKEND_CUDA;
    options.inspect_only = true;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    char error[256] = {0};
    ds4_vision_embedding embedding = {0};
    if (!ds4_engine_vision_encode_file(engine, argv[3], &embedding,
                                       error, sizeof(error))) {
        fprintf(stderr, "Qwen3-VL encode failed: %s\n", error);
        ds4_engine_close(engine);
        return 1;
    }
    const size_t count = (size_t)embedding.token_count * 5120u;
    int valid = embedding.data != NULL && embedding.token_count != 0u &&
        embedding.grid_width != 0u && embedding.grid_height != 0u &&
        (uint64_t)embedding.grid_width * embedding.grid_height ==
            embedding.token_count;
    double sum = 0.0;
    for (size_t i = 0; valid && i < count; i++) {
        valid = isfinite(embedding.data[i]);
        sum += embedding.data[i];
    }
    if (!valid) {
        fprintf(stderr, "Qwen3-VL embedding is empty, malformed, or nonfinite\n");
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    if (argc == 5 && !write_f32(argv[4], embedding.data, count)) {
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    printf("%ux%u -> %ux%u, grid %ux%u, %u tokens, sum %.9f\n",
           embedding.width, embedding.height,
           embedding.content_width, embedding.content_height,
           embedding.grid_width, embedding.grid_height,
           embedding.token_count, sum);

    const char *pair_paths[] = {argv[3], argv[3]};
    ds4_vision_embedding pair = {0};
    if (!ds4_engine_vision_encode_frame_files(
            engine, pair_paths, 2u, &pair, error, sizeof(error)) ||
        pair.grid_time != 1u || pair.token_count != embedding.token_count ||
        memcmp(pair.data, embedding.data, count * sizeof(float)) != 0) {
        fprintf(stderr, "Qwen3-VL identical frame-pair mismatch: %s\n", error);
        ds4_vision_embedding_free(&pair);
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    ds4_vision_embedding_free(&pair);

    const char *odd_paths[] = {argv[3], argv[3], argv[3]};
    ds4_vision_embedding odd = {0};
    if (!ds4_engine_vision_encode_frame_files(
            engine, odd_paths, 3u, &odd, error, sizeof(error)) ||
        odd.grid_time != 2u ||
        odd.token_count != 2u * embedding.token_count ||
        memcmp(odd.data, embedding.data, count * sizeof(float)) != 0 ||
        memcmp(odd.data + count, embedding.data, count * sizeof(float)) != 0) {
        fprintf(stderr, "Qwen3-VL odd frame-sequence mismatch: %s\n", error);
        ds4_vision_embedding_free(&odd);
        ds4_vision_embedding_free(&embedding);
        ds4_engine_close(engine);
        return 1;
    }
    ds4_vision_embedding_free(&odd);
    puts("Qwen3-VL temporal frame merge PASS");
    ds4_vision_embedding_free(&embedding);
    ds4_engine_close(engine);
    return 0;
}
