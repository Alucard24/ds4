/* Warm-process Qwen3.8 CUDA throughput probe. This is a reporting benchmark,
 * not a fixed performance gate: compare runs on the same GPU and clocks. */
#include "ds4.h"

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static void fail(const char *message) {
    fprintf(stderr, "FAIL: %s\n", message);
    exit(1);
}

static double now_seconds(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) fail("clock_gettime");
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
}

static char *read_all(const char *path) {
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "%s: %s\n", path, strerror(errno)); exit(1); }
    if (fseek(fp, 0, SEEK_END) != 0) fail("seek prompt");
    long n = ftell(fp);
    if (n < 0 || fseek(fp, 0, SEEK_SET) != 0) fail("size prompt");
    char *text = malloc((size_t)n + 1u);
    if (!text) fail("prompt allocation");
    if (n && fread(text, 1, (size_t)n, fp) != (size_t)n) fail("read prompt");
    text[n] = '\0';
    fclose(fp);
    return text;
}

int main(int argc, char **argv) {
    if (argc < 3 || argc > 4)
        fail("usage: test_qwen38_cuda_perf MODEL PROMPT_FILE [DECODE_TOKENS]");
    const int n_decode = argc == 4 ? atoi(argv[3]) : 128;
    if (n_decode <= 0 || n_decode > 2048) fail("invalid decode token count");
    char *text = read_all(argv[2]);
    ds4_engine_options opt = {
        .model_path = argv[1], .backend = DS4_BACKEND_CUDA,
        .context_size = 4096, .power_percent = 100,
    };
    ds4_engine *engine = NULL;
    double t0 = now_seconds();
    if (ds4_engine_open(&engine, &opt) != 0) fail("engine open");
    double t1 = now_seconds();
    if (ds4_engine_model_id(engine) != DS4_MODEL_ID_QWEN38)
        fail("requires Qwen3.8 model");
    ds4_tokens prompt = {0};
    ds4_tokenize_text(engine, text, &prompt);
    free(text);
    if (prompt.len <= 0 || prompt.len + n_decode >= 4096) fail("prompt/context size");
    ds4_session *session = NULL;
    if (ds4_session_create(&session, engine, 4096) != 0) fail("session create");
    char err[256] = {0};
    double c0 = now_seconds();
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) fail(err);
    double c1 = now_seconds();
    /* First sync includes lazy CUDA library/scratch initialization. Rebuild the
     * same prompt in the same session for a steady-state prefill measurement. */
    ds4_session_invalidate(session);
    double p0 = now_seconds();
    if (ds4_session_sync(session, &prompt, err, sizeof(err)) != 0) fail(err);
    double p1 = now_seconds();
    uint64_t rng = 1;
    double d0 = now_seconds();
    for (int i = 0; i < n_decode; i++) {
        const int token = ds4_session_sample(session, 0.0f, 0, 1.0f, 0.0f, &rng);
        if (token < 0 || ds4_session_eval(session, token, err, sizeof(err)) != 0)
            fail(err[0] ? err : "decode");
    }
    double d1 = now_seconds();
    printf("LOAD_SECONDS %.6f\n", t1 - t0);
    printf("COLD_PREFILL_TOKENS %d SECONDS %.6f TPS %.3f\n",
           prompt.len, c1 - c0, prompt.len / (c1 - c0));
    printf("PREFILL_TOKENS %d SECONDS %.6f TPS %.3f\n",
           prompt.len, p1 - p0, prompt.len / (p1 - p0));
    printf("DECODE_TOKENS %d SECONDS %.6f TPS %.3f\n",
           n_decode, d1 - d0, n_decode / (d1 - d0));
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    ds4_engine_close(engine);
    return 0;
}
