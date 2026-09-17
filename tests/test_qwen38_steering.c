/* Qwen steering through the public API: no-op equivalence, live scale changes,
 * sublayer liveness, and cache incompatibility before state mutation.
 * Runs model instances sequentially, with private temporary direction files. */
#include "ds4.h"
#ifndef DS4_NO_GPU
#include "ds4_gpu.h"
#endif
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define WIDTH 5120u
#define LAYERS 64u
static char error[256];
static int nv;
static void check(int ok, const char *what) {
    if (!ok) { fprintf(stderr, "FAIL: %s (%s)\n", what, error); exit(1); }
}
static float *logits(ds4_session *s) {
    float *v = malloc((size_t)nv * sizeof(float));
    check(v != NULL, "malloc");
    check(ds4_session_copy_logits(s, v, nv) == nv, "copy logits");
    for (int i = 0; i < nv; i++) check(isfinite(v[i]), "finite logits");
    return v;
}
static void compare(ds4_session *s, const float *want, int equal) {
    float *v = logits(s);
    double delta = 0;
    for (int i = 0; i < nv; i++) delta = fmax(delta, fabs((double)v[i] - want[i]));
    printf("LOGITS %s max_delta=%.9g\n", equal ? "identical" : "changed", delta);
    check(equal ? memcmp(v, want, (size_t)nv * sizeof(float)) == 0 : delta > 1e-4,
          equal ? "bit-identical logits" : "steering liveness");
    free(v);
}
static ds4_session *session(ds4_engine *e, ds4_tokens *p) {
    ds4_session *s = NULL;
    check(ds4_session_create(&s, e, 128) == 0, "create session");
    check(ds4_session_sync(s, p, error, sizeof(error)) == 0, "prefill");
    return s;
}
static void save(ds4_session *s, FILE *f) {
    check(ds4_session_save_payload(s, f, error, sizeof(error)) == 0, "save payload");
    check((uint64_t)ftello(f) == ds4_session_payload_bytes(s), "payload size");
}
static int load(ds4_session *s, FILE *f, uint64_t bytes) {
    rewind(f);
    return ds4_session_load_payload(s, f, bytes, error, sizeof(error));
}
static void write_dirs(const char *path, const float *d) {
    FILE *f = fopen(path, "wb");
    check(f != NULL, "open directions");
    check(fwrite(d, sizeof(float), WIDTH * LAYERS, f) == WIDTH * LAYERS, "write directions");
    check(fclose(f) == 0, "close directions");
}
#ifndef DS4_NO_GPU
static void projection_oracle(void) {
    const uint32_t rows = 17;
    const size_t n = rows * WIDTH;
    float *x = malloc(n * sizeof(float)), *got = malloc(n * sizeof(float));
    float *dirs = calloc(2 * WIDTH, sizeof(float));
    check(x && got && dirs, "oracle allocations");
    for (size_t i = 0; i < n; i++) x[i] = sinf((float)i * 0.03f);
    double norm = 0;
    for (uint32_t i = 0; i < WIDTH; i++) {
        dirs[WIDTH + i] = cosf((float)i * 0.013f);
        norm += (double)dirs[WIDTH + i] * dirs[WIDTH + i];
    }
    for (uint32_t i = 0; i < WIDTH; i++) dirs[WIDTH + i] /= sqrt(norm);
    check(ds4_gpu_init(), "GPU init");
    ds4_gpu_tensor *tx = ds4_gpu_tensor_alloc(n * sizeof(float));
    ds4_gpu_tensor *td = ds4_gpu_tensor_alloc(2 * WIDTH * sizeof(float));
    check(tx && td, "GPU oracle allocations");
    check(ds4_gpu_tensor_write(td, 0, dirs, 2 * WIDTH * sizeof(float)), "write oracle dirs");
    const float scales[] = {1.0f, -0.7f};
    for (int k = 0; k < 2; k++) {
        check(ds4_gpu_tensor_write(tx, 0, x, n * sizeof(float)), "write oracle input");
        check(ds4_gpu_directional_steering_project_tensor(tx, td, 1, WIDTH, rows, scales[k]),
              "project CUDA rows");
        check(ds4_gpu_tensor_read(tx, 0, got, n * sizeof(float)), "read projection");
        double maxerr = 0;
        for (uint32_t r = 0; r < rows; r++) {
            double dot = 0;
            for (uint32_t i = 0; i < WIDTH; i++) dot += (double)x[r * WIDTH + i] * dirs[WIDTH + i];
            for (uint32_t i = 0; i < WIDTH; i++) {
                double ref = x[r * WIDTH + i] - scales[k] * dot * dirs[WIDTH + i];
                check(isfinite(got[r * WIDTH + i]), "finite projection");
                maxerr = fmax(maxerr, fabs(got[r * WIDTH + i] - ref));
            }
        }
        printf("PROJECTION scale=%g rows=%u max_error=%.9g\n", scales[k], rows, maxerr);
        check(maxerr < 2e-6, "CUDA projection versus double CPU oracle");
    }
    ds4_gpu_tensor_free(tx); ds4_gpu_tensor_free(td);
    free(x); free(got); free(dirs);
}
#endif
int main(int argc, char **argv) {
    check(argc == 2, "usage: test_qwen38_steering MODEL");
    const int cpu = getenv("DS4_TEST_STEERING_CPU") != NULL;
#ifndef DS4_NO_GPU
    if (!cpu) projection_oracle();
#endif
    ds4_engine_options opt = {.model_path=argv[1], .context_size=128, .power_percent=100,
        .backend=cpu ? DS4_BACKEND_CPU : DS4_BACKEND_CUDA};
    char path[] = "/tmp/ds4-steering-test-XXXXXX";
    int fd = mkstemp(path); check(fd >= 0, "mkstemp"); close(fd);
    float *dirs = calloc(WIDTH * LAYERS, sizeof(float)); check(dirs != NULL, "dirs");
    ds4_engine *e = NULL;
    check(ds4_engine_open(&e, &opt) == 0, "baseline engine");
    nv = (int)ds4_engine_vocab_size(e);
    ds4_tokens p = {0};
    ds4_tokenize_text(e, "Hello world.", &p);
    check(p.len > 1, "chunk prompt");
    ds4_session *s = session(e, &p);
    float *base = logits(s);
    FILE *normal = tmpfile(); check(normal != NULL, "normal tmpfile"); save(s, normal);
    uint64_t normal_bytes = ds4_session_payload_bytes(s);
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "baseline decode");
    float *base_decode = logits(s);
    ds4_session_free(s); ds4_engine_close(e);

    /* A zero direction at nonzero scale must remain a numerical no-op. */
    write_dirs(path, dirs); opt.directional_steering_file = path; opt.directional_steering_ffn = 1;
    check(ds4_engine_open(&e, &opt) == 0, "zero direction engine");
    s = session(e, &p); compare(s, base, 1);
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "zero decode");
    compare(s, base_decode, 1);
    ds4_session_free(s); ds4_engine_close(e);

    /* Deterministic, normalized directions. No behavior-targeted dataset. */
    uint32_t rng = 12345;
    for (uint32_t l = 0; l < LAYERS; l++) {
        double norm = 0;
        for (uint32_t i = 0; i < WIDTH; i++) {
            rng = rng * 1664525u + 1013904223u;
            dirs[l * WIDTH + i] = (float)(int32_t)(rng >> 8) / 8388608.0f - 1.0f;
            norm += (double)dirs[l * WIDTH + i] * dirs[l * WIDTH + i];
        }
        for (uint32_t i = 0; i < WIDTH; i++) dirs[l * WIDTH + i] /= sqrt(norm);
    }
    write_dirs(path, dirs); opt.directional_steering_ffn = 0;
    check(ds4_engine_open(&e, &opt) == 0, "zero scale engine");
    s = session(e, &p); compare(s, base, 1);
    check(load(s, normal, normal_bytes) == 0, "zero scale accepts normal payload");
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "zero scale decode");
    compare(s, base_decode, 1);
    check(ds4_session_set_directional_steering_ffn(s, 1) == 0, "live scale");
    check(ds4_session_directional_steering_ffn(s) == 1 && ds4_session_pos(s) == 0,
          "getter and invalidated prefix");
    check(ds4_session_sync(s, &p, error, sizeof(error)) == 0, "steered rebuild");
    compare(s, base, 0);
    ds4_session *other = session(e, &p); compare(other, base, 1); /* session-local scale */
    ds4_session_free(other);
    float *steered = logits(s);
    FILE *cache = tmpfile(); check(cache != NULL, "steered tmpfile"); save(s, cache);
    uint64_t bytes = ds4_session_payload_bytes(s);
    check(load(s, normal, normal_bytes) != 0, "normal -> steered rejected");
    compare(s, steered, 1);
    /* Exact same shape/scale/directions but another GGUF identity must fail
     * before mutating the live timeline. The extension starts after 13 u32s. */
    check(fseeko(cache, 13u * sizeof(uint32_t), SEEK_SET) == 0, "identity seek");
    uint64_t identity;
    check(fread(&identity, sizeof(identity), 1, cache) == 1, "identity read");
    uint64_t wrong = identity ^ 1u;
    check(fseeko(cache, 13u * sizeof(uint32_t), SEEK_SET) == 0, "identity rewind");
    check(fwrite(&wrong, sizeof(wrong), 1, cache) == 1, "identity tamper");
    check(load(s, cache, bytes) != 0, "model identity mismatch rejected"); compare(s, steered, 1);
    check(fseeko(cache, 13u * sizeof(uint32_t), SEEK_SET) == 0, "identity restore seek");
    check(fwrite(&identity, sizeof(identity), 1, cache) == 1, "identity restore");
    const char *export_path = getenv("DS4_TEST_STEERING_EXPORT");
    if (export_path) {
        FILE *out = fopen(export_path, "wb"); check(out != NULL, "export open");
        save(s, out); check(fclose(out) == 0, "export close");
    }
    const char *import_path = getenv("DS4_TEST_STEERING_IMPORT");
    if (import_path) {
        FILE *in = fopen(import_path, "rb"); check(in != NULL, "import open");
        check(fseeko(in, 0, SEEK_END) == 0, "import size");
        uint64_t size = (uint64_t)ftello(in);
        check(load(s, in, size) == 0, "cross-backend payload import");
        check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "cross-backend continuation");
        float *v = logits(s); free(v); fclose(in);
        check(load(s, cache, bytes) == 0, "restore after import");
        puts("CROSS-BACKEND payload accepted and continued with finite logits");
    }
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "steered decode");
    compare(s, base_decode, 0);
    float *next = logits(s);
    check(load(s, cache, bytes) == 0, "matching steered payload"); compare(s, steered, 1);
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "steered continuation");
    compare(s, next, 1);
    check(load(s, cache, bytes - 1) != 0, "truncated steered payload"); compare(s, next, 1);
    check(ds4_session_set_directional_steering_ffn(s, -1) == 0, "negative scale");
    check(ds4_session_sync(s, &p, error, sizeof(error)) == 0, "negative prefill");
    float *negative = logits(s);
    check(load(s, cache, bytes) != 0, "scale mismatch rejected"); compare(s, negative, 1);
    check(ds4_session_set_directional_steering_ffn(s, 0) == 0, "disable steering");
    check(ds4_session_sync(s, &p, error, sizeof(error)) == 0, "normal rebuild"); compare(s, base, 1);
    check(load(s, cache, bytes) != 0, "steered -> normal rejected"); compare(s, base, 1);
    ds4_session_free(s); ds4_engine_close(e);

    opt.directional_steering_ffn = 1;
    dirs[0] += 0.01f; write_dirs(path, dirs);
    check(ds4_engine_open(&e, &opt) == 0, "different direction engine");
    s = session(e, &p);
    float *changed = logits(s);
    check(load(s, cache, bytes) != 0, "direction contents mismatch rejected"); compare(s, changed, 1);
    ds4_session_free(s); ds4_engine_close(e);

    opt.directional_steering_ffn = 0; opt.directional_steering_attn = 1;
    check(ds4_engine_open(&e, &opt) == 0, "attention engine");
    s = session(e, &p); compare(s, base, 0);
    check(ds4_session_eval(s, p.v[0], error, sizeof(error)) == 0, "attention decode");
    compare(s, base_decode, 0);
    ds4_session_free(s); ds4_engine_close(e);

    /* File validation must run even with both scales zero. */
    opt.directional_steering_attn = 0;
    dirs[0] = NAN; write_dirs(path, dirs);
    check(ds4_engine_open(&e, &opt) != 0, "NaN direction rejected");
    check(truncate(path, 12) == 0, "truncate direction");
    check(ds4_engine_open(&e, &opt) != 0, "wrong shape rejected");
    unlink(path); fclose(cache); fclose(normal); ds4_tokens_free(&p);
    free(base); free(base_decode); free(dirs); free(steered); free(next); free(negative); free(changed);
    puts("Qwen steering PASS");
    return 0;
}
