/* Boundary-shape drive of the Qwen GA prefill and widening kernels.
 *
 * Everything here is small on purpose: the point is to reach the index
 * arithmetic at launch time - the last position of the cache, tails shorter than
 * a vector load, both KV formats - with tiny tensors so that compute-sanitizer's
 * memcheck fits and reports the kernel and the address if anything steps outside
 * an allocation.  This is the harness the Xid incident needs
 * (docs/CUDA_XID_INCIDENT.md); it is a diagnostic, not a gate.
 *
 * Build (objects from a normal `make ds4`):
 *   nvcc -O3 -arch=sm_120a -I. -o /tmp/kv_bounds tests/qwen38_kv_bounds.cu \
 *     ds4.o ds4_image.o ds4_video.o ds4_distributed.o ds4_tp.o ds4_ssd.o \
 *     ds4_cuda.o ds4_layer_pack.o cuda/mmq/*.o \
 *     -lm -lcudart -lcublas -L/opt/cuda/targets/sbsa-linux/lib -L/opt/cuda/lib64
 * Run:
 *   ~/opt/cuda-cs/bin/compute-sanitizer --tool memcheck --log-file /tmp/cs-bounds.log \
 *     /tmp/kv_bounds MODEL.gguf
 */
#include "ds4_gpu.h"

#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define HEADS     24u   /* query heads */
#define HEADS_KV   4u   /* key/value heads */
#define HEAD_DIM 256u
#define Q_VALUES (HEADS * HEAD_DIM * 2u)  /* query plus its gate: 12288 */
#define OUT_VALUES (HEADS * HEAD_DIM)
#define ROPE_VALUES 3u

static uint64_t kv_row_bytes(int fmt) {
    const uint64_t values = HEADS_KV * HEAD_DIM;
    if (fmt == 2) return (values / 32u) * 18u;   /* q4_0 */
    if (fmt == 1) return (values / 32u) * 34u;   /* q8_0 */
    return values * sizeof(_Float16);            /* f16   */
}

static int failures;
static uint64_t g_qnorm = 1607885600, g_knorm = 1578127136;

static void drive(const void *map, uint64_t map_size, uint32_t ctx,
                  uint32_t start_pos, uint32_t n_tokens) {
    for (int fmt = 0; fmt <= 2; fmt++) {
        ds4_gpu_qwen38_set_kv_fmt(fmt);
        const uint64_t row = kv_row_bytes(fmt);
        ds4_gpu_tensor *k_cache = ds4_gpu_tensor_alloc((uint64_t)ctx * row);
        ds4_gpu_tensor *v_cache = ds4_gpu_tensor_alloc((uint64_t)ctx * row);
        ds4_gpu_tensor *q_full = ds4_gpu_tensor_alloc((uint64_t)n_tokens * Q_VALUES * sizeof(float));
        ds4_gpu_tensor *k = ds4_gpu_tensor_alloc((uint64_t)n_tokens * HEADS_KV * HEAD_DIM * sizeof(float));
        ds4_gpu_tensor *v = ds4_gpu_tensor_alloc((uint64_t)n_tokens * HEADS_KV * HEAD_DIM * sizeof(float));
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n_tokens * OUT_VALUES * sizeof(float));
        ds4_gpu_tensor *rope = ds4_gpu_tensor_alloc((uint64_t)n_tokens * ROPE_VALUES * sizeof(uint32_t));
        if (!k_cache || !v_cache || !q_full || !k || !v || !out || !rope) {
            printf("  alloc failed ctx=%u n=%u fmt=%d\n", ctx, n_tokens, fmt);
            failures++;
            return;
        }
        /* A fill of 0.25 is finite and boring; values are irrelevant to bounds. */
        float *zeros = (float *)calloc((size_t)Q_VALUES * n_tokens, sizeof(float));
        for (uint64_t i = 0; i < (uint64_t)Q_VALUES * n_tokens; i++) zeros[i] = 0.25f;
        ds4_gpu_tensor_write(q_full, 0, zeros, (uint64_t)Q_VALUES * n_tokens * sizeof(float));
        ds4_gpu_tensor_write(k, 0, zeros, (uint64_t)HEADS_KV * HEAD_DIM * n_tokens * sizeof(float));
        ds4_gpu_tensor_write(v, 0, zeros, (uint64_t)HEADS_KV * HEAD_DIM * n_tokens * sizeof(float));
        (void)0;
        uint32_t *pos = (uint32_t *)calloc((size_t)n_tokens * ROPE_VALUES, sizeof(uint32_t));
        for (uint32_t t = 0; t < n_tokens; t++) {
            pos[t * 3u] = pos[t * 3u + 1u] = pos[t * 3u + 2u] = start_pos + t;
        }
        ds4_gpu_tensor_write(rope, 0, pos, (uint64_t)n_tokens * ROPE_VALUES * sizeof(uint32_t));
        free(zeros);
        free(pos);

        /* Norm-weight offsets of 0 read the mapped file's header instead of the
         * real tensors: the data is nonsense, the addresses are valid. */
        const int s1 = ds4_gpu_qwen38_ga_prepare_chunk(q_full, k_cache, v_cache, k, v,
                                                      map, map_size, g_qnorm, g_knorm, rope,
                                                      start_pos, n_tokens, ctx);
        const int s2 = s1 ? ds4_gpu_qwen38_ga_chunk(out, q_full, k_cache, v_cache,
                                                    start_pos, n_tokens, ctx) : 0;
        const int s3 = s2 ? ds4_gpu_qwen38_ga_decode(out, q_full, k_cache, v_cache,
                                                     start_pos + n_tokens - 1u, ctx) : 0;
        const int s4 = s3 ? (ds4_gpu_synchronize() != 0) : 0;
        const int ok = s4;
        printf("  ctx=%u start=%u tokens=%u fmt=%s prepare=%d chunk=%d decode=%d sync=%d\n",
               ctx, start_pos, n_tokens, fmt == 2 ? "q4_0" : fmt ? "q8_0" : "f16", s1, s2, s3, s4);
        if (!ok) failures++;
        ds4_gpu_tensor_free(k_cache); ds4_gpu_tensor_free(v_cache);
        ds4_gpu_tensor_free(q_full); ds4_gpu_tensor_free(k);
        ds4_gpu_tensor_free(v); ds4_gpu_tensor_free(out);
        ds4_gpu_tensor_free(rope);
    }
}


/* The GDN half of the layers: same recipe, boundary token counts, and the four
 * real weight offsets of block 0 (conv1d, a, dt, norm) read from the model file. */
static void drive_gdn(const void *map, uint64_t map_size, uint64_t conv_off,
                      uint64_t a_off, uint64_t dt_off, uint64_t norm_off) {
    const uint32_t shapes[] = { 1u, 7u, 16u, 17u, 512u };
    const uint64_t conv_dim = 10240u, z_dim = 6144u;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        const uint32_t n = shapes[i];
        ds4_gpu_tensor *out = ds4_gpu_tensor_alloc((uint64_t)n * z_dim * sizeof(float));
        ds4_gpu_tensor *conv = ds4_gpu_tensor_alloc(3ull * conv_dim * sizeof(float));
        ds4_gpu_tensor *state = ds4_gpu_tensor_alloc(48ull * 128ull * 128ull * sizeof(float));
        ds4_gpu_tensor *qkv = ds4_gpu_tensor_alloc((uint64_t)n * conv_dim * sizeof(float));
        ds4_gpu_tensor *z = ds4_gpu_tensor_alloc((uint64_t)n * z_dim * sizeof(float));
        ds4_gpu_tensor *alpha = ds4_gpu_tensor_alloc((uint64_t)n * 48u * sizeof(float));
        ds4_gpu_tensor *beta = ds4_gpu_tensor_alloc((uint64_t)n * 48u * sizeof(float));
        float *buf = (float *)malloc((size_t)n * conv_dim * sizeof(float));
        if (!out || !conv || !state || !qkv || !z || !alpha || !beta || !buf) {
            printf("  GDN alloc failed n=%u\n", n);
            failures++;
            return;
        }
        for (size_t j = 0; j < (size_t)n * conv_dim; j++) buf[j] = 0.25f;
        ds4_gpu_tensor_write(qkv, 0, buf, (uint64_t)n * conv_dim * sizeof(float));
        ds4_gpu_tensor_write(z, 0, buf, (uint64_t)n * z_dim * sizeof(float));
        ds4_gpu_tensor_write(alpha, 0, buf, (uint64_t)n * 48u * sizeof(float));
        ds4_gpu_tensor_write(beta, 0, buf, (uint64_t)n * 48u * sizeof(float));
        free(buf);
        const int s1 = ds4_gpu_qwen38_gdn_chunk(out, conv, state, qkv, z, alpha, beta,
                                                map, map_size, conv_off, a_off, dt_off,
                                                norm_off, n);
        const int s2 = s1 ? ds4_gpu_qwen38_gdn_decode(out, conv, state, qkv, z, alpha,
                                                      beta, map, map_size, conv_off,
                                                      a_off, dt_off, norm_off) : 0;
        const int s3 = s2 ? (ds4_gpu_synchronize() != 0) : 0;
        printf("  GDN tokens=%u chunk=%d decode=%d sync=%d\n", n, s1, s2, s3);
        if (!s3) failures++;
        ds4_gpu_tensor_free(out); ds4_gpu_tensor_free(conv); ds4_gpu_tensor_free(state);
        ds4_gpu_tensor_free(qkv); ds4_gpu_tensor_free(z);
        ds4_gpu_tensor_free(alpha); ds4_gpu_tensor_free(beta);
    }
}


/* The elementwise family: norm rows, SwiGLU and the residual add, with widths and
 * row counts at and around the vector widths the kernels block on.  The norm weight
 * offset passed is a real tensor in the file (256 floats) while n is 5120, so the
 * weight read lands inside the mapping with nonsense values: this probe is about
 * the in/out tensors' bounds, not about the arithmetic. */
static void drive_elementwise(const void *map, uint64_t map_size, uint64_t norm_off) {
    const uint32_t rows[] = { 1u, 7u, 16u, 17u, 512u };
    const uint32_t n = 5120u;
    for (size_t i = 0; i < sizeof(rows) / sizeof(rows[0]); i++) {
        const uint64_t count = (uint64_t)n * rows[i];
        ds4_gpu_tensor *a = ds4_gpu_tensor_alloc(count * sizeof(float));
        ds4_gpu_tensor *b = ds4_gpu_tensor_alloc(count * sizeof(float));
        ds4_gpu_tensor *o = ds4_gpu_tensor_alloc(count * sizeof(float));
        float *buf = (float *)malloc((size_t)count * sizeof(float));
        if (!a || !b || !o || !buf) { printf("  elem alloc failed rows=%u\n", rows[i]); failures++; return; }
        for (uint64_t j = 0; j < count; j++) buf[j] = 0.25f;
        ds4_gpu_tensor_write(a, 0, buf, count * sizeof(float));
        ds4_gpu_tensor_write(b, 0, buf, count * sizeof(float));
        free(buf);
        const int s1 = ds4_gpu_add_tensor(o, a, b, (uint32_t)count);
        const int s2 = s1 ? ds4_gpu_swiglu_tensor(o, a, b, (uint32_t)count, 0.0f, 1.0f) : 0;
        const int s3 = s2 ? ds4_gpu_rms_norm_weight_rows_tensor(o, a, map, map_size,
                                                                norm_off, n, rows[i],
                                                                1.0e-6f) : 0;
        const int s4 = s3 ? (ds4_gpu_synchronize() != 0) : 0;
        printf("  ELEM rows=%u n=%u add=%d swiglu=%d norm=%d sync=%d\n", rows[i], n, s1, s2, s3, s4);
        if (!s4) failures++;
        ds4_gpu_tensor_free(a); ds4_gpu_tensor_free(b); ds4_gpu_tensor_free(o);
    }
}

int main(int argc, char **argv) {
    if (argc != 2) { fprintf(stderr, "usage: %s MODEL.gguf\n", argv[0]); return 2; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open model"); return 2; }
    struct stat st;
    if (fstat(fd, &st) != 0) { perror("stat"); return 2; }
    void *map = mmap(NULL, (size_t)st.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (map == MAP_FAILED) { perror("mmap"); return 2; }
    if (!ds4_gpu_init()) { fprintf(stderr, "ds4_gpu_init failed\n"); return 2; }

    /* Small cache with tails shorter than the 8-half vector load, and the two
     * shapes where the chunk ends exactly on the last position of the cache. */
    const uint32_t shapes[][3] = {
        {256, 0, 1}, {256, 0, 7}, {256, 0, 16}, {256, 0, 17},
        {256, 239, 17}, {256, 241, 15}, {256, 255, 1},
        {512, 0, 512}, {512, 5, 507}, {512, 508, 4}, {512, 511, 1},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++)
        drive(map, (uint64_t)st.st_size, shapes[i][0], shapes[i][1], shapes[i][2]);

    drive_gdn(map, (uint64_t)st.st_size, 1256301920ull, 1255318688ull,
              1256465760ull, 1256465952ull);
    drive_elementwise(map, (uint64_t)st.st_size, g_qnorm);
    printf("kv bounds drive done, %d failures\n", failures);
    munmap(map, (size_t)st.st_size);
    close(fd);
    return failures ? 1 : 0;
}
