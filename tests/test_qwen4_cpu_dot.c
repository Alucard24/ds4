/* A/B the Qwen3.8 CPU MoE SIMD expert kernels against the scalar reference
 * (dequantize-then-sum) on synthetic packed blocks.  No model, no GPU: this
 * is the "CPU-ref exactness before speed" gate for the CPU expert path. */
#define _POSIX_C_SOURCE 200809L
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Provided by ds4.c under DS4_TEST_HOOKS. */
float ds4_test_qwen4_cpu_dot(uint32_t type, const void *row, const float *x, uint32_t k);
float ds4_test_qwen4_ref_dot(uint32_t type, const void *row, const float *x, uint32_t k);
uint32_t ds4_test_qwen4_block_bytes(uint32_t type, uint32_t *values);
void ds4_test_qwen4_dequant_row(uint32_t type, const void *row, float *out, uint32_t k);

/* Q8_K side of the IQ2_S int8 kernel (also ds4.c, DS4_TEST_HOOKS). */
uint32_t ds4_test_qwen4_q8k_block_bytes(void);
const char *ds4_test_qwen4_q8k_kernel_name(void);
void ds4_test_qwen4_quantize_row_q8k(const float *x, void *yq, uint32_t k);
void ds4_test_qwen4_dequant_row_q8k(const void *yq, float *out, uint32_t k);
float ds4_test_qwen4_cpu_dot_q8k(uint32_t type, const void *row, const void *yq, uint32_t k);
void ds4_test_qwen4_cpu_dot_q8k_batch(uint32_t type, const void *row, const void *const *xq,
                                      float *out, uint32_t n, uint32_t k);
void ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_maddubs(const void *row, const void *const *xq,
                                                     float *out, uint32_t n, uint32_t k);
void ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_vnni(const void *row, const void *const *xq,
                                                  float *out, uint32_t n, uint32_t k);
void ds4_test_qwen4_cpu_dot_q8k_batch2(const void *row0, const void *row1, const void *const *xq,
                                       float *out0, float *out1, uint32_t n, uint32_t k);
void ds4_test_qwen4_cpu_dot_fp32_batch(uint32_t type, const void *row, const float *const *xs,
                                       float *const *outs, uint32_t n, uint32_t k);
float ds4_test_qwen4_ref_dot_q8k(uint32_t type, const void *row, const void *yq, uint32_t k);
float ds4_test_qwen4_cpu_dot_q8k_maddubs(uint32_t type, const void *row, const void *yq, uint32_t k);
float ds4_test_qwen4_cpu_dot_q8k_vnni(uint32_t type, const void *row, const void *yq, uint32_t k);

#define IQ4_NL 20u
#define IQ2_S  22u
#define IQ3_S  21u
#define IQ2_XXS 16u
#define IQ2_XS  17u
#define IQ3_XXS 18u
#define Q2_0    42u

static uint32_t rng = 20260918u;
static uint32_t rnd(void) {
    rng = rng * 1664525u + 1013904223u;
    return rng;
}
static float frnd(void) { return ((int32_t)(rnd() >> 16) - 32768) / 32768.0f; }

/* One synthetic row: a valid f16 scale (1.0) per block, random payload. */
static void fill_row(uint32_t type, uint8_t *row, uint32_t k) {
    uint32_t per_block = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per_block);
    const uint32_t nb = k / per_block;
    for (uint32_t b = 0; b < nb; b++) {
        uint8_t *blk = row + (uint64_t)b * bs;
        blk[0] = 0x00;
        blk[1] = 0x3C; /* f16 1.0 */
        for (uint32_t i = 2; i < bs; i++) blk[i] = (uint8_t)(rnd() >> 13);
    }
}

static int check_type(uint32_t type, const char *name, uint32_t k) {
    uint32_t per_block = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per_block);
    if (!bs || k % per_block) {
        fprintf(stderr, "%s: bad geometry\n", name);
        return 1;
    }
    uint8_t *row = malloc((size_t)(k / per_block) * bs);
    float *x = malloc((size_t)k * sizeof(float));
    float *vals = malloc((size_t)k * sizeof(float));
    if (!row || !x || !vals) return 1;
    for (uint32_t i = 0; i < k; i++) x[i] = frnd();
    double worst = 0;
    for (int trial = 0; trial < 8; trial++) {
        fill_row(type, row, k);
        /* True reference: dequantize the row and sum in double precision. */
        ds4_test_qwen4_dequant_row(type, row, vals, k);
        double ref = 0;
        for (uint32_t j = 0; j < k; j++) ref += (double)vals[j] * (double)x[j];
        const float ref_alt = ds4_test_qwen4_ref_dot(type, row, x, k);
        const float got = ds4_test_qwen4_cpu_dot(type, row, x, k);
        const double rel = fabs((double)got - ref) / (1.0 + fabs(ref));
        if (rel > worst) worst = rel;
        if (trial == 0)
            printf("  %s k=%u: ref=%.9g vecdot=%.9g simd=%.9g\n", name, k, ref, (double)ref_alt, (double)got);
    }
    printf("  %s k=%u: worst rel diff %.3g %s\n", name, k, worst,
           worst < 1e-4 ? "ok" : "MISMATCH");
    free(row);
    free(x);
    free(vals);
    return worst < 1e-4 ? 0 : 1;
}

/* 256-value int8/Q8_K kernels: the packed kernel must reproduce the *exact*
 * integer dot of the Q8_K-quantized activation (the test dequantizes the
 * Q8_K blocks itself), so fp32 rounding is the only allowed difference.  The
 * Q8_K step is separately measured against the fp32 activation to expose the
 * quantization loss it trades for the speed. */
static int check_q8k(uint32_t type, const char *name, uint32_t k) {
    uint32_t per_block = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per_block);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || per_block != 256 || k % per_block || k % 256) {
        fprintf(stderr, "%s q8k: bad geometry\n", name);
        return 1;
    }
    const uint32_t nb = k / per_block;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    float *vals = malloc((size_t)k * sizeof(float));
    float *xrec = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc((size_t)(k / 256) * q8bs);
    if (!row || !x || !vals || !xrec || !xq) return 1;
    double worst_kernel = 0, worst_quant = 0;
    for (int trial = 0; trial < 8; trial++) {
        fill_row(type, row, k);
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq, k);
        ds4_test_qwen4_dequant_row_q8k(xq, xrec, k);
        ds4_test_qwen4_dequant_row(type, row, vals, k);
        double ref_q = 0, ref_fp = 0;
        for (uint32_t j = 0; j < k; j++) {
            ref_q += (double)vals[j] * (double)xrec[j];
            ref_fp += (double)vals[j] * (double)x[j];
        }
        const double quant_rel = fabs(ref_q - ref_fp) / (1.0 + fabs(ref_fp));
        if (quant_rel > worst_quant) worst_quant = quant_rel;
        const struct { const char *name; float v; } got[] = {
            {"kernel", ds4_test_qwen4_cpu_dot_q8k(type, row, xq, k)},
            {"ref", ds4_test_qwen4_ref_dot_q8k(type, row, xq, k)},
            {"maddubs", ds4_test_qwen4_cpu_dot_q8k_maddubs(type, row, xq, k)},
            {"vnni", ds4_test_qwen4_cpu_dot_q8k_vnni(type, row, xq, k)},
        };
        for (size_t i = 0; i < sizeof(got) / sizeof(got[0]); i++) {
            if (i > 1 && got[i].v == 0.0f) continue; /* variant not compiled */
            const double rel = fabs((double)got[i].v - ref_q) / (1.0 + fabs(ref_q));
            if (rel > worst_kernel) worst_kernel = rel;
            if (trial == 0)
                printf("  %s q8k k=%u [%s] %-7s: %.9g (ref %.9g)\n", name, k,
                       ds4_test_qwen4_q8k_kernel_name(), got[i].name, (double)got[i].v, ref_q);
        }
    }
    printf("  %s q8k k=%u: worst kernel rel diff %.3g (tol 1e-4), worst Q8_K loss %.3g %s\n",
           name, k, worst_kernel, worst_quant, worst_kernel < 1e-4 ? "ok" : "MISMATCH");
    free(row);
    free(x);
    free(vals);
    free(xrec);
    free(xq);
    return worst_kernel < 1e-4 ? 0 : 1;
}

#include <time.h>
#include <pthread.h>

static double now_s(void);

/* Same pattern as time_model_like, but with N concurrent workers: the model
 * runs 16 threads and gets 8.9 GMAC/s per thread while one thread of this exact
 * pattern does 31.  The scaling curve tells whether the wall is the shared L2
 * (two SMT siblings, each streaming an expert's ~1 MB of weights into a 1 MB
 * private L2) or aggregate L3/DRAM bandwidth. */
typedef struct {
    uint32_t k, ntok, rows, nb, bs;
    size_t stride;
    uint8_t *w;
    uint8_t *scratch;
    uint32_t *list;
    int reps;
    int tworow;
    uint32_t chunk;
    double sink;
} ml_mt_ctx;

static void *ml_mt_worker(void *arg) {
    ml_mt_ctx *c = (ml_mt_ctx *)arg;
    const void *ptrs[16];
    float out[16], out2[16];
    double sink = 0;
    for (int rep = 0; rep < c->reps; rep++) {
        for (uint32_t row = 0; row < c->rows; row += (c->tworow ? 2u : 1u)) {
            const void *wr = (const void *)(c->w + (size_t)row * c->nb * c->bs);
            for (uint32_t base = 0; base < c->ntok; base += c->chunk) {
                const uint32_t m = (c->ntok - base) < c->chunk ? (c->ntok - base) : c->chunk;
                for (uint32_t b = 0; b < m; b++)
                    ptrs[b] = (const void *)(c->scratch + (size_t)c->list[base + b] * c->stride);
                if (c->tworow) {
                    ds4_test_qwen4_cpu_dot_q8k_batch2(wr, (const char *)wr + c->nb * c->bs, ptrs,
                                                      out, out2, m, c->k);
                    sink += out[0] + out2[0];
                } else {
                    ds4_test_qwen4_cpu_dot_q8k_batch(IQ2_S, wr, ptrs, out, m, c->k);
                    sink += out[0];
                }
            }
        }
    }
    c->sink = sink;
    return NULL;
}

/* Down phase of the model: fp32 mid vector per token (k_in=640), one weight row
 * of the expert at a time (k_out=2560 rows of IQ4_NL), token chunk of 8 outer
 * and rows inner - exactly the worker's structure.  Tells whether the down is
 * already at the machine's ceiling like the mid (142 vs 180 GMAC/s aggregate). */
typedef struct {
    uint32_t k_in, k_out, ntok;
    uint64_t rb;
    const uint8_t *w;
    float *mid;
    float *part;
    uint32_t *list;
    int reps;
    double sink;
} dl_mt_ctx;

static void *dl_mt_worker(void *arg) {
    dl_mt_ctx *c = (dl_mt_ctx *)arg;
    const float *xs[8];
    float *outs0[8], *outs[8];
    double sink = 0;
    for (int rep = 0; rep < c->reps; rep++) {
        for (uint32_t base = 0; base < c->ntok; base += 8u) {
            const uint32_t m = (c->ntok - base) < 8u ? (c->ntok - base) : 8u;
            for (uint32_t b = 0; b < m; b++) {
                const uint32_t t = c->list[base + b];
                xs[b] = c->mid + (size_t)t * c->k_in;
                outs0[b] = c->part + (size_t)t * c->k_out;
            }
            for (uint32_t r = 0; r < c->k_out; r++) {
                for (uint32_t b = 0; b < m; b++) outs[b] = outs0[b] + r;
                ds4_test_qwen4_cpu_dot_fp32_batch(IQ4_NL, c->w + (size_t)r * c->rb, xs, outs, m, c->k_in);
            }
        }
        for (uint32_t b = 0; b < c->ntok; b++) sink += c->part[(size_t)c->list[b] * c->k_out];
    }
    c->sink = sink;
    return NULL;
}

static void time_down_mt(uint32_t nthreads) {
    const uint32_t k_in = 640u, k_out = 2560u, ntok = 34u, pool = 64u;
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ4_NL, &per);
    if (!bs || k_in % per || nthreads < 1 || nthreads > 32) return;
    const uint64_t rb = (uint64_t)(k_in / per) * bs;
    dl_mt_ctx *c = calloc(nthreads, sizeof(*c));
    pthread_t *th = calloc(nthreads, sizeof(*th));
    int ok = c && th;
    for (uint32_t t = 0; ok && t < nthreads; t++) {
        c[t].k_in = k_in; c[t].k_out = k_out; c[t].ntok = ntok; c[t].rb = rb; c[t].reps = 120;
        uint8_t *w = malloc((size_t)k_out * rb);
        float *mid = malloc((size_t)pool * k_in * sizeof(float));
        float *part = malloc((size_t)pool * k_out * sizeof(float));
        uint32_t *list = malloc((size_t)ntok * sizeof(*list));
        ok = w && mid && part && list;
        if (ok) {
            for (uint32_t r = 0; r < k_out; r++) fill_row(IQ4_NL, w + (size_t)r * rb, k_in);
            for (uint32_t i = 0; i < pool * k_in; i++) mid[i] = frnd();
            memset(part, 0, (size_t)pool * k_out * sizeof(float));
            for (uint32_t i = 0; i < ntok; i++) list[i] = (uint32_t)((uint64_t)i * pool / ntok);
            c[t].w = w; c[t].mid = mid; c[t].part = part; c[t].list = list;
        } else {
            free(w); free(mid); free(part); free(list);
        }
    }
    if (!ok) {
        if (c) for (uint32_t t = 0; t < nthreads; t++) { free((void *)c[t].w); free(c[t].mid); free(c[t].part); free(c[t].list); }
        free(c); free(th);
        return;
    }
    const double t0 = now_s();
    for (uint32_t t = 1; t < nthreads; t++) pthread_create(&th[t], NULL, dl_mt_worker, &c[t]);
    dl_mt_worker(&c[0]);
    for (uint32_t t = 1; t < nthreads; t++) pthread_join(th[t], NULL);
    const double dt = now_s() - t0;
    double sink = 0;
    for (uint32_t t = 0; t < nthreads; t++) sink += c[t].sink;
    printf("  down IQ4_NL MT: %2u thread: %6.1f GMAC/s aggregati (%5.1f per thread, sink %g)\n",
           nthreads, (double)k_in * k_out * ntok * c[0].reps * nthreads / dt / 1e9,
           (double)k_in * k_out * ntok * c[0].reps / dt / 1e9, sink);
    for (uint32_t t = 0; t < nthreads; t++) { free((void *)c[t].w); free(c[t].mid); free(c[t].part); free(c[t].list); }
    free(c); free(th);
}

static void time_model_like_mt(uint32_t k, uint32_t ntok, uint32_t rows, uint32_t nthreads, int tworow,
                               uint32_t chunk) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ2_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per || k % 256 || nthreads < 1 || nthreads > 32) return;
    const uint32_t nb = k / per, nq = k / 256, scratch_tokens = 256u;
    const size_t stride = (size_t)nq * q8bs;
    ml_mt_ctx *c = calloc(nthreads, sizeof(*c));
    pthread_t *th = calloc(nthreads, sizeof(*th));
    float *x = malloc((size_t)k * sizeof(float));
    int ok = c && th && x;
    for (uint32_t t = 0; ok && t < nthreads; t++) {
        c[t].k = k; c[t].ntok = ntok; c[t].rows = rows; c[t].nb = nb; c[t].bs = bs;
        c[t].stride = stride; c[t].reps = 120; c[t].tworow = tworow; c[t].chunk = chunk;
        c[t].w = malloc((size_t)nb * bs * rows);
        c[t].scratch = malloc((size_t)scratch_tokens * stride);
        c[t].list = malloc((size_t)ntok * sizeof(*c[t].list));
        ok = c[t].w && c[t].scratch && c[t].list;
    }
    for (uint32_t t = 0; ok && t < nthreads; t++) {
        for (uint32_t r = 0; r < rows; r++) fill_row(IQ2_S, c[t].w + (size_t)r * nb * bs, k);
        for (uint32_t v = 0; v < scratch_tokens; v++) {
            for (uint32_t i = 0; i < k; i++) x[i] = frnd();
            ds4_test_qwen4_quantize_row_q8k(x, c[t].scratch + (size_t)v * stride, k);
        }
        for (uint32_t i = 0; i < ntok; i++)
            c[t].list[i] = (uint32_t)((uint64_t)i * scratch_tokens / ntok);
    }
    if (!ok) {
        if (c) for (uint32_t t = 0; t < nthreads; t++) { free(c[t].w); free(c[t].scratch); free(c[t].list); }
        free(c); free(th); free(x);
        return;
    }
    const double t0 = now_s();
    for (uint32_t t = 1; t < nthreads; t++) pthread_create(&th[t], NULL, ml_mt_worker, &c[t]);
    ml_mt_worker(&c[0]);
    for (uint32_t t = 1; t < nthreads; t++) pthread_join(th[t], NULL);
    const double dt = now_s() - t0;
    double sink = 0;
    for (uint32_t t = 0; t < nthreads; t++) sink += c[t].sink;
    printf("  IQ2_S model-like MT: %2u thread%-9s chunk %u, %u token sparsi, riga nuova: %6.1f GMAC/s aggregati (%5.1f per thread, sink %g)\n",
           nthreads, tworow ? " (2 righe)" : "", chunk, ntok,
           (double)k * ntok * rows * c[0].reps * nthreads / dt / 1e9,
           (double)k * ntok * rows * c[0].reps / dt / 1e9, sink);
    for (uint32_t t = 0; t < nthreads; t++) { free(c[t].w); free(c[t].scratch); free(c[t].list); }
    free(c); free(th); free(x);
}
static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Throughput of one packed row dot (single thread): reports GB/s of packed
 * weights and MACs/s, the numbers that decide the CPU-MoE budget. */
static void time_type(uint32_t type, const char *name, uint32_t k) {
    uint32_t per_block = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per_block);
    const uint32_t nb = k / per_block;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    if (!row || !x) return;
    fill_row(type, row, k);
    for (uint32_t i = 0; i < k; i++) x[i] = frnd();
    const int reps = 2000;
    float sink = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; r++) sink += ds4_test_qwen4_cpu_dot(type, row, x, k);
    const double dt = now_s() - t0;
    const double bytes = (double)nb * bs * reps;
    printf("  %s k=%u: %.2f GB/s packed, %.2f GMAC/s single-thread (sink %g)\n",
           name, k, bytes / dt / 1e9, (double)k * reps / dt / 1e9, (double)sink);
    free(row);
    free(x);
}

/* Same, for the 256-value int8 kernels: the activation is quantized once,
 * then the row dot is timed on its own. */
static float *g_bench_mid;   /* mid-like scratch: what the model writes into */
static int g_bench_midstore; /* emulate the model's scattered 4-byte stores */

/* Batched mid kernel in isolation: n activations per call.  nbufs == n means
 * the same activation set every rep (xq hot in L1); a large nbufs streams the
 * activation set from outside the cache, which is what the model does when a
 * row is walked over the expert's whole token list. */
static void time_batch_q8k(uint32_t type, const char *name, uint32_t k, uint32_t ntokens, uint32_t nbufs) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    /* Per-type cap: asking for more than the kernel supports silently clamps
     * inside the kernel while GMAC/s keeps dividing by ntokens, inflating the
     * number.  IQ2_S now takes 16, the others 8. */
    const uint32_t maxn = (type == IQ2_S) ? 16u : 8u;
    if (!bs || !q8bs || k % per || k % 256 || nbufs < ntokens || ntokens > maxn) return;
    const uint32_t nb = k / per, nq = k / 256;
    const size_t stride = (size_t)nq * q8bs;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc((size_t)nbufs * stride);
    const void **ptrs = malloc((size_t)ntokens * sizeof(*ptrs));
    float *out = malloc((size_t)ntokens * sizeof(*out));
    if (!row || !x || !xq || !ptrs || !out) {
        free(row); free(x); free(xq); free(ptrs); free(out);
        return;
    }
    fill_row(type, row, k);
    for (uint32_t v = 0; v < nbufs; v++) {
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)v * stride, k);
    }
    const int reps = 300;
    float sink = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        const uint32_t base = (nbufs == ntokens) ? 0u : (uint32_t)((uint64_t)r * ntokens % nbufs);
        for (uint32_t b = 0; b < ntokens; b++) ptrs[b] = xq + (size_t)((base + b) % nbufs) * stride;
        ds4_test_qwen4_cpu_dot_q8k_batch(type, row, ptrs, out, ntokens, k);
        for (uint32_t b = 0; b < ntokens; b++) sink += out[b];
        if (g_bench_midstore) {
            /* The model writes each row into a different token's mid row, so the
             * stores are 4 bytes each, k_ff*4 apart, i.e. one line per token. */
            for (uint32_t b = 0; b < ntokens; b++)
                g_bench_mid[(size_t)b * 2560u + (uint32_t)(r & 1023u)] = out[b];
        }
    }
    const double dt = now_s() - t0;
    printf("  %s batch x%u k=%u [%s] nbuf=%u %s%s: %.2f GMAC/s, %.1f GB/s q8 (sink %g)\n",
           name, ntokens, k, ds4_test_qwen4_q8k_kernel_name(), nbufs,
           nbufs == ntokens ? "L1-hot" : "streamed",
           g_bench_midstore ? " +mid-stores" : "",
           (double)k * ntokens * reps / dt / 1e9,
           (double)ntokens * (double)stride * reps / dt / 1e9, sink);
    free(row); free(x); free(xq); free(ptrs); free(out);
}

/* Direct IQ3_S A/B in one process and one allocation.  The generic timing
 * harness normally dispatches just the default kernel; this alternates the
 * explicit AVX2 and VNNI implementations so frequency/thermal drift cannot
 * turn two separate runs into a false win. */
typedef void (*iq3s_batch_fn)(const void *row, const void *const *xq, float *out,
                              uint32_t n, uint32_t k);

static void time_iq3s_batch_ab(uint32_t nbufs) {
    const uint32_t k = 2560u, ntok = 8u;
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ3_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per || nbufs < ntok) return;
    const uint32_t nb = k / per, nq = k / 256u;
    const size_t stride = (size_t)nq * q8bs;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(*x));
    uint8_t *xq = malloc((size_t)nbufs * stride);
    const void *ptrs[8];
    float out[8];
    if (!row || !x || !xq) { free(row); free(x); free(xq); return; }
    fill_row(IQ3_S, row, k);
    for (uint32_t v = 0; v < nbufs; v++) {
        for (uint32_t j = 0; j < k; j++) x[j] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)v * stride, k);
    }
    const struct { const char *name; iq3s_batch_fn fn; } variant[2] = {
        {"maddubs", ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_maddubs},
        {"vnni",    ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_vnni},
    };
    const uint32_t reps = nbufs == ntok ? 50000u : 30000u;
    double best[2] = {1e30, 1e30};
    double sink[2] = {0, 0};
    /* Reverse the order on alternating passes. */
    for (uint32_t pass = 0; pass < 6u; pass++) {
        for (uint32_t order = 0; order < 2u; order++) {
            const uint32_t v = (pass + order) & 1u;
            const double t0 = now_s();
            double local = 0;
            for (uint32_t r = 0; r < reps; r++) {
                const uint32_t base = nbufs == ntok ? 0u : (r * ntok) % nbufs;
                for (uint32_t b = 0; b < ntok; b++)
                    ptrs[b] = xq + (size_t)((base + b) % nbufs) * stride;
                variant[v].fn(row, ptrs, out, ntok, k);
                local += out[r & 7u];
            }
            const double dt = now_s() - t0;
            if (dt < best[v]) best[v] = dt;
            sink[v] += local;
        }
    }
    for (uint32_t v = 0; v < 2u; v++) {
        printf("  IQ3_S A/B %-7s x8 nbuf=%-4u %s: %.2f GMAC/s (sink %g)\n",
               variant[v].name, nbufs, nbufs == ntok ? "L1-hot" : "streamed",
               (double)k * ntok * reps / best[v] / 1e9, sink[v]);
    }
    free(row); free(x); free(xq);
}

/* Both explicit IQ3_S batch implementations must agree with the trusted
 * single-token packed kernel.  The dispatch check below additionally covers
 * the default/env selection. */
static int check_iq3s_batch_variants(void) {
    const uint32_t k = 2560u, ntok = 8u;
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ3_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per) return 1;
    const uint32_t nb = k / per, nq = k / 256u;
    const size_t stride = (size_t)nq * q8bs;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(*x));
    uint8_t *xq = malloc((size_t)ntok * stride);
    const void *ptrs[8];
    float got[8];
    if (!row || !x || !xq) { free(row); free(x); free(xq); return 1; }
    fill_row(IQ3_S, row, k);
    for (uint32_t b = 0; b < ntok; b++) {
        for (uint32_t j = 0; j < k; j++) x[j] = frnd();
        ptrs[b] = xq + (size_t)b * stride;
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)b * stride, k);
    }
    const struct { const char *name; iq3s_batch_fn fn; } variant[2] = {
        {"maddubs", ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_maddubs},
        {"vnni",    ds4_test_qwen4_cpu_dot_iq3_s_q8k_batch_vnni},
    };
    int rc = 0;
    for (uint32_t v = 0; v < 2u; v++) {
        double worst = 0;
        variant[v].fn(row, ptrs, got, ntok, k);
        for (uint32_t b = 0; b < ntok; b++) {
            const double ref = ds4_test_qwen4_cpu_dot_q8k(IQ3_S, row, ptrs[b], k);
            const double rel = fabs(got[b] - ref) / (fabs(ref) + 1e-6);
            if (rel > worst) worst = rel;
        }
        printf("  IQ3_S %-7s batch x8 vs singolo: worst rel diff %.2e %s\n", variant[v].name,
               worst, worst < 1e-4 ? "ok" : "**FAIL**");
        if (!(worst < 1e-4)) rc = 1;
    }
    free(row); free(x); free(xq);
    return rc;
}

/* Same activation-load count per call as time_batch_q8k with ntokens=8, but
 * two weight rows per 4 tokens: if the mid is bound by the activation stream,
 * this reads half the bytes per MAC. */
static void time_batch2_q8k(uint32_t k, uint32_t ntokens, uint32_t nbufs) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ2_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per || k % 256 || nbufs < ntokens || ntokens > 8) return;
    const uint32_t nb = k / per, nq = k / 256;
    const size_t stride = (size_t)nq * q8bs;
    uint8_t *row0 = malloc((size_t)nb * bs), *row1 = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc((size_t)nbufs * stride);
    const void **ptrs = malloc((size_t)ntokens * sizeof(*ptrs));
    float *o0 = malloc((size_t)ntokens * sizeof(*o0)), *o1 = malloc((size_t)ntokens * sizeof(*o1));
    if (!row0 || !row1 || !x || !xq || !ptrs || !o0 || !o1) {
        free(row0); free(row1); free(x); free(xq); free(ptrs); free(o0); free(o1);
        return;
    }
    fill_row(IQ2_S, row0, k);
    fill_row(IQ2_S, row1, k);
    for (uint32_t v = 0; v < nbufs; v++) {
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)v * stride, k);
    }
    const int reps = 300;
    float sink = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        const uint32_t base = (nbufs == ntokens) ? 0u : (uint32_t)((uint64_t)r * ntokens % nbufs);
        for (uint32_t b = 0; b < ntokens; b++) ptrs[b] = xq + (size_t)((base + b) % nbufs) * stride;
        ds4_test_qwen4_cpu_dot_q8k_batch2(row0, row1, ptrs, o0, o1, ntokens, k);
        for (uint32_t b = 0; b < ntokens; b++) sink += o0[b] + o1[b];
    }
    const double dt = now_s() - t0;
    printf("  IQ2_S batch2 x%u (2 righe) k=%u [%s] nbuf=%u %s: %.2f GMAC/s, %.1f GB/s q8 (sink %g)\n",
           ntokens, k, ds4_test_qwen4_q8k_kernel_name(), nbufs,
           nbufs == ntokens ? "L1-hot" : "streamed",
           (double)k * ntokens * 2.0 * reps / dt / 1e9,
           (double)ntokens * (double)stride * reps / dt / 1e9, sink);
    free(row0); free(row1); free(x); free(xq); free(ptrs); free(o0); free(o1);
}

/* The 2-row kernel must agree with the single-token one on every element. */
/* Il percorso batch (quello che il prefill usa davvero) non era coperto da
 * nessun test di correttezza: solo IQ2_S aveva un controllo.  Qui ogni kernel
 * batch della famiglia viene confrontato, token per token, col kernel a token
 * singolo - che a sua volta e' validato contro il riferimento indipendente in
 * check_q8k().  Se un batch legge gli offset sbagliati, questo lo scopre. */
static int check_batch_any(uint32_t type, const char *name, uint32_t k) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per || k % 256) {
        fprintf(stderr, "%s batch: bad geometry\n", name);
        return 1;
    }
    const uint32_t nb = k / per, nq = k / 256, ntok = 8u;
    const size_t stride = (size_t)nq * q8bs;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc(stride * ntok);
    const void *ptrs[8];
    float got[8];
    if (!row || !x || !xq) {
        free(row); free(x); free(xq);
        return 1;
    }
    fill_row(type, row, k);
    for (uint32_t v = 0; v < ntok; v++) {
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)v * stride, k);
        ptrs[v] = xq + (size_t)v * stride;
    }
    ds4_test_qwen4_cpu_dot_q8k_batch(type, row, ptrs, got, ntok, k);
    double worst = 0;
    for (uint32_t v = 0; v < ntok; v++) {
        const double a = ds4_test_qwen4_cpu_dot_q8k(type, row, ptrs[v], k);
        const double d = fabs(got[v] - a) / (fabs(a) + 1e-6);
        if (d > worst) worst = d;
    }
    printf("  %-8s batch x%u vs singolo: worst rel diff %.2e %s\n", name, ntok, worst,
           worst < 1e-4 ? "ok" : "**FAIL**");
    free(row); free(x); free(xq);
    return worst < 1e-4 ? 0 : 1;
}

/* Percorso di produzione del down: i kernel fp32 batch (IQ4_NL/Q2_0) erano usati
 * solo per il timing e non confrontati col kernel a token singolo.  E' la stessa
 * lacuna che ha lasciato passare il bug di IQ2_XXS nel percorso q8k. */
static int check_fp32_batch(uint32_t type, const char *name, uint32_t k) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per);
    if (!bs || k % per) {
        fprintf(stderr, "%s fp32 batch: bad geometry\n", name);
        return 1;
    }
    const uint32_t ntok = 8u;
    uint8_t *row = malloc((size_t)(k / per) * bs);
    float *x = malloc((size_t)k * ntok * sizeof(float));
    const float *xs[8];
    float *outs[8];
    float got[8];
    if (!row || !x) {
        free(row); free(x);
        return 1;
    }
    fill_row(type, row, k);
    for (uint32_t v = 0; v < ntok; v++) {
        for (uint32_t i = 0; i < k; i++) x[(size_t)v * k + i] = frnd();
        xs[v] = x + (size_t)v * k;
        outs[v] = &got[v];
    }
    ds4_test_qwen4_cpu_dot_fp32_batch(type, row, xs, outs, ntok, k);
    double worst = 0;
    for (uint32_t v = 0; v < ntok; v++) {
        const double a = ds4_test_qwen4_cpu_dot(type, row, xs[v], k);
        const double d = fabs(got[v] - a) / (fabs(a) + 1e-6);
        if (d > worst) worst = d;
    }
    printf("  %-8s fp32 batch x%u vs singolo: worst rel diff %.2e %s\n", name, ntok, worst,
           worst < 1e-4 ? "ok" : "**FAIL**");
    free(row); free(x);
    return worst < 1e-4 ? 0 : 1;
}

static int check_batch2(void) {
    const uint32_t k = 2560;
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ2_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    const uint32_t nq = k / 256;
    const size_t rbytes = (size_t)bs * (k / per), stride = (size_t)q8bs * nq;
    uint8_t *r0 = malloc(rbytes), *r1 = malloc(rbytes);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc(stride * 16u);
    float o0[16], o1[16];
    const void *ptrs[16];
    if (!r0 || !r1 || !x || !xq) {
        free(r0); free(r1); free(x); free(xq);
        return 1;
    }
    fill_row(IQ2_S, r0, k);
    fill_row(IQ2_S, r1, k);
    for (uint32_t v = 0; v < 16u; v++) {
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, xq + (size_t)v * stride, k);
        ptrs[v] = xq + (size_t)v * stride;
    }
    int rc = 0;
    {
        /* The batch kernel at its 16-token cap must agree too. */
        float o[16];
        ds4_test_qwen4_cpu_dot_q8k_batch(IQ2_S, r0, ptrs, o, 16u, k);
        double worst16 = 0;
        for (uint32_t v = 0; v < 16u; v++) {
            const double a = ds4_test_qwen4_cpu_dot_q8k(IQ2_S, r0, ptrs[v], k);
            const double d = fabs(o[v] - a) / (fabs(a) + 1e-6);
            if (d > worst16) worst16 = d;
        }
        printf("  IQ2_S batch x16: worst rel diff %.2e (tol 1e-4) %s\n", worst16,
               worst16 < 1e-4 ? "ok" : "FAIL");
        if (!(worst16 < 1e-4)) rc = 1;
    }
    for (uint32_t ntok = 4; ntok <= 8u; ntok += 4u) {
        ds4_test_qwen4_cpu_dot_q8k_batch2(r0, r1, ptrs, o0, o1, ntok, k);
        double worst = 0;
        for (uint32_t v = 0; v < ntok; v++) {
            const double a0 = ds4_test_qwen4_cpu_dot_q8k(IQ2_S, r0, ptrs[v], k);
            const double a1 = ds4_test_qwen4_cpu_dot_q8k(IQ2_S, r1, ptrs[v], k);
            const double d0 = fabs(o0[v] - a0) / (fabs(a0) + 1e-6);
            const double d1 = fabs(o1[v] - a1) / (fabs(a1) + 1e-6);
            if (d0 > worst) worst = d0;
            if (d1 > worst) worst = d1;
        }
        printf("  IQ2_S batch2 x%u: worst rel diff %.2e (tol 1e-4) %s\n", ntok, worst,
               worst < 1e-4 ? "ok" : "FAIL");
        if (!(worst < 1e-4)) rc = 1;
    }
    free(r0); free(r1); free(x); free(xq);
    return rc;
}

/* The model runs the mid at ~9 GMAC/s per thread while this microbenchmark
 * does 12 even when streaming 12 MB from DRAM.  Something structural in the
 * model path must cost more than memory.  Emulate it here: an expert's token
 * list is a small set of *scattered* pointers inside a T-sized q8 scratch
 * (like tok_idx), and every call uses a fresh weight row (like walking all
 * k_ff rows per chunk).  Knobs isolate which of the two matters. */
static void time_model_like(uint32_t k, uint32_t ntok, uint32_t rows, int scatter, int fresh_row, int midstore) {
    uint32_t per = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(IQ2_S, &per);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per || k % 256 || ntok < 8 || ntok > 64 || rows < 1) return;
    const uint32_t nb = k / per, nq = k / 256, scratch_tokens = 1739u;
    const size_t stride = (size_t)nq * q8bs;
    const size_t wbytes = (size_t)nb * bs * (fresh_row ? rows : 1u);
    /* Model-shaped mid buffer: T tokens x 10 slots x k_ff floats (35.6 MB). */
    float *mid = midstore ? calloc((size_t)scratch_tokens * 10u * rows, sizeof(float)) : NULL;
    uint8_t *w = malloc(wbytes);
    uint8_t *scratch = malloc((size_t)scratch_tokens * stride);
    float *x = malloc((size_t)k * sizeof(float));
    uint32_t *list = malloc((size_t)ntok * sizeof(*list));
    if (!w || !scratch || !x || !list || (midstore && !mid)) {
        free(w); free(scratch); free(x); free(list); free(mid);
        return;
    }
    fill_row(IQ2_S, w, k);
    for (uint32_t r = 1; r < (fresh_row ? rows : 1u); r++) fill_row(IQ2_S, w + (size_t)r * nb * bs, k);
    for (uint32_t v = 0; v < scratch_tokens; v++) {
        for (uint32_t i = 0; i < k; i++) x[i] = frnd();
        ds4_test_qwen4_quantize_row_q8k(x, scratch + (size_t)v * stride, k);
    }
    /* The model's list is a random subset of the tokens; keep the same set
     * across reps so only the access pattern varies. */
    for (uint32_t i = 0; i < ntok; i++)
        list[i] = scatter ? (uint32_t)((uint64_t)i * scratch_tokens / ntok) : i;
    const float *ptrs[8];
    float out[8];
    const int reps = 60;
    float sink = 0;
    const double t0 = now_s();
    for (int rep = 0; rep < reps; rep++) {
        for (uint32_t row = 0; row < rows; row++) {
            const void *wr = (const void *)(w + (size_t)(fresh_row ? row : 0u) * nb * bs);
            for (uint32_t base = 0; base < ntok; base += 8) {
                const uint32_t m = (ntok - base) < 8u ? (ntok - base) : 8u;
                for (uint32_t b = 0; b < m; b++)
                    ptrs[b] = (const float *)(scratch + (size_t)list[base + b] * stride);
                ds4_test_qwen4_cpu_dot_q8k_batch(IQ2_S, wr, (const void **)ptrs, out, m, k);
                sink += out[0];
                if (mid) {
                    /* The model writes each row into its token's mid row: 4 bytes
                     * per token, one line each, into a 35.6 MB buffer. */
                    for (uint32_t b = 0; b < m; b++)
                        mid[((size_t)list[base + b] * 10u + ((base + b) % 10u)) * rows + row] = out[b];
                }
            }
        }
    }
    const double dt = now_s() - t0;
    printf("  IQ2_S model-like ntok=%u rows=%u %-10s %-10s %-10s: %.2f GMAC/s (sink %g)\n",
           ntok, rows, scatter ? "scattered" : "dense", fresh_row ? "fresh-row" : "same-row",
           midstore ? "+mid35MB" : "no-mid",
           (double)k * ntok * rows * reps / dt / 1e9, sink);
    free(w); free(scratch); free(x); free(list); free(mid);
}

static void time_type_q8k(uint32_t type, const char *name, uint32_t k) {
    uint32_t per_block = 0;
    const uint32_t bs = ds4_test_qwen4_block_bytes(type, &per_block);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    if (!bs || !q8bs || k % per_block || k % 256) return;
    const uint32_t nb = k / per_block;
    uint8_t *row = malloc((size_t)nb * bs);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc((size_t)(k / 256) * q8bs);
    if (!row || !x || !xq) return;
    fill_row(type, row, k);
    for (uint32_t i = 0; i < k; i++) x[i] = frnd();
    ds4_test_qwen4_quantize_row_q8k(x, xq, k);
    const int reps = 2000;
    float sink = 0;
    const double t0 = now_s();
    for (int r = 0; r < reps; r++) sink += ds4_test_qwen4_cpu_dot_q8k(type, row, xq, k);
    const double dt = now_s() - t0;
    const double bytes = (double)nb * bs * reps;
    printf("  %s q8k k=%u [%s]: %.2f GB/s packed, %.2f GMAC/s single-thread (sink %g)\n",
           name, k, ds4_test_qwen4_q8k_kernel_name(), bytes / dt / 1e9,
           (double)k * reps / dt / 1e9, (double)sink);
    free(row);
    free(x);
    free(xq);
}
/* Parallel throughput on a large working set: rows spread over `mb` MiB so
 * the data comes from RAM, not L1.  Shows whether the CPU path is compute
 * bound (scales with threads) or memory bound (flat). */
typedef struct {
    const uint8_t *rows;
    uint32_t n_rows;
    uint32_t row_bytes;
    uint32_t k;
    const float *x;
    const void *xq;
    int use_q8k;
    uint64_t iters;
    uint64_t start;
    double sink;
} par_job;

static void *par_worker(void *arg) {
    par_job *j = (par_job *)arg;
    double acc = 0;
    uint32_t st = 12345;
    if (getenv("DS4_BENCH_SEQ")) {
        for (uint64_t i = 0; i < j->iters; i++) {
            const uint32_t r = (uint32_t)((j->start + i) % j->n_rows);
            const uint8_t *row = j->rows + (uint64_t)r * j->row_bytes;
            acc += j->use_q8k ? ds4_test_qwen4_cpu_dot_q8k(IQ2_S, row, j->xq, j->k)
                              : ds4_test_qwen4_cpu_dot(IQ2_S, row, j->x, j->k);
        }
    } else {
        for (uint64_t i = 0; i < j->iters; i++) {
            st = st * 1103515245u + 12345u;
            const uint32_t r = (st >> 8) % j->n_rows;
            const uint8_t *row = j->rows + (uint64_t)r * j->row_bytes;
            acc += j->use_q8k ? ds4_test_qwen4_cpu_dot_q8k(IQ2_S, row, j->xq, j->k)
                              : ds4_test_qwen4_cpu_dot(IQ2_S, row, j->x, j->k);
        }
    }
    j->sink = acc;
    return NULL;
}

static void bench_parallel(uint32_t mb, int threads) {
    const uint32_t k = 2560, rb = 82u * (2560u / 256u);
    const uint32_t q8bs = ds4_test_qwen4_q8k_block_bytes();
    const uint32_t n_rows = (mb << 20) / rb;
    uint8_t *rows = malloc((size_t)n_rows * rb);
    float *x = malloc((size_t)k * sizeof(float));
    uint8_t *xq = malloc((size_t)(k / 256) * (q8bs ? q8bs : 1));
    if (!rows || !x || !xq) return;
    for (uint32_t i = 0; i < n_rows; i++) fill_row(IQ2_S, rows + (uint64_t)i * rb, k);
    for (uint32_t i = 0; i < k; i++) x[i] = frnd();
    ds4_test_qwen4_quantize_row_q8k(x, xq, k);
    const uint64_t total = (uint64_t)n_rows * 4;
    const uint64_t per = total / (threads > 0 ? (uint64_t)threads : 1ull);
    pthread_t th[64];
    par_job jobs[64];
    for (int mode = 0; mode < 2; mode++) {
        const double t0 = now_s();
        for (int t = 0; t < threads; t++) {
            jobs[t].rows = rows; jobs[t].n_rows = n_rows; jobs[t].row_bytes = rb;
            jobs[t].k = k; jobs[t].x = x; jobs[t].xq = xq; jobs[t].use_q8k = mode;
            jobs[t].iters = per; jobs[t].start = per * (uint64_t)t; jobs[t].sink = 0;
            pthread_create(&th[t], NULL, par_worker, &jobs[t]);
        }
        for (int t = 0; t < threads; t++) pthread_join(th[t], NULL);
        const double dt = now_s() - t0;
        const double bytes = (double)per * rb * threads;
        printf("  RAM bench %u MiB, %2d threads [%s]: %.2f GB/s packed, %.2f GMAC/s total, %.2f GB/s per thread\n",
               mb, threads, mode ? "iq2s-q8k" : "iq2s-fp32", bytes / dt / 1e9,
               bytes / rb * k / dt / 1e9, bytes / dt / 1e9 / threads);
    }
    free(rows);
    free(x);
    free(xq);
}

int main(void) {
    int rc = 0;
    printf("test_qwen4_cpu_dot\n");
    if (getenv("DS4_BENCH_RAM")) {
        for (int t = 1; t <= 16; t *= (t == 1 ? 8 : 2)) bench_parallel(256, t);
    }
    if (getenv("DS4_BENCH_DOT")) {
        g_bench_mid = calloc((size_t)16u * 2560u, sizeof(float));
        /* Mixed shapes as they appear in a layer: gate+up (IQ2_S x2) and
         * down (IQ4_NL).  Timing only; values are not checked here. */
        time_type(IQ2_S, "IQ2_S", 2560);
        time_type_q8k(IQ2_S, "IQ2_S", 2560);
        time_type_q8k(IQ2_XXS, "IQ2_XXS", 2560);
        time_type_q8k(IQ2_XS, "IQ2_XS", 2560);
        time_type_q8k(IQ3_XXS, "IQ3_XXS", 2560);
        time_type_q8k(IQ3_S, "IQ3_S", 2560);
        time_type(IQ4_NL, "IQ4_NL", 640);
        time_type(Q2_0, "Q2_0", 640);
        time_type(IQ3_S, "IQ3_S", 2560);
        /* Batched mid path: hot vs streamed activation set. */
        time_batch_q8k(IQ2_S, "IQ2_S", 2560, 8, 8);
        time_batch_q8k(IQ2_S, "IQ2_S", 2560, 16, 16);
        g_bench_midstore = 1;
        time_batch_q8k(IQ2_S, "IQ2_S", 2560, 8, 8);
        g_bench_midstore = 0;
        time_batch_q8k(IQ2_S, "IQ2_S", 2560, 8, 4096);
        time_batch_q8k(IQ2_S, "IQ2_S", 2560, 16, 4096);
        time_batch_q8k(IQ3_S, "IQ3_S", 2560, 8, 8);
        time_batch_q8k(IQ3_S, "IQ3_S", 2560, 8, 4096);
        time_iq3s_batch_ab(8);
        time_iq3s_batch_ab(4096);
        /* Tipi di ISTA: se il divario con UD e' la famiglia di kernel (maddubs vs
         * VNNI), questi due numeri lo dicono. */
        time_batch_q8k(IQ2_XXS, "IQ2_XXS", 2560, 8, 8);
        time_batch_q8k(IQ2_XXS, "IQ2_XXS", 2560, 8, 4096);
        time_batch_q8k(IQ2_XS, "IQ2_XS", 2560, 8, 8);
        time_batch_q8k(IQ2_XS, "IQ2_XS", 2560, 8, 4096);
        time_batch_q8k(IQ3_XXS, "IQ3_XXS", 2560, 8, 8);
        time_batch_q8k(IQ3_XXS, "IQ3_XXS", 2560, 8, 4096);
        /* 2 rows x 4 tokens = same 8 MAC-vectors per call as batch x8. */
        time_batch2_q8k(2560, 4, 4);
        time_batch2_q8k(2560, 8, 8);
        time_batch2_q8k(2560, 4, 4096);
        time_batch2_q8k(2560, 8, 4096);
        /* Which structural factor makes the model 5x slower than this bench? */
        time_model_like(2560, 34, 640, 0, 0, 0);
        time_model_like(2560, 34, 640, 1, 1, 0);
        time_model_like(2560, 34, 640, 0, 0, 1);
        time_model_like(2560, 34, 640, 1, 1, 1);
        for (uint32_t nt = 1; nt <= 16u; nt <<= 1) time_model_like_mt(2560, 34, 640, nt, 0, 8);
        for (uint32_t nt = 1; nt <= 16u; nt <<= 1) time_model_like_mt(2560, 34, 640, nt, 0, 16);
        for (uint32_t nt = 1; nt <= 16u; nt <<= 1) time_model_like_mt(2560, 34, 640, nt, 1, 8);
        for (uint32_t nt = 1; nt <= 16u; nt <<= 1) time_down_mt(nt);
    }
    rc |= check_type(IQ4_NL, "IQ4_NL", 640);   /* down experts (ff -> embd) */
    rc |= check_fp32_batch(IQ4_NL, "IQ4_NL", 640);
    rc |= check_fp32_batch(Q2_0, "Q2_0", 640);
    rc |= check_fp32_batch(IQ4_NL, "IQ4_NL", 2560);
    rc |= check_fp32_batch(Q2_0, "Q2_0", 2560);
    rc |= check_type(IQ4_NL, "IQ4_NL", 2560);
    rc |= check_type(IQ2_S, "IQ2_S", 2560);    /* gate/up experts */
    rc |= check_type(IQ3_S, "IQ3_S", 2560);
    rc |= check_type(Q2_0, "Q2_0", 640);       /* ISTA down experts */
    rc |= check_type(Q2_0, "Q2_0", 2560);
    rc |= check_q8k(IQ2_S, "IQ2_S", 2560);     /* int8 gate/up kernels */
    rc |= check_q8k(IQ2_XXS, "IQ2_XXS", 2560);
    rc |= check_q8k(IQ2_XS, "IQ2_XS", 2560);
    rc |= check_q8k(IQ3_XXS, "IQ3_XXS", 2560);
    rc |= check_q8k(IQ3_S, "IQ3_S", 2560);
    rc |= check_batch_any(IQ2_XXS, "IQ2_XXS", 2560);
    rc |= check_batch_any(IQ2_XS, "IQ2_XS", 2560);
    rc |= check_batch_any(IQ3_XXS, "IQ3_XXS", 2560);
    rc |= check_batch_any(IQ3_S, "IQ3_S", 2560);
    rc |= check_iq3s_batch_variants();
    rc |= check_batch_any(IQ2_S, "IQ2_S", 2560);
    rc |= check_batch2();
    if (rc) {
        printf("test_qwen4_cpu_dot: FAIL\n");
        return 1;
    }
    printf("test_qwen4_cpu_dot: ok\n");
    return 0;
}
