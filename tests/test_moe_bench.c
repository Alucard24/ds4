#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t state = 1234567;
static uint32_t random_bits(void) {
    state = state * 1664525u + 1013904223u;
    return state;
}
static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

/* Microbench the decode MoE row kernels (moe_mv) with synthetic weights.
 * Timing only (values random): validates the harness against the server
 * row-profile (mid ~2.5ms, down ~2.0ms per layer) and gives a seconds-fast
 * iteration loop for future kernel A/B (no server boot). */
int main(void) {
    const uint32_t K = 2560, M = 640, NE = 512, T = 1, NS = 10, TYPE = 18;
    const uint32_t DK = 640, DM = 2560, DTYPE = 20; /* UD3 down is IQ4_NL */
    const uint64_t rb = (uint64_t)(K / 256u) * 98u;       /* expert_row_bytes(18, K) */
    const uint64_t drb = (uint64_t)(DK / 32u) * 18u;      /* expert_row_bytes(20, DK) */
    const uint64_t gate_bytes = (uint64_t)NE * M * rb;
    const uint64_t down_bytes = (uint64_t)NE * DM * drb;
    const uint64_t total = gate_bytes + down_bytes;       /* up reuses gate range layout */
    unsigned char *model = malloc(total ? total : 1);
    CHECK(model);
    for (uint64_t i = 0; i < total; i++) model[i] = (unsigned char)(random_bits() >> 19);
    /* Valid IQ3_XXS scales (1.0) per 98B superblock, random payload. */
    for (uint64_t base = 0; base < gate_bytes; base += rb)
        for (uint32_t sb = 0; sb < K / 256u; sb++) {
            model[base + (uint64_t)sb * 98u] = 0x00;
            model[base + (uint64_t)sb * 98u + 1] = 0x3C;
        }
    /* Valid IQ4_NL scales (1.0) per 18B block-of-32. */
    for (uint64_t base = gate_bytes; base < total; base += drb)
        for (uint32_t b = 0; b < DK / 32u; b++) {
            model[base + (uint64_t)b * 18u] = 0x00;
            model[base + (uint64_t)b * 18u + 1] = 0x3C;
        }
    float *x = malloc((size_t)K * sizeof(float));
    int *sel = malloc((size_t)NS * sizeof(int));
    CHECK(x && sel);
    for (uint32_t i = 0; i < K; i++) x[i] = ((int32_t)(random_bits() >> 16) - 32768) / 32768.0f;
    for (uint32_t s = 0; s < NS; s++) sel[s] = (int)((s * 37) % 512);

    CHECK(ds4_gpu_init());
    CHECK(ds4_gpu_set_model_map(model, total));
    ds4_gpu_tensor *dx = ds4_gpu_tensor_alloc((size_t)K * sizeof(float));
    ds4_gpu_tensor *dsel = ds4_gpu_tensor_alloc((size_t)NS * sizeof(int));
    ds4_gpu_tensor *dmid = ds4_gpu_tensor_alloc((size_t)NS * M * sizeof(float));
    ds4_gpu_tensor *ddown_in = ds4_gpu_tensor_alloc((size_t)NS * DK * sizeof(float));
    ds4_gpu_tensor *dpart = ds4_gpu_tensor_alloc((size_t)NS * DM * sizeof(float));
    CHECK(dx && dsel && dmid && ddown_in && dpart);
    CHECK(ds4_gpu_tensor_write(dx, 0, x, (size_t)K * sizeof(float)));
    CHECK(ds4_gpu_tensor_write(dsel, 0, sel, (size_t)NS * sizeof(int)));
    CHECK(ds4_gpu_tensor_fill_f32(dmid, 0.0f, (size_t)NS * M));
    CHECK(ds4_gpu_tensor_fill_f32(dpart, 0.0f, (size_t)NS * DM));

    const uint32_t ST = 0xFFFFFFFFu; /* no shared expert: isolate the routed path */
    const int REPS = 50;
    for (int i = 0; i < 5; i++)
        CHECK(ds4_gpu_qwen4_moe_mid_tensor(dmid, dx, dsel, model, total, 0, 0,
                                           TYPE, NE, T, NS, K, M, 0, 0, ST));
    CHECK(ds4_gpu_synchronize());
    double t0 = now_sec();
    for (int i = 0; i < REPS; i++)
        CHECK(ds4_gpu_qwen4_moe_mid_tensor(dmid, dx, dsel, model, total, 0, 0,
                                           TYPE, NE, T, NS, K, M, 0, 0, ST));
    CHECK(ds4_gpu_synchronize());
    double mid_ms = (now_sec() - t0) * 1000.0 / REPS;

    for (int i = 0; i < 5; i++)
        CHECK(ds4_gpu_qwen4_moe_down_tensor(dpart, ddown_in, dsel, model, total, gate_bytes,
                                            DTYPE, NE, T, NS, DK, DM, 0, ST));
    CHECK(ds4_gpu_synchronize());
    t0 = now_sec();
    for (int i = 0; i < REPS; i++)
        CHECK(ds4_gpu_qwen4_moe_down_tensor(dpart, ddown_in, dsel, model, total, gate_bytes,
                                            DTYPE, NE, T, NS, DK, DM, 0, ST));
    CHECK(ds4_gpu_synchronize());
    double down_ms = (now_sec() - t0) * 1000.0 / REPS;

    printf("moe_bench T=%u NS=%u K=%u M=%u type=%u: mid=%.3fms down=%.3fms sum/layer=%.3fms x48=%.1fms\n",
           T, NS, K, M, TYPE, mid_ms, down_ms, mid_ms + down_ms, (mid_ms + down_ms) * 48.0);
    /* v1-vs-v2 numeric check: same inputs, max abs diff of mid outputs. */
    {
        float *got1 = malloc((size_t)NS * M * sizeof(float));
        float *got2 = malloc((size_t)NS * M * sizeof(float));
        CHECK(got1 && got2);
        unsetenv("DS4_QWEN4_MOE_V2");
        CHECK(ds4_gpu_qwen4_moe_mid_tensor(dmid, dx, dsel, model, total, 0, 0,
                                           TYPE, NE, T, NS, K, M, 0, 0, ST));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(dmid, 0, got1, (size_t)NS * M * sizeof(float)));
        setenv("DS4_QWEN4_MOE_V2", "1", 1);
        CHECK(ds4_gpu_qwen4_moe_mid_tensor(dmid, dx, dsel, model, total, 0, 0,
                                           TYPE, NE, T, NS, K, M, 0, 0, ST));
        CHECK(ds4_gpu_synchronize());
        CHECK(ds4_gpu_tensor_read(dmid, 0, got2, (size_t)NS * M * sizeof(float)));
        double maxdiff = 0, maxrel = 0;
        for (uint64_t i = 0; i < (uint64_t)NS * M; i++) {
            double d = fabs((double)got2[i] - (double)got1[i]);
            if (d > maxdiff) maxdiff = d;
            double r = d / (1.0 + fabs((double)got1[i]));
            if (r > maxrel) maxrel = r;
        }
        printf("moe_bench v1-vs-v2 mid maxdiff=%g maxrel=%g %s\n", maxdiff, maxrel,
               maxrel <= 1e-3 ? "MATCH" : "MISMATCH");
        printf("moe_bench sample v1: %g %g %g %g | v2: %g %g %g %g\n",
               got1[0], got1[1], got1[2], got1[3], got2[0], got2[1], got2[2], got2[3]);
        free(got1); free(got2);
    }
    printf("test_moe_bench: ok\n");
    free(model); free(x); free(sel);
    return 0;
}
