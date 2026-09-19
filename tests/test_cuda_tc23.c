#define _POSIX_C_SOURCE 200809L
#include "ds4_gpu.h"
#include <math.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "%s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
static uint32_t state = 918273;
static uint32_t random_bits(void) {
    state = state * 1664525u + 1013904223u;
    return state;
}
/* host IQ4_XS dequant (one row), mirrors ds4_dequant_row_iq4_xs layout:
 * 256 values in 136 bytes: scales then nibbles (see ds4.c). */
static float dev_f16(uint16_t h) {
    uint32_t s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023, f;
    if (e == 0) f = (s << 31) | (m ? ((m << 13) + 0x38800000u) : 0);
    else if (e == 31) f = (s << 31) | 0x7f800000u | (m << 13);
    else f = (s << 31) | ((e + 112) << 23) | (m << 13);
    float v; memcpy(&v, &f, 4); return v;
}
static const int8_t iq4nl_grid[16] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113};
static float iq4xs_value(const uint8_t *row, unsigned i) {
    const unsigned j = i % 256, ib = j / 32, r = j % 32;
    uint16_t dh, sh;
    memcpy(&dh, row + (i / 256) * 136, 2);
    memcpy(&sh, row + (i / 256) * 136 + 2, 2);
    const unsigned ls = ((row[(i / 256) * 136 + 4 + ib / 2] >> (4 * (ib % 2))) & 15u) |
                        (((sh >> (2 * ib)) & 3u) << 4);
    const float dl = dev_f16(dh) * ((float)ls - 32.0f);
    const unsigned q = r < 16 ? (row[(i / 256) * 136 + 8 + ib * 16 + r] & 15u)
                              : (row[(i / 256) * 136 + 8 + ib * 16 + r - 16] >> 4);
    return dl * (float)iq4nl_grid[q];
}
static void check_tc23(uint32_t K, uint32_t M, uint32_t T) {
    CHECK(ds4_gpu_init());
    const uint64_t rb = (K / 256u) * 136u;
    unsigned char *model = malloc(rb * M);
    float *input = malloc((size_t)T * K * sizeof(float));
    float *ref = malloc((size_t)T * M * sizeof(float));
    float *got = malloc((size_t)T * M * sizeof(float));
    CHECK(model && input && ref && got);
    for (uint64_t b = 0; b < (uint64_t)M * (K / 256u); b++) {
        uint8_t *blk = model + b * 136u;
        uint16_t d = 0x3C00u; /* 1.0 */
        memcpy(blk, &d, 2);
        uint16_t sh = 0xAAAAu;
        memcpy(blk + 2, &sh, 2);
        for (int i = 4; i < 136; i++) blk[i] = (uint8_t)(random_bits() >> 17);
    }
    for (uint64_t i = 0; i < (uint64_t)T * K; i++)
        input[i] = ((int32_t)(random_bits() >> 16) - 32768) / 32768.0f;
    for (uint32_t t = 0; t < T; t++)
        for (uint32_t m = 0; m < M; m++) {
            double acc = 0;
            for (uint32_t k = 0; k < K; k++)
                acc += (double)iq4xs_value(model + (uint64_t)m * rb, k) * (double)input[(uint64_t)t * K + k];
            ref[(uint64_t)t * M + m] = (float)acc;
        }
    CHECK(ds4_gpu_set_model_map(model, rb * M));
    ds4_gpu_tensor *x = ds4_gpu_tensor_alloc((size_t)T * K * sizeof(float));
    ds4_gpu_tensor *y = ds4_gpu_tensor_alloc((size_t)T * M * sizeof(float));
    CHECK(x && y && ds4_gpu_tensor_write(x, 0, input, (size_t)T * K * sizeof(float)));
    CHECK(ds4_gpu_tensor_fill_f32(y, 0.0f, (size_t)T * M));
    CHECK(ds4_gpu_qwen4_dense_mm_tensor(y, x, model, rb * M, 0, 23, T, K, M));
    CHECK(ds4_gpu_tensor_read(y, 0, got, (size_t)T * M * sizeof(float)));
    double maxdiff = 0;
    uint64_t nbad = 0;
    for (uint64_t i = 0; i < (uint64_t)T * M; i++) {
        double d = fabs((double)got[i] - (double)ref[i]);
        if (!(d <= 0.5 * (1.0 + fabs((double)ref[i])))) {
            if (nbad < 5) fprintf(stderr, "K=%u M=%u T=%u i=%llu got=%g ref=%g\n",
                                  K, M, T, (unsigned long long)i, got[i], ref[i]);
            nbad++;
        }
        if (d > maxdiff) maxdiff = d;
    }
    printf("TC23 K=%u M=%u T=%u: maxdiff=%g bad=%llu/%llu %s\n", K, M, T, maxdiff,
           (unsigned long long)nbad, (unsigned long long)((uint64_t)T * M),
           nbad ? "MISMATCH" : "ok");
    free(model); free(input); free(ref); free(got);
}
#include <sys/wait.h>
#include <unistd.h>
int main(void) {
    /* NOTE: one shape per process. ds4_gpu_set_model_map rebind keeps
     * stale ranges for a previous same-offset mapping, so sequential
     * shapes in one process read stale weights (test artifact, not a
     * kernel bug: the server binds once). Fork isolates each shape. */
    const struct { uint32_t K, M, T; } shapes[] = {
        {2560, 10240, 13}, {2560, 10240, 1}, {6144, 2560, 13}, {6144, 2560, 1},
    };
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        pid_t pid = fork();
        CHECK(pid >= 0);
        if (pid == 0) {
            check_tc23(shapes[i].K, shapes[i].M, shapes[i].T);
            fflush(stdout);
            _exit(0);
        }
        int status = 0;
        CHECK(waitpid(pid, &status, 0) == pid);
        CHECK(WIFEXITED(status) && WEXITSTATUS(status) == 0);
    }
    printf("test_cuda_tc23: ok\n");
    return 0;
}
