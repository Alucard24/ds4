/* CPU reference decodes for the Unsloth quants: no GPU, no model files.
 *
 * Q6_K rows are checked against an independent element-indexed oracle plus
 * the trusted ds4_vec_dot_q6_K_f32 dot; IQ2_S wiring is checked for correct
 * row stride/dispatch onto ds4_dequant_row_iq2_s; IQ4_NL rows are checked
 * against hand-computed grid values (low nibble first) including strides.
 */
#include "../ds4.c"
#include <assert.h>
#include <math.h>

static uint64_t g_rng = 0x123456789abcdefULL;
static uint64_t next_rand(void) {
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return g_rng >> 11;
}

static ds4_model g_m;
static ds4_tensor g_t;

static void setup_row(const void *buf, uint64_t bytes, uint32_t type, uint64_t dim0) {
    memset(&g_m, 0, sizeof(g_m));
    memset(&g_t, 0, sizeof(g_t));
    g_m.map = (const uint8_t *)buf;
    g_m.size = bytes;
    g_t.type = type;
    g_t.ndim = 2;
    g_t.dim[0] = dim0;
    g_t.dim[1] = 2;
    g_t.abs_offset = 0;
    g_t.file_offset = 0;
}

/* Element-indexed Q6_K oracle written straight from the block layout
 * (ql = low nibbles, qh = high bits, one int8 scale per 16 values),
 * deliberately not mirroring the loop structure under test. */
static float q6k_oracle(const uint8_t *blk, uint32_t i) {
    uint16_t dh;
    memcpy(&dh, blk + 208, sizeof(dh));
    const float d = f16_to_f32(dh);
    const uint32_t h = i >> 7, j = i & 127u;
    const uint32_t l = j & 31u, s = (j >> 5) & 3u;
    const uint8_t *ql = blk + (uint64_t)h * 64u;
    const uint8_t *qh = blk + 128u + (uint64_t)h * 32u;
    const int8_t *sc = (const int8_t *)(blk + 192u + (uint64_t)h * 8u);
    uint32_t low, high;
    if (s == 0) { low = ql[l] & 0x0Fu; high = (qh[l] >> 0) & 3u; }
    else if (s == 1) { low = ql[l + 32] & 0x0Fu; high = (qh[l] >> 2) & 3u; }
    else if (s == 2) { low = ql[l] >> 4; high = (qh[l] >> 4) & 3u; }
    else { low = ql[l + 32] >> 4; high = (qh[l] >> 6) & 3u; }
    const int q = (int)(low | (high << 4)) - 32;
    return d * (float)sc[l / 16u + 2u * s] * (float)q;
}

static void check_q6k(void) {
    static uint8_t blk[2 * 210];
    for (size_t i = 0; i < sizeof(blk); i++) blk[i] = (uint8_t)next_rand();
    /* Pin block 0 to hand-checkable values: d = 1, scales = 1. */
    memset(blk, 0, 210);
    blk[208] = 0x00;
    blk[209] = 0x3C; /* 1.0f */
    /* Keep block 1 random but finite: random f16 scale bits could be NaN. */
    blk[210 + 208] = 0x00;
    blk[210 + 209] = 0x3C;
    for (int i = 0; i < 16; i++) blk[192 + i] = 1;
    blk[0] = 0x12;
    blk[128] = 0xE4;
    setup_row(blk, sizeof(blk), DS4_TENSOR_Q6_K, 256);
    float out[512];
    qwen4_ref_row(&g_m, &g_t, 0, out);
    /* q1: low 0x2, high 0 -> 2 - 32 = -30. q3: low 0x1, high 2 -> 33 - 32 = 1. */
    assert(out[0] == -30.0f);
    assert(out[64] == 1.0f);
    /* Full oracle comparison on both blocks (block 1 stays random). */
    for (uint32_t r = 0; r < 2; r++) {
        qwen4_ref_row(&g_m, &g_t, r, out);
        for (uint32_t i = 0; i < 256; i++) {
            float want = q6k_oracle(blk + (size_t)r * 210u, i);
            assert(fabsf(out[i] - want) <= 1e-5f * (1.0f + fabsf(want)));
        }
    }
    /* Dot-product cross-check against the trusted CPU dot (row 0). */
    float y[256], acc = 0.0f;
    qwen4_ref_row(&g_m, &g_t, 0, out);
    for (int i = 0; i < 256; i++) {
        y[i] = (float)((int)(next_rand() % 2000) - 1000) * 0.01f;
        acc += out[i] * y[i];
    }
    float dot = ds4_vec_dot_q6_K_f32(256, (const block_q6_K *)blk, y);
    assert(fabsf(acc - dot) <= 1e-2f * (1.0f + fabsf(dot)));
}

static void check_iq2s_stride(void) {
    static uint8_t buf[2 * 82];
    for (size_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t)next_rand();
    setup_row(buf, sizeof(buf), DS4_TENSOR_IQ2_S, 256);
    float ref[256], direct[256];
    for (uint32_t r = 0; r < 2; r++) {
        qwen4_ref_row(&g_m, &g_t, r, ref);
        ds4_dequant_row_iq2_s((const block_iq2_s *)(buf + (size_t)r * 82u), direct, 256);
        for (int i = 0; i < 256; i++) assert(ref[i] == direct[i]);
    }
    /* Rows must actually differ: guards a test that passes vacuously. */
    bool same = true;
    for (int i = 0; i < 256; i++) {
        float a, b;
        qwen4_ref_row(&g_m, &g_t, 0, ref);
        a = ref[i];
        qwen4_ref_row(&g_m, &g_t, 1, ref);
        if (a != ref[i]) { same = false; break; }
        (void)b;
    }
    assert(!same);
}

static const int8_t g_grid[16] = {
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
};

/* Wiring proofs for the ISTA/UD4 expert types: correct row stride and
 * dispatch onto the long-tested row dequantizers, plus a dot cross-check
 * where a trusted CPU dot exists. */
static void check_superblock_stride(uint32_t type, uint64_t stride,
                                    void (*dequant)(const void *, float *, int64_t),
                                    float (*vecdot)(int, const void *, const float *)) {
    static uint8_t buf[2 * 256];
    for (size_t i = 0; i < 2 * stride; i++) buf[i] = (uint8_t)next_rand();
    buf[0] = 0x00; buf[1] = 0x3C; /* finite f16 scale in both blocks */
    buf[stride] = 0x00; buf[stride + 1] = 0x3C;
    setup_row(buf, 2 * stride, type, 256);
    float ref[256], direct[256];
    for (uint32_t r = 0; r < 2; r++) {
        qwen4_ref_row(&g_m, &g_t, r, ref);
        dequant(buf + (size_t)r * stride, direct, 256);
        for (int i = 0; i < 256; i++) assert(ref[i] == direct[i]);
    }
    if (vecdot) {
        float y[256], acc = 0.0f;
        qwen4_ref_row(&g_m, &g_t, 0, ref); /* row 0 matches buf below */
        for (int i = 0; i < 256; i++) {
            y[i] = (float)((int)(next_rand() % 2000) - 1000) * 0.01f;
            acc += ref[i] * y[i];
        }
        float dot = vecdot(256, buf, y);
        assert(fabsf(acc - dot) <= 1e-2f * (1.0f + fabsf(dot)));
    }
}

static void check_q5k(void) {
    /* Independent element-indexed oracle straight from the block layout. */
    static uint8_t blk[2 * 176];
    for (size_t i = 0; i < sizeof(blk); i++) blk[i] = (uint8_t)next_rand();
    for (uint32_t r = 0; r < 2; r++) {
        uint8_t *b = blk + (size_t)r * 176u;
        b[0] = 0x00; b[1] = 0x3C; /* d = 1 */
        b[2] = 0x00; b[3] = 0x3C; /* dmin = 1 */
    }
    setup_row(blk, sizeof(blk), DS4_TENSOR_Q5_K, 256);
    float out[256];
    for (uint32_t r = 0; r < 2; r++) {
        uint8_t *b = blk + (size_t)r * 176u;
        qwen4_ref_row(&g_m, &g_t, r, out);
        const float d = f16_to_f32((uint16_t)(b[0] | ((uint16_t)b[1] << 8)));
        const float dmin = f16_to_f32((uint16_t)(b[2] | ((uint16_t)b[3] << 8)));
        for (uint32_t i = 0; i < 256; i++) {
            const uint32_t g = i / 32u, l = i % 32u;
            uint32_t sc, mn;
            if (g < 4) { sc = b[4 + g] & 63u; mn = b[4 + g + 4] & 63u; }
            else {
                sc = (b[4 + g + 4] & 15u) | ((b[4 + g - 4] >> 6) << 4);
                mn = (b[4 + g + 4] >> 4) | ((b[4 + g] >> 6) << 4);
            }
            const uint32_t q = ((b[48 + (g >> 1) * 32 + l] >> ((g & 1u) * 4u)) & 15u) |
                               (((b[16 + l] >> g) & 1u) << 4);
            const float want = d * (float)sc * (float)q - dmin * (float)mn;
            assert(fabsf(out[i] - want) <= 1e-4f * (1.0f + fabsf(want)));
        }
    }
}

static void check_q2_0(void) {
    /* ggml Q2_0: 2-bit values 00=-1..11=+2 times the f16 scale. */
    static uint8_t buf[2 * 18];
    buf[0] = 0x00; buf[1] = 0x3C; /* d = 1.0 */
    buf[2] = 0xE4;               /* bits 11|10|01|00 for j=3..0 -> 2,1,0,-1 reversed: -1,0,1,2 */
    memset(buf + 3, 0x55, 15);   /* 01 -> 0 */
    buf[18] = 0x00; buf[19] = 0x38; /* d = 0.5 */
    memset(buf + 20, 0xFF, 16);  /* 11 -> 2 * 0.5 = 1 */
    setup_row(buf, sizeof(buf), DS4_TENSOR_Q2_0, 64);
    float out[128];
    qwen4_ref_row(&g_m, &g_t, 0, out);
    assert(out[0] == -1.0f && out[1] == 0.0f && out[2] == 1.0f && out[3] == 2.0f);
    for (int i = 4; i < 64; i++) assert(out[i] == 0.0f);
    qwen4_ref_row(&g_m, &g_t, 1, out);
    for (int i = 0; i < 64; i++) assert(out[i] == 1.0f);
}

static void check_iq4xs_stride(void) {
    check_superblock_stride(DS4_TENSOR_IQ4_XS, 136,
        (void (*)(const void *, float *, int64_t))ds4_dequant_row_iq4_xs,
        (float (*)(int, const void *, const float *))ds4_vec_dot_iq4_xs_f32);
}

static void check_iq2xs_stride(void) {
    check_superblock_stride(DS4_TENSOR_IQ2_XS, 74,
        (void (*)(const void *, float *, int64_t))ds4_dequant_row_iq2_xs,
        (float (*)(int, const void *, const float *))ds4_vec_dot_iq2_xs_f32);
}

static void check_iq3xxs_stride(void) {
    check_superblock_stride(DS4_TENSOR_IQ3_XXS, 98,
        (void (*)(const void *, float *, int64_t))ds4_dequant_row_iq3_xxs,
        (float (*)(int, const void *, const float *))ds4_vec_dot_iq3_xxs_f32);
}
static void check_iq3s_stride(void) {
    /* Layer 2 of UD-IQ3_XXS mixes IQ3_S gate/up experts in: same wiring
     * proof as IQ2_S, plus a dot cross-check against the trusted CPU dot. */
    static uint8_t buf[2 * 110];
    for (size_t i = 0; i < sizeof(buf); i++) buf[i] = (uint8_t)next_rand();
    /* Keep both scale halves finite: random f16 d could be NaN. */
    buf[0] = 0x00; buf[1] = 0x3C;
    buf[110] = 0x00; buf[111] = 0x3C;
    setup_row(buf, sizeof(buf), DS4_TENSOR_IQ3_S, 256);
    float ref[256], direct[256];
    for (uint32_t r = 0; r < 2; r++) {
        qwen4_ref_row(&g_m, &g_t, r, ref);
        ds4_dequant_row_iq3_s((const block_iq3_s *)(buf + (size_t)r * 110u), direct, 256);
        for (int i = 0; i < 256; i++) assert(ref[i] == direct[i]);
    }
    float y[256], acc = 0.0f;
    qwen4_ref_row(&g_m, &g_t, 0, ref);
    for (int i = 0; i < 256; i++) {
        y[i] = (float)((int)(next_rand() % 2000) - 1000) * 0.01f;
        acc += ref[i] * y[i];
    }
    float dot = ds4_vec_dot_iq3_s_f32(256, (const block_iq3_s *)buf, y);
    assert(fabsf(acc - dot) <= 1e-2f * (1.0f + fabsf(dot)));
}

static void check_iq4nl(void) {
    static uint8_t buf[2 * 18];
    /* Row 0: d = 1.0, every nibble 0xB/0xA -> grid[11]/grid[10]. */
    buf[0] = 0x00;
    buf[1] = 0x3C;
    memset(buf + 2, 0xAB, 16);
    /* Row 1: d = 0.5, first byte 0x12 (low nibble first), rest 0x00. */
    buf[18] = 0x00;
    buf[19] = 0x38;
    buf[20] = 0x12;
    memset(buf + 21, 0x00, 15);
    setup_row(buf, sizeof(buf), DS4_TENSOR_IQ4_NL, 32);
    float out[64];
    qwen4_ref_row(&g_m, &g_t, 0, out);
    for (int i = 0; i < 16; i++) {
        assert(out[i] == 38.0f);
        assert(out[i + 16] == 25.0f);
    }
    qwen4_ref_row(&g_m, &g_t, 1, out);
    assert(out[0] == -41.5f);  /* 0.5 * grid[2] */
    assert(out[16] == -52.0f); /* 0.5 * grid[1] */
    for (int i = 1; i < 16; i++) {
        assert(out[i] == -63.5f);  /* 0.5 * grid[0] */
        assert(out[i + 16] == -63.5f);
    }
    /* Grid in the test matches the compiled-in table. */
    for (int i = 0; i < 16; i++) assert(g_grid[i] == kvalues_iq4nl[i]);
}

int main(void) {
    check_q6k();
    check_iq2s_stride();
    check_iq4nl();
    check_iq3s_stride();
    check_iq4xs_stride();
    check_iq2xs_stride();
    check_iq3xxs_stride();
    check_q5k();
    check_q2_0();
    printf("test_qwen4_dequant: ok\n");
    return 0;
}
