#include "../ds4.c"
#include <assert.h>
#include <stdio.h>
#include <string.h>

/* Pure CPU side of the gathered MTP draft head: row geometry per type and
 * the raw row gather. GPU scoring/upload stay on the device path. */

static void test_row_bytes(void) {
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q8_0, 2560) == 2720ull);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q8_0, 32) == 34ull);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q6_K, 2560) == 2100ull);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q6_K, 512) == 420ull);
    /* misaligned or unsupported types have no gathered geometry */
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q8_0, 100) == 0);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q6_K, 640) == 0);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q4_K, 2560) == 0);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_F32, 2560) == 0);
    assert(qwen4_mtp_draft_row_bytes(99, 2560) == 0);
    assert(qwen4_mtp_draft_row_bytes(DS4_TENSOR_Q8_0, 0) == 0);
}

static void test_gather_q8(void) {
    /* fake 4-row Q8_0 head: dim0=32 -> 34 bytes/row, each row filled with its index */
    uint8_t backing[4 * 34];
    for (uint32_t r = 0; r < 4; r++) memset(backing + r * 34u, (int)(0x10 + r), 34);
    ds4_model m;
    memset(&m, 0, sizeof(m));
    m.map = backing;
    m.size = sizeof(backing);
    ds4_tensor w;
    memset(&w, 0, sizeof(w));
    w.type = DS4_TENSOR_Q8_0;
    w.ndim = 2;
    w.dim[0] = 32;
    w.dim[1] = 4;
    w.abs_offset = 0;
    const int32_t ids[3] = { 3, 0, 3 };
    uint8_t out[3 * 34];
    memset(out, 0, sizeof(out));
    assert(qwen4_mtp_draft_gather_rows(&m, &w, ids, 3, out));
    assert(memcmp(out + 0 * 34u, backing + 3 * 34u, 34) == 0);
    assert(memcmp(out + 1 * 34u, backing + 0 * 34u, 34) == 0);
    assert(memcmp(out + 2 * 34u, backing + 3 * 34u, 34) == 0);
    /* id out of range, nulls, empty, and truncated mappings are refused */
    const int32_t bad[1] = { 4 };
    assert(!qwen4_mtp_draft_gather_rows(&m, &w, bad, 1, out));
    const int32_t neg[1] = { -1 };
    assert(!qwen4_mtp_draft_gather_rows(&m, &w, neg, 1, out));
    assert(!qwen4_mtp_draft_gather_rows(&m, &w, ids, 0, out));
    assert(!qwen4_mtp_draft_gather_rows(&m, &w, ids, 3, NULL));
    assert(!qwen4_mtp_draft_gather_rows(NULL, &w, ids, 3, out));
    assert(!qwen4_mtp_draft_gather_rows(&m, NULL, ids, 3, out));
    ds4_model small = m;
    small.size = sizeof(backing) - 1;
    assert(!qwen4_mtp_draft_gather_rows(&small, &w, ids, 3, out));
    ds4_tensor off = w;
    off.abs_offset = 34;
    assert(!qwen4_mtp_draft_gather_rows(&m, &off, ids, 3, out));
}

static void test_gather_q6k_offset(void) {
    /* Q6_K rows (dim0=256 -> 210 bytes) gathered through a nonzero offset */
    static uint8_t backing[64 + 3 * 210];
    for (uint32_t r = 0; r < 3; r++) {
        uint8_t *row = backing + 64u + r * 210u;
        for (uint32_t i = 0; i < 210; i++) row[i] = (uint8_t)(r * 37u + i);
    }
    ds4_model m;
    memset(&m, 0, sizeof(m));
    m.map = backing;
    m.size = sizeof(backing);
    ds4_tensor w;
    memset(&w, 0, sizeof(w));
    w.type = DS4_TENSOR_Q6_K;
    w.ndim = 2;
    w.dim[0] = 256;
    w.dim[1] = 3;
    w.abs_offset = 64;
    const int32_t ids[2] = { 2, 1 };
    uint8_t out[2 * 210];
    assert(qwen4_mtp_draft_gather_rows(&m, &w, ids, 2, out));
    assert(memcmp(out, backing + 64u + 2 * 210u, 210) == 0);
    assert(memcmp(out + 210u, backing + 64u + 1 * 210u, 210) == 0);
}

int main(void) {
    test_row_bytes();
    test_gather_q8();
    test_gather_q6k_offset();
    printf("test_qwen4_drafthead: ok\n");
    return 0;
}
