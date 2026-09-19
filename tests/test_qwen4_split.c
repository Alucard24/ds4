/* Split-GGUF loader fixtures: header-only synthetic shards, no GPU.
 *
 * Builds a 2-shard qwen4exp file (shard 0 = metadata only, like the Unsloth
 * packs) with the n-gram table placed mid-file in shard 1, then checks that
 * model_open merges the directories, resolves cross-shard tensor_data, opens
 * the table through its own shard fd, and survives warm + close. Death cases
 * (duplicate names, opening the second shard) run in a fork like the Engram
 * fixture test.
 */
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put16(FILE *fp, uint16_t v) { assert(fwrite(&v, 2, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }
static void putstr(FILE *fp, const char *s) {
    put64(fp, strlen(s));
    assert(fwrite(s, 1, strlen(s), fp) == strlen(s));
}

static void string_kv(FILE *fp, const char *key, const char *value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_STRING); putstr(fp, value);
}

static void u16_kv(FILE *fp, const char *key, uint16_t value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_UINT16); put16(fp, value);
}

static void i32_kv(FILE *fp, const char *key, int32_t value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_INT32);
    assert(fwrite(&value, 4, 1, fp) == 1);
}

static void tensor_entry(FILE *fp, const char *name, uint32_t type,
                         uint64_t d0, uint64_t d1, uint64_t offset) {
    putstr(fp, name); put32(fp, 2); put64(fp, d0); put64(fp, d1);
    put32(fp, type); put64(fp, offset);
}

static uint16_t f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return (uint16_t)(u >> 16);
}

static char g_dir[] = "/tmp/ds4split.XXXXXX";
static char g_p1[256], g_p2[256];

static void write_shards(void) {
    assert(mkdtemp(g_dir) != NULL);
    snprintf(g_p1, sizeof(g_p1), "%s/m-00001-of-00002.gguf", g_dir);
    snprintf(g_p2, sizeof(g_p2), "%s/m-00002-of-00002.gguf", g_dir);

    /* Shard 0: metadata only. Tensors: a.weight F32 [4,2] (8 elems, 32 B). */
    FILE *f1 = fopen(g_p1, "w+b");
    assert(f1);
    put32(f1, DS4_GGUF_MAGIC); put32(f1, 3); put64(f1, 1); put64(f1, 5);
    string_kv(f1, "general.architecture", "qwen4exp");
    u16_kv(f1, "split.no", 0);
    u16_kv(f1, "split.count", 2);
    i32_kv(f1, "split.tensors.count", 3);
    string_kv(f1, "general.name", "split-fixture");
    tensor_entry(f1, "a.weight", DS4_TENSOR_F32, 4, 2, 0);
    long data1 = ftell(f1);
    long pad1 = (32 - (data1 % 32)) % 32;
    for (long i = 0; i < pad1; i++) fputc(0, f1);
    long base1 = ftell(f1);
    assert(base1 == ((data1 + 31) / 32) * 32);
    float a[8];
    for (int i = 0; i < 8; i++) a[i] = 1.0f + (float)i;
    assert(fwrite(a, sizeof(a), 1, f1) == 1);
    assert(fflush(f1) == 0);
    fclose(f1);

    /* Shard 1: mid-file BF16 table [4,3], then b.weight F32 [4,1]. */
    FILE *f2 = fopen(g_p2, "w+b");
    assert(f2);
    put32(f2, DS4_GGUF_MAGIC); put32(f2, 3); put64(f2, 2); put64(f2, 4);
    string_kv(f2, "general.architecture", "qwen4exp");
    u16_kv(f2, "split.no", 1);
    u16_kv(f2, "split.count", 2);
    i32_kv(f2, "split.tensors.count", 3);
    tensor_entry(f2, "per_layer_token_embd.weight", DS4_TENSOR_BF16, 4, 3, 0);
    tensor_entry(f2, "b.weight", DS4_TENSOR_F32, 4, 1, 24);
    long data2 = ftell(f2);
    long pad2 = (32 - (data2 % 32)) % 32;
    for (long i = 0; i < pad2; i++) fputc(0, f2);
    assert(ftell(f2) == ((data2 + 31) / 32) * 32);
    /* Table rows: row r holds 4+r, 8+r, 12+r, 16+r as BF16. */
    for (uint32_t r = 0; r < 3; r++) {
        for (uint32_t i = 0; i < 4; i++) {
            float v = (float)(4 * (i + 1) + r);
            uint16_t b = f32_to_bf16(v);
            assert(fwrite(&b, 2, 1, f2) == 1);
        }
    }
    float b[4] = {101.0f, 102.0f, 103.0f, 104.0f};
    assert(fwrite(b, sizeof(b), 1, f2) == 1);
    assert(fflush(f2) == 0);
    fclose(f2);
}

static void check_open(void) {
    ds4_model m;
    model_open(&m, g_p1, false, true);
    assert(m.n_shards == 2);
    assert(m.n_tensors == 3);
    assert(m.ngram_fd >= 0 && m.ngram_tensor != NULL);

    const ds4_tensor *a = model_find_tensor(&m, "a.weight");
    const ds4_tensor *b = model_find_tensor(&m, "b.weight");
    const ds4_tensor *t = model_find_tensor(&m, "per_layer_token_embd.weight");
    assert(a && b && t);
    assert(a->shard == 0 && b->shard == 1 && t->shard == 1);
    assert(a->bytes == 32 && b->bytes == 16 && t->bytes == 24);
    /* Shard-local file offsets: data starts at the first 32-aligned byte. */
    assert(t->file_offset + 24 == b->file_offset);
    assert(b->file_offset + 16 <= m.shards[1].size);

    /* Cross-shard tensor_data resolves through the single reservation. */
    const float *ad = tensor_data(&m, a);
    for (int i = 0; i < 8; i++) assert(ad[i] == 1.0f + (float)i);
    const float *bd = tensor_data(&m, b);
    for (int i = 0; i < 4; i++) assert(bd[i] == 101.0f + (float)i);

    /* The mid-file table reads back through its shard fd at file offsets. */
    assert(t == m.ngram_tensor);
    for (uint32_t r = 0; r < 3; r++) {
        float row[4];
        assert(qwen4_ngram_row(&m, r, row));
        for (uint32_t i = 0; i < 4; i++)
            assert(row[i] == (float)(4 * (i + 1) + r));
    }

    /* Warm walks both shards while skipping the punched table pages. */
    model_warm_weights(&m);
    model_close(&m);
}

static void expect_death(void (*fn)(void)) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        fn();
        _exit(0);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) != 0);
}

static void open_second_shard(void) {
    ds4_model m;
    model_open(&m, g_p2, false, false);
    model_close(&m);
}

static void write_dup_shards(char *d1, char *d2) {
    FILE *f1 = fopen(d1, "w+b");
    assert(f1);
    put32(f1, DS4_GGUF_MAGIC); put32(f1, 3); put64(f1, 1); put64(f1, 4);
    string_kv(f1, "general.architecture", "qwen4exp");
    u16_kv(f1, "split.no", 0);
    u16_kv(f1, "split.count", 2);
    i32_kv(f1, "split.tensors.count", 2);
    tensor_entry(f1, "dup.weight", DS4_TENSOR_F32, 4, 1, 0);
    long data = ftell(f1);
    for (long i = 0; i < (32 - (data % 32)) % 32; i++) fputc(0, f1);
    float z[4] = {0};
    assert(fwrite(z, sizeof(z), 1, f1) == 1);
    assert(fflush(f1) == 0);
    fclose(f1);

    FILE *f2 = fopen(d2, "w+b");
    assert(f2);
    put32(f2, DS4_GGUF_MAGIC); put32(f2, 3); put64(f2, 1); put64(f2, 4);
    string_kv(f2, "general.architecture", "qwen4exp");
    u16_kv(f2, "split.no", 1);
    u16_kv(f2, "split.count", 2);
    i32_kv(f2, "split.tensors.count", 2);
    tensor_entry(f2, "dup.weight", DS4_TENSOR_F32, 4, 1, 0);
    data = ftell(f2);
    for (long i = 0; i < (32 - (data % 32)) % 32; i++) fputc(0, f2);
    assert(fwrite(z, sizeof(z), 1, f2) == 1);
    assert(fflush(f2) == 0);
    fclose(f2);
}

static char g_d1[256], g_d2[256];

static void open_dup_shards(void) {
    ds4_model m;
    model_open(&m, g_d1, false, false);
    model_close(&m);
}

/* Single-file qwen4exp model with a mid-file IQ4_NL n-gram table [32,2]:
 * the relaxed gate must accept it and reads must dequantize on the fly. */
static char g_nl[256];

static void write_iq4nl_table(void) {
    FILE *fp = fopen(g_nl, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 3); put64(fp, 1);
    string_kv(fp, "general.architecture", "qwen4exp");
    tensor_entry(fp, "w.weight", DS4_TENSOR_F32, 4, 1, 0);
    tensor_entry(fp, "per_layer_token_embd.weight", DS4_TENSOR_IQ4_NL, 32, 2, 16);
    tensor_entry(fp, "z.weight", DS4_TENSOR_F32, 4, 1, 52);
    long data = ftell(fp);
    for (long i = 0; i < (32 - (data % 32)) % 32; i++) fputc(0, fp);
    float w[4] = {7.0f, 8.0f, 9.0f, 10.0f};
    assert(fwrite(w, sizeof(w), 1, fp) == 1);
    /* Row 0: d = 1.0, nibbles 0xB/0xA. Row 1: d = 0.5, first byte 0x12. */
    uint8_t row0[18];
    row0[0] = 0x00; row0[1] = 0x3C;
    memset(row0 + 2, 0xAB, 16);
    uint8_t row1[18];
    row1[0] = 0x00; row1[1] = 0x38; row1[2] = 0x12;
    memset(row1 + 3, 0x00, 15);
    assert(fwrite(row0, sizeof(row0), 1, fp) == 1);
    assert(fwrite(row1, sizeof(row1), 1, fp) == 1);
    float z[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    assert(fwrite(z, sizeof(z), 1, fp) == 1);
    assert(fflush(fp) == 0);
    fclose(fp);
}

static void check_iq4nl_table(void) {
    ds4_model m;
    model_open(&m, g_nl, false, true);
    assert(m.n_shards == 1);
    assert(m.ngram_fd >= 0 && m.ngram_tensor);
    assert(m.ngram_tensor->type == DS4_TENSOR_IQ4_NL);
    float row[32];
    assert(qwen4_ngram_row(&m, 0, row));
    for (int i = 0; i < 16; i++) {
        assert(row[i] == 38.0f);
        assert(row[i + 16] == 25.0f);
    }
    assert(qwen4_ngram_row(&m, 1, row));
    assert(row[0] == -41.5f);
    assert(row[16] == -52.0f);
    /* The neighbor after the mid-file table still resolves. */
    const float *zd =
        tensor_data(&m, model_find_tensor(&m, "z.weight"));
    for (int i = 0; i < 4; i++) assert(zd[i] == 1.0f + (float)i);
    model_warm_weights(&m);
    model_close(&m);
}

int main(void) {
    write_shards();
    check_open();
    expect_death(open_second_shard);

    snprintf(g_d1, sizeof(g_d1), "%s/d-00001-of-00002.gguf", g_dir);
    snprintf(g_d2, sizeof(g_d2), "%s/d-00002-of-00002.gguf", g_dir);
    write_dup_shards(g_d1, g_d2);
    expect_death(open_dup_shards);

    snprintf(g_nl, sizeof(g_nl), "%s/nl.gguf", g_dir);
    write_iq4nl_table();
    check_iq4nl_table();

    unlink(g_p1);
    unlink(g_p2);
    unlink(g_d1);
    unlink(g_d2);
    unlink(g_nl);
    rmdir(g_dir);
    printf("test_qwen4_split: ok\n");
    return 0;
}
