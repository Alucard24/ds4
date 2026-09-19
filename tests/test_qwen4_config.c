/* qwen4exp trunk-shape fixtures: no tensors, only metadata.
 *
 * Proves the no-MTP trim in config_validate_qwen4_model: an upstream-style
 * file (block_count 48, no nextn_predict_layers key) yields a 48-layer trunk
 * with nextn disabled, while a recipe-style file (block_count 49,
 * nextn_predict_layers 1) keeps the 49-layer + MTP profile. Validation runs
 * in a forked child since it rewrites the global shape.
 */
#include "../ds4.c"
#include <assert.h>
#include <sys/wait.h>

static int g_nkv;

static void put32(FILE *fp, uint32_t v) { assert(fwrite(&v, 4, 1, fp) == 1); }
static void put64(FILE *fp, uint64_t v) { assert(fwrite(&v, 8, 1, fp) == 1); }
static void putstr(FILE *fp, const char *s) {
    put64(fp, strlen(s));
    assert(fwrite(s, 1, strlen(s), fp) == strlen(s));
}

static void u32_kv(FILE *fp, const char *key, uint32_t value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_UINT32); put32(fp, value);
    g_nkv++;
}

static void f32_kv(FILE *fp, const char *key, float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    putstr(fp, key); put32(fp, GGUF_VALUE_FLOAT32); put32(fp, bits);
    g_nkv++;
}

static void u64array_kv(FILE *fp, const char *key, const uint64_t *vals, size_t n) {
    putstr(fp, key);
    put32(fp, GGUF_VALUE_ARRAY);
    put32(fp, GGUF_VALUE_UINT64);
    put64(fp, (uint64_t)n);
    for (size_t i = 0; i < n; i++) put64(fp, vals[i]);
    g_nkv++;
}

static void string_kv(FILE *fp, const char *key, const char *value) {
    putstr(fp, key); put32(fp, GGUF_VALUE_STRING); putstr(fp, value);
    g_nkv++;
}

/* EXP profile values, copied from the on-disk Unsloth metadata. */
static void write_fixture(const char *path, uint32_t block_count, bool with_nextn) {
    FILE *fp = fopen(path, "w+b");
    assert(fp);
    put32(fp, DS4_GGUF_MAGIC); put32(fp, 3); put64(fp, 0);
    long nkv_pos = ftell(fp);
    put64(fp, 0);
    g_nkv = 0;

    string_kv(fp, "general.architecture", "qwen4exp");
    u32_kv(fp, "qwen4exp.embedding_length", 2560);
    u32_kv(fp, "qwen4exp.block_count", block_count);
    if (with_nextn) u32_kv(fp, "qwen4exp.nextn_predict_layers", 1);
    u32_kv(fp, "qwen4exp.attention.head_count", 24);
    u32_kv(fp, "qwen4exp.attention.head_count_kv", 2);
    u32_kv(fp, "qwen4exp.attention.key_length", 256);
    u32_kv(fp, "qwen4exp.attention.value_length", 256);
    u32_kv(fp, "qwen4exp.rope.dimension_count", 64);
    f32_kv(fp, "qwen4exp.rope.freq_base", 10000000.0f);
    u32_kv(fp, "qwen4exp.context_length", 262144);
    f32_kv(fp, "qwen4exp.attention.layer_norm_rms_epsilon", 0.000001f);
    u32_kv(fp, "qwen4exp.expert_count", 512);
    u32_kv(fp, "qwen4exp.expert_used_count", 10);
    u32_kv(fp, "qwen4exp.expert_feed_forward_length", 640);
    u32_kv(fp, "qwen4exp.expert_shared_feed_forward_length", 640);
    u32_kv(fp, "qwen4exp.ssm.conv_kernel", 4);
    u32_kv(fp, "qwen4exp.ssm.state_size", 128);
    u32_kv(fp, "qwen4exp.ssm.group_count", 16);
    u32_kv(fp, "qwen4exp.ssm.time_step_rank", 48);
    u32_kv(fp, "qwen4exp.ssm.inner_size", 6144);
    u32_kv(fp, "qwen4exp.full_attention_interval", 4);
    u32_kv(fp, "qwen4exp.hyper_connection.count", 4);
    u32_kv(fp, "qwen4exp.hyper_connection.low_rank", 320);
    u32_kv(fp, "qwen4exp.attention.indexer.head_count", 4);
    u32_kv(fp, "qwen4exp.attention.indexer.key_length", 128);
    u32_kv(fp, "qwen4exp.attention.indexer.top_k", 2048);

    uint64_t ratios[49];
    for (uint32_t i = 0; i < block_count; i++)
        ratios[i] = ((i + 1u) % 4u == 0u) ? 4u : 0u;
    if (with_nextn) ratios[block_count - 1] = 4; /* the MTP block runs QSA */
    u64array_kv(fp, "qwen4exp.attention.compress_ratios", ratios, block_count);

    uint64_t one[1] = {1};
    u64array_kv(fp, "qwen4exp.ple.layers", one, 1);
    u32_kv(fp, "qwen4exp.ple.ngram_size", 3);
    u32_kv(fp, "qwen4exp.ple.heads_per_ngram", 8);
    u32_kv(fp, "qwen4exp.ple.conv_kernel", 4);
    u32_kv(fp, "qwen4exp.ple.eos_token_id", 248044);
    u32_kv(fp, "qwen4exp.embedding_length_per_layer_input", 160);
    uint64_t mult[3] = {23703573157769ull, 20109073645365ull, 8052911324071ull};
    u64array_kv(fp, "qwen4exp.ple.layer_multipliers", mult, 3);
    uint64_t offs[16], vocs[16];
    for (int i = 0; i < 16; i++) {
        offs[i] = (uint64_t)i * 20000000ull;
        vocs[i] = 20000000ull + (uint64_t)i;
    }
    u64array_kv(fp, "qwen4exp.ple.head_offsets", offs, 16);
    u64array_kv(fp, "qwen4exp.ple.head_vocab_sizes", vocs, 16);

    assert(fseek(fp, nkv_pos, SEEK_SET) == 0);
    put64(fp, (uint64_t)g_nkv);
    assert(fflush(fp) == 0);
    fclose(fp);
}

static void check_shape_sidecar(const char *path, uint32_t layers, uint32_t nextn, bool sidecar);

static void check_shape(const char *path, uint32_t layers, uint32_t nextn) {
    check_shape_sidecar(path, layers, nextn, false);
}

static void check_shape_sidecar(const char *path, uint32_t layers, uint32_t nextn, bool sidecar) {
    pid_t pid = fork();
    assert(pid >= 0);
    if (pid == 0) {
        ds4_model m;
        model_open(&m, path, false, false);
        config_validate_qwen4_model(&m, sidecar);
        assert(DS4_N_LAYER == layers);
        assert(DS4_N_NEXTN_PREDICT == nextn);
        /* Trunk predicates follow the trimmed shape, not the recipe one. */
        assert(ds4_qwen4_layer_is_linear(0));
        assert(!ds4_qwen4_layer_is_linear(3));
        assert(ds4_qwen4_layer_is_ple(1) && !ds4_qwen4_layer_is_ple(0));
        assert(directional_steering_layer_count() == layers - nextn);
        if (nextn == 0) {
            assert(!ds4_qwen4_layer_is_nextn(layers - 1));
        } else {
            assert(!ds4_qwen4_layer_is_nextn(layers - 2));
            assert(ds4_qwen4_layer_is_nextn(layers - 1));
        }
        model_close(&m);
        _exit(0);
    }
    int status;
    assert(waitpid(pid, &status, 0) == pid);
    assert(WIFEXITED(status) && WEXITSTATUS(status) == 0);
}

int main(void) {
    char p48[] = "/tmp/ds4-qwen-cfg48-XXXXXX";
    char p49[] = "/tmp/ds4-qwen-cfg49-XXXXXX";
    int fd = mkstemp(p48);
    assert(fd >= 0);
    close(fd);
    fd = mkstemp(p49);
    assert(fd >= 0);
    close(fd);
    /* Upstream layout: 48 blocks, no nextn key at all. */
    write_fixture(p48, 48, false);
    /* Recipe layout: 48 + MTP block, nextn key present. */
    write_fixture(p49, 49, true);
    check_shape(p48, 48, 0);
    check_shape(p49, 49, 1);
    /* Sidecar layout: 48-block trunk + external head restores 49/1. */
    check_shape_sidecar(p48, 49, 1, true);
    unlink(p48);
    unlink(p49);
    printf("test_qwen4_config: ok\n");
    return 0;
}
