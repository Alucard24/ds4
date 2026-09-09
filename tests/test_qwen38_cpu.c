/* Test the actual private CPU kernels without widening ds4.h.
 * Linux: tests/run_qwen38_cpu.sh [path/to/libggml-cpu.so [model.gguf]]
 * Optional ggml comparison is a test-only dependency, never an engine link. */
#include "../ds4.c"
#include <dlfcn.h>

static void check_close(double got, double want, double tol, const char *name) {
    if (!isfinite(got) || !isfinite(want) || fabs(got - want) > tol) {
        fprintf(stderr, "%s: got %.9g, expected %.9g (tol %.3g)\n", name, got, want, tol);
        exit(1);
    }
}

static void test_rope(void) {
    float v[256], original[256];
    for (int i = 0; i < 256; i++) original[i] = (i - 123) / 71.0f;
    const uint32_t positions[] = {0, 1, 17, 255};
    for (unsigned p = 0; p < sizeof(positions)/sizeof(*positions); p++) {
        memcpy(v, original, sizeof(v));
        qwen38_rope_inplace(v, 64, positions[p]);
        for (int j = 0; j < 32; j++) {
            double a = positions[p] * pow(1e7, -2.0*j/64);
            check_close(v[j], original[j]*cos(a) - original[j+32]*sin(a), 3e-5, "rope low");
            check_close(v[j+32], original[j]*sin(a) + original[j+32]*cos(a), 3e-5, "rope high");
        }
        for (int j = 64; j < 256; j++) check_close(v[j], original[j], 0, "rope tail");
    }
}

static float conv_input(int t, int d) {
    return t < 0 ? 0 : ((t * 7 + d * 3) % 31 - 15) / 9.0f;
}

static void test_conv(void) {
    float *h = calloc(3 * QWEN38_CONV_DIM, sizeof(float));
    float *w = malloc(4 * QWEN38_CONV_DIM * sizeof(float));
    float *x = malloc(QWEN38_CONV_DIM * sizeof(float));
    if (!h || !w || !x) exit(1);
    for (unsigned d = 0; d < QWEN38_CONV_DIM; d++)
        for (int k = 0; k < 4; k++) w[4*d+k] = ((d + k*5) % 13 - 6.0f) / 17;
    for (int t = 0; t < 8; t++) {
        for (unsigned d = 0; d < QWEN38_CONV_DIM; d++) x[d] = conv_input(t, d);
        qwen38_conv_step(x, h, w);
        for (unsigned d = 0; d < QWEN38_CONV_DIM; d++) {
            double y = 0;
            for (int k = 0; k < 4; k++) y += (double)w[4*d+k] * conv_input(t-3+k, d);
            check_close(x[d], y/(1+exp(-y)), 1e-6, "conv output");
            for (int k = 0; k < 3; k++)
                check_close(h[k*QWEN38_CONV_DIM+d], conv_input(t-2+k, d), 0, "conv history");
        }
    }
    free(h); free(w); free(x);
}

static void test_gqa(void) {
    float k[3*1024], v[3*1024], q[24*512] = {0}, out[24*256], scores[24*3];
    for (int t = 0; t < 3; t++) for (int h = 0; h < 4; h++) for (int d = 0; d < 256; d++) {
        k[t*1024+h*256+d] = d == 0 ? (t-1)*0.25f : 0;
        v[t*1024+h*256+d] = h*10 + t + d*0.001f;
    }
    for (int h = 0; h < 24; h++) {
        q[h*512] = (h+1)*0.1f;
        for (int d = 0; d < 256; d++) q[h*512+256+d] = (d%5-2)*0.2f;
    }
    for (int n = 1; n <= 3; n++) {
        qwen38_ga_head_ctx c = {.scores=scores, .kcache=k, .vcache=v, .q_full=q,
            .out=out, .n_attend=n, .ctx_size=3, .kq_scale=1.0f/16};
        qwen38_ga_head_worker(&c, 0, 24);
        for (int h = 0; h < 24; h++) for (int d = 0; d < 256; d++) {
            double sum = 0, weighted = 0;
            for (int t = 0; t < n; t++) {
                double e = exp((double)q[h*512] * k[t*1024+(h/6)*256] / 16);
                sum += e;
                weighted += e * v[t*1024+(h/6)*256+d];
            }
            double want = weighted/sum / (1+exp(-q[h*512+256+d]));
            check_close(out[h*256+d], want, 5e-6, "GQA head/causal/gate");
        }
    }
}

static void test_gdn(void) {
    const size_t ns = 48*128*128;
    float *state = calloc(ns, sizeof(float));
    double *ref = calloc(ns, sizeof(double));
    float q[16*128], k[16*128], v[48*128], z[48*128], out[48*128];
    float alpha[48], beta[48], a[48], dt[48], norm[128];
    if (!state || !ref) exit(1);
    for (int j = 0; j < 128; j++) norm[j] = 0.5f+j/128.0f;
    qwen38_gdn_head_ctx c = {.S=state, .q=q, .k=k, .v=v, .z=z, .o=out,
        .alpha=alpha, .beta=beta, .a=a, .dt=dt, .ssm_norm=norm,
        .q_scale=1.0f/sqrtf(128), .eps=1e-6f};
    for (int t = 0; t < 4; t++) {
        for (int h = 0; h < 16; h++) for (int j = 0; j < 128; j++) {
            q[h*128+j] = sinf((h*128+j+t)*0.1f)/8;
            k[h*128+j] = cosf((h*128+j+t)*0.13f)/8;
        }
        for (int h = 0; h < 48; h++) {
            a[h] = -0.1f-h*0.01f; dt[h] = h*0.003f;
            alpha[h] = (h%7-3)*0.2f; beta[h] = (h%5-2)*0.1f;
            double decay = exp(a[h]*log1p(exp((double)alpha[h]+dt[h])));
            double b = 1/(1+exp(-beta[h])), raw[128], sumsq = 0;
            for (int j = 0; j < 128; j++) {
                v[h*128+j] = sinf((h*128+j+t)*0.23f);
                z[h*128+j] = cosf((h*128+j+t)*0.17f);
                double pred = 0;
                for (int i = 0; i < 128; i++) pred += decay*ref[h*128*128+i*128+j]*k[(h%16)*128+i];
                double delta = b*(v[h*128+j]-pred), o = 0;
                for (int i = 0; i < 128; i++) {
                    size_t idx = h*128*128+i*128+j;
                    ref[idx] = decay*ref[idx] + k[(h%16)*128+i]*delta;
                    o += ref[idx]*q[(h%16)*128+i]*c.q_scale;
                }
                raw[j] = o; sumsq += o*o;
            }
            for (int j = 0; j < 128; j++) raw[j] = raw[j]/sqrt(sumsq/128+c.eps)*norm[j]*
                z[h*128+j]/(1+exp(-z[h*128+j]));
            qwen38_gdn_head_worker(&c, h, h+1);
            for (int j = 0; j < 128; j++) check_close(out[h*128+j], raw[j], 2e-5, "GDN output");
        }
        for (size_t j = 0; j < ns; j++) check_close(state[j], ref[j], 1e-6, "GDN state");
    }
    free(state); free(ref);
}

static uint32_t rng = 0x41789321;
static uint32_t next_u32(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
    return rng;
}

typedef void (*ggml_dequant_fn)(const void *, float *, int64_t);
static const struct { uint32_t type; const char *name; } quants[] = {
    {DS4_TENSOR_IQ2_XXS, "iq2_xxs"}, {DS4_TENSOR_IQ2_XS, "iq2_xs"},
    {DS4_TENSOR_IQ2_S, "iq2_s"}, {DS4_TENSOR_IQ3_XXS, "iq3_xxs"},
    {DS4_TENSOR_IQ3_S, "iq3_s"}, {DS4_TENSOR_IQ4_XS, "iq4_xs"},
    {DS4_TENSOR_IQ1_M, "iq1_m"}, {DS4_TENSOR_Q2_K, "q2_K"},
    {DS4_TENSOR_Q4_K, "q4_K"}
};

static void compare_row(uint32_t type, ggml_dequant_fn ref, const void *row, int n) {
    float *a = malloc(n*sizeof(float)), *b = malloc(n*sizeof(float)), *x = malloc(n*sizeof(float));
    if (!a || !b || !x) exit(1);
    ref(row, b, n);
    qwen38_dequant_row(type, row, n, a);
    double dot = 0, abs_sum = 0;
    for (int j = 0; j < n; j++) {
        check_close(a[j], b[j], 2e-6*fmax(1, fabs(b[j])), "ggml dequant");
        x[j] = (int)(next_u32()%2001) / 1000.0f - 1;
        dot += (double)b[j]*x[j];
        abs_sum += fabs((double)b[j]*x[j]);
    }
    check_close(qwen38_row_dot(type, n, row, x), dot, 2e-5*fmax(1, abs_sum), "ggml float dot");
    free(a); free(b); free(x);
}

static void test_quants(const char *library, const char *model_path) {
    void *lib = dlopen(library, RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "%s\n", dlerror()); exit(1); }
    ds4_model m = {0};
    if (model_path) model_open(&m, model_path, false, false);
    for (unsigned q = 0; q < sizeof(quants)/sizeof(*quants); q++) {
        char symbol[96];
        snprintf(symbol, sizeof(symbol), "dequantize_row_%s", quants[q].name);
        ggml_dequant_fn ref = (ggml_dequant_fn)dlsym(lib, symbol);
        if (!ref) { fprintf(stderr, "missing %s\n", symbol); exit(1); }
        uint64_t bytes;
        if (!tensor_nbytes(quants[q].type, 256, &bytes)) exit(1);
        uint8_t *data = malloc(bytes*4);
        if (!data) exit(1);
        for (int trial = 0; trial < 64; trial++) {
            for (uint64_t i = 0; i < bytes*4; i++) data[i] = next_u32();
            for (int block = 0; block < 4; block++) {
                uint8_t *b = data + block*bytes;
                uint16_t scale = (uint16_t)(0x1000 + next_u32()%0x2800);
                if (quants[q].type == DS4_TENSOR_IQ1_M) {
                    /* Global fp16 scale is packed in four high nibbles. */
                    block_iq1_m *iq = (block_iq1_m *)b;
                    uint16_t s[4]; memcpy(s, iq->scales, sizeof(s));
                    for (int i = 0; i < 4; i++) s[i] = (s[i]&0xfff) | (((scale>>(4*i))&15)<<12);
                    memcpy(iq->scales, s, sizeof(s));
                } else if (quants[q].type == DS4_TENSOR_Q2_K) {
                    block_q2_K *k = (block_q2_K *)b; k->d = scale; k->dmin = 0x2000;
                } else {
                    memcpy(b, &scale, 2);
                    if (quants[q].type == DS4_TENSOR_Q4_K) ((block_q4_K *)b)->dmin = 0x2000;
                }
            }
            compare_row(quants[q].type, ref, data, 1024);
        }
        free(data);
        unsigned tensors = 0;
        if (model_path) for (uint64_t i = 0; i < m.n_tensors; i++) {
            const ds4_tensor *t = &m.tensors[i];
            if (t->type != quants[q].type || t->ndim != 2) continue;
            uint64_t stride;
            if (!tensor_nbytes(t->type, t->dim[0], &stride)) exit(1);
            const uint8_t *base = tensor_data(&m, t);
            compare_row(t->type, ref, base, t->dim[0]);
            compare_row(t->type, ref, base + (t->dim[1]-1)*stride, t->dim[0]);
            tensors++;
        }
        printf("%s: 64 synthetic rows, %u real tensors PASS\n", quants[q].name, tensors);
    }
    if (model_path) model_close(&m);
    dlclose(lib);
}

int main(int argc, char **argv) {
    test_rope(); test_conv(); test_gqa(); test_gdn();
    puts("Qwen CPU: RoPE, conv history, gated GQA, multi-step GDN PASS");
    if (argc > 1) test_quants(argv[1], argc > 2 ? argv[2] : NULL);
    return 0;
}
