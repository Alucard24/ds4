/* Real-model CPU/CUDA test through the public engine boundary. Build after make cpu:
 * cc -O2 -std=c99 -I. tests/test_qwen38_session.c ds4_cpu.o ds4_image.o \
 *   ds4_distributed.o ds4_tp.o ds4_ssd.o ds4_layer_pack.o -lm -pthread -o /tmp/qwen-session
 * /tmp/qwen-session MODEL 'The capital of France is Paris.'
 * Emits next-token log probabilities; compares every final logit after sync,
 * eval, rewrite/replay, and a no-op sync. No concurrent model instances. */
#include "ds4.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int ok, const char *message) {
    if (!ok) { fprintf(stderr, "FAIL: %s\n", message); exit(1); }
}

static double rebuild_logit_tolerance;

static void check_qwen_tokenizer(ds4_engine *e) {
    static const int ids0[] = {
        9419, 11, 50203, 1892, 220, 99986, 171405, 59720, 102, 9008,
        237, 121, 373, 235, 88995, 119, 198, 248045, 74455, 198,
    };
    static const int ids1[] = {
        727, 50203, 2007, 1590, 198, 827, 5046, 95789, 763, 328,
        9008, 239, 102, 9008, 237, 121, 373, 235, 88995, 119, 487,
        328, 77, 763, 220, 18, 13, 16, 19, 92, 198,
    };
    static const int ids2[] = {
        248045, 74455, 198, 248068, 198, 3965, 40312, 198, 248069,
        198, 248058, 198, 27, 1628, 21402, 956, 29, 198, 27, 15704,
        28, 5454, 29, 198, 24751, 198, 510, 15704, 29, 198, 510,
        1628, 29, 198, 248059, 248046, 198,
    };
    static const int ids3[] = {
        68, 52033, 3825, 59720, 101, 373, 235, 9008, 239, 102, 373,
        235, 9008, 239, 100, 373, 235, 9008, 239, 99, 190488, 150127,
        177453, 181204, 190925, 211075, 198,
    };
    static const int ids4[] = {248053, 248056, 248054, 248057};
    static const struct {
        const char *text;
        const int *ids;
        size_t count;
        int rendered;
    } cases[] = {
        {"Hello, café — 中文 العربية 👩🏽‍💻\n<|im_start|>assistant\n",
         ids0, sizeof(ids0) / sizeof(ids0[0]), 1},
        {"def café(x):\n\treturn {\"中\": \"👩🏽‍💻\", \"n\": 3.14}\n",
         ids1, sizeof(ids1) / sizeof(ids1[0]), 0},
        {"<|im_start|>assistant\n<think>\nragiona\n</think>\n<tool_call>\n"
         "<function=bash>\n<parameter=command>\npwd\n</parameter>\n"
         "</function>\n</tool_call><|im_end|>\n",
         ids2, sizeof(ids2) / sizeof(ids2[0]), 1},
        {"é é 👨‍👩‍👧‍👦 हिन्दी ไทย 한국어 русский\n",
         ids3, sizeof(ids3) / sizeof(ids3[0]), 0},
        {"<|vision_start|><|image_pad|><|vision_end|><|video_pad|>",
         ids4, sizeof(ids4) / sizeof(ids4[0]), 1},
    };
    for (size_t i = 0; i < sizeof(cases) / sizeof(cases[0]); i++) {
        ds4_tokens got = {0};
        if (cases[i].rendered)
            ds4_tokenize_rendered_chat(e, cases[i].text, &got);
        else
            ds4_tokenize_text(e, cases[i].text, &got);
        if (got.len != (int)cases[i].count ||
            memcmp(got.v, cases[i].ids, cases[i].count * sizeof(int)) != 0) {
            fprintf(stderr, "FAIL: tokenizer case %zu differs from llama.cpp\n", i);
            exit(1);
        }
        ds4_tokens_free(&got);
    }
}

static void same_logits(ds4_session *s, const float *want, int n,
                        double tolerance) {
    float *got = malloc(n*sizeof(float));
    require(got != NULL, "logit allocation");
    require(ds4_session_copy_logits(s, got, n) == n, "copy logits");
    double maxerr = 0;
    int got_best = 0, want_best = 0;
    for (int i = 0; i < n; i++) {
        require(isfinite(got[i]) && isfinite(want[i]), "non-finite logits");
        maxerr = fmax(maxerr, fabs((double)got[i]-want[i]));
        if (got[i] > got[got_best]) got_best = i;
        if (want[i] > want[want_best]) want_best = i;
    }
    printf("LOGIT_MAX_ERROR %.9g ARGMAX %d\n", maxerr, got_best);
    require(got_best == want_best, "sync/eval/reset changed argmax");
    require(maxerr <= tolerance, "sync/eval/reset mismatch");
    free(got);
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: test_qwen38_session MODEL TEXT");
    const int use_cuda = getenv("DS4_TEST_QWEN38_CUDA") != NULL;
    /* Batched MMQ prefill changes reduction order from MMVQ decode. Its
     * rebuild gate is numerical + argmax-stable; CPU remains bit-exact. */
    rebuild_logit_tolerance = use_cuda ? 0.5 : 0.0;
    ds4_engine_options opt = {.model_path=argv[1],
        .backend=use_cuda ? DS4_BACKEND_CUDA : DS4_BACKEND_CPU,
        .context_size=128, .power_percent=100};
    ds4_engine *e = NULL;
    require(ds4_engine_open(&e, &opt) == 0, "engine open");
    require(ds4_engine_model_id(e) == 4, "requires Qwen3.8 model");
    check_qwen_tokenizer(e);
    ds4_tokens tokens = {0};
    ds4_tokenize_text(e, argv[2], &tokens);
    require(tokens.len > 1 && tokens.len < 127, "text must encode to 2..126 tokens");
    ds4_session *s = NULL;
    require(ds4_session_create(&s, e, 128) == 0, "session create");
    char err[256] = {0};
    ds4_tokens prefix = tokens; prefix.len = 1;
    require(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0, err);
    double nll = 0;
    for (int i = 1; i < tokens.len; i++) {
        ds4_token_score score;
        require(ds4_session_token_logprob(s, tokens.v[i], &score) == 1, "token logprob");
        printf("LP %d %d %.9g\n", i, tokens.v[i], score.logprob);
        nll -= score.logprob;
        require(ds4_session_eval(s, tokens.v[i], err, sizeof(err)) == 0, err);
    }
    const double mean_nll = nll / (tokens.len - 1);
    printf("MEAN_NLL %.9g TOKENS %d\n", mean_nll, tokens.len - 1);
    if (use_cuda && !strcmp(argv[2],
            "The capital of France is Paris. The largest ocean on Earth is "
            "the Pacific Ocean.")) {
        require(fabs(mean_nll - 1.81334038) <= 1.0e-5,
                "CUDA reference NLL drift");
    }
    const int nv = 248320;
    float *last = malloc(nv*sizeof(float));
    require(last != NULL, "logit allocation");
    require(ds4_session_copy_logits(s, last, nv) == nv, "copy logits");
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv, rebuild_logit_tolerance); /* unchanged prefix */
    /* Force a rewrite to a different first token, then rebuild the full input. */
    int other = tokens.v[0] == 0 ? 1 : 0;
    ds4_tokens rewrite = {.v=&other, .len=1, .cap=1};
    require(ds4_session_sync(s, &rewrite, err, sizeof(err)) == 0, err);
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv, rebuild_logit_tolerance);
    /* Shortening resets recurrent history; extending must recover the same state. */
    require(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0, err);
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv, rebuild_logit_tolerance);

    /* A payload must restore logits and every persistent GDN/GA state needed
     * for an exactly reproducible next token. CUDA writes its native F16 GA KV;
     * the CPU oracle writes F32. */
    FILE *cache = tmpfile();
    require(cache != NULL, "temporary cache");
    float *saved = malloc(nv * sizeof(float));
    require(saved != NULL, "saved logit allocation");
    require(ds4_session_copy_logits(s, saved, nv) == nv, "saved logits");
    const uint64_t payload_bytes = ds4_session_payload_bytes(s);
    require(payload_bytes > 150u * 1024u * 1024u, "Qwen payload size");
    require(ds4_session_save_payload(s, cache, err, sizeof(err)) == 0, err);
    require((uint64_t)ftello(cache) == payload_bytes, "payload byte count");

    const int probe = tokens.v[tokens.len / 2];
    require(ds4_session_eval(s, probe, err, sizeof(err)) == 0, err);
    float *continued = malloc(nv * sizeof(float));
    require(continued != NULL, "continuation logit allocation");
    require(ds4_session_copy_logits(s, continued, nv) == nv,
            "continuation logits");

    require(fseeko(cache, 0, SEEK_SET) == 0, "payload rewind");
    require(ds4_session_load_payload(s, cache, payload_bytes,
                                     err, sizeof(err)) == 0, err);
    require(ds4_session_pos(s) == tokens.len, "restored checkpoint position");
    same_logits(s, saved, nv, 0.0);
    require(ds4_session_eval(s, probe, err, sizeof(err)) == 0, err);
    same_logits(s, continued, nv, 0.0);

    /* Size validation must reject truncation before mutating the live state. */
    require(fseeko(cache, 0, SEEK_SET) == 0, "truncated payload rewind");
    require(ds4_session_load_payload(s, cache, payload_bytes - 1u,
                                     err, sizeof(err)) != 0,
            "truncated payload accepted");
    require(ds4_session_pos(s) == tokens.len + 1,
            "rejected payload changed checkpoint");
    same_logits(s, continued, nv, 0.0);
    free(continued);
    free(saved);
    fclose(cache);
    free(last);
    ds4_session_free(s);
    ds4_tokens_free(&tokens);
    ds4_engine_close(e);
    printf("Qwen %s session PASS\n", use_cuda ? "CUDA" : "CPU");
    return 0;
}
