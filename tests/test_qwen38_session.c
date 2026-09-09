/* Real-model CPU test through the public engine boundary. Build after make cpu:
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

static void same_logits(ds4_session *s, const float *want, int n) {
    float *got = malloc(n*sizeof(float));
    require(got != NULL, "logit allocation");
    require(ds4_session_copy_logits(s, got, n) == n, "copy logits");
    double maxerr = 0;
    for (int i = 0; i < n; i++) {
        require(isfinite(got[i]) && isfinite(want[i]), "non-finite logits");
        maxerr = fmax(maxerr, fabs((double)got[i]-want[i]));
    }
    printf("LOGIT_MAX_ERROR %.9g\n", maxerr);
    require(maxerr == 0, "sync/eval/reset mismatch");
    free(got);
}

int main(int argc, char **argv) {
    require(argc == 3, "usage: test_qwen38_session MODEL TEXT");
    ds4_engine_options opt = {.model_path=argv[1], .backend=DS4_BACKEND_CPU,
        .context_size=128, .power_percent=100};
    ds4_engine *e = NULL;
    require(ds4_engine_open(&e, &opt) == 0, "engine open");
    require(ds4_engine_model_id(e) == 4, "requires Qwen3.8 model");
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
    printf("MEAN_NLL %.9g TOKENS %d\n", nll/(tokens.len-1), tokens.len-1);
    const int nv = 248320;
    float *last = malloc(nv*sizeof(float));
    require(last != NULL, "logit allocation");
    require(ds4_session_copy_logits(s, last, nv) == nv, "copy logits");
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv); /* unchanged prefix */
    /* Force a rewrite to a different first token, then rebuild the full input. */
    int other = tokens.v[0] == 0 ? 1 : 0;
    ds4_tokens rewrite = {.v=&other, .len=1, .cap=1};
    require(ds4_session_sync(s, &rewrite, err, sizeof(err)) == 0, err);
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv);
    /* Shortening resets recurrent history; extending must recover the same state. */
    require(ds4_session_sync(s, &prefix, err, sizeof(err)) == 0, err);
    require(ds4_session_sync(s, &tokens, err, sizeof(err)) == 0, err);
    same_logits(s, last, nv);
    /* Do not write a misleading DeepSeek-shaped cache for recurrent state. */
    FILE *cache = tmpfile();
    require(cache != NULL, "temporary cache");
    require(ds4_session_payload_bytes(s) == 0, "unsupported payload size");
    require(ds4_session_save_payload(s, cache, err, sizeof(err)) != 0, "save must reject unsupported codec");
    require(ftell(cache) == 0, "rejected save wrote data");
    require(ds4_session_load_payload(s, cache, 0, err, sizeof(err)) != 0, "load must reject unsupported codec");
    fclose(cache);
    same_logits(s, last, nv);
    free(last);
    ds4_session_free(s);
    ds4_tokens_free(&tokens);
    ds4_engine_close(e);
    puts("Qwen CPU session PASS");
    return 0;
}
