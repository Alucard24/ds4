/* Optional independent NLL probe against a local llama.cpp build (C API).
 * cc -O2 -I$LLAMA/include -I$LLAMA/ggml/include tests/qwen38_llama_reference.c \
 *   -L$LLAMA/build/bin -Wl,-rpath,$LLAMA/build/bin -lllama -lm -o /tmp/qwen-ref
 * /tmp/qwen-ref MODEL TEXT
 * Compare LP lines with test_qwen38_session: identical raw token IDs, no BOS,
 * no template, single-token decode, CPU weights and no operation offload. */
#include "llama.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) return 1;
    llama_backend_init();
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    struct llama_model *model = llama_model_load_from_file(argv[1], mp);
    if (!model) return 1;
    const struct llama_vocab *vocab = llama_model_get_vocab(model);
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 128;
    cp.offload_kqv = false;
    cp.op_offload = false;
    struct llama_context *ctx = llama_init_from_model(model, cp);
    if (!ctx) return 1;
    llama_token tokens[128];
    int n = llama_tokenize(vocab, argv[2], strlen(argv[2]), tokens, 128, false, false);
    if (n < 2 || n >= 128) return 1;
    struct llama_batch b = llama_batch_init(1, 0, 1);
    double nll = 0;
    for (int i = 0; i < n; i++) {
        if (i > 0) {
            const float *logits = llama_get_logits_ith(ctx, -1);
            if (!logits) return 1;
            const int nv = llama_vocab_n_tokens(vocab);
            double max = -INFINITY, z = 0;
            for (int j = 0; j < nv; j++) if (logits[j] > max) max = logits[j];
            for (int j = 0; j < nv; j++) z += exp(logits[j]-max);
            double lp = logits[tokens[i]]-max-log(z);
            if (!isfinite(lp)) return 1;
            nll -= lp;
            printf("LP %d %d %.9g\n", i, tokens[i], lp);
        }
        b.n_tokens=1; b.token[0]=tokens[i]; b.pos[0]=i;
        b.n_seq_id[0]=1; b.seq_id[0][0]=0; b.logits[0]=1;
        if (llama_decode(ctx, b) != 0) return 1;
    }
    printf("MEAN_NLL %.9g TOKENS %d\n", nll/(n-1), n-1);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
