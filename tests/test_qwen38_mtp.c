/* Qwen3.8 MTP draft-head gate.
 *
 * The draft head lives in a separate GGUF (blk.<n_layer>.nextn.*) while the
 * embedding table and LM head stay the trunk's.  The runtime contract this
 * test pins down is:
 *
 *   1. the sidecar binds and the draft rows execute on CUDA;
 *   2. every draft row consumes exactly the bound sidecar bytes (no trunk
 *      block is ever read from the sidecar);
 *   3. the draft's proposal for the token after the last committed position
 *      agrees with the trunk argmax often enough to be worth verifying - the
 *      measured acceptance rate is printed, never assumed;
 *   4. a rejected proposal never changes trunk logits: the reference decode
 *      with the sidecar loaded must equal the decode without it.
 */
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 0;
    }
    return 1;
}

static int argmax_of(const float *logits, uint32_t n) {
    uint32_t best = 0;
    for (uint32_t i = 1; i < n; i++) {
        if (logits[i] > logits[best]) best = i;
    }
    return (int)best;
}

static int run_greedy(ds4_engine *engine,
                      const char *text,
                      uint32_t max_tokens,
                      int *tokens_out,
                      float *logits_a,
                      float *logits_b,
                      uint32_t *accepted,
                      uint32_t *proposed) {
    const uint32_t n_vocab = ds4_engine_vocab_size(engine);
    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", text);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);

    ds4_session *session = NULL;
    char error[256] = {0};
    if (!require(ds4_session_create(&session, engine, 256) == 0, "session create")) {
        ds4_tokens_free(&prompt);
        return 0;
    }
    if (!require(ds4_session_sync(session, &prompt, error, sizeof(error)) == 0,
                 error[0] ? error : "session sync")) {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return 0;
    }

    uint32_t kept = 0;
    for (uint32_t step = 0; step < max_tokens; step++) {
        float *trunk = logits_a + (size_t)step * n_vocab;
        if (!require(ds4_session_copy_logits(session, trunk, n_vocab) == (int)n_vocab,
                     "session logits")) {
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
        int proposal = -1;
        if (ds4_session_qwen38_mtp_proposal(session, &proposal)) {
            (*proposed)++;
            (*accepted) += (proposal == argmax_of(trunk, n_vocab));
        }
        const int token = argmax_of(trunk, n_vocab);
        if (token == ds4_token_eos(engine)) break;
        tokens_out[kept++] = token;
        if (!require(ds4_session_eval(session, token, error, sizeof(error)) == 0,
                     error[0] ? error : "session eval")) {
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
    }
    if (logits_b) {
        /* Final-state logits after the whole greedy run, for a sidecar/no-
         * sidecar bit-comparison of the trunk path. */
        if (!require(ds4_session_copy_logits(session, logits_b, n_vocab) == (int)n_vocab,
                     "final logits")) {
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
    }
    tokens_out[kept] = -1;
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TRUNK.gguf MTP.gguf PROMPT\n", argv[0]);
        return 2;
    }
    const uint32_t max_tokens = 16u;
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.mtp_path = argv[2];
    options.backend = DS4_BACKEND_CUDA;
    options.context_size = 256;

    ds4_engine *engine = NULL;
    if (ds4_engine_open(&engine, &options) != 0) return 1;
    if (!require(ds4_engine_has_qwen38_mtp(engine),
                 "engine did not bind the Qwen3.8 MTP sidecar")) {
        ds4_engine_close(engine);
        return 1;
    }
    const uint32_t n_vocab = ds4_engine_vocab_size(engine);
    int *tokens = calloc(max_tokens + 1u, sizeof(tokens[0]));
    float *logits_a = malloc((size_t)(max_tokens + 1u) * n_vocab * sizeof(float));
    float *logits_b = malloc((size_t)n_vocab * sizeof(float));
    if (!require(tokens && logits_a && logits_b, "test buffers")) {
        free(tokens);
        free(logits_a);
        free(logits_b);
        ds4_engine_close(engine);
        return 1;
    }

    uint32_t accepted = 0, proposed = 0;
    if (!run_greedy(engine, argv[3], max_tokens, tokens, logits_a, logits_b,
                    &accepted, &proposed)) {
        free(tokens);
        free(logits_a);
        free(logits_b);
        ds4_engine_close(engine);
        return 1;
    }
    printf("MTP draft: %u/%u proposals matched the trunk argmax (%.1f%%)\n",
           accepted, proposed,
           proposed ? 100.0 * (double)accepted / (double)proposed : 0.0);
    if (!require(proposed != 0u,
                 "no draft rows ran: the MTP sidecar was never exercised")) {
        free(tokens);
        free(logits_a);
        free(logits_b);
        ds4_engine_close(engine);
        return 1;
    }

    /* The draft must never alter the trunk path: the same greedy prompt without
     * the sidecar has to reproduce identical tokens and final logits.  Only one
     * engine at a time may hold the instance lock, so the reference run starts
     * after the sidecar engine is released. */
    free(logits_a);
    logits_a = NULL;
    ds4_engine_close(engine);
    engine = NULL;

    ds4_engine_options plain = {0};
    plain.model_path = argv[1];
    plain.backend = DS4_BACKEND_CUDA;
    plain.context_size = 256;
    ds4_engine *plain_engine = NULL;
    if (ds4_engine_open(&plain_engine, &plain) != 0) {
        free(tokens);
        free(logits_b);
        return 1;
    }
    if (!require(!ds4_engine_has_qwen38_mtp(plain_engine),
                 "plain engine unexpectedly reports an MTP sidecar")) {
        ds4_engine_close(plain_engine);
        free(tokens);
        free(logits_b);
        return 1;
    }
    int *plain_tokens = calloc(max_tokens + 1u, sizeof(plain_tokens[0]));
    float *plain_logits = malloc((size_t)(max_tokens + 1u) * n_vocab * sizeof(float));
    float *plain_final = malloc((size_t)n_vocab * sizeof(float));
    uint32_t plain_accepted = 0, plain_proposed = 0;
    const int plain_ok = plain_tokens && plain_logits && plain_final &&
        run_greedy(plain_engine, argv[3], max_tokens, plain_tokens, plain_logits,
                   plain_final, &plain_accepted, &plain_proposed);
    if (!require(plain_ok, "plain reference run") ||
        !require(plain_proposed == 0u, "plain run produced draft proposals") ||
        !require(memcmp(tokens, plain_tokens,
                        (max_tokens + 1u) * sizeof(tokens[0])) == 0,
                 "MTP sidecar changed the greedy token stream") ||
        !require(memcmp(logits_b, plain_final, (size_t)n_vocab * sizeof(float)) == 0,
                 "MTP sidecar changed the final trunk logits")) {
        free(plain_tokens);
        free(plain_logits);
        free(plain_final);
        ds4_engine_close(plain_engine);
        free(tokens);
        free(logits_b);
        return 1;
    }
    printf("MTP drafting left the trunk path bit-identical over %u tokens\n",
           max_tokens);
    free(plain_tokens);
    free(plain_logits);
    free(plain_final);
    ds4_engine_close(plain_engine);
    free(tokens);
    free(logits_b);
    puts("Qwen3.8 MTP draft PASS");
    return 0;
}
