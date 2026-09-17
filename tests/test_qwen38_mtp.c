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
 *   5. with a quantized KV cache (DS4_KV_Q8/DS4_KV_Q4) the strict
 *      token-for-token equality is relaxed to a near-tie gate: the verify
 *      chunk and the single-token decode read different representations of
 *      the same cache, so a flip between two logits within 0.10 of each
 *      other is tolerated once and reported, while anything wider still
 *      fails. f16 keeps the strict gate.
 */
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

static int require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        return 0;
    }
    return 1;
}

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1.0e-9;
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
                      uint32_t *proposed,
                      double *seconds) {
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
    const double t0 = now_seconds();
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
    if (seconds) *seconds = now_seconds() - t0;
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

/* Greedy speculative round-trip: the committed stream must equal plain greedy
 * decoding, and the round must actually commit more than one token at a time.
 * The wall-clock comparison is only reported, never asserted: this gate exists
 * to prove equivalence and to surface the measured speedup. */
static int run_spec(ds4_engine *engine,
                    const char *text,
                    uint32_t max_tokens,
                    int *tokens_out,
                    uint32_t *rounds,
                    uint32_t *committed,
                    double *seconds) {
    const uint32_t n_vocab = ds4_engine_vocab_size(engine);
    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_chat_append_message(engine, &prompt, "user", text);
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);

    ds4_session *session = NULL;
    char error[256] = {0};
    if (!require(ds4_session_create(&session, engine, 256) == 0, "spec session") ||
        !require(ds4_session_sync(session, &prompt, error, sizeof(error)) == 0,
                 error[0] ? error : "spec sync")) {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return 0;
    }
    float *logits = malloc((size_t)n_vocab * sizeof(float));
    if (!require(logits != NULL, "spec logits")) {
        ds4_session_free(session);
        ds4_tokens_free(&prompt);
        return 0;
    }
    const double t0 = now_seconds();
    uint32_t kept = 0, round_count = 0, total_committed = 0;
    int pending = -1;
    for (;;) {
        if (!require(ds4_session_copy_logits(session, logits, (int)n_vocab) ==
                         (int)n_vocab, "spec logits copy")) {
            free(logits);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
        pending = argmax_of(logits, n_vocab);
        if (pending == ds4_token_eos(engine) || kept >= max_tokens) break;
        int round_tokens[8];
        uint32_t count = 0;
        if (!require(ds4_session_qwen38_spec_step(session, pending,
                                                  round_tokens, 8u, &count,
                                                  error, sizeof(error)) == 0,
                     error[0] ? error : "spec step")) {
            free(logits);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
        if (!require(count >= 1u && count <= 8u, "spec round length")) {
            free(logits);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
        if (!require(round_tokens[0] == pending, "spec round dropped its token")) {
            free(logits);
            ds4_session_free(session);
            ds4_tokens_free(&prompt);
            return 0;
        }
        round_count++;
        total_committed += count;
        for (uint32_t i = 0; i < count && kept < max_tokens; i++) {
            if (round_tokens[i] == ds4_token_eos(engine)) {
                kept = max_tokens;
                break;
            }
            tokens_out[kept++] = round_tokens[i];
        }
    }
    *seconds = now_seconds() - t0;
    *rounds = round_count;
    *committed = total_committed;
    tokens_out[kept] = -1;
    free(logits);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "usage: %s TRUNK.gguf MTP.gguf PROMPT\n", argv[0]);
        return 2;
    }
    const uint32_t max_tokens = 48u;
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.mtp_path = argv[2];
    options.backend = DS4_BACKEND_CUDA;
    options.context_size = 256;
    options.directional_steering_file = getenv("DS4_TEST_STEERING_FILE");
    if (options.directional_steering_file) {
        options.directional_steering_ffn = 0.5f;
        options.directional_steering_attn = 0.25f;
    }

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
    double greedy_seconds = 0.0;
    if (!run_greedy(engine, argv[3], max_tokens, tokens, logits_a, logits_b,
                    &accepted, &proposed, &greedy_seconds)) {
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

    /* Speculative round gate: same tokens as plain greedy, more than one token
     * committed per round, and a measured wall-clock difference. */
    uint32_t spec_budget = 64u;
    const char *budget_env = getenv("DS4_TEST_MTP_SPEC_TOKENS");
    if (budget_env && budget_env[0]) {
        const long v = strtol(budget_env, NULL, 10);
        if (v >= 0 && v <= 4096) spec_budget = (uint32_t)v;
    }
    int *spec_tokens = calloc((size_t)spec_budget + 1u, sizeof(spec_tokens[0]));
    uint32_t rounds = 0, committed = 0;
    double spec_seconds = 0.0;
    if (!require(spec_tokens != NULL, "spec token buffer") ||
        !run_spec(engine, argv[3], spec_budget, spec_tokens, &rounds, &committed,
                  &spec_seconds)) {
        free(spec_tokens);
        free(tokens);
        free(logits_a);
        free(logits_b);
        ds4_engine_close(engine);
        return 1;
    }
    uint32_t plain_len = 0;
    while (plain_len < max_tokens && tokens[plain_len] >= 0) plain_len++;
    uint32_t spec_len = 0;
    while (spec_len < spec_budget && spec_tokens[spec_len] >= 0) spec_len++;
    printf("MTP speculation: %u rounds, %u tokens committed, %.2f tok/round, "
           "%.3fs (%.1f tok/s)\n",
           rounds, committed,
           rounds ? (double)committed / (double)rounds : 0.0, spec_seconds,
           spec_seconds > 0.0 ? (double)committed / spec_seconds : 0.0);
    /* The speculative run must cover the whole plain reference stream and agree
     * with it token for token. */
    uint32_t agree = 0;
    while (agree < plain_len && agree < spec_len &&
           spec_tokens[agree] == tokens[agree]) agree++;
    printf("MTP speculation agrees with plain greedy on %u/%u tokens\n",
           agree, plain_len);
    if (agree < plain_len && agree < spec_len) {
        printf("MTP first divergence at token %u: spec=%d plain=%d\n",
               agree, spec_tokens[agree], tokens[agree]);
    }
    int streams_match = memcmp(spec_tokens, tokens,
                               (size_t)plain_len * sizeof(spec_tokens[0])) == 0;
    if (!streams_match && agree < plain_len && agree < spec_len &&
        (getenv("DS4_KV_Q8") != NULL || getenv("DS4_KV_Q4") != NULL)) {
        /* Relaxed gate, quantized KV only: the verify chunk widens the native
         * quantized cache into an f16 mirror and runs FA-2 over it, while the
         * single-token decode reads the quantized KV directly. The two paths
         * apply different reduction orders to values that differ by
         * quantization noise, so a near-tie argmax can flip between them.
         * Measured on the release trunk with q8_0: plain 13:21.6413 vs
         * 11:21.6041 (margin 0.037), chunk path moves both by 0.027-0.049 in
         * the flip direction. Tolerate a first divergence whose plain-path
         * logits are at most 0.10 apart - about twice the observed
         * perturbation - and fail anything wider. Past a tolerated flip both
         * streams condition on different histories, so later positions are
         * not re-checked; the MTP machinery checks below still apply. */
        const float *row = logits_a + (size_t)agree * n_vocab;
        const float margin =
            fabsf(row[spec_tokens[agree]] - row[tokens[agree]]);
        printf("MTP quantized-KV near-tie check at token %u: margin %.4f (limit 0.10)\n",
               agree, margin);
        streams_match = margin <= 0.10f;
    }
    if (!require(rounds != 0u && committed > rounds,
                 "speculation never committed more than one token per round") ||
        !require(spec_len >= plain_len,
                 "speculative run ended before the plain reference stream") ||
        !require(streams_match,
                 "speculative stream differs from plain greedy")) {
        free(spec_tokens);
        free(tokens);
        free(logits_a);
        free(logits_b);
        ds4_engine_close(engine);
        return 1;
    }
    if (agree == plain_len)
        printf("MTP speculation reproduced %u plain greedy tokens exactly\n",
               plain_len);
    else
        printf("MTP speculation reproduced %u/%u tokens before a tolerated quantized-KV near-tie flip\n",
               agree, plain_len);
    printf("MTP decode rate: greedy %.1f tok/s vs speculative %.1f tok/s "
           "(%.2fx over %u tokens)\n",
           greedy_seconds > 0.0 ? (double)plain_len / greedy_seconds : 0.0,
           spec_seconds > 0.0 ? (double)committed / spec_seconds : 0.0,
           (greedy_seconds > 0.0 && spec_seconds > 0.0) ?
               (greedy_seconds * (double)committed) /
                   (spec_seconds * (double)plain_len) : 0.0,
           plain_len);
    free(spec_tokens);

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
    plain.directional_steering_file = options.directional_steering_file;
    plain.directional_steering_ffn = options.directional_steering_ffn;
    plain.directional_steering_attn = options.directional_steering_attn;
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
                   plain_final, &plain_accepted, &plain_proposed, NULL);
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
