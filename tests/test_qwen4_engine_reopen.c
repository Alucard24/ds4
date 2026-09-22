/* CUDA engine-lifecycle regression for Qwen Flash-Next.
 *
 * Run with the real Qwen4/Flash-Next GGUF.  Each close resets the CUDA
 * context, so the next open must rebuild MMQ state and context-local dynamic
 * shared-memory attributes before reproducing the baseline decode result.
 */
#define _POSIX_C_SOURCE 200809L
#include "ds4.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum {
    k_context = 512,
    k_reset_reopens = 3,
    k_topk = 8,
};

static const char k_prompt[] =
    "Explain why a CUDA engine must release every context-owned cache before "
    "cudaDeviceReset. Mention matrix-multiplication workspaces, dynamic "
    "shared-memory function attributes, and safe lazy reinitialization. "
    "Use precise technical language. Then restate the same requirement with "
    "an example involving a quantized mixture-of-experts model, a long prompt, "
    "and a decode token after the engine is reopened in the same process. "
    "Correctness requires the recreated engine to produce a finite, stable "
    "next-token distribution without retaining pointers or initialization "
    "flags from the destroyed CUDA context.";

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static void force_lifecycle_paths(void) {
    /* Establish the paths whose CUDA state is context-local before the first
     * engine opens. The test process owns its environment, so these settings
     * cannot affect a server or a subsequent shell command. Clear diagnostic
     * variants so an inherited benchmark setting cannot bypass the tiled path. */
    static const char *const inherited_overrides[] = {
        "DS4_CUDA_NO_EXACT_SCORE_SPLIT_DECODE",
        "DS4_CUDA_NO_SCORE_TILE",
        "DS4_CUDA_DECODE_SCORE4",
        "DS4_CUDA_DECODE_SCORE8",
        "DS4_CUDA_EXACT_SCORE_SPLIT_GRAPH",
        "DS4_CUDA_EXACT_SCORE_SPLIT_LDG",
        "DS4_CUDA_EXACT_SCORE_SPLIT_VEC4",
        "DS4_CUDA_EXACT_SCORE_SPLIT_VEC4_PLAIN",
        "DS4_CUDA_EXACT_SCORE_SPLIT_DIM2",
        "DS4_CUDA_NO_EXACT_SCORE_SPLIT_DIM2",
        "DS4_CUDA_EXACT_SCORE_SPLIT_FUSE_INV_ROPE",
        "DS4_CUDA_EXACT_SCORE_SPLIT_MIN_SCORE",
        "DS4_CUDA_EXACT_SCORE_SPLIT_CHUNK",
        "DS4_CUDA_EXACT_SCORE_SPLIT_S_FLOOR",
        "DS4_CUDA_EXACT_SCORE_SPLIT_S_MAX",
        "DS4_CUDA_EXACT_SCORE_SPLIT_S",
    };
    const size_t override_count =
        sizeof(inherited_overrides) / sizeof(inherited_overrides[0]);
    for (size_t i = 0; i < override_count; i++) {
        require(unsetenv(inherited_overrides[i]) == 0,
                "clear inherited CUDA score-split override");
    }
    require(setenv("DS4_CUDA_MMQ", "1", 1) == 0, "enable MMQ");
    require(setenv("DS4_CUDA_EXACT_SCORE_SPLIT_DECODE", "1", 1) == 0,
            "enable exact score split decode");
}

static void require_finite_scores(const ds4_token_score *scores, int n) {
    for (int i = 0; i < n; i++) {
        require(scores[i].id >= 0, "invalid top-logprob token");
        require(isfinite(scores[i].logit), "non-finite top-logprob logit");
        require(isfinite(scores[i].logprob), "non-finite top-logprob logprob");
    }
}

int main(int argc, char **argv) {
    require(argc == 2 || argc == 3,
            "usage: test_qwen4_engine_reopen MODEL [MTP_SIDECAR]");
    force_lifecycle_paths();

    ds4_tokens prompt = {0};
    int baseline_probe = -1;
    int baseline_top1 = -1;
    float baseline_top1_logit = 0.0f;

    /* Baseline plus three in-process reopen cycles.  Keeping the token vector
     * makes every evaluation use byte-for-byte identical rendered input while
     * every engine/session/GPU context is newly constructed. */
    for (int cycle = 0; cycle <= k_reset_reopens; cycle++) {
        const ds4_engine_options opt = {
            .model_path = argv[1],
            .mtp_path = argc == 3 ? argv[2] : NULL,
            .backend = DS4_BACKEND_CUDA,
            /* A supplied sidecar must exercise its actual MTP decode path,
             * not merely load and validate its tensors. */
            .glm_mtp = argc == 3,
            .n_threads = 16,
            .context_size = k_context,
            .power_percent = 100,
        };
        ds4_engine *engine = NULL;
        ds4_session *session = NULL;
        char error[256] = {0};

        require(ds4_engine_open(&engine, &opt) == 0, "Qwen4 CUDA engine open");
        require(ds4_engine_is_qwen4(engine), "requires Qwen4/Flash-Next model");
        if (argc == 3) {
            require(ds4_engine_mtp_draft_tokens(engine) > 0,
                    "Qwen4 MTP sidecar must enable GPU draft decoding");
        }

        if (cycle == 0) {
            ds4_encode_chat_prompt(engine, NULL, k_prompt, DS4_THINK_NONE, &prompt);
            require(prompt.len >= 64 && prompt.len < k_context - 2,
                    "lifecycle prompt must exercise decode attention");
        }

        require(ds4_session_create(&session, engine, k_context) == 0,
                "Qwen4 CUDA session create");
        require(ds4_session_sync(session, &prompt, error, sizeof(error)) == 0,
                "Qwen4 CUDA prompt prefill");

        const int current_probe = ds4_session_argmax(session);
        require(current_probe >= 0, "Qwen4 CUDA prompt argmax");
        if (cycle == 0) {
            baseline_probe = current_probe;
        } else {
            require(current_probe == baseline_probe,
                    "reopened prompt changed decode probe argmax");
        }
        if (argc == 3) {
            int accepted[2] = {-1, -1};
            const int n_accepted = ds4_session_eval_speculative_argmax(
                session, baseline_probe, 2, -1, accepted, 2, error, sizeof(error));
            require(n_accepted >= 1 && n_accepted <= 2,
                    "Qwen4 CUDA MTP decode after reopen");
            require(accepted[0] == baseline_probe,
                    "Qwen4 CUDA MTP must commit the probe token");
        } else {
            require(ds4_session_eval(session, baseline_probe, error, sizeof(error)) == 0,
                    "Qwen4 CUDA decode after reopen");
        }

        ds4_token_score scores[k_topk];
        require(ds4_session_top_logprobs(session, scores, k_topk) == k_topk,
                "Qwen4 CUDA top logprobs after reopen");
        require_finite_scores(scores, k_topk);

        if (cycle == 0) {
            baseline_top1 = scores[0].id;
            baseline_top1_logit = scores[0].logit;
            printf("RESET_REOPEN_BASELINE prompt_tokens=%d probe=%d top1=%d\n",
                   prompt.len, baseline_probe, baseline_top1);
        } else {
            const float delta = fabsf(scores[0].logit - baseline_top1_logit);
            require(scores[0].id == baseline_top1,
                    "reopened decode changed final argmax");
            /* Different prefill/MMQ reductions may move logits slightly.  The
             * invariant is finite output and the exact greedy result; print
             * the delta for diagnosis without turning normal numeric drift into
             * a lifecycle-test flake. */
            printf("RESET_REOPEN_CYCLE %d prompt_tokens=%d probe=%d top1=%d "
                   "top1_logit_delta=%g PASS\n",
                   cycle, prompt.len, baseline_probe, scores[0].id, delta);
        }

        ds4_session_free(session);
        ds4_engine_close(engine);
    }

    ds4_tokens_free(&prompt);
    printf("Qwen4 CUDA engine reset/reopen PASS\n");
    return 0;
}
