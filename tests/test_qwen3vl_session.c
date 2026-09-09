#include "ds4.h"

#include <ctype.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void require(int ok, const char *message) {
    if (!ok) {
        fprintf(stderr, "FAIL: %s\n", message);
        exit(1);
    }
}

static int contains_white(const char *s) {
    const char needle[] = "white";
    for (; *s; s++) {
        size_t i = 0;
        while (needle[i] && s[i] &&
               tolower((unsigned char)s[i]) == needle[i]) i++;
        if (!needle[i]) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    require(argc == 4,
            "usage: test_qwen3vl_session MAIN.gguf MMPROJ.gguf WHITE_IMAGE");
    ds4_engine_options options = {0};
    options.model_path = argv[1];
    options.vision_path = argv[2];
    options.backend = DS4_BACKEND_CUDA;
    options.context_size = 192;
    options.power_percent = 100;
    ds4_engine *engine = NULL;
    require(ds4_engine_open(&engine, &options) == 0, "engine open");
    require(ds4_engine_has_vision(engine), "vision sidecar not available");

    char error[256] = {0};
    ds4_vision_embedding image = {0};
    require(ds4_engine_vision_encode_file(engine, argv[3], &image,
                                          error, sizeof(error)),
            error[0] ? error : "image encode");
    require(image.token_count == 49u && image.grid_width == 7u &&
            image.grid_height == 7u, "expected a 224x224 white fixture");

    ds4_tokens prompt = {0};
    ds4_chat_begin(engine, &prompt);
    ds4_vision_span span = {0};
    const char *parts[2] = {"", ""};
    require(ds4_chat_append_multimodal_message(
                engine, &prompt, "user", parts, &image, 1u, &span,
                error, sizeof(error)),
            error[0] ? error : "multimodal prompt");
    ds4_chat_append_assistant_prefix(engine, &prompt, DS4_THINK_NONE);
    require(span.embedding.data != NULL && image.data == NULL,
            "vision embedding ownership did not move to prompt span");
    require(prompt.len > (int)span.embedding.token_count && prompt.len < 191,
            "invalid multimodal token count");

    ds4_session *session = NULL;
    require(ds4_session_create(&session, engine, 192) == 0, "session create");
    require(ds4_session_sync_multimodal(session, &prompt, &span, 1u,
                                        error, sizeof(error)) == 0,
            error[0] ? error : "multimodal sync");
    const int n_vocab = 248320;
    float *before = malloc((size_t)n_vocab * sizeof(float));
    float *after = malloc((size_t)n_vocab * sizeof(float));
    require(before && after, "logit buffers");
    require(ds4_session_copy_logits(session, before, n_vocab) == n_vocab,
            "initial logits");
    require(ds4_session_sync_multimodal(session, &prompt, &span, 1u,
                                        error, sizeof(error)) == 0,
            error[0] ? error : "no-op image sync");
    require(ds4_session_copy_logits(session, after, n_vocab) == n_vocab,
            "no-op logits");
    require(memcmp(before, after, (size_t)n_vocab * sizeof(float)) == 0,
            "no-op image sync changed logits");

    char generated[4096] = {0};
    size_t generated_len = 0;
    uint64_t rng = 1;
    for (uint32_t i = 0; i < 24u; i++) {
        const int token = ds4_session_sample(session, 0.0f, 1, 1.0f, 0.0f, &rng);
        if (token == ds4_token_eos(engine)) break;
        size_t piece_len = 0;
        char *piece = ds4_token_text(engine, token, &piece_len);
        require(piece != NULL, "token decode");
        if (piece_len > sizeof(generated) - 1u - generated_len)
            piece_len = sizeof(generated) - 1u - generated_len;
        memcpy(generated + generated_len, piece, piece_len);
        generated_len += piece_len;
        generated[generated_len] = '\0';
        free(piece);
        require(ds4_session_eval(session, token, error, sizeof(error)) == 0,
                error[0] ? error : "vision continuation decode");
    }
    printf("GREEDY %s\n", generated);
    require(contains_white(generated) || strstr(generated, "\xe7\x99\xbd") != NULL,
            "greedy white-image smoke did not identify the image color");

    free(after);
    free(before);
    ds4_session_free(session);
    ds4_tokens_free(&prompt);

    ds4_vision_embedding multi_images[2] = {span.embedding, span.embedding};
    memset(&span.embedding, 0, sizeof(span.embedding));
    const size_t image_floats = (size_t)multi_images[0].token_count *
                                (size_t)ds4_engine_embd_dim(engine);
    multi_images[1].data = malloc(image_floats * sizeof(float));
    require(multi_images[1].data != NULL, "second-image embedding clone");
    memcpy(multi_images[1].data, multi_images[0].data,
           image_floats * sizeof(float));
    ds4_vision_span multi_spans[2] = {0};
    const char *multi_parts[3] = {"", " and ", ""};
    ds4_tokens multi_prompt = {0};
    ds4_chat_begin(engine, &multi_prompt);
    require(ds4_chat_append_multimodal_message(
                engine, &multi_prompt, "user", multi_parts, multi_images, 2u,
                multi_spans, error, sizeof(error)),
            error[0] ? error : "two-image prompt");
    ds4_chat_append_assistant_prefix(engine, &multi_prompt, DS4_THINK_NONE);
    require(multi_spans[0].token_start < multi_spans[1].token_start &&
            multi_spans[0].embedding.token_count == 49u &&
            multi_spans[1].embedding.token_count == 49u,
            "two-image span layout");
    require(ds4_session_create(&session, engine, 192) == 0,
            "two-image session create");
    require(ds4_session_sync_multimodal(session, &multi_prompt, multi_spans, 2u,
                                        error, sizeof(error)) == 0,
            error[0] ? error : "two-image sync");
    require(ds4_session_sync_multimodal(session, &multi_prompt, multi_spans, 2u,
                                        error, sizeof(error)) == 0,
            error[0] ? error : "two-image no-op sync");

    ds4_session_free(session);
    ds4_tokens_free(&multi_prompt);
    ds4_vision_embedding_free(&multi_spans[0].embedding);
    ds4_vision_embedding_free(&multi_spans[1].embedding);
    ds4_engine_close(engine);
    puts("Qwen3-VL CUDA session PASS");
    return 0;
}
