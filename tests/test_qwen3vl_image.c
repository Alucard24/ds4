#include "ds4_image.h"

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

static void check_white(uint32_t width, uint32_t height,
                        uint32_t want_width, uint32_t want_height,
                        uint32_t want_tokens) {
    ds4_image image = {0};
    image.width = width;
    image.height = height;
    image.rgb = malloc((size_t)width * height * 3u);
    require(image.rgb != NULL, "image allocation");
    memset(image.rgb, 255, (size_t)width * height * 3u);
    ds4_image_patches patches = {0};
    char error[160] = {0};
    require(ds4_image_preprocess_qwen3vl(
                &patches, &image, 8u, 4096u, error, sizeof(error)),
            error[0] ? error : "Qwen3-VL preprocessing");
    require(patches.content_width == want_width &&
            patches.content_height == want_height,
            "unexpected Qwen3-VL smart-resize dimensions");
    require(patches.grid_width == want_width / 16u &&
            patches.grid_height == want_height / 16u,
            "unexpected patch grid");
    require(patches.patch_count == patches.grid_width * patches.grid_height &&
            patches.image_token_count == want_tokens,
            "unexpected merged image token count");
    const size_t count = (size_t)patches.patch_count * 768u;
    for (size_t i = 0; i < count; i++) {
        require(isfinite(patches.patches[i]) &&
                fabsf(patches.patches[i] - 1.0f) < 1.0e-6f,
                "white image normalization or patch order failed");
    }
    ds4_image_patches_free(&patches);
    ds4_image_free(&image);
}

int main(void) {
    check_white(224u, 224u, 224u, 224u, 49u);
    check_white(512u, 507u, 512u, 512u, 256u);
    check_white(1u, 1u, 96u, 96u, 9u);
    puts("Qwen3-VL image preprocessing PASS");
    return 0;
}
