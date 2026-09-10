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

static void check_resized_pattern(void) {
    ds4_image image = {0};
    image.width = 300u;
    image.height = 200u;
    image.rgb = malloc((size_t)image.width * image.height * 3u);
    require(image.rgb != NULL, "pattern image allocation");
    for (uint32_t y = 0; y < image.height; y++) {
        for (uint32_t x = 0; x < image.width; x++) {
            uint8_t *pixel = image.rgb + ((size_t)y * image.width + x) * 3u;
            pixel[0] = (uint8_t)(x * 17u + y * 31u + (x * y) % 251u);
            pixel[1] = (uint8_t)(x * 7u + y * 13u);
            pixel[2] = (uint8_t)(x * 3u + y * 29u + (x ^ y));
        }
    }
    ds4_image_patches patches = {0};
    char error[160] = {0};
    require(ds4_image_preprocess_qwen3vl(
                &patches, &image, 8u, 4096u, error, sizeof(error)),
            error[0] ? error : "Qwen3-VL pattern preprocessing");
    require(patches.content_width == 288u &&
            patches.content_height == 192u &&
            patches.grid_width == 18u && patches.grid_height == 12u &&
            patches.image_token_count == 54u,
            "unexpected resized-pattern dimensions");
    uint64_t hash = UINT64_C(14695981039346656037);
    const size_t count = (size_t)patches.patch_count * 768u;
    for (size_t i = 0; i < count; i++) {
        int pixel = (int)lrintf((patches.patches[i] + 1.0f) * 127.5f);
        require(pixel >= 0 && pixel <= 255,
                "resized-pattern sample is outside byte range");
        hash ^= (uint8_t)pixel;
        hash *= UINT64_C(1099511628211);
    }
    require(hash == UINT64_C(0x6333a6756db1a4d2),
            "Pillow-compatible Qwen3-VL resize changed");
    ds4_image_patches_free(&patches);
    ds4_image_free(&image);
}

static void check_white(uint32_t width, uint32_t height,
                        uint32_t want_width, uint32_t want_height,
                        uint32_t want_tokens,
                        uint32_t want_padding_rows) {
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
    size_t black = 0, white = 0;
    for (size_t i = 0; i < count; i++) {
        require(isfinite(patches.patches[i]),
                "nonfinite Qwen3-VL patch value");
        if (fabsf(patches.patches[i] + 1.0f) < 1.0e-6f) black++;
        else if (fabsf(patches.patches[i] - 1.0f) < 1.0e-6f) white++;
        else require(0, "white image normalization or patch order failed");
    }
    const size_t want_black =
        (size_t)want_width * want_padding_rows * 3u;
    require(black == want_black && white == count - want_black,
            "unexpected Qwen3-VL aspect padding");
    if (want_padding_rows) {
        const uint32_t top_padding = want_padding_rows / 2u;
        const size_t last_patch = (size_t)(patches.patch_count - 1u) * 768u;
        require(fabsf(patches.patches[0] + 1.0f) < 1.0e-6f &&
                fabsf(patches.patches[(size_t)top_padding * 16u] - 1.0f) <
                    1.0e-6f &&
                fabsf(patches.patches[last_patch + 15u * 16u] + 1.0f) <
                    1.0e-6f,
                "Qwen3-VL padding is not centered in patch order");
    }
    ds4_image_patches_free(&patches);
    ds4_image_free(&image);
}

int main(void) {
    check_white(224u, 224u, 224u, 224u, 49u, 0u);
    check_white(512u, 507u, 512u, 512u, 256u, 5u);
    check_white(1u, 1u, 96u, 96u, 9u, 0u);
    check_resized_pattern();
    puts("Qwen3-VL image preprocessing PASS");
    return 0;
}
