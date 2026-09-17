/* Qwen3-VL vision encoder for the Qwen3.8 CUDA path.
 *
 * The graph mirrors llama.cpp's MIT-licensed qwen3vl mtmd graph, while using
 * ds4's own tensor and BF16 GEMM primitives. Activations remain FP32 and the
 * sidecar's BF16 weights stay in the auxiliary CUDA model mapping.
 */

#ifndef DS4_QWEN3VL_VISION_STREAM
#define DS4_QWEN3VL_VISION_STREAM 0
#endif

#ifndef DS4_QWEN3VL_VISION_TYPES_DEFINED
#define DS4_QWEN3VL_VISION_TYPES_DEFINED
#define DS4_QWEN3VL_VISION_LAYERS 27u
typedef struct {
    uint64_t norm1_weight, norm1_bias;
    uint64_t qkv_weight, qkv_bias;
    uint64_t attn_out_weight, attn_out_bias;
    uint64_t norm2_weight, norm2_bias;
    uint64_t ffn_up_weight, ffn_up_bias;
    uint64_t ffn_down_weight, ffn_down_bias;
} ds4_qwen3vl_vision_layer_weights;
typedef struct {
    uint64_t patch_weight_0, patch_weight_1, patch_bias;
    uint64_t position_embedding;
    uint64_t post_norm_weight, post_norm_bias;
    uint64_t merger_up_weight, merger_up_bias;
    uint64_t merger_down_weight, merger_down_bias;
    ds4_qwen3vl_vision_layer_weights layer[DS4_QWEN3VL_VISION_LAYERS];
} ds4_qwen3vl_vision_weights;
#endif

enum {
    QWEN3VL_PATCH_DIM = 768,
    QWEN3VL_WIDTH = 1152,
    QWEN3VL_QKV = 3456,
    QWEN3VL_FFN = 4304,
    QWEN3VL_HEADS = 16,
    QWEN3VL_HEAD_DIM = 72,
    QWEN3VL_MERGED = 4608,
    QWEN3VL_OUTPUT = 5120,
};

__device__ __forceinline__ static float qwen3vl_f32(const float *p) {
    return *p;
}

__global__ static void qwen3vl_patch_pos_kernel(
        float          *out,
        const float    *patch_1,
        const float    *bias,
        const float    *position,
        uint32_t        rows,
        uint32_t        grid_h,
        uint32_t        grid_w) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t count = (uint64_t)rows * QWEN3VL_WIDTH;
    if (i >= count) return;
    const uint32_t row = (uint32_t)(i / QWEN3VL_WIDTH);
    const uint32_t d = (uint32_t)(i % QWEN3VL_WIDTH);
    const uint32_t merge_w = grid_w / 2u;
    const uint32_t group = row / 4u;
    const uint32_t within = row & 3u;
    const uint32_t y = (group / merge_w) * 2u + within / 2u;
    const uint32_t x = (group % merge_w) * 2u + within % 2u;

    const float scale_y = grid_h > 1u ? (float)(grid_h - 1u) / 47.0f : 1.0f;
    const float scale_x = grid_w > 1u ? (float)(grid_w - 1u) / 47.0f : 1.0f;
    const float fy = grid_h > 1u ? (float)y / scale_y : 0.0f;
    const float fx = grid_w > 1u ? (float)x / scale_x : 0.0f;
    const uint32_t y0 = (uint32_t)floorf(fy);
    const uint32_t x0 = (uint32_t)floorf(fx);
    const uint32_t y1 = y0 < 47u ? y0 + 1u : y0;
    const uint32_t x1 = x0 < 47u ? x0 + 1u : x0;
    const float wy = fy - (float)y0;
    const float wx = fx - (float)x0;
    const float p00 = qwen3vl_f32(position + ((uint64_t)y0 * 48u + x0) * QWEN3VL_WIDTH + d);
    const float p01 = qwen3vl_f32(position + ((uint64_t)y0 * 48u + x1) * QWEN3VL_WIDTH + d);
    const float p10 = qwen3vl_f32(position + ((uint64_t)y1 * 48u + x0) * QWEN3VL_WIDTH + d);
    const float p11 = qwen3vl_f32(position + ((uint64_t)y1 * 48u + x1) * QWEN3VL_WIDTH + d);
    const float pos = p00 * (1.0f - wx) * (1.0f - wy) +
                      p01 * wx * (1.0f - wy) +
                      p10 * (1.0f - wx) * wy +
                      p11 * wx * wy;
    float value = __fadd_rn(out[i], patch_1[i]);
    value = __fadd_rn(value, qwen3vl_f32(bias + d));
    out[i] = __fadd_rn(value, pos);
}

__device__ __forceinline__ static float2 qwen3vl_warp_sum(float2 v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v.x += __shfl_xor_sync(0xffffffffu, v.x, offset, 32);
        v.y += __shfl_xor_sync(0xffffffffu, v.y, offset, 32);
    }
    return v;
}

__global__ static void qwen3vl_layernorm_kernel(
        float          *out,
        const float    *x,
        const float    *weight,
        const float    *bias,
        uint32_t        rows,
        uint32_t        width,
        float           eps) {
    __shared__ float2 warp_sums[32];
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    if (row >= rows) return;
    const float *xr = x + (uint64_t)row * width;
    float *yr = out + (uint64_t)row * width;
    float2 mean_var = make_float2(0.0f, 0.0f);
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        const float v = xr[d];
        mean_var.x += v;
        mean_var.y += v * v;
    }
    mean_var = qwen3vl_warp_sum(mean_var);
    const uint32_t lane = tid & 31u;
    if (lane == 0u) warp_sums[tid >> 5u] = mean_var;
    __syncthreads();
    mean_var = lane < 32u ? warp_sums[lane] : make_float2(0.0f, 0.0f);
    mean_var = qwen3vl_warp_sum(mean_var);
    const float mean = mean_var.x / (float)width;
    const float variance = mean_var.y / (float)width - mean * mean;
    const float inv = ds4_cuda_rsqrtf(variance + eps);
    for (uint32_t d = tid; d < width; d += blockDim.x) {
        const float centered = __fsub_rn(xr[d], mean);
        const float normed = __fmul_rn(centered, inv);
        const float weighted = __fmul_rn(normed, qwen3vl_f32(weight + d));
        yr[d] = __fadd_rn(weighted, qwen3vl_f32(bias + d));
    }
}

__global__ static void qwen3vl_bias_rope_kernel(
        float          *qkv,
        const float    *bias,
        uint32_t        rows,
        uint32_t        grid_w,
        float           theta_scale) {
    const uint32_t row = blockIdx.x;
    const uint32_t head = blockIdx.y;
    const uint32_t lane = threadIdx.x;
    if (row >= rows || head >= QWEN3VL_HEADS || lane >= QWEN3VL_HEAD_DIM) return;
    const uint64_t base = (uint64_t)row * QWEN3VL_QKV +
                          (uint64_t)head * QWEN3VL_HEAD_DIM;
    qkv[base + lane] += qwen3vl_f32(
            bias + (uint64_t)head * QWEN3VL_HEAD_DIM + lane);
    qkv[base + QWEN3VL_WIDTH + lane] += qwen3vl_f32(
            bias + QWEN3VL_WIDTH +
            (uint64_t)head * QWEN3VL_HEAD_DIM + lane);
    qkv[base + 2u * QWEN3VL_WIDTH + lane] += qwen3vl_f32(
            bias + 2u * QWEN3VL_WIDTH +
            (uint64_t)head * QWEN3VL_HEAD_DIM + lane);
    __syncthreads();
    if (lane >= 36u) return;

    const uint32_t merge_w = grid_w / 2u;
    const uint32_t group = row / 4u;
    const uint32_t within = row & 3u;
    const uint32_t py = (group / merge_w) * 2u + within / 2u;
    const uint32_t px = (group % merge_w) * 2u + within % 2u;
    const uint32_t freq = lane < 18u ? lane : lane - 18u;
    const uint32_t pos = lane < 18u ? py : px;
    const float angle = (float)pos * powf(theta_scale, (float)freq);
    const float cs = cosf(angle);
    const float sn = sinf(angle);

    float q0 = qkv[base + lane];
    float q1 = qkv[base + lane + 36u];
    float k0 = qkv[base + QWEN3VL_WIDTH + lane];
    float k1 = qkv[base + QWEN3VL_WIDTH + lane + 36u];
    qkv[base + lane] = q0 * cs - q1 * sn;
    qkv[base + lane + 36u] = q0 * sn + q1 * cs;
    qkv[base + QWEN3VL_WIDTH + lane] = k0 * cs - k1 * sn;
    qkv[base + QWEN3VL_WIDTH + lane + 36u] = k0 * sn + k1 * cs;
}

#define QWEN3VL_ATTN_QUERY_WARPS 16u
#define QWEN3VL_ATTN_KEY_TILE 32u

/* Sixteen query warps share each K/V tile. Every warp retains the original
 * key order, FP32 online softmax and score-reduction order, so tiling reduces
 * global K/V traffic without an N x N workspace or a precision-policy change. */
__global__ static void qwen3vl_attention_kernel(
        float       *out,
        const float *qkv,
        uint32_t     rows) {
    __shared__ float kv[QWEN3VL_ATTN_KEY_TILE][2u * QWEN3VL_HEAD_DIM];
    const uint32_t tid = threadIdx.x;
    const uint32_t lane = tid & 31u;
    const uint32_t warp = tid >> 5u;
    const uint32_t row = blockIdx.x * QWEN3VL_ATTN_QUERY_WARPS + warp;
    const uint32_t head = blockIdx.y;
    const bool active = row < rows && head < QWEN3VL_HEADS;
    const uint64_t qbase = (uint64_t)row * QWEN3VL_QKV +
                           (uint64_t)head * QWEN3VL_HEAD_DIM;
    float q0 = 0.0f, q1 = 0.0f, q2 = 0.0f;
    if (active) {
        q0 = qkv[qbase + lane];
        q1 = qkv[qbase + lane + 32u];
        if (lane < 8u) q2 = qkv[qbase + lane + 64u];
    }
    float acc[3] = {0.0f, 0.0f, 0.0f};
    float max_score = -INFINITY;
    float denom = 0.0f;

    for (uint32_t first = 0; first < rows;
         first += QWEN3VL_ATTN_KEY_TILE) {
        const uint32_t tile_rows = rows - first < QWEN3VL_ATTN_KEY_TILE ?
            rows - first : QWEN3VL_ATTN_KEY_TILE;
        const uint32_t tile_values = tile_rows * 2u * QWEN3VL_HEAD_DIM;
        for (uint32_t i = tid; i < tile_values; i += blockDim.x) {
            const uint32_t key = i / (2u * QWEN3VL_HEAD_DIM);
            const uint32_t d = i % (2u * QWEN3VL_HEAD_DIM);
            const uint64_t base = (uint64_t)(first + key) * QWEN3VL_QKV +
                                  QWEN3VL_WIDTH +
                                  (uint64_t)head * QWEN3VL_HEAD_DIM;
            kv[key][d] = d < QWEN3VL_HEAD_DIM ?
                qkv[base + d] :
                qkv[base + QWEN3VL_WIDTH + d - QWEN3VL_HEAD_DIM];
        }
        __syncthreads();

        if (active) {
            for (uint32_t key = 0; key < tile_rows; key++) {
                float partial = __fmul_rn(q0, kv[key][lane]);
                if (lane < 8u) {
                    partial = __fadd_rn(partial,
                        __fmul_rn(q2, kv[key][lane + 64u]));
                }
                partial = __fadd_rn(partial,
                    __fmul_rn(q1, kv[key][lane + 32u]));
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    const float other = __shfl_down_sync(
                        0xffffffffu, partial, offset, 32);
                    if (lane < (uint32_t)offset)
                        partial = __fadd_rn(partial, other);
                }
                const float dot = __shfl_sync(0xffffffffu, partial, 0, 32);
                const float score = dot * 0.11785113019775793f;
                const float next_max = fmaxf(max_score, score);
                const float old_scale = (first + key) == 0u ? 0.0f :
                    expf(max_score - next_max);
                const float new_scale = expf(score - next_max);
                denom = denom * old_scale + new_scale;
#pragma unroll
                for (uint32_t slot = 0; slot < 3u; slot++) {
                    const uint32_t d = lane + slot * 32u;
                    if (d < QWEN3VL_HEAD_DIM) {
                        acc[slot] = acc[slot] * old_scale +
                            new_scale * kv[key][QWEN3VL_HEAD_DIM + d];
                    }
                }
                max_score = next_max;
            }
        }
        __syncthreads();
    }

    if (active) {
#pragma unroll
        for (uint32_t slot = 0; slot < 3u; slot++) {
            const uint32_t d = lane + slot * 32u;
            if (d < QWEN3VL_HEAD_DIM) {
                out[(uint64_t)row * QWEN3VL_WIDTH +
                    (uint64_t)head * QWEN3VL_HEAD_DIM + d] =
                        acc[slot] / denom;
            }
        }
    }
}

__global__ static void qwen3vl_bias_residual_kernel(
        float          *x,
        const float    *bias,
        const float    *residual,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    x[i] += qwen3vl_f32(bias + i % width) + residual[i];
}

__global__ static void qwen3vl_gelu_bias_kernel(
        float          *x,
        const float    *bias,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const float v = x[i] + qwen3vl_f32(bias + i % width);
    const float c = 0.7978845608028654f;
    x[i] = 0.5f * v * (1.0f + tanhf(c * v * (1.0f + 0.044715f * v * v)));
}

__global__ static void qwen3vl_bias_kernel(
        float          *x,
        const float    *bias,
        uint64_t        count,
        uint32_t        width) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) x[i] += qwen3vl_f32(bias + i % width);
}

__global__ static void qwen3vl_f16_to_f32_kernel(
        float *out, const __half *x, uint64_t count) {
    const uint64_t i = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) out[i] = __half2float(x[i]);
}

static const float *qwen3vl_weight(
        const void *model_map,
        uint64_t    model_size,
        uint64_t    offset,
        uint64_t    elements,
        const char *label) {
    if (!model_map || elements > UINT64_MAX / sizeof(float) ||
        offset > model_size) return NULL;
    const uint64_t bytes = elements * sizeof(float);
    if (bytes > model_size - offset) return NULL;
    return (const float *)cuda_resolve_weight_ptr(
            model_map, offset, bytes, 0, label);
}

static int qwen3vl_launch_ok(const char *label) {
    return cuda_ok(cudaGetLastError(), label);
}

/* ggml's Qwen3-VL patch convolution requests an F16 im2col output. Its CUDA
 * MUL_MAT therefore converts both the nominally-F32 patch weights and patches
 * to F16, accumulates to F16, then exposes the result as F32. Match that
 * boundary before adding the F32 bias and interpolated position embedding. */
static int qwen3vl_patch_matmul_f16(
        ds4_gpu_tensor *out,
        const void     *model_map,
        uint64_t        model_size,
        uint64_t        weight_offset,
        const ds4_gpu_tensor *x,
        uint32_t        rows) {
    const uint64_t weight_count =
        (uint64_t)QWEN3VL_PATCH_DIM * QWEN3VL_WIDTH;
    const uint64_t input_count = (uint64_t)rows * QWEN3VL_PATCH_DIM;
    const uint64_t output_count = (uint64_t)rows * QWEN3VL_WIDTH;
    if (!out || !x || !g_cublas_ready ||
        x->bytes < input_count * sizeof(float) ||
        out->bytes < output_count * sizeof(float) ||
        weight_offset > model_size ||
        weight_count * sizeof(float) > model_size - weight_offset) {
        return 0;
    }
    const float *weights = (const float *)cuda_resolve_weight_ptr(
        model_map, weight_offset, weight_count * sizeof(float), 0,
        "Qwen3-VL patch weight");
    const uint64_t weight_bytes = weight_count * sizeof(__half);
    const uint64_t input_offset = (weight_bytes + 255u) & ~255ull;
    const uint64_t input_bytes = input_count * sizeof(__half);
    const uint64_t output_offset =
        (input_offset + input_bytes + 255u) & ~255ull;
    const uint64_t output_bytes = output_count * sizeof(__half);
    if (input_offset < weight_bytes || output_offset < input_offset ||
        output_offset > UINT64_MAX - output_bytes) return 0;
    unsigned char *scratch = (unsigned char *)cuda_tmp_alloc_on(
        0, output_offset + output_bytes, "Qwen3-VL F16 patch matmul");
    if (!weights || !scratch) return 0;
    __half *weights_f16 = (__half *)scratch;
    __half *input_f16 = (__half *)(scratch + input_offset);
    __half *output_f16 = (__half *)(scratch + output_offset);
    f32_to_f16_kernel<<<
        (weight_count + 255u) / 256u, 256u, 0,
        DS4_QWEN3VL_VISION_STREAM>>>(weights_f16, weights, weight_count);
    f32_to_f16_kernel<<<
        (input_count + 255u) / 256u, 256u, 0,
        DS4_QWEN3VL_VISION_STREAM>>>(input_f16, (const float *)x->ptr,
                                    input_count);
    if (!qwen3vl_launch_ok("Qwen3-VL patch F16 conversion")) return 0;
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    cublasStatus_t status = cublasGemmEx(
        cuda_cublas_for_tier(0), CUBLAS_OP_T, CUBLAS_OP_N,
        QWEN3VL_WIDTH, (int)rows, QWEN3VL_PATCH_DIM,
        &alpha,
        weights_f16, CUDA_R_16F, QWEN3VL_PATCH_DIM,
        input_f16, CUDA_R_16F, QWEN3VL_PATCH_DIM,
        &beta,
        output_f16, CUDA_R_16F, QWEN3VL_WIDTH,
        CUBLAS_COMPUTE_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if (!cublas_ok(status, "Qwen3-VL F16 patch matmul")) return 0;
    qwen3vl_f16_to_f32_kernel<<<
        (output_count + 255u) / 256u, 256u, 0,
        DS4_QWEN3VL_VISION_STREAM>>>(
            (float *)out->ptr, output_f16, output_count);
    return qwen3vl_launch_ok("Qwen3-VL patch F16 output conversion");
}

extern "C" int ds4_gpu_qwen3vl_vision_encode_pair(
        float                            *out,
        const float                      *patches_0,
        const float                      *patches_1,
        uint32_t                          grid_h,
        uint32_t                          grid_w,
        const void                       *model_map,
        uint64_t                          model_size,
        const ds4_qwen3vl_vision_weights *weights) {
    if (!out || !patches_0 || !patches_1 || !model_map || !weights ||
        grid_h == 0u ||
        grid_w == 0u || (grid_h & 1u) != 0u || (grid_w & 1u) != 0u ||
        grid_h > UINT32_MAX / grid_w) return 0;
    const uint32_t rows = grid_h * grid_w;
    const uint32_t merged_rows = rows / 4u;
    const uint64_t row768 = (uint64_t)rows * QWEN3VL_PATCH_DIM;
    const uint64_t row1152 = (uint64_t)rows * QWEN3VL_WIDTH;
    const uint64_t row3456 = (uint64_t)rows * QWEN3VL_QKV;
    const uint64_t row4304 = (uint64_t)rows * QWEN3VL_FFN;
    const uint64_t merged4608 = (uint64_t)merged_rows * QWEN3VL_MERGED;
    const uint64_t merged5120 = (uint64_t)merged_rows * QWEN3VL_OUTPUT;
    if (row4304 > SIZE_MAX / sizeof(float) ||
        merged5120 > SIZE_MAX / sizeof(float)) return 0;

    ds4_gpu_tensor *patch_0 = NULL, *patch_1 = NULL;
    ds4_gpu_tensor *a = NULL, *b = NULL, *qkv = NULL;
    ds4_gpu_tensor *attn = NULL, *ffn = NULL, *merged = NULL, *output = NULL;
    ds4_gpu_tensor *cur = NULL, *tmp = NULL;
    const float    *bias = NULL, *position = NULL;
    const float *merger_bias = NULL, *output_bias = NULL;
    int ok = 0;
#define QWEN3VL_ALLOC(name_, count_) do { \
        name_ = ds4_gpu_tensor_alloc((count_) * sizeof(float)); \
        if (!(name_)) goto cleanup; \
    } while (0)
    QWEN3VL_ALLOC(patch_0, row768);
    QWEN3VL_ALLOC(patch_1, row768);
    QWEN3VL_ALLOC(a, row1152);
    QWEN3VL_ALLOC(b, row1152);
    QWEN3VL_ALLOC(qkv, row3456);
    QWEN3VL_ALLOC(attn, row1152);
    QWEN3VL_ALLOC(ffn, row4304);
    QWEN3VL_ALLOC(merged, merged4608);
    QWEN3VL_ALLOC(output, merged5120);
#undef QWEN3VL_ALLOC

    if (!ds4_gpu_tensor_write(patch_0, 0, patches_0,
                              row768 * sizeof(float)) ||
        !ds4_gpu_tensor_write(patch_1, 0, patches_1,
                              row768 * sizeof(float)) ||
        !ds4_gpu_begin_commands()) goto cleanup;
    ok = qwen3vl_patch_matmul_f16(
            a, model_map, model_size, weights->patch_weight_0, patch_0, rows);
    if (ok) ok = qwen3vl_patch_matmul_f16(
            b, model_map, model_size, weights->patch_weight_1, patch_1, rows);
    if (ok) {
        bias = qwen3vl_weight(model_map, model_size, weights->patch_bias,
                              QWEN3VL_WIDTH, "Qwen3-VL patch bias");
        position = qwen3vl_weight(model_map, model_size,
                                  weights->position_embedding,
                                  48u * 48u * QWEN3VL_WIDTH,
                                  "Qwen3-VL position embedding");
        if (!bias || !position) ok = 0;
    }
    if (ok) {
        qwen3vl_patch_pos_kernel<<<
            (unsigned)((row1152 + 255u) / 256u), 256u, 0,
            DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)a->ptr, (const float *)b->ptr, bias, position,
                rows, grid_h, grid_w);
        ok = qwen3vl_launch_ok("Qwen3-VL patch and position embedding");
    }

    cur = a;
    tmp = b;
    for (uint32_t il = 0; ok && il < DS4_QWEN3VL_VISION_LAYERS; il++) {
        const ds4_qwen3vl_vision_layer_weights *w = &weights->layer[il];
        const float *norm_w = qwen3vl_weight(
                model_map, model_size, w->norm1_weight, QWEN3VL_WIDTH,
                "Qwen3-VL norm1 weight");
        const float *norm_b = qwen3vl_weight(
                model_map, model_size, w->norm1_bias, QWEN3VL_WIDTH,
                "Qwen3-VL norm1 bias");
        if (!norm_w || !norm_b) { ok = 0; break; }
        qwen3vl_layernorm_kernel<<<rows, 1024u, 0,
            DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)tmp->ptr, (const float *)cur->ptr,
                norm_w, norm_b, rows, QWEN3VL_WIDTH, 1.0e-6f);
        ok = qwen3vl_launch_ok("Qwen3-VL norm1");
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                qkv, model_map, model_size, w->qkv_weight,
                QWEN3VL_WIDTH, QWEN3VL_QKV, tmp, rows);
        const float *qkv_bias = NULL;
        if (ok) {
            qkv_bias = qwen3vl_weight(
                    model_map, model_size, w->qkv_bias, QWEN3VL_QKV,
                    "Qwen3-VL QKV bias");
            if (!qkv_bias) ok = 0;
        }
        if (ok) {
            qwen3vl_bias_rope_kernel<<<dim3(rows, QWEN3VL_HEADS, 1u),
                QWEN3VL_HEAD_DIM, 0, DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)qkv->ptr, qkv_bias, rows, grid_w,
                    powf(10000.0f, -2.0f / 36.0f));
            ok = qwen3vl_launch_ok("Qwen3-VL QKV RoPE");
        }
        if (ok) {
            qwen3vl_attention_kernel<<<
                dim3((rows + QWEN3VL_ATTN_QUERY_WARPS - 1u) /
                         QWEN3VL_ATTN_QUERY_WARPS,
                     QWEN3VL_HEADS, 1u),
                QWEN3VL_ATTN_QUERY_WARPS * 32u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)attn->ptr, (const float *)qkv->ptr, rows);
            ok = qwen3vl_launch_ok("Qwen3-VL attention");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->attn_out_weight,
                QWEN3VL_WIDTH, QWEN3VL_WIDTH, attn, rows);
        const float *attn_bias = NULL;
        if (ok) {
            attn_bias = qwen3vl_weight(
                    model_map, model_size, w->attn_out_bias, QWEN3VL_WIDTH,
                    "Qwen3-VL attention bias");
            if (!attn_bias) ok = 0;
        }
        if (ok) {
            qwen3vl_bias_residual_kernel<<<
                (unsigned)((row1152 + 255u) / 256u), 256u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)tmp->ptr, attn_bias, (const float *)cur->ptr,
                    row1152, QWEN3VL_WIDTH);
            ok = qwen3vl_launch_ok("Qwen3-VL attention residual");
        }
        ds4_gpu_tensor *swap = cur; cur = tmp; tmp = swap;

        if (ok) {
            norm_w = qwen3vl_weight(
                    model_map, model_size, w->norm2_weight, QWEN3VL_WIDTH,
                    "Qwen3-VL norm2 weight");
            norm_b = qwen3vl_weight(
                    model_map, model_size, w->norm2_bias, QWEN3VL_WIDTH,
                    "Qwen3-VL norm2 bias");
            if (!norm_w || !norm_b) ok = 0;
        }
        if (ok) {
            qwen3vl_layernorm_kernel<<<rows, 1024u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm_w, norm_b, rows, QWEN3VL_WIDTH, 1.0e-6f);
            ok = qwen3vl_launch_ok("Qwen3-VL norm2");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                ffn, model_map, model_size, w->ffn_up_weight,
                QWEN3VL_WIDTH, QWEN3VL_FFN, tmp, rows);
        const float *up_bias = NULL;
        if (ok) {
            up_bias = qwen3vl_weight(
                    model_map, model_size, w->ffn_up_bias, QWEN3VL_FFN,
                    "Qwen3-VL FFN up bias");
            if (!up_bias) ok = 0;
        }
        if (ok) {
            qwen3vl_gelu_bias_kernel<<<
                (unsigned)((row4304 + 255u) / 256u), 256u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)ffn->ptr, up_bias, row4304, QWEN3VL_FFN);
            ok = qwen3vl_launch_ok("Qwen3-VL FFN GELU");
        }
        if (ok) ok = ds4_gpu_glm53_matmul_bf16(
                tmp, model_map, model_size, w->ffn_down_weight,
                QWEN3VL_FFN, QWEN3VL_WIDTH, ffn, rows);
        const float *down_bias = NULL;
        if (ok) {
            down_bias = qwen3vl_weight(
                    model_map, model_size, w->ffn_down_bias, QWEN3VL_WIDTH,
                    "Qwen3-VL FFN down bias");
            if (!down_bias) ok = 0;
        }
        if (ok) {
            qwen3vl_bias_residual_kernel<<<
                (unsigned)((row1152 + 255u) / 256u), 256u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)tmp->ptr, down_bias, (const float *)cur->ptr,
                    row1152, QWEN3VL_WIDTH);
            ok = qwen3vl_launch_ok("Qwen3-VL FFN residual");
        }
        swap = cur; cur = tmp; tmp = swap;
    }

    if (ok) {
        const float *norm_w = qwen3vl_weight(
                model_map, model_size, weights->post_norm_weight,
                QWEN3VL_WIDTH, "Qwen3-VL post norm weight");
        const float *norm_b = qwen3vl_weight(
                model_map, model_size, weights->post_norm_bias,
                QWEN3VL_WIDTH, "Qwen3-VL post norm bias");
        if (!norm_w || !norm_b) ok = 0;
        else {
            qwen3vl_layernorm_kernel<<<rows, 1024u, 0,
                DS4_QWEN3VL_VISION_STREAM>>>(
                    (float *)tmp->ptr, (const float *)cur->ptr,
                    norm_w, norm_b, rows, QWEN3VL_WIDTH, 1.0e-6f);
            ok = qwen3vl_launch_ok("Qwen3-VL post norm");
        }
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            merged, model_map, model_size, weights->merger_up_weight,
            QWEN3VL_MERGED, QWEN3VL_MERGED, tmp, merged_rows);
    if (ok) {
        merger_bias = qwen3vl_weight(
                model_map, model_size, weights->merger_up_bias,
                QWEN3VL_MERGED, "Qwen3-VL merger up bias");
        if (!merger_bias) ok = 0;
    }
    if (ok) {
        qwen3vl_gelu_bias_kernel<<<
            (unsigned)((merged4608 + 255u) / 256u), 256u, 0,
            DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)merged->ptr, merger_bias,
                merged4608, QWEN3VL_MERGED);
        ok = qwen3vl_launch_ok("Qwen3-VL merger GELU");
    }
    if (ok) ok = ds4_gpu_glm53_matmul_bf16(
            output, model_map, model_size, weights->merger_down_weight,
            QWEN3VL_MERGED, QWEN3VL_OUTPUT, merged, merged_rows);
    if (ok) {
        output_bias = qwen3vl_weight(
                model_map, model_size, weights->merger_down_bias,
                QWEN3VL_OUTPUT, "Qwen3-VL merger down bias");
        if (!output_bias) ok = 0;
    }
    if (ok) {
        qwen3vl_bias_kernel<<<
            (unsigned)((merged5120 + 255u) / 256u), 256u, 0,
            DS4_QWEN3VL_VISION_STREAM>>>(
                (float *)output->ptr, output_bias,
                merged5120, QWEN3VL_OUTPUT);
        ok = qwen3vl_launch_ok("Qwen3-VL merger down bias");
    }
    if (ds4_gpu_end_commands() == 0) ok = 0;
    if (ok) ok = ds4_gpu_tensor_read(
            output, 0, out, merged5120 * sizeof(float));

cleanup:
    ds4_gpu_tensor_free(output);
    ds4_gpu_tensor_free(merged);
    ds4_gpu_tensor_free(ffn);
    ds4_gpu_tensor_free(attn);
    ds4_gpu_tensor_free(qkv);
    ds4_gpu_tensor_free(b);
    ds4_gpu_tensor_free(a);
    ds4_gpu_tensor_free(patch_1);
    ds4_gpu_tensor_free(patch_0);
    return ok;
}

extern "C" int ds4_gpu_qwen3vl_vision_encode(
        float                            *out,
        const float                      *patches,
        uint32_t                          grid_h,
        uint32_t                          grid_w,
        const void                       *model_map,
        uint64_t                          model_size,
        const ds4_qwen3vl_vision_weights *weights) {
    return ds4_gpu_qwen3vl_vision_encode_pair(
        out, patches, patches, grid_h, grid_w,
        model_map, model_size, weights);
}

#undef DS4_QWEN3VL_VISION_STREAM
