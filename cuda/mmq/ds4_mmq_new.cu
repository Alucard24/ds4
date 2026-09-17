/* Dense MMQ entry for the ported (later upstream) kernels.
 *
 * Same C ABI and the same argument convention as ds4_mmq_quant_dense, so
 * ds4_cuda.cu can pick between the two revisions with one switch.  Differences
 * that matter:
 *
 * - the later revision takes its tile width, K iteration and SRAM layout from a
 *   per-architecture config table (ds4n_mmq_get_config) instead of choosing a
 *   tile at runtime, so the launch is dispatched through
 *   ds4n_mul_mat_q_case<type>();
 * - stream-k decomposition is a property of that config rather than a caller
 *   flag, so ds4n_mmq_args has no use_stream_k field;
 * - the activation quantizer signature is unchanged, so the vendored one is
 *   reused (quantize_mmq_q8_1_cuda).
 *
 * The body mirrors cuda/mmq/ds4_mmq.cu's dense entry on purpose: both must
 * produce the same numbers, and the mixed-IQ oracle compares them against the
 * ggml CPU dequantization.
 */
#include "ds4_ggml_stubs.h"
#include "ds4_mmq.h"

#include "new/mmq.cuh"

// The activation quantizer kept both its signature and its C++ linkage between
// the two revisions, so the vendored object provides the definition. Declaring
// it here (instead of including the vendored quantize.cuh) keeps this
// translation unit on the new tree's headers only: the two common.cuh revisions
// cannot coexist in one TU.
void quantize_mmq_q8_1_cuda(const float * x, const int32_t * ids, void * vy,
                            ggml_type type_src0, int64_t ne00, int64_t s01,
                            int64_t s02, int64_t s03, int64_t ne0, int64_t ne1,
                            int64_t ne2, int64_t ne3, cudaStream_t stream);

#include <cstdio>
#include <cstdint>

// ds4_mmq.cu keeps its helper in an anonymous namespace, so this translation
// unit creates its own context.  A second context is harmless: each one owns
// its pool, and every caller stays inside the revision it started in.
static ggml_backend_cuda_context * ds4_mmq_new_ctx(int device) {
    static ggml_backend_cuda_context * cached[GGML_CUDA_MAX_DEVICES] = {};
    if (device < 0 || device >= GGML_CUDA_MAX_DEVICES) return nullptr;
    if (!cached[device]) cached[device] = new ggml_backend_cuda_context(device);
    return cached[device];
}

template <ggml_type type>
static int ds4_mmq_new_dense_impl(const char * tag, const void * W,
                                  const float * X_f32, float * out_f32,
                                  int M, int N, int K, cudaStream_t stream) {
    if (!W || !X_f32 || !out_f32 || M <= 0 || N <= 0 || K <= 0) return -1;
    if (K % 256 != 0) {
        // mmq's inner tile loop expects K to be a multiple of QK_K = 256.
        fprintf(stderr, "%s: K=%d must be a multiple of 256\n", tag, K);
        return -1;
    }

    const int dev = ggml_cuda_get_device();
    const int cc  = ggml_cuda_info().devices[dev].cc;

    ggml_backend_cuda_context * ctx = ds4_mmq_new_ctx(dev);
    if (!ctx) {
        fprintf(stderr, "%s: failed to get cuda context for device %d\n", tag, dev);
        return -1;
    }
    // The pool's async alloc/free must be ordered on the kernels' stream.
    ds4_pool_set_stream(stream);

    const int64_t ne00        = K;
    const int64_t ne10_padded = GGML_PAD((int64_t)K, MATRIX_ROW_PADDING);
    const int64_t ne11        = N;
    const bool fallback = (M % 128) != 0;

    const size_t y_block_size       = sizeof(ds4n_block_q8_1_mmq);
    const size_t y_values_per_block = QK8_1_MMQ;
    const size_t nbytes_src1_q8_1 =
        ne11 * ne10_padded * y_block_size / y_values_per_block +
        ds4n_mmq_get_J_max(type, fallback, cc, ne11) * sizeof(ds4n_block_q8_1_mmq);

    ggml_cuda_pool_alloc<char> src1_q8_1_pool(ctx->pool(), nbytes_src1_q8_1);
    char *src1_q8_1 = src1_q8_1_pool.get();
    if (!src1_q8_1) {
        fprintf(stderr, "%s: failed to allocate %zu bytes of q8_1 scratch\n",
                tag, nbytes_src1_q8_1);
        return -1;
    }
    // The kernel reads the whole last column tile while the quantizer writes
    // only the valid columns, so the tail must be deterministic.
    (void)cudaMemsetAsync(src1_q8_1, 0, nbytes_src1_q8_1, stream);

    quantize_mmq_q8_1_cuda(
        X_f32, /*ids=*/nullptr, (void *)src1_q8_1, type,
        /*ne00=*/K, /*s01=*/(int64_t)K, /*s02=*/0, /*s03=*/0,
        /*ne0=*/ne10_padded, /*ne1=*/ne11, /*ne2=*/1, /*ne3=*/1, stream);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: quantize failed: %s\n", tag, cudaGetErrorString(err));
        return -2;
    }

    const int64_t blck = ggml_blck_size(type);
    const int64_t s01  = (int64_t)K / blck;   /* weight blocks per row */
    const int64_t s1   = (int64_t)M;          /* dst row stride, in floats */
    const int64_t s12  = ne11 * ne10_padded * y_block_size /
                         (y_values_per_block * sizeof(int));
    const int64_t s13  = s12;

    const ds4n_mmq_args args = {
        /*x=*/(const char *)W,
        /*type_x=*/type,
        /*y=*/(const int *)src1_q8_1,
        /*ids_dst=*/nullptr,
        /*expert_bounds=*/nullptr,
        /*dst=*/out_f32,
        /*y_scale=*/nullptr,
        /*ncols_x=*/ne00,   /*nrows_x=*/(int64_t)M,  /*ncols_dst=*/ne11,
        /*stride_row_x=*/s01, /*ncols_y=*/ne11,       /*nrows_dst=*/s1,
        /*nchannels_x=*/1,  /*nchannels_y=*/1,
        /*stride_channel_x=*/0, /*stride_channel_y=*/s12,
        /*stride_channel_dst=*/0,
        /*nsamples_x=*/1,   /*nsamples_y=*/1,
        /*stride_sample_x=*/0, /*stride_sample_y=*/s13,
        /*stride_sample_dst=*/0,
        /*ncols_max=*/ne11,
    };

    ds4n_mul_mat_q_case<type>(*ctx, args, stream);

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "%s: ds4n_mul_mat_q_case failed: %s\n", tag,
                cudaGetErrorString(err));
        return -3;
    }
    return 0;
}

extern "C" int ds4_mmq_quant_dense_new(
        const void  * W,
        uint32_t      weight_type,
        const float * X_f32,
        float       * out_f32,
        int           M,
        int           N,
        int           K,
        cudaStream_t  stream) {
    const char *tag = "ds4-mmq-new";
    switch ((ggml_type)weight_type) {
    case GGML_TYPE_Q2_K:    return ds4_mmq_new_dense_impl<GGML_TYPE_Q2_K>   (tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_Q4_K:    return ds4_mmq_new_dense_impl<GGML_TYPE_Q4_K>   (tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ2_XXS: return ds4_mmq_new_dense_impl<GGML_TYPE_IQ2_XXS>(tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ2_XS:  return ds4_mmq_new_dense_impl<GGML_TYPE_IQ2_XS> (tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ3_XXS: return ds4_mmq_new_dense_impl<GGML_TYPE_IQ3_XXS>(tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ3_S:   return ds4_mmq_new_dense_impl<GGML_TYPE_IQ3_S>  (tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ2_S:   return ds4_mmq_new_dense_impl<GGML_TYPE_IQ2_S>  (tag, W, X_f32, out_f32, M, N, K, stream);
    case GGML_TYPE_IQ4_XS:  return ds4_mmq_new_dense_impl<GGML_TYPE_IQ4_XS> (tag, W, X_f32, out_f32, M, N, K, stream);
    default:
        return -1;   /* caller falls back to the vendored revision */
    }
}
