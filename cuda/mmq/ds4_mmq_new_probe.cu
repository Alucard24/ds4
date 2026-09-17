/* Compile probe for the ported MMQ kernels.
 *
 * This translation unit exists to answer one question without touching the
 * working build: how much of the ggml runtime surface does the current upstream
 * MMQ need beyond what cuda/mmq/ds4_ggml_stubs.{h,cu} already provides?  It
 * includes the adapter and the ported headers and instantiates the dense case
 * for one type, so the compiler lists every missing symbol.
 *
 * When the port lands, this file becomes cuda/mmq/ds4_mmq_new.cu with a real
 * dense entry point; until then it deliberately stays out of the build.
 */
#include "ds4_ggml_stubs.h"
#include "new/mmq.cuh"

/* Force one instantiation: the dense case ds4 needs for IQ3_S at M=256..512. */
template void ds4n_mul_mat_q_case<GGML_TYPE_IQ3_S>(
        ggml_backend_cuda_context & ctx, const ds4n_mmq_args & args,
        cudaStream_t stream);
